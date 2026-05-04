"""Tests for connectivity endpoints: Tailscale status and remote access info."""

from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import exo.api.main as api_main
import exo.connectivity.remote_access as remote_access_module
from exo.api.main import API
from exo.connectivity.tailscale import TailscaleStatus
from exo.shared.election import ElectionMessage
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.state import State
from exo.utils.channels import channel


def _json_object(response: httpx.Response) -> dict[str, object]:
    return cast(dict[str, object], cast(object, response.json()))


def _build_api(node_id: str = "local-node") -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId(node_id),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def _tailscale_running(
    *,
    ip: str = "100.1.2.3",
    dns_name: str | None = "my-node.tailnet-abc.ts.net",
) -> TailscaleStatus:
    return TailscaleStatus(
        running=True,
        self_ip=ip,
        hostname="my-node",
        dns_name=dns_name,
        tailnet="tailnet-abc.ts.net" if dns_name else None,
        version="1.66.1",
    )


def _tailscale_not_running() -> TailscaleStatus:
    return TailscaleStatus(running=False)


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stub for proxy tests."""

    def __init__(self, responses: dict[str, httpx.Response]) -> None:
        self._responses = responses

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, _et: object, _exc: object, _tb: object) -> None:
        return None

    async def get(self, url: str) -> httpx.Response:
        return self._responses[url]


def _state_with_lan_ip(node_id: str, ip: str) -> State:
    """Minimal state with one LAN interface for node_id."""
    iface = NetworkInterfaceInfo(name="en0", ip_address=ip)
    return State().model_copy(
        update={"node_network": {NodeId(node_id): NodeNetworkInfo(interfaces=[iface])}}
    )


# ── GET /v1/connectivity/tailscale (local) ──────────────────────────────────


def test_tailscale_local_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local Tailscale status returned directly when no node_id given."""

    api = _build_api()
    client = TestClient(api.app)

    # api/main.py imports query_tailscale_status by name — patch the binding there
    monkeypatch.setattr(api_main, "query_tailscale_status", AsyncMock(return_value=_tailscale_running()))

    response = client.get("/v1/connectivity/tailscale")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["running"] is True
    assert body["selfIp"] == "100.1.2.3"
    assert body["dnsName"] == "my-node.tailnet-abc.ts.net"


def test_tailscale_local_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns running=false when tailscaled is not running."""

    api = _build_api()
    client = TestClient(api.app)

    monkeypatch.setattr(api_main, "query_tailscale_status", AsyncMock(return_value=_tailscale_not_running()))

    response = client.get("/v1/connectivity/tailscale")

    assert response.status_code == 200
    assert _json_object(response)["running"] is False


# ── GET /v1/connectivity/tailscale?node_id= (proxy) ────────────────────────


def test_tailscale_proxy_local_node_id_is_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing the local node's own ID does not proxy — returns local status."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    monkeypatch.setattr(api_main, "query_tailscale_status", AsyncMock(return_value=_tailscale_running()))

    response = client.get("/v1/connectivity/tailscale?node_id=local-node")

    assert response.status_code == 200
    assert _json_object(response)["running"] is True


def test_tailscale_proxy_unreachable_node_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown node_id that is not in reachable peers returns 404."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _no_peers() -> dict[str, str]:
        return {}

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _no_peers)

    response = client.get("/v1/connectivity/tailscale?node_id=ghost-node")

    assert response.status_code == 404
    assert "ghost-node" in response.text


def test_tailscale_proxy_forwards_to_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxied request returns the peer node's Tailscale status."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _peers() -> dict[str, str]:
        return {"peer-node": "http://peer-node:52415"}

    peer_status = _tailscale_running(ip="100.9.9.9", dns_name="peer.tailnet-abc.ts.net")

    def _build_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient(
            {
                "http://peer-node:52415/v1/connectivity/tailscale": httpx.Response(
                    200,
                    json=peer_status.model_dump(mode="json", by_alias=True),
                    request=httpx.Request("GET", "http://peer-node:52415/v1/connectivity/tailscale"),
                )
            }
        )

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _peers)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_client)

    response = client.get("/v1/connectivity/tailscale?node_id=peer-node")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["selfIp"] == "100.9.9.9"
    assert body["dnsName"] == "peer.tailnet-abc.ts.net"


def test_tailscale_proxy_peer_error_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-2xx response from peer is forwarded with the peer's status code."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _peers() -> dict[str, str]:
        return {"peer-node": "http://peer-node:52415"}

    def _build_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient(
            {
                "http://peer-node:52415/v1/connectivity/tailscale": httpx.Response(
                    503,
                    json={"detail": "unavailable"},
                    request=httpx.Request("GET", "http://peer-node:52415/v1/connectivity/tailscale"),
                )
            }
        )

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _peers)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_client)

    response = client.get("/v1/connectivity/tailscale?node_id=peer-node")

    assert response.status_code == 503


# ── GET /v1/connectivity/remote-access ─────────────────────────────────────


def test_remote_access_tailscale_running_uses_magic_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preferredUrl and operatorUrl use MagicDNS name when Tailscale is running."""

    api = _build_api("local-node")
    api.state = _state_with_lan_ip("local-node", "192.168.1.5")
    client = TestClient(api.app)

    # remote_access.py imports query_tailscale_status by name — patch there
    monkeypatch.setattr(remote_access_module, "query_tailscale_status", AsyncMock(return_value=_tailscale_running()))

    response = client.get("/v1/connectivity/remote-access")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["preferredUrl"] == "http://my-node.tailnet-abc.ts.net:52415"
    assert body["operatorUrl"] == "http://my-node.tailnet-abc.ts.net:52415/operator"
    assert cast(dict[str, object], body["tailscale"])["running"] is True
    assert cast(dict[str, object], body["local"])["ip"] == "192.168.1.5"


def test_remote_access_tailscale_running_no_dns_falls_back_to_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MagicDNS is absent, preferredUrl uses the raw 100.x.x.x IP."""

    api = _build_api("local-node")
    api.state = _state_with_lan_ip("local-node", "192.168.1.5")
    client = TestClient(api.app)

    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running(ip="100.1.2.3", dns_name=None)),
    )

    response = client.get("/v1/connectivity/remote-access")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["preferredUrl"] == "http://100.1.2.3:52415"
    assert body["operatorUrl"] == "http://100.1.2.3:52415/operator"


def test_remote_access_tailscale_not_running_falls_back_to_lan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Tailscale is not running, preferredUrl is the LAN address."""

    api = _build_api("local-node")
    api.state = _state_with_lan_ip("local-node", "10.0.0.42")
    client = TestClient(api.app)

    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_not_running()),
    )

    response = client.get("/v1/connectivity/remote-access")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["preferredUrl"] == "http://10.0.0.42:52415"
    assert body["operatorUrl"] == "http://10.0.0.42:52415/operator"
    assert cast(dict[str, object], body["tailscale"])["running"] is False


def test_remote_access_no_network_info_returns_null_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both URLs are null when Tailscale is off and no LAN IP is known."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_not_running()),
    )

    response = client.get("/v1/connectivity/remote-access")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["preferredUrl"] is None
    assert body["operatorUrl"] is None


def test_remote_access_response_uses_camel_case_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON keys in the response are camelCase (aliased), not snake_case."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_not_running()),
    )

    response = client.get("/v1/connectivity/remote-access")

    assert response.status_code == 200
    body = _json_object(response)
    assert "preferredUrl" in body
    assert "operatorUrl" in body
    assert "local" in body
    assert "tailscale" in body
    ts = cast(dict[str, object], body["tailscale"])
    assert "dnsName" in ts
    assert "dns_name" not in ts
