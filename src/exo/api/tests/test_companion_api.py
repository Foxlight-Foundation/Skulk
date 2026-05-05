"""Tests for native companion pairing and read-only overview endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import exo.api.companion as companion_module
import exo.api.main as api_main
import exo.connectivity.remote_access as remote_access_module
from exo.api.companion import CompanionPairingManager
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


def _build_api(tmp_path: Path, node_id: str = "local-node") -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        NodeId(node_id),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )
    api._companion_pairing = CompanionPairingManager(  # pyright: ignore[reportPrivateUsage]
        node_id=api.node_id,
        key_path=tmp_path / "companion_cluster.key",
        credentials_path=tmp_path / "companion_credentials.json",
    )
    return api


def _tailscale_running() -> TailscaleStatus:
    return TailscaleStatus(
        running=True,
        self_ip="100.1.2.3",
        hostname="my-node",
        dns_name="my-node.tailnet-abc.ts.net",
        tailnet="tailnet-abc.ts.net",
        version="1.66.1",
    )


def _state_with_lan_ip(node_id: str, ip: str) -> State:
    iface = NetworkInterfaceInfo(name="en0", ip_address=ip)
    return State().model_copy(
        update={"node_network": {NodeId(node_id): NodeNetworkInfo(interfaces=[iface])}}
    )


def _non_loopback_request(_: Request) -> bool:
    return False


def test_create_pairing_session_returns_companion_qr_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    api.state = _state_with_lan_ip("local-node", "192.168.1.5")
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )

    response = client.post(
        "/v1/companion/pairing-sessions",
        json={"clusterName": "Kitchen Cluster"},
    )

    assert response.status_code == 200
    body = _json_object(response)
    payload = cast(dict[str, object], body["qrPayload"])
    assert payload["version"] == 1
    assert payload["clusterName"] == "Kitchen Cluster"
    assert payload["pairingNonce"]
    assert payload["clusterPublicKey"]
    assert payload["lanUrl"] == "http://192.168.1.5:52415"
    assert payload["tailscaleUrl"] == "http://my-node.tailnet-abc.ts.net:52415"
    assert str(payload["exchangeUrl"]).endswith("/exchange")


def test_pairing_session_creation_requires_operator_token_for_nonlocal_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(api_main, "_is_loopback_client", _non_loopback_request)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )

    response = client.post("/v1/companion/pairing-sessions", json={})

    assert response.status_code == 403
    assert (
        cast(dict[str, object], _json_object(response)["error"])["message"]
        == "operator_auth_required"
    )


def test_pairing_session_creation_rejects_forwarded_nonlocal_loopback_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )

    response = client.post(
        "/v1/companion/pairing-sessions",
        headers={"X-Forwarded-For": "203.0.113.10"},
        json={},
    )

    assert response.status_code == 403
    assert (
        cast(dict[str, object], _json_object(response)["error"])["message"]
        == "operator_auth_required"
    )


def test_pairing_session_creation_accepts_forwarded_nonlocal_request_with_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setenv("SKULK_COMPANION_PAIRING_TOKEN", "operator-token")
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )

    response = client.post(
        "/v1/companion/pairing-sessions",
        headers={
            "X-Forwarded-For": "203.0.113.10",
            "X-Skulk-Operator-Token": "operator-token",
        },
        json={},
    )

    assert response.status_code == 200


def test_pairing_session_creation_accepts_configured_operator_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(api_main, "_is_loopback_client", _non_loopback_request)
    monkeypatch.setenv("SKULK_COMPANION_PAIRING_TOKEN", "operator-token")
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )

    response = client.post(
        "/v1/companion/pairing-sessions",
        headers={"X-Skulk-Operator-Token": "operator-token"},
        json={},
    )

    assert response.status_code == 200
    assert cast(dict[str, object], _json_object(response)["qrPayload"])[
        "pairingNonce"
    ]


def test_companion_cluster_id_is_derived_from_persisted_cluster_key(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "companion_cluster.key"
    first = CompanionPairingManager(
        node_id=NodeId("first-node"),
        key_path=key_path,
        credentials_path=tmp_path / "first_credentials.json",
    )
    second = CompanionPairingManager(
        node_id=NodeId("second-node"),
        key_path=key_path,
        credentials_path=tmp_path / "second_credentials.json",
    )

    assert first.cluster_id == second.cluster_id


def test_pairing_exchange_returns_read_only_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )
    session_response = client.post("/v1/companion/pairing-sessions", json={})
    nonce = cast(
        str,
        cast(dict[str, object], _json_object(session_response)["qrPayload"])[
            "pairingNonce"
        ],
    )

    response = client.post(
        f"/v1/companion/pairing-sessions/{nonce}/exchange",
        json={"clientName": "Thomas iPhone"},
    )

    assert response.status_code == 200
    body = _json_object(response)
    assert body["token"]
    assert body["credentialId"]
    assert body["scopes"] == [
        "cluster:read",
        "nodes:read",
        "models:read",
        "events:read",
    ]


def test_pairing_nonce_is_single_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )
    session_response = client.post("/v1/companion/pairing-sessions", json={})
    nonce = cast(
        str,
        cast(dict[str, object], _json_object(session_response)["qrPayload"])[
            "pairingNonce"
        ],
    )

    first = client.post(f"/v1/companion/pairing-sessions/{nonce}/exchange", json={})
    second = client.post(f"/v1/companion/pairing-sessions/{nonce}/exchange", json={})

    assert first.status_code == 200
    assert second.status_code == 400
    assert (
        cast(dict[str, object], _json_object(second)["error"])["message"]
        == "already_used"
    )


def test_pairing_nonce_expiry_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    now = datetime(2026, 5, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(companion_module, "_utc_now", lambda: now)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )
    session_response = client.post("/v1/companion/pairing-sessions", json={})
    nonce = cast(
        str,
        cast(dict[str, object], _json_object(session_response)["qrPayload"])[
            "pairingNonce"
        ],
    )
    monkeypatch.setattr(
        companion_module,
        "_utc_now",
        lambda: now + timedelta(minutes=6),
    )

    response = client.post(f"/v1/companion/pairing-sessions/{nonce}/exchange", json={})

    assert response.status_code == 400
    assert (
        cast(dict[str, object], _json_object(response)["error"])["message"]
        == "expired_code"
    )


def test_companion_overview_requires_bearer_token(tmp_path: Path) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)

    response = client.get("/v1/companion/overview")

    assert response.status_code == 401
    assert (
        cast(dict[str, object], _json_object(response)["error"])["message"]
        == "missing_token"
    )


def test_companion_overview_returns_safe_read_only_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    api.state = _state_with_lan_ip("local-node", "192.168.1.5")
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )
    session_response = client.post("/v1/companion/pairing-sessions", json={})
    nonce = cast(
        str,
        cast(dict[str, object], _json_object(session_response)["qrPayload"])[
            "pairingNonce"
        ],
    )
    exchange_response = client.post(
        f"/v1/companion/pairing-sessions/{nonce}/exchange",
        json={},
    )
    token = cast(str, _json_object(exchange_response)["token"])

    response = client.get(
        "/v1/companion/overview",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = _json_object(response)
    assert set(body) == {
        "cluster",
        "connection",
        "nodes",
        "runningModels",
        "recentEvents",
    }
    assert cast(dict[str, object], body["connection"])["authenticated"] is True
    nodes = cast(list[dict[str, object]], body["nodes"])
    assert nodes[0]["nodeId"] == "local-node"
    assert "admin" not in body


def test_revoked_companion_credential_cannot_authenticate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )
    session_response = client.post("/v1/companion/pairing-sessions", json={})
    nonce = cast(
        str,
        cast(dict[str, object], _json_object(session_response)["qrPayload"])[
            "pairingNonce"
        ],
    )
    exchange_response = client.post(
        f"/v1/companion/pairing-sessions/{nonce}/exchange",
        json={},
    )
    exchange_body = _json_object(exchange_response)
    token = cast(str, exchange_body["token"])
    credential_id = cast(str, exchange_body["credentialId"])
    manager = api._companion_pairing  # pyright: ignore[reportPrivateUsage]
    assert manager.revoke_credential(credential_id) is True

    response = client.get(
        "/v1/companion/overview",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert (
        cast(dict[str, object], _json_object(response)["error"])["message"]
        == "auth_failed"
    )


def test_companion_credential_hash_survives_manager_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    client = TestClient(api.app)
    monkeypatch.setattr(
        remote_access_module,
        "query_tailscale_status",
        AsyncMock(return_value=_tailscale_running()),
    )
    session_response = client.post("/v1/companion/pairing-sessions", json={})
    nonce = cast(
        str,
        cast(dict[str, object], _json_object(session_response)["qrPayload"])[
            "pairingNonce"
        ],
    )
    exchange_response = client.post(
        f"/v1/companion/pairing-sessions/{nonce}/exchange",
        json={},
    )
    token = cast(str, _json_object(exchange_response)["token"])

    restarted = CompanionPairingManager(
        node_id=api.node_id,
        key_path=tmp_path / "companion_cluster.key",
        credentials_path=tmp_path / "companion_credentials.json",
    )

    assert restarted.authenticate_bearer(token).scopes == (
        "cluster:read",
        "nodes:read",
        "models:read",
        "events:read",
    )


def test_corrupted_companion_credential_store_does_not_block_startup(
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "companion_credentials.json"
    credentials_path.write_text("{")

    manager = CompanionPairingManager(
        node_id=NodeId("local-node"),
        key_path=tmp_path / "companion_cluster.key",
        credentials_path=credentials_path,
    )

    assert manager.revoke_credential("missing") is False
