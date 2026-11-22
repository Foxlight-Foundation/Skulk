"""Original owner identity comes from the actual Core HTTP boundary."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from skulk.api.operator_auth import authenticate_capability_owner
from skulk.api.operator_gateway import OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY
from skulk.shared.types.common import NodeId


def _application(*, relay_authorized: bool = False) -> FastAPI:
    app = FastAPI()

    async def peer_verified(address: str) -> bool:
        return address == "100.101.102.103"

    async def identity(request: Request) -> dict[str, str]:
        if relay_authorized:
            request.scope[OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY] = True
        caller = await authenticate_capability_owner(
            request, NodeId("actual-serving-node"), peer_verified
        )
        return {
            "identifier": caller.identifier,
            "kind": caller.kind,
            "issuer_node_id": str(caller.issuer_node_id),
        }

    app.post("/identity")(identity)
    return app


@pytest.mark.parametrize(
    ("peer", "origin"),
    [
        ("127.0.0.1", "http://127.0.0.1:52415"),
        ("::1", "http://[::1]:52415"),
        ("100.101.102.103", "https://node.example.ts.net"),
    ],
)
def test_owner_identity_is_issued_by_the_serving_host(
    peer: str, origin: str
) -> None:
    """Verified direct requests cannot choose a different caller or issuer."""
    client = TestClient(
        _application(), base_url=origin, client=(peer, 50000)
    )
    response = client.post(
        "/identity",
        headers={
            "Origin": origin,
            "X-Skulk-Dashboard": "pairing-v1",
            "X-Skulk-Caller-Id": "another-caller",
            "X-Skulk-Caller-Kind": "service",
            "X-Skulk-Issuer-Node-Id": "another-node",
        },
        json={"caller_id": "another-caller", "issuer_node_id": "another-node"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "identifier": "direct-owner",
        "kind": "operator",
        "issuer_node_id": "actual-serving-node",
    }


@pytest.mark.parametrize(
    ("peer", "origin", "extra_headers"),
    [
        ("192.0.2.20", "http://127.0.0.1:52415", {}),
        ("100.101.102.104", "https://node.example.ts.net", {}),
        ("127.0.0.1", "http://example.invalid", {}),
        ("127.0.0.1", "http://127.0.0.1:52415", {"Forwarded": "for=127.0.0.1"}),
        ("127.0.0.1", "http://127.0.0.1:52415", {"X-Forwarded-For": "127.0.0.1"}),
        ("127.0.0.1", "http://127.0.0.1:52415", {"Authorization": "Bearer paired"}),
    ],
)
def test_untrusted_requests_cannot_establish_an_owner(
    peer: str, origin: str, extra_headers: dict[str, str]
) -> None:
    """Caller claims cannot upgrade LAN, proxy, unverified or paired access."""
    client = TestClient(
        _application(), base_url=origin, client=(peer, 50000)
    )
    response = client.post(
        "/identity",
        headers={
            "Origin": origin,
            "X-Skulk-Dashboard": "pairing-v1",
            "X-Skulk-Caller-Id": "direct-owner",
            **extra_headers,
        },
        json={"caller_kind": "operator"},
    )
    assert response.status_code == 403
    assert "actual-serving-node" not in response.text


def test_relay_authentication_does_not_become_direct_owner_identity() -> None:
    """A real relay admission marker never inherits the node owner's realm."""
    origin = "http://127.0.0.1:52415"
    client = TestClient(
        _application(relay_authorized=True),
        base_url=origin,
        client=("127.0.0.1", 50000),
    )
    response = client.post(
        "/identity",
        headers={"Origin": origin, "X-Skulk-Dashboard": "pairing-v1"},
    )
    assert response.status_code == 403
