"""HTTP contract tests for host-authorized operator pairing."""

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skulk.api.operator_auth import create_operator_auth_router
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import (
    OperatorPairingService,
    pairing_invitation_signature_message,
    pairing_signature_message,
)
from skulk.operator.relay import (
    OperatorRelayConfigurationRepository,
    OperatorRelayProvisioning,
)


async def _verified_tailnet_peer(peer_host: str) -> bool:
    """Stand in for tailscaled membership proof in HTTP boundary tests."""

    return peer_host in {"100.101.102.103", "fd7a:115c:a1e0::1234"}


def _base64url(value: bytes) -> str:
    """Encode bytes for pairing JSON payloads."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _dashboard_invitation_service(
    tmp_path: Path,
) -> OperatorPairingService:
    """Return one relay-configured service suitable for dashboard invitations."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "dashboard-key.bin")
    store = EncryptedAuthorityStore(provider, tmp_path / "dashboard-authority.sqlite3")
    repository = OperatorRelayConfigurationRepository(
        store,
        certificate_path=tmp_path / "dashboard-relay-cert.pem",
        private_key_path=tmp_path / "dashboard-relay-key.pem",
    )
    service = OperatorPairingService(
        store,
        provider,
        relay_repository=repository,
        now=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
    )
    service.configure_relay(
        OperatorRelayProvisioning(
            version=1,
            app_websocket_url="wss://relay.example.invalid/v1/carrier/app",
            gateway_websocket_url="wss://relay.example.invalid/v1/carrier/gateway",
            routing_locator=_base64url(bytes(range(32))),
            app_carrier_credential=_base64url(bytes(range(32, 64))),
            gateway_carrier_credential=_base64url(bytes(range(64, 96))),
            lane_count=1,
        ),
        operator_api_port=52416,
        cluster_name="Fox Den",
    )
    return service


def test_dashboard_can_create_list_and_revoke_pairing_invitation(
    tmp_path: Path,
) -> None:
    """The node-local dashboard owns the complete invitation lifecycle."""

    service = _dashboard_invitation_service(tmp_path)
    app = FastAPI()
    app.include_router(
        create_operator_auth_router(
            service,
            tailnet_peer_verifier=_verified_tailnet_peer,
        )
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1:52415",
        client=("127.0.0.1", 50000),
    )
    headers = {
        "Origin": "http://127.0.0.1:52415",
        "X-Skulk-Dashboard": "pairing-v1",
    }

    created = client.post(
        "/v1/auth/pairing-invitations",
        headers=headers,
        json={"validForSeconds": 7 * 24 * 60 * 60, "maxPairings": 3},
    )
    assert created.status_code == 200
    assert created.headers["cache-control"] == "no-store, max-age=0"
    assert created.headers["pragma"] == "no-cache"
    body = cast(dict[str, object], created.json())
    pairing_code = body["pairingCode"]
    assert isinstance(pairing_code, str)
    assert pairing_code.startswith("skulk://pair?z=")
    invitation = cast(dict[str, object], body["invitation"])
    assert invitation["maxPairings"] == 3
    invitation_id = str(invitation["invitationId"])

    listed = client.get("/v1/auth/pairing-invitations", headers=headers)
    assert listed.status_code == 200
    listed_body = cast(list[dict[str, object]], listed.json())
    assert listed_body == [invitation]
    assert pairing_code not in listed.text
    assert "nonce" not in listed.text.casefold()

    revoked = client.delete(
        f"/v1/auth/pairing-invitations/{invitation_id}",
        headers=headers,
    )
    assert revoked.status_code == 204
    relisted = cast(
        list[dict[str, object]],
        client.get("/v1/auth/pairing-invitations", headers=headers).json(),
    )
    assert relisted[0]["state"] == "revoked"


def test_dashboard_invitation_management_requires_direct_browser_authority(
    tmp_path: Path,
) -> None:
    """Loopback and same-origin tailnet browsers work while other peers do not."""

    service = _dashboard_invitation_service(tmp_path)
    app = FastAPI()
    app.include_router(
        create_operator_auth_router(
            service,
            tailnet_peer_verifier=_verified_tailnet_peer,
        )
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1:52415",
        client=("127.0.0.1", 50000),
    )
    payload = {"validForSeconds": 300, "maxPairings": 1}

    assert client.post("/v1/auth/pairing-invitations", json=payload).status_code == 403
    assert (
        client.post(
            "/v1/auth/pairing-invitations",
            headers={
                "Origin": "https://hostile.example",
                "X-Skulk-Dashboard": "pairing-v1",
            },
            json=payload,
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/auth/pairing-invitations",
            headers={
                "Origin": "http://127.0.0.1:52415",
                "X-Skulk-Dashboard": "pairing-v1",
                "X-Forwarded-For": "192.0.2.10",
            },
            json=payload,
        ).status_code
        == 403
    )
    listed = client.get(
        "/v1/auth/pairing-invitations",
        headers={
            "Referer": "http://127.0.0.1:52415/cluster",
            "X-Skulk-Dashboard": "pairing-v1",
        },
    )
    assert listed.status_code == 200

    tailnet_client = TestClient(
        app,
        base_url="http://kite1:52415",
        client=("100.101.102.103", 50000),
    )
    tailnet_listed = tailnet_client.get(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "http://kite1:52415",
            "X-Skulk-Dashboard": "pairing-v1",
        },
    )
    assert tailnet_listed.status_code == 200

    tailnet_fqdn_client = TestClient(
        app,
        base_url="https://kite1.example-tailnet.ts.net:52415",
        client=("fd7a:115c:a1e0::1234", 50000),
    )
    assert (
        tailnet_fqdn_client.get(
            "/v1/auth/pairing-invitations",
            headers={
                "Origin": "https://kite1.example-tailnet.ts.net:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
        ).status_code
        == 200
    )

    tailnet_literal_client = TestClient(
        app,
        base_url="http://100.110.120.125:52415",
        client=("100.101.102.103", 50000),
    )
    assert (
        tailnet_literal_client.get(
            "/v1/auth/pairing-invitations",
            headers={
                "Origin": "http://100.110.120.125:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
        ).status_code
        == 200
    )

    cross_origin = tailnet_client.post(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "https://hostile.example",
            "X-Skulk-Dashboard": "pairing-v1",
        },
        json=payload,
    )
    assert cross_origin.status_code == 403

    mismatched_port = tailnet_client.post(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "http://kite1:52416",
            "X-Skulk-Dashboard": "pairing-v1",
        },
        json=payload,
    )
    assert mismatched_port.status_code == 403

    lan_client = TestClient(
        app,
        base_url="http://kite1:52415",
        client=("192.168.1.20", 50000),
    )
    blocked_lan = lan_client.post(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "http://kite1:52415",
            "X-Skulk-Dashboard": "pairing-v1",
        },
        json=payload,
    )
    assert blocked_lan.status_code == 403

    unverified_cgnat_client = TestClient(
        app,
        base_url="http://kite1:52415",
        client=("100.64.0.9", 50000),
    )
    blocked_cgnat = unverified_cgnat_client.post(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "http://kite1:52415",
            "X-Skulk-Dashboard": "pairing-v1",
        },
        json=payload,
    )
    assert blocked_cgnat.status_code == 403
    assert "Tailscale" in blocked_lan.text
    assert "localhost" in blocked_lan.text
    assert "ordinary LAN access" in blocked_lan.text


def test_dashboard_invitation_creation_requires_configured_gateway(
    tmp_path: Path,
) -> None:
    """A non-gateway dashboard cannot mint unusable remote invitations."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "unconfigured-key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "unconfigured-authority.sqlite3"),
        provider,
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))
    client = TestClient(
        app,
        base_url="http://127.0.0.1:52415",
        client=("127.0.0.1", 50000),
    )

    response = client.post(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "http://127.0.0.1:52415",
            "X-Skulk-Dashboard": "pairing-v1",
        },
        json={"validForSeconds": 300, "maxPairings": 1},
    )

    assert response.status_code == 409
    assert "configured operator gateway" in response.text
    assert "Tailscale or localhost" in response.text
    assert "pairingCode" not in response.text

    listed = client.get(
        "/v1/auth/pairing-invitations",
        headers={
            "Origin": "http://127.0.0.1:52415",
            "X-Skulk-Dashboard": "pairing-v1",
        },
    )
    assert listed.status_code == 409
    assert "configured operator gateway" in listed.text


def _pair_device(
    client: TestClient,
    service: OperatorPairingService,
    *,
    device_name: str = "Test phone",
) -> dict[str, object]:
    """Pair one generated device through the public HTTP contract."""

    package = service.create_session(exchange_url="https://example.invalid")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    challenge_response = client.post(
        "/v1/auth/pairing-sessions/challenge",
        json={
            "nonce": package.nonce,
            "deviceName": device_name,
            "devicePublicKey": _base64url(public_key),
        },
    )
    assert challenge_response.status_code == 200
    challenge_body = cast(dict[str, object], challenge_response.json())
    signature = private_key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=str(challenge_body["challenge"]),
        )
    )
    exchange_response = client.post(
        "/v1/auth/pairing-sessions/exchange",
        json={"nonce": package.nonce, "signature": _base64url(signature)},
    )
    assert exchange_response.status_code == 200
    return cast(dict[str, object], exchange_response.json())


def test_pairing_routes_issue_credentials_and_reject_reuse(tmp_path: Path) -> None:
    """The HTTP path mirrors the single-use service state machine."""

    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"),
        provider,
        now=lambda: now,
    )
    package = service.create_session(exchange_url="https://example.invalid")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))
    client = TestClient(app)

    challenge_response = client.post(
        "/v1/auth/pairing-sessions/challenge",
        json={
            "nonce": package.nonce,
            "deviceName": "Test phone",
            "devicePublicKey": _base64url(public_key),
        },
    )
    assert challenge_response.status_code == 200
    challenge_body = cast(dict[str, object], challenge_response.json())
    challenge = str(challenge_body["challenge"])
    signature = private_key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=challenge,
        )
    )

    exchange_response = client.post(
        "/v1/auth/pairing-sessions/exchange",
        json={"nonce": package.nonce, "signature": _base64url(signature)},
    )
    assert exchange_response.status_code == 200
    exchange_body = cast(dict[str, object], exchange_response.json())
    assert isinstance(exchange_body["accessToken"], str)
    assert isinstance(exchange_body["refreshToken"], str)

    reused_response = client.post(
        "/v1/auth/pairing-sessions/exchange",
        json={"nonce": package.nonce, "signature": _base64url(signature)},
    )
    assert reused_response.status_code == 409


def test_reusable_invitation_routes_return_and_require_attempt_identity(
    tmp_path: Path,
) -> None:
    """Version-three exchanges bind credentials to one independent HTTP attempt."""

    now = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"),
        provider,
        now=lambda: now,
    )
    package = service.create_invitation(
        lifetime=timedelta(days=90),
        exchange_url="https://example.invalid",
    )
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))
    client = TestClient(app)

    challenged = client.post(
        "/v1/auth/pairing-sessions/challenge",
        json={
            "nonce": package.nonce,
            "invitationId": str(package.invitation_id),
            "deviceName": "Review phone",
            "devicePublicKey": _base64url(public_key),
        },
    )
    assert challenged.status_code == 200, challenged.text
    challenge_body = cast(dict[str, object], challenged.json())
    attempt_id = UUID(str(challenge_body["attemptId"]))
    signature = private_key.sign(
        pairing_invitation_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            invitation_id=package.invitation_id,
            nonce=package.nonce,
            attempt_id=attempt_id,
            challenge=str(challenge_body["challenge"]),
        )
    )
    exchanged = client.post(
        "/v1/auth/pairing-sessions/exchange",
        json={
            "nonce": package.nonce,
            "invitationId": str(package.invitation_id),
            "attemptId": str(attempt_id),
            "signature": _base64url(signature),
        },
    )
    assert exchanged.status_code == 200, exchanged.text
    assert client.post(
        "/v1/auth/pairing-sessions/exchange",
        json={
            "nonce": package.nonce,
            "invitationId": str(package.invitation_id),
            "attemptId": str(attempt_id),
            "signature": _base64url(signature),
        },
    ).status_code == 409


def test_reusable_invitation_active_attempt_capacity_returns_retry_after(
    tmp_path: Path,
) -> None:
    """The public challenge route exposes temporary capacity as HTTP 429."""

    now = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"),
        provider,
        now=lambda: now,
    )
    package = service.create_invitation(
        lifetime=timedelta(days=1),
        exchange_url="https://example.invalid",
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))
    client = TestClient(app)
    for index in range(11):
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        response = client.post(
            "/v1/auth/pairing-sessions/challenge",
            json={
                "nonce": package.nonce,
                "invitationId": str(package.invitation_id),
                "deviceName": f"Review phone {index}",
                "devicePublicKey": _base64url(public_key),
            },
        )
        if index < 10:
            assert response.status_code == 200
        else:
            assert response.status_code == 429
            assert response.headers["retry-after"] == "300"


def test_pairing_routes_are_present_in_openapi(tmp_path: Path) -> None:
    """Every pairing and invitation route remains visible in the API contract."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"),
        provider,
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))

    paths = cast(dict[str, object], app.openapi()["paths"])
    assert "/v1/auth/pairing-sessions/challenge" in paths
    assert "/v1/auth/pairing-sessions/exchange" in paths
    assert "/v1/auth/token" in paths
    assert "/v1/auth/devices" in paths
    assert "/v1/auth/devices/{device_id}" in paths
    assert "/v1/auth/pairing-invitations" in paths
    assert "/v1/auth/pairing-invitations/{invitation_id}" in paths


def test_token_rotation_and_device_revocation_http_contract(tmp_path: Path) -> None:
    """Bearer routes rotate credentials and make revocation immediate."""

    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"),
        provider,
        now=lambda: now,
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))
    client = TestClient(app)
    paired = _pair_device(client, service)
    device_id = str(paired["deviceId"])
    access_token = str(paired["accessToken"])
    refresh_token = str(paired["refreshToken"])

    missing_bearer = client.get("/v1/auth/devices")
    assert missing_bearer.status_code == 401
    assert missing_bearer.headers["www-authenticate"] == "Bearer"

    listed = client.get(
        "/v1/auth/devices",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert listed.status_code == 200
    listed_body = cast(dict[str, object], listed.json())
    assert isinstance(listed_body["devices"], list)

    rotated = client.post(
        "/v1/auth/token",
        json={"deviceId": device_id, "refreshToken": refresh_token},
    )
    assert rotated.status_code == 200, rotated.text
    rotated_body = cast(dict[str, object], rotated.json())
    next_access_token = str(rotated_body["accessToken"])

    assert client.get(
        "/v1/auth/devices",
        headers={"Authorization": f"Bearer {access_token}"},
    ).status_code == 401
    assert client.post(
        "/v1/auth/token",
        json={"deviceId": device_id, "refreshToken": refresh_token},
    ).status_code == 401

    revoked = client.delete(
        f"/v1/auth/devices/{device_id}",
        headers={"Authorization": f"Bearer {next_access_token}"},
    )
    assert revoked.status_code == 204
    assert client.get(
        "/v1/auth/devices",
        headers={"Authorization": f"Bearer {next_access_token}"},
    ).status_code == 401


def test_lifecycle_routes_fail_closed_on_an_uninitialized_gateway(
    tmp_path: Path,
) -> None:
    """A non-designated API node returns 503 instead of leaking an error 500."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    service = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"),
        provider,
    )
    app = FastAPI()
    app.include_router(create_operator_auth_router(service))
    client = TestClient(app)

    listed = client.get(
        "/v1/auth/devices",
        headers={"Authorization": "Bearer unknown-access-token"},
    )
    assert listed.status_code == 503
    assert listed.json() == {"detail": "operator gateway is not initialized"}
