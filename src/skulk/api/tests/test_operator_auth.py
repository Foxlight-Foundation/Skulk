"""HTTP contract tests for host-authorized operator pairing."""

import base64
from datetime import datetime, timezone
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
    pairing_signature_message,
)


def _base64url(value: bytes) -> str:
    """Encode bytes for pairing JSON payloads."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


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


def test_pairing_routes_are_present_in_openapi(tmp_path: Path) -> None:
    """Both HTTP operations remain visible in the generated API contract."""

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
