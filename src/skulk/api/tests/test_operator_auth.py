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
