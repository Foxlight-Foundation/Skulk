"""Real authority and generated-response checks for isolated app observation."""

import base64
from pathlib import Path
from typing import cast
from uuid import UUID

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from bench.operator_fixture_app import create_fixture_app, generated_responses
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import OperatorPairingService, pairing_signature_message


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _service(path: Path) -> OperatorPairingService:
    provider = LocalFileAuthorityKeyProvider(path / "key.bin")
    return OperatorPairingService(
        EncryptedAuthorityStore(provider, path / "authority.sqlite3"), provider
    )


def _pair(client: TestClient, service: OperatorPairingService) -> dict[str, object]:
    package = service.create_session(exchange_url="http://127.0.0.1:52415")
    key = Ed25519PrivateKey.generate()
    challenge = client.post(
        "/v1/auth/pairing-sessions/challenge",
        json={
            "nonce": package.nonce,
            "deviceName": "Generated test client",
            "devicePublicKey": _base64url(
                key.public_key().public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw
                )
            ),
        },
    )
    assert challenge.status_code == 200
    challenge_body = cast(dict[str, object], challenge.json())
    proof = key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=str(challenge_body["challenge"]),
        )
    )
    exchange = client.post(
        "/v1/auth/pairing-sessions/exchange",
        json={"nonce": package.nonce, "signature": _base64url(proof)},
    )
    assert exchange.status_code == 200
    return cast(dict[str, object], exchange.json())


def test_generated_bodies_are_fresh_and_explicitly_synthetic() -> None:
    """Mutable responses cannot bleed between fixture sessions."""
    first = generated_responses()
    first["/state"].clear()
    second = generated_responses()
    assert second["/state"]["topology"] == {
        "nodes": ["synthetic-node"],
        "connections": {},
    }
    assert len(second) == 6


def test_pairing_refresh_and_revocation_use_real_authority(tmp_path: Path) -> None:
    """Canonical fixture data stays behind real bearer lifecycle validation."""
    service = _service(tmp_path)
    client = TestClient(create_fixture_app(service))
    assert client.get("/state").status_code in {401, 503}
    credentials = _pair(client, service)
    headers = {"Authorization": f"Bearer {credentials['accessToken']}"}
    for path, expected in generated_responses().items():
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert response.json() == expected
    assert (
        client.get("/v1/auth/pairing-invitations", headers=headers).status_code == 404
    )
    refreshed = client.post(
        "/v1/auth/token",
        json={
            "deviceId": credentials["deviceId"],
            "refreshToken": credentials["refreshToken"],
        },
    )
    assert refreshed.status_code == 200
    assert client.get("/state", headers=headers).status_code == 401
    replacement = cast(dict[str, object], refreshed.json())
    headers = {"Authorization": f"Bearer {replacement['accessToken']}"}
    assert client.get("/state", headers=headers).status_code == 200
    assert (
        client.delete(
            f"/v1/auth/devices/{credentials['deviceId']}", headers=headers
        ).status_code
        == 204
    )
    assert client.get("/state", headers=headers).status_code == 401


def test_streams_never_echo_input_and_mutations_are_unavailable(tmp_path: Path) -> None:
    """Generated chat and playable silence never call inference or cluster commands."""
    service = _service(tmp_path)
    client = TestClient(create_fixture_app(service))
    credentials = _pair(client, service)
    headers = {"Authorization": f"Bearer {credentials['accessToken']}"}
    marker = "do-not-retain-this-prompt"
    chat = client.post("/v1/chat/completions", headers=headers, json={"input": marker})
    assert chat.status_code == 200
    assert "[DONE]" in chat.text
    assert marker not in chat.text
    speech = client.post("/v1/audio/speech", headers=headers, json={"input": marker})
    assert speech.status_code == 200
    assert speech.headers["x-audio-sample-rate"] == "24000"
    assert speech.headers["x-audio-sample-format"] == "s16le"
    assert speech.content == bytes(144000)
    assert client.post("/place_instance", headers=headers, json={}).status_code == 404


def test_oversized_input_is_rejected_without_starting_a_stream(tmp_path: Path) -> None:
    """Oversized authenticated and pre-auth bodies fail with one bounded response."""
    service = _service(tmp_path)
    client = TestClient(create_fixture_app(service))
    credentials = _pair(client, service)
    headers = {"Authorization": f"Bearer {credentials['accessToken']}"}
    for path in ("/v1/chat/completions", "/v1/auth/pairing-sessions/challenge"):
        response = client.post(path, headers=headers, content=bytes(65537))
        assert response.status_code == 413
        assert response.json() == {"detail": "fixture request limit"}
