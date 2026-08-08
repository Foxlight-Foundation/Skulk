"""Tests for host-authorized single-gateway pairing."""

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import (
    AuthorityKeyUnavailableError,
    LocalFileAuthorityKeyProvider,
)
from skulk.operator.pairing import (
    OperatorCredentialExpiredError,
    OperatorCredentialInvalidError,
    OperatorDeviceNotFoundError,
    OperatorPairingService,
    OperatorTokenRequest,
    PairingChallengeRequest,
    PairingExchangeRequest,
    PairingExchangeResponse,
    PairingGatewayNotInitializedError,
    PairingProofError,
    PairingSessionExpiredError,
    PairingSessionStateError,
    pairing_signature_message,
)


def _base64url(value: bytes) -> str:
    """Encode test key material like the wire contract."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _service(
    tmp_path: Path,
    clock: list[datetime],
) -> OperatorPairingService:
    """Build an isolated pairing service with a mutable deterministic clock."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "authority-key.bin")
    store = EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3")
    return OperatorPairingService(store, provider, now=lambda: clock[0])


def _device_key() -> tuple[Ed25519PrivateKey, str]:
    """Return one candidate private key and its wire public key."""

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_key, _base64url(public_key)


def _complete_pairing(
    service: OperatorPairingService,
    *,
    device_name: str = "Phone",
) -> PairingExchangeResponse:
    """Complete one host-authorized pairing against the service."""

    package = service.create_session(exchange_url="https://example.invalid")
    private_key, public_key = _device_key()
    challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            device_name=device_name,
            device_public_key=public_key,
        )
    )
    signature = private_key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=challenge.challenge,
        )
    )
    return service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            signature=_base64url(signature),
        )
    )


def test_pairing_challenge_exchange_consumes_session(tmp_path: Path) -> None:
    """A device proves its key and receives credentials exactly once."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_session(
        exchange_url="https://example.invalid/operator",
        cluster_name="Fox Den",
    )
    private_key, public_key = _device_key()

    challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            device_name="  Tom's   iPhone  ",
            device_public_key=public_key,
        ),
    )
    signature = private_key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=challenge.challenge,
        )
    )
    result = service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            signature=_base64url(signature),
        ),
    )

    assert result.cluster.name == "Fox Den"
    assert result.access_token
    assert result.refresh_token
    assert result.access_token != result.refresh_token
    assert result.scopes == (
        "cluster:read",
        "models:read",
        "chat:write",
        "operations:write",
        "devices:manage",
    )
    assert package.nonce.encode("ascii") not in (
        tmp_path / "authority.sqlite3"
    ).read_bytes()
    with pytest.raises(PairingSessionStateError, match="not awaiting"):
        service.exchange(
            PairingExchangeRequest(
                nonce=package.nonce,
                signature=_base64url(signature),
            ),
        )


def test_pairing_rejects_wrong_device_proof(tmp_path: Path) -> None:
    """A signature from a key other than the challenged key is rejected."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_session(exchange_url="https://example.invalid")
    _, public_key = _device_key()
    wrong_private_key, _ = _device_key()
    challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            device_name="Phone",
            device_public_key=public_key,
        ),
    )
    signature = wrong_private_key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=challenge.challenge,
        )
    )

    with pytest.raises(PairingProofError, match="invalid"):
        service.exchange(
            PairingExchangeRequest(
                nonce=package.nonce,
                signature=_base64url(signature),
            ),
        )


def test_pairing_expires_after_five_minutes(tmp_path: Path) -> None:
    """Expired QR capabilities cannot bind a device key."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_session(exchange_url="https://example.invalid")
    _, public_key = _device_key()
    clock[0] += timedelta(minutes=5)

    with pytest.raises(PairingSessionExpiredError, match="expired"):
        service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                device_name="Phone",
                device_public_key=public_key,
            ),
        )


def test_pairing_package_contains_no_durable_credential(tmp_path: Path) -> None:
    """The QR carries identity and one short-lived capability only."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    package = _service(tmp_path, clock).create_session(
        exchange_url="https://example.invalid"
    )
    payload = package.model_dump(mode="json", by_alias=True)

    assert "accessToken" not in payload
    assert "refreshToken" not in payload
    assert package.as_url().startswith("skulk://pair?payload=")


def test_pairing_rejects_cleartext_non_loopback_exchange_url(
    tmp_path: Path,
) -> None:
    """Remote pairing credentials cannot be issued over cleartext HTTP."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)

    with pytest.raises(ValueError, match="must use HTTPS"):
        service.create_session(exchange_url="http://gateway.example.invalid")

    assert not (tmp_path / "authority-key.bin").exists()
    assert not (tmp_path / "authority.sqlite3").exists()


def test_pairing_allows_cleartext_loopback_for_local_development(
    tmp_path: Path,
) -> None:
    """Loopback remains available for isolated local development."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    package = _service(tmp_path, clock).create_session(
        exchange_url="http://127.0.0.1:52415"
    )

    assert str(package.exchange_url) == "http://127.0.0.1:52415/"


def test_existing_authority_fails_closed_when_local_key_is_lost(
    tmp_path: Path,
) -> None:
    """An existing authority database never receives a replacement data key."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    service.create_session(exchange_url="https://example.invalid")
    (tmp_path / "authority-key.bin").unlink()

    with pytest.raises(AuthorityKeyUnavailableError, match="not initialized"):
        service.create_session(exchange_url="https://example.invalid")


def test_refresh_rotates_both_credentials_and_rejects_replay(tmp_path: Path) -> None:
    """Refresh rotation invalidates both members of the previous token pair."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    paired = _complete_pairing(service)

    refreshed = service.refresh(
        OperatorTokenRequest(
            device_id=paired.device_id,
            refresh_token=paired.refresh_token,
        )
    )

    assert refreshed.access_token != paired.access_token
    assert refreshed.refresh_token != paired.refresh_token
    with pytest.raises(OperatorCredentialInvalidError, match="access"):
        service.validate_access_token(paired.access_token)
    with pytest.raises(OperatorCredentialInvalidError, match="refresh"):
        service.refresh(
            OperatorTokenRequest(
                device_id=paired.device_id,
                refresh_token=paired.refresh_token,
            )
        )
    context = service.validate_access_token(
        refreshed.access_token,
        required_scopes=("cluster:read", "devices:manage"),
    )
    assert context.device_id == paired.device_id


def test_access_and_refresh_credentials_expire_independently(tmp_path: Path) -> None:
    """The service enforces the short access and longer refresh lifetimes."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    paired = _complete_pairing(service)
    clock[0] += timedelta(minutes=15)

    with pytest.raises(OperatorCredentialExpiredError, match="access"):
        service.validate_access_token(paired.access_token)

    refreshed = service.refresh(
        OperatorTokenRequest(
            device_id=paired.device_id,
            refresh_token=paired.refresh_token,
        )
    )
    clock[0] += timedelta(days=30)
    with pytest.raises(OperatorCredentialExpiredError, match="refresh"):
        service.refresh(
            OperatorTokenRequest(
                device_id=paired.device_id,
                refresh_token=refreshed.refresh_token,
            )
        )


def test_device_listing_and_revocation_expose_no_credentials(tmp_path: Path) -> None:
    """An authorized device can inspect and immediately revoke a peer."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    first = _complete_pairing(service, device_name="First phone")
    clock[0] += timedelta(seconds=1)
    second = _complete_pairing(service, device_name="Second phone")

    before = service.devices(second.access_token)
    assert [device.name for device in before.devices] == [
        "First phone",
        "Second phone",
    ]
    assert [device.current for device in before.devices] == [False, True]

    service.revoke_device(second.access_token, first.device_id)
    after = service.devices(second.access_token)
    revoked = next(device for device in after.devices if device.device_id == first.device_id)
    assert revoked.state == "revoked"
    assert revoked.refresh_expires_at is None
    with pytest.raises(OperatorCredentialInvalidError, match="access"):
        service.validate_access_token(first.access_token)
    with pytest.raises(OperatorCredentialInvalidError, match="revoked"):
        service.refresh(
            OperatorTokenRequest(
                device_id=first.device_id,
                refresh_token=first.refresh_token,
            )
        )
    with pytest.raises(OperatorDeviceNotFoundError, match="not found"):
        service.revoke_device(second.access_token, UUID(int=0))


def test_malformed_unicode_credentials_fail_safely(tmp_path: Path) -> None:
    """Unexpected Unicode bearer material is unknown rather than a server error."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    _complete_pairing(service)

    with pytest.raises(OperatorCredentialInvalidError, match="access"):
        service.validate_access_token("snowman-☃")


def test_lifecycle_rejects_an_uninitialized_gateway_safely(tmp_path: Path) -> None:
    """Credential operations never leak a missing authority store as an error 500."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)

    with pytest.raises(PairingGatewayNotInitializedError, match="not initialized"):
        service.validate_access_token("unknown-access-token")
