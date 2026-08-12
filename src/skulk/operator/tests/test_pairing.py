"""Tests for host-authorized single-gateway pairing."""

import base64
from concurrent.futures import ThreadPoolExecutor
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
    PairingChallengeResponse,
    PairingExchangeRequest,
    PairingExchangeResponse,
    PairingGatewayNotInitializedError,
    PairingInvitationCapacityError,
    PairingInvitationPackage,
    PairingProofError,
    PairingSessionExpiredError,
    PairingSessionStateError,
    pairing_invitation_signature_message,
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


def _complete_invitation_pairing(
    service: OperatorPairingService,
    package: PairingInvitationPackage,
    *,
    device_name: str = "Review phone",
) -> PairingExchangeResponse:
    """Complete one independent attempt under a reusable invitation."""

    private_key, public_key = _device_key()
    challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name=device_name,
            device_public_key=public_key,
        )
    )
    assert challenge.attempt_id is not None
    signature = private_key.sign(
        pairing_invitation_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            invitation_id=package.invitation_id,
            nonce=package.nonce,
            attempt_id=challenge.attempt_id,
            challenge=challenge.challenge,
        )
    )
    return service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            attempt_id=challenge.attempt_id,
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


def test_reusable_invitation_issues_independent_attempts(tmp_path: Path) -> None:
    """Simultaneous scans receive distinct challenges and both can pair."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_invitation(
        lifetime=timedelta(days=90),
        max_pairings=10,
        exchange_url="https://example.invalid/operator",
        cluster_name="Review Fabric",
    )
    first_private, first_public = _device_key()
    second_private, second_public = _device_key()
    first_challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name="First reviewer",
            device_public_key=first_public,
        )
    )
    second_challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name="Second reviewer",
            device_public_key=second_public,
        )
    )
    assert first_challenge.attempt_id is not None
    assert second_challenge.attempt_id is not None
    assert first_challenge.attempt_id != second_challenge.attempt_id
    assert first_challenge.expires_at == clock[0] + timedelta(minutes=5)

    first = service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            attempt_id=first_challenge.attempt_id,
            signature=_base64url(
                first_private.sign(
                    pairing_invitation_signature_message(
                        cluster_id=UUID(str(package.cluster_id)),
                        invitation_id=package.invitation_id,
                        nonce=package.nonce,
                        attempt_id=first_challenge.attempt_id,
                        challenge=first_challenge.challenge,
                    )
                )
            ),
        )
    )
    second = service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            attempt_id=second_challenge.attempt_id,
            signature=_base64url(
                second_private.sign(
                    pairing_invitation_signature_message(
                        cluster_id=UUID(str(package.cluster_id)),
                        invitation_id=package.invitation_id,
                        nonce=package.nonce,
                        attempt_id=second_challenge.attempt_id,
                        challenge=second_challenge.challenge,
                    )
                )
            ),
        )
    )

    assert first.device_id != second.device_id
    summary = service.invitations()[0]
    assert summary.successful_pairings == 2
    assert summary.active_attempts == 0
    assert summary.total_attempts == 2
    assert summary.state == "active"
    assert package.nonce.encode("ascii") not in (
        tmp_path / "authority.sqlite3"
    ).read_bytes()


def test_invitation_exhaustion_and_revocation_are_separate_from_devices(
    tmp_path: Path,
) -> None:
    """Invitation limits block new scans while issued device credentials remain valid."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    exhausted = service.create_invitation(
        lifetime=timedelta(days=1),
        max_pairings=1,
        exchange_url="https://example.invalid",
    )
    paired = _complete_invitation_pairing(service, exhausted)
    _, public_key = _device_key()
    with pytest.raises(PairingSessionExpiredError, match="exhausted"):
        service.create_challenge(
            PairingChallengeRequest(
                nonce=exhausted.nonce,
                invitation_id=exhausted.invitation_id,
                device_name="Another phone",
                device_public_key=public_key,
            )
        )
    assert service.validate_access_token(paired.access_token).device_id == paired.device_id

    revoked = service.create_invitation(
        lifetime=timedelta(days=1),
        max_pairings=2,
        exchange_url="https://example.invalid",
    )
    existing_device = _complete_invitation_pairing(service, revoked)
    unfinished_private_key, unfinished_public_key = _device_key()
    unfinished = service.create_challenge(
        PairingChallengeRequest(
            nonce=revoked.nonce,
            invitation_id=revoked.invitation_id,
            device_name="Unfinished phone",
            device_public_key=unfinished_public_key,
        )
    )
    assert unfinished.attempt_id is not None
    service.revoke_invitation(revoked.invitation_id)
    with pytest.raises(PairingSessionExpiredError, match="revoked"):
        service.create_challenge(
            PairingChallengeRequest(
                nonce=revoked.nonce,
                invitation_id=revoked.invitation_id,
                device_name="Blocked phone",
                device_public_key=public_key,
            )
        )
    unfinished_signature = unfinished_private_key.sign(
        pairing_invitation_signature_message(
            cluster_id=UUID(str(revoked.cluster_id)),
            invitation_id=revoked.invitation_id,
            nonce=revoked.nonce,
            attempt_id=unfinished.attempt_id,
            challenge=unfinished.challenge,
        )
    )
    with pytest.raises(PairingSessionExpiredError, match="revoked"):
        service.exchange(
            PairingExchangeRequest(
                nonce=revoked.nonce,
                invitation_id=revoked.invitation_id,
                attempt_id=unfinished.attempt_id,
                signature=_base64url(unfinished_signature),
            )
        )
    assert (
        service.validate_access_token(existing_device.access_token).device_id
        == existing_device.device_id
    )
    listed = service.devices(existing_device.access_token)
    assert {device.device_id for device in listed.devices} == {
        paired.device_id,
        existing_device.device_id,
    }
    summaries = {summary.invitation_id: summary for summary in service.invitations()}
    assert summaries[exhausted.invitation_id].state == "exhausted"
    assert summaries[revoked.invitation_id].state == "revoked"
    assert summaries[revoked.invitation_id].successful_pairings == 1
    assert summaries[revoked.invitation_id].active_attempts == 1


def test_invitation_and_pending_attempt_survive_service_restart(tmp_path: Path) -> None:
    """Durable authority records preserve reusable pairing across a restart."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    before_restart = _service(tmp_path, clock)
    package = before_restart.create_invitation(
        lifetime=timedelta(days=90),
        exchange_url="https://example.invalid/operator",
    )
    private_key, public_key = _device_key()
    challenge = before_restart.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name="Restarted phone",
            device_public_key=public_key,
        )
    )
    assert challenge.attempt_id is not None

    after_restart = _service(tmp_path, clock)
    signature = private_key.sign(
        pairing_invitation_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            invitation_id=package.invitation_id,
            nonce=package.nonce,
            attempt_id=challenge.attempt_id,
            challenge=challenge.challenge,
        )
    )
    paired = after_restart.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            attempt_id=challenge.attempt_id,
            signature=_base64url(signature),
        )
    )

    assert after_restart.validate_access_token(paired.access_token).device_id == paired.device_id
    summary = after_restart.invitations()[0]
    assert summary.successful_pairings == 1
    assert summary.total_attempts == 1


def test_invitation_attempts_expire_and_active_capacity_is_bounded(
    tmp_path: Path,
) -> None:
    """Unfinished scans do not consume pairing slots and cannot grow without bound."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_invitation(
        lifetime=timedelta(days=1),
        exchange_url="https://example.invalid",
    )
    for index in range(10):
        _, public_key = _device_key()
        service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                device_name=f"Phone {index}",
                device_public_key=public_key,
            )
        )
    _, overflow_public_key = _device_key()
    with pytest.raises(PairingInvitationCapacityError) as raised:
        service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                device_name="Overflow phone",
                device_public_key=overflow_public_key,
            )
        )
    assert raised.value.retry_after_seconds == 300
    assert service.invitations()[0].state == "attempt-limit"

    clock[0] += timedelta(minutes=5)
    replacement = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name="Replacement phone",
            device_public_key=overflow_public_key,
        )
    )
    assert replacement.attempt_id is not None
    summary = service.invitations()[0]
    assert summary.successful_pairings == 0
    assert summary.active_attempts == 1
    assert summary.total_attempts == 11


def test_invitation_total_attempts_are_bounded(tmp_path: Path) -> None:
    """The journal stops growing while its final live attempt can still finish."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_invitation(
        lifetime=timedelta(days=90),
        exchange_url="https://example.invalid",
    )
    final_private_key: Ed25519PrivateKey | None = None
    final_challenge: PairingChallengeResponse | None = None
    for index in range(100):
        private_key, public_key = _device_key()
        challenge = service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                device_name=f"Attempt {index}",
                device_public_key=public_key,
            )
        )
        if index == 99:
            final_private_key = private_key
            final_challenge = challenge
        else:
            clock[0] += timedelta(minutes=5)

    _, overflow_public_key = _device_key()
    with pytest.raises(PairingSessionExpiredError, match="attempt limit"):
        service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                device_name="Attempt 101",
                device_public_key=overflow_public_key,
            )
        )
    summary = service.invitations()[0]
    assert summary.total_attempts == 100
    assert summary.successful_pairings == 0
    assert summary.state == "attempt-limit"

    assert final_private_key is not None
    assert final_challenge is not None
    assert final_challenge.attempt_id is not None
    signature = final_private_key.sign(
        pairing_invitation_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            invitation_id=package.invitation_id,
            nonce=package.nonce,
            attempt_id=final_challenge.attempt_id,
            challenge=final_challenge.challenge,
        )
    )
    paired = service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            attempt_id=final_challenge.attempt_id,
            signature=_base64url(signature),
        )
    )
    assert service.validate_access_token(paired.access_token).device_id == paired.device_id
    assert service.invitations()[0].successful_pairings == 1


def test_invitation_proof_is_bound_to_its_attempt(tmp_path: Path) -> None:
    """A valid signature for one attempt cannot complete another attempt."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_invitation(
        lifetime=timedelta(days=1),
        exchange_url="https://example.invalid",
    )
    private_key, public_key = _device_key()
    first = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name="Phone",
            device_public_key=public_key,
        )
    )
    second = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            invitation_id=package.invitation_id,
            device_name="Phone",
            device_public_key=public_key,
        )
    )
    assert first.attempt_id is not None
    assert second.attempt_id is not None
    signature = private_key.sign(
        pairing_invitation_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            invitation_id=package.invitation_id,
            nonce=package.nonce,
            attempt_id=first.attempt_id,
            challenge=first.challenge,
        )
    )
    with pytest.raises(PairingProofError, match="invalid"):
        service.exchange(
            PairingExchangeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                attempt_id=second.attempt_id,
                signature=_base64url(signature),
            )
        )


def test_concurrent_invitation_exchanges_cannot_exceed_success_limit(
    tmp_path: Path,
) -> None:
    """Global authority fencing admits only one racing final pairing slot."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    package = service.create_invitation(
        lifetime=timedelta(days=1),
        max_pairings=1,
        exchange_url="https://example.invalid",
    )
    exchanges: list[PairingExchangeRequest] = []
    for name in ("First", "Second"):
        private_key, public_key = _device_key()
        challenge = service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                device_name=name,
                device_public_key=public_key,
            )
        )
        assert challenge.attempt_id is not None
        exchanges.append(
            PairingExchangeRequest(
                nonce=package.nonce,
                invitation_id=package.invitation_id,
                attempt_id=challenge.attempt_id,
                signature=_base64url(
                    private_key.sign(
                        pairing_invitation_signature_message(
                            cluster_id=UUID(str(package.cluster_id)),
                            invitation_id=package.invitation_id,
                            nonce=package.nonce,
                            attempt_id=challenge.attempt_id,
                            challenge=challenge.challenge,
                        )
                    )
                ),
            )
        )

    def exchange(request: PairingExchangeRequest) -> str:
        """Return the safe result category from one racing exchange."""

        try:
            service.exchange(request)
        except PairingSessionExpiredError:
            return "unavailable"
        return "paired"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(exchange, exchanges))

    assert sorted(results) == ["paired", "unavailable"]
    summary = service.invitations()[0]
    assert summary.successful_pairings == 1
    assert summary.state == "exhausted"


@pytest.mark.parametrize(
    "lifetime,max_pairings",
    [
        (timedelta(seconds=59), 10),
        (timedelta(days=90, seconds=1), 10),
        (timedelta(days=1), 0),
        (timedelta(days=1), 21),
    ],
)
def test_invitation_policy_bounds_are_enforced(
    tmp_path: Path,
    lifetime: timedelta,
    max_pairings: int,
) -> None:
    """Hosts cannot create unbounded reusable bearer invitations."""

    clock = [datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)
    with pytest.raises(ValueError):
        service.create_invitation(
            lifetime=lifetime,
            max_pairings=max_pairings,
            exchange_url="https://example.invalid",
        )


def test_pairing_package_contains_no_durable_credential(tmp_path: Path) -> None:
    """A direct QR carries identity and one short-lived capability only."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    package = _service(tmp_path, clock).create_session(
        exchange_url="https://example.invalid"
    )
    payload = package.model_dump(mode="json", by_alias=True)

    assert "accessToken" not in payload
    assert "refreshToken" not in payload
    assert package.version == 1
    assert package.remote_access is None
    assert package.as_url().startswith("skulk://pair?payload=")
    assert "remoteAccess" not in _pairing_url_payload(package.as_url())


def test_pairing_requires_direct_url_before_relay_configuration(
    tmp_path: Path,
) -> None:
    """A host cannot print an unreachable pairing package before provisioning."""

    clock = [datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)]
    service = _service(tmp_path, clock)

    with pytest.raises(ValueError, match="exchange_url is required"):
        service.create_session()

    assert not (tmp_path / "authority-key.bin").exists()
    assert not (tmp_path / "authority.sqlite3").exists()


def _pairing_url_payload(value: str) -> str:
    """Decode a generated QR URL for wire-shape assertions."""

    encoded = value.split("payload=", maxsplit=1)[1]
    return base64.urlsafe_b64decode(f"{encoded}{'=' * (-len(encoded) % 4)}").decode(
        "utf-8"
    )


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
