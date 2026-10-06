"""Tests for the paired-device CLI and the device limit at the terminal."""

import base64
from pathlib import Path
from uuid import UUID

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import skulk.operator.cli as operator_cli
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import (
    OperatorPairingService,
    PairingChallengeRequest,
    PairingExchangeRequest,
    PairingExchangeResponse,
    pairing_signature_message,
)


def _base64url(value: bytes) -> str:
    """Encode bytes like the pairing wire contract."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _service(tmp_path: Path) -> OperatorPairingService:
    """Build an isolated pairing service on the real clock the CLI displays."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "authority-key.bin")
    store = EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3")
    return OperatorPairingService(store, provider)


def _pair(service: OperatorPairingService, device_name: str) -> PairingExchangeResponse:
    """Complete one single-use pairing against the service."""

    package = service.create_session(exchange_url="https://example.invalid")
    private_key = Ed25519PrivateKey.generate()
    public_key = _base64url(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
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
        PairingExchangeRequest(nonce=package.nonce, signature=_base64url(signature))
    )


def _use_service(
    monkeypatch: pytest.MonkeyPatch,
    service: OperatorPairingService,
) -> None:
    """Route the CLI's default-path service to the isolated test service."""

    def service_from_default_paths(
        _service_type: type[OperatorPairingService],
    ) -> OperatorPairingService:
        """Return the isolated service for this CLI invocation."""

        return service

    monkeypatch.setattr(
        OperatorPairingService,
        "from_default_paths",
        classmethod(service_from_default_paths),
    )


def test_devices_list_on_an_unpaired_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A node that never paired reports every slot free instead of failing."""

    _use_service(monkeypatch, _service(tmp_path))

    assert operator_cli.main(["devices", "list"]) == 0
    output = capsys.readouterr().out
    assert "0 of 5 device slots in use; 5 free." in output
    assert "No paired devices." in output


def test_devices_list_reports_slots_and_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Revoked devices are listed but do not occupy a slot."""

    service = _service(tmp_path)
    kitchen = _pair(service, "Kitchen phone")
    _pair(service, "Office tablet")
    service.owner_revoke_device(kitchen.device_id)
    _use_service(monkeypatch, service)

    assert operator_cli.main(["devices", "list"]) == 0
    output = capsys.readouterr().out
    assert "1 of 5 device slots in use; 4 free." in output
    assert "revoked  Kitchen phone" in output
    assert "active   Office tablet" in output
    assert "accessToken" not in output


def test_devices_revoke_frees_a_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Headless operators can free a slot without a dashboard."""

    service = _service(tmp_path)
    paired = _pair(service, "Old phone")
    _use_service(monkeypatch, service)

    assert operator_cli.main(["devices", "revoke", str(paired.device_id)]) == 0
    assert f"Revoked paired device {paired.device_id}." in capsys.readouterr().out
    assert service.pairing_capacity().available_slots == 5

    with pytest.raises(SystemExit):
        operator_cli.main(["devices", "revoke", "00000000-0000-4000-8000-000000000009"])
    assert "paired device was not found" in capsys.readouterr().err


def test_pair_refuses_at_the_device_limit_with_terminal_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The refusal names the commands that free a slot."""

    service = _service(tmp_path)
    for index in range(5):
        _pair(service, f"Phone {index}")
    _use_service(monkeypatch, service)
    payloads: list[str] = []
    monkeypatch.setattr(operator_cli, "_print_pairing_qr", payloads.append)

    with pytest.raises(SystemExit):
        operator_cli.main(["pair", "--exchange-url", "https://example.invalid"])

    assert payloads == []
    assert "skulk operator devices revoke" in capsys.readouterr().err


def test_pair_reports_an_invitation_reduced_to_free_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A larger request is reduced to the free slots and the output says why."""

    service = _service(tmp_path)
    for index in range(3):
        _pair(service, f"Phone {index}")
    _use_service(monkeypatch, service)
    payloads: list[str] = []
    monkeypatch.setattr(operator_cli, "_print_pairing_qr", payloads.append)

    assert (
        operator_cli.main(
            [
                "pair",
                "--exchange-url",
                "https://example.invalid",
                "--max-pairings",
                "10",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "permits up to 2 successful pairings" in output
    assert "Reduced from 10 because the cluster has 2 free device slots" in output
    assert len(payloads) == 1
