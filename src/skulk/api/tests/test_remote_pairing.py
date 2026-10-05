"""Tests for dashboard phone-pairing status and turning relay pairing on and off."""

from __future__ import annotations

import base64
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast, final
from uuid import UUID

import pytest

import skulk.operator.relay_registration as relay_registration
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skulk.api.operator_auth import create_operator_auth_router
from skulk.api.operator_remote_access import OperatorRemoteAccessState
from skulk.api.remote_pairing import (
    RelayLink,
    RemotePairingController,
    RemotePairingRefusedError,
)
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import (
    OperatorPairingService,
    PairingChallengeRequest,
    PairingExchangeRequest,
    pairing_signature_message,
)
from skulk.operator.relay import (
    OperatorRelayConfigurationRepository,
    OperatorRelayProvisioning,
)
from skulk.operator.relay_registration import (
    RelayRegistrationError,
    RelayRegistrationMaterial,
)
from skulk.store.config import RelayConnectivityConfig

_DASHBOARD = {"Origin": "http://127.0.0.1:52415", "X-Skulk-Dashboard": "pairing-v1"}


def _encode(value: bytes) -> str:
    """Encode canonical unpadded base64url."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


@final
class _Node:
    """One gateway node with injectable relay, ingress, and cluster facts."""

    def __init__(self, tmp_path: Path) -> None:
        provider = LocalFileAuthorityKeyProvider(tmp_path / "authority-key.bin")
        store = EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3")
        repository = OperatorRelayConfigurationRepository(
            store,
            certificate_path=tmp_path / "relay-cert.pem",
            private_key_path=tmp_path / "relay-key.pem",
        )
        self.clock = [datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)]
        self.service = OperatorPairingService(
            store,
            provider,
            relay_repository=repository,
            now=lambda: self.clock[0],
        )
        self.settings: RelayConnectivityConfig | None = RelayConnectivityConfig(
            registration_url="https://relay.example.invalid"
        )
        self.offline = False
        self.remote_access_state: OperatorRemoteAccessState = "not_configured"
        self.link: RelayLink | None = None
        self.elsewhere: tuple[str, str | None] | None = None
        self.checks = 0
        self.registrations = 0
        self.failure: RelayRegistrationError | None = None
        self.during_registration: list[object] = []
        self.monotonic = [1_000.0]
        self.controller = RemotePairingController(
            service=self.service,
            relay_settings=lambda: self.settings,
            offline=lambda: self.offline,
            remote_access_state=lambda: self.remote_access_state,
            relay_link=lambda: self.link,
            request_remote_access_check=self._check,
            pairing_gateway_elsewhere=lambda: self.elsewhere,
            register=self.register_route,
            monotonic_seconds=lambda: self.monotonic[0],
        )

    def _check(self) -> None:
        self.checks += 1

    def register_route(
        self, origin: str, material: RelayRegistrationMaterial
    ) -> OperatorRelayProvisioning:
        """Answer a registration like the relay, recording when it happens."""

        assert origin == "https://relay.example.invalid"
        self.registrations += 1
        self.during_registration.append(self.controller.status().state)
        if self.failure is not None:
            raise self.failure
        return OperatorRelayProvisioning(
            version=2,
            app_websocket_url="wss://relay.example.invalid/v1/carrier/app",
            gateway_control_websocket_url=(
                "wss://relay.example.invalid/v1/connector/control"
            ),
            gateway_data_websocket_url="wss://relay.example.invalid/v1/connector/data",
            routing_locator=_encode(secrets.token_bytes(32)),
            app_carrier_credential=material.app_carrier_credential,
            gateway_carrier_credential=material.gateway_carrier_credential,
            connector_authority_private_key_pkcs8=(
                material.connector_authority_private_key_pkcs8
            ),
            connector_authority_key_id=material.connector_authority_key_id,
            connector_region=_encode(b"region01"),
            connector_authority_epoch=material.connector_authority_epoch,
        )

    def pair_directly(self, name: str) -> None:
        """Pair one device over a direct exchange URL, without the relay."""

        package = self.service.create_session(exchange_url="https://example.invalid")
        private_key = Ed25519PrivateKey.generate()
        challenge = self.service.create_challenge(
            PairingChallengeRequest(
                nonce=package.nonce,
                device_name=name,
                device_public_key=_encode(
                    private_key.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ),
            )
        )
        signature = private_key.sign(
            pairing_signature_message(
                cluster_id=UUID(str(package.cluster_id)),
                nonce=package.nonce,
                challenge=challenge.challenge,
            )
        )
        self.service.exchange(
            PairingExchangeRequest(nonce=package.nonce, signature=_encode(signature))
        )

    def client(self) -> TestClient:
        """Return a loopback dashboard client for the pairing routes."""

        async def verified(_peer: str) -> bool:
            return False

        app = FastAPI()
        app.include_router(
            create_operator_auth_router(
                self.service,
                remote_pairing=self.controller,
                tailnet_peer_verifier=verified,
            )
        )
        return TestClient(
            app, base_url="http://127.0.0.1:52415", client=("127.0.0.1", 50000)
        )


def test_a_fresh_node_is_not_set_up_and_can_register(tmp_path: Path) -> None:
    """A never-paired node reads as neutral, with the relay it would use."""

    node = _Node(tmp_path)

    status = node.controller.status()

    assert status.state == "not_set_up"
    assert status.registration_available is True
    assert status.registration_blocked_reason is None
    assert status.relay_host == "relay.example.invalid"


@pytest.mark.parametrize(
    ("settings", "offline", "reason"),
    [
        (RelayConnectivityConfig(enabled=False), False, "disabled"),
        (RelayConnectivityConfig(registration_url="https://r.invalid"), True, "offline"),
    ],
)
def test_registration_blocked_reasons(
    tmp_path: Path,
    settings: RelayConnectivityConfig | None,
    offline: bool,
    reason: str,
) -> None:
    """The status names why this node cannot register, and enabling refuses."""

    node = _Node(tmp_path)
    node.settings = settings
    node.offline = offline

    status = node.controller.status()
    assert (status.registration_available, status.registration_blocked_reason) == (
        False,
        reason,
    )
    with pytest.raises(RelayRegistrationError) as raised:
        node.controller.enable()
    assert raised.value.failure == reason
    assert node.registrations == 0


def test_enabling_registers_once_and_starts_remote_access(tmp_path: Path) -> None:
    """Pair a phone registers, asks ingress to start, and is idempotent."""

    node = _Node(tmp_path)

    status = node.controller.enable()

    assert node.registrations == 1
    assert node.during_registration == ["registering"]
    assert node.checks == 1
    assert status.state == "connecting"
    assert node.service.relay_configuration() is not None

    again = node.controller.enable()
    assert node.registrations == 1
    assert again.state == "connecting"


def test_route_states_follow_ingress_and_relay_liveness(tmp_path: Path) -> None:
    """Connecting, connected, unreachable, and revoked come from the live session."""

    node = _Node(tmp_path)
    node.controller.enable()

    node.remote_access_state = "running"
    node.link = RelayLink(connected=False, disconnected_seconds=5.0)
    assert node.controller.status().state == "connecting"

    node.link = RelayLink(connected=True, disconnected_seconds=0.0)
    assert node.controller.status().state == "connected"

    node.link = RelayLink(connected=False, disconnected_seconds=60.0)
    assert node.controller.status().state == "relay_unreachable"

    node.remote_access_state = "failed"
    node.link = None
    assert node.controller.status().state == "relay_unreachable"

    node.remote_access_state = "rejected"
    assert node.controller.status().state == "revoked"


def test_pairing_managed_on_another_node_is_shown_and_respected(
    tmp_path: Path,
) -> None:
    """A second node never becomes a separate pairing gateway by accident."""

    node = _Node(tmp_path)
    node.elsewhere = ("node-b", "Kitchen Mac")

    status = node.controller.status()
    assert status.state == "managed_elsewhere"
    assert (status.managed_on_node_id, status.managed_on_node_name) == (
        "node-b",
        "Kitchen Mac",
    )
    with pytest.raises(RemotePairingRefusedError) as raised:
        node.controller.enable()
    assert raised.value.refusal == "managed_elsewhere"
    assert "Kitchen Mac" in str(raised.value)
    assert node.registrations == 0


def test_a_full_cluster_does_not_register_a_route(tmp_path: Path) -> None:
    """The device limit is checked before contacting the relay."""

    node = _Node(tmp_path)
    for index in range(5):
        node.pair_directly(f"Phone {index}")

    with pytest.raises(RemotePairingRefusedError) as raised:
        node.controller.enable()

    assert raised.value.refusal == "device_limit"
    assert node.registrations == 0


def test_a_refused_registration_is_reported_with_its_delay(tmp_path: Path) -> None:
    """The last failure and the remaining delay stay visible until retried."""

    node = _Node(tmp_path)
    node.failure = RelayRegistrationError("rate_limited", retry_after_seconds=30)

    with pytest.raises(RelayRegistrationError):
        node.controller.enable()

    status = node.controller.status()
    assert status.state == "not_set_up"
    assert status.last_failure == "rate_limited"
    assert status.retry_after_seconds == 31
    node.monotonic[0] += 31
    assert node.controller.status().retry_after_seconds is None

    node.failure = None
    node.controller.enable()
    assert node.controller.status().last_failure is None


def test_a_concurrent_enable_returns_the_in_flight_status(tmp_path: Path) -> None:
    """Double clicks never register twice; turning off waits for the first."""

    node = _Node(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = node.register_route

    def slow_register(
        origin: str, material: RelayRegistrationMaterial
    ) -> OperatorRelayProvisioning:
        entered.set()
        assert release.wait(5)
        return original(origin, material)

    node.controller = RemotePairingController(
        service=node.service,
        relay_settings=lambda: node.settings,
        offline=lambda: node.offline,
        remote_access_state=lambda: node.remote_access_state,
        relay_link=lambda: node.link,
        request_remote_access_check=lambda: None,
        pairing_gateway_elsewhere=lambda: None,
        register=slow_register,
    )
    first = threading.Thread(target=node.controller.enable)
    first.start()
    assert entered.wait(5)

    assert node.controller.enable().state == "registering"
    with pytest.raises(RemotePairingRefusedError) as raised:
        node.controller.disable()
    assert raised.value.refusal == "busy"

    release.set()
    first.join(5)
    assert node.registrations == 1


def test_turning_off_forgets_the_route_and_its_invitations(tmp_path: Path) -> None:
    """Relay invitations die with the route; direct invitations survive."""

    node = _Node(tmp_path)
    node.controller.enable()
    relay_invitation = node.service.create_invitation(lifetime=timedelta(days=1))
    direct_invitation = node.service.create_invitation(
        lifetime=timedelta(days=1), exchange_url="https://gateway.example.invalid"
    )
    checks_before = node.checks

    status = node.controller.disable()

    assert status.state == "not_set_up"
    assert node.service.relay_configuration() is None
    assert node.checks == checks_before + 1
    states = {item.invitation_id: item.state for item in node.service.invitations()}
    assert states[relay_invitation.invitation_id] == "revoked"
    assert states[direct_invitation.invitation_id] == "active"
    assert node.controller.disable().state == "not_set_up"


def test_routes_are_dashboard_only_and_report_stable_codes(tmp_path: Path) -> None:
    """The HTTP surface mirrors the controller with documented error bodies."""

    node = _Node(tmp_path)
    client = node.client()

    assert client.get("/v1/auth/remote-pairing").status_code == 403
    assert client.post("/v1/auth/remote-pairing").status_code == 403
    assert client.delete("/v1/auth/remote-pairing").status_code == 403

    status = client.get("/v1/auth/remote-pairing", headers=_DASHBOARD)
    assert status.status_code == 200
    assert status.headers["cache-control"] == "no-store"
    assert status.json()["state"] == "not_set_up"
    assert status.json()["registrationAvailable"] is True

    node.failure = RelayRegistrationError("rate_limited", retry_after_seconds=30)
    limited = client.post("/v1/auth/remote-pairing", headers=_DASHBOARD)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "30"
    assert limited.json()["code"] == "rate_limited"
    assert limited.json()["retryAfterSeconds"] == 30

    node.failure = RelayRegistrationError("registration_unsupported")
    unsupported = client.post("/v1/auth/remote-pairing", headers=_DASHBOARD)
    assert unsupported.status_code == 502
    assert unsupported.json()["code"] == "registration_unsupported"

    node.failure = None
    enabled = client.post("/v1/auth/remote-pairing", headers=_DASHBOARD)
    assert enabled.status_code == 200
    assert enabled.json()["state"] == "connecting"

    disabled = client.delete("/v1/auth/remote-pairing", headers=_DASHBOARD)
    assert disabled.status_code == 200
    assert disabled.json()["state"] == "not_set_up"

    node.elsewhere = ("node-b", None)
    elsewhere = client.post("/v1/auth/remote-pairing", headers=_DASHBOARD)
    assert elsewhere.status_code == 409
    assert elsewhere.json()["code"] == "managed_elsewhere"


def test_unreadable_pairing_state_is_a_safe_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost data key or damaged record never leaks details or a 500."""

    node = _Node(tmp_path)

    def unreadable() -> None:
        raise RuntimeError("protected path detail")

    monkeypatch.setattr(node.service, "relay_configuration", unreadable)
    response = node.client().get("/v1/auth/remote-pairing", headers=_DASHBOARD)

    assert response.status_code == 503
    body = cast(dict[str, object], response.json())
    assert body["code"] == "pairing_state_unavailable"
    assert "protected path detail" not in response.text


def test_unconfigured_node_uses_the_default_relay(tmp_path: Path) -> None:
    """With no relay settings, a node registers with Foxlight's relay."""

    node = _Node(tmp_path)
    node.settings = None

    status = node.controller.status()
    assert (status.registration_available, status.registration_blocked_reason) == (
        True,
        None,
    )
    assert status.relay_host == "relay.foxlight.ai"


def test_build_without_a_default_relay_reports_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that unsets the default needs skulk.yaml to name a relay."""

    monkeypatch.setattr(relay_registration, "DEFAULT_RELAY_REGISTRATION_ORIGIN", None)
    node = _Node(tmp_path)
    node.settings = None

    status = node.controller.status()
    assert status.registration_blocked_reason == "not_configured"
    with pytest.raises(RelayRegistrationError) as raised:
        node.controller.enable()
    assert raised.value.failure == "not_configured"
    assert node.registrations == 0

