"""Tests for self-service relay registration against an in-process contract fake."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import cast, final

import httpx
import pytest
from cryptography.hazmat.primitives import serialization

import skulk.operator.cli as operator_cli
import skulk.operator.pairing as pairing_module
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import OperatorPairingService
from skulk.operator.relay import (
    OperatorRelayAlreadyConfiguredError,
    OperatorRelayConfigurationRepository,
    OperatorRelayProvisioning,
    OperatorRelayUnavailableError,
)
from skulk.operator.relay_registration import (
    RelayRegistrationError,
    RelayRegistrationMaterial,
    register_relay_route,
    resolve_registration_origin,
)
from skulk.store.config import ConnectivityConfig, RelayConnectivityConfig, SkulkConfig

_RELAY_ORIGIN = "https://relay.example.invalid"


def _encode(value: bytes) -> str:
    """Encode canonical unpadded base64url like the wire contract."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_exact(value: object, size: int) -> bytes:
    """Decode one canonical unpadded base64url value of an exact size, or fail."""

    if not isinstance(value, str) or "=" in value:
        raise ValueError("not canonical")
    decoded = base64.urlsafe_b64decode(f"{value}{'=' * (-len(value) % 4)}")
    if len(decoded) != size or _encode(decoded) != value:
        raise ValueError("not canonical")
    return decoded


@final
class _ContractRelay:
    """In-process relay implementing registration contract v2 exactly."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[str, str, str]] = {}
        self.requests: list[dict[str, object]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Serve ``POST /v1/registrations`` with the contract's semantics."""

        if request.method != "POST" or request.url.path != "/v1/registrations":
            return httpx.Response(404, json={"code": "invalid_request"})
        try:
            body = cast(object, json.loads(request.content))
        except ValueError:
            return httpx.Response(400, json={"code": "invalid_request"})
        if not isinstance(body, dict):
            return httpx.Response(400, json={"code": "invalid_request"})
        fields = cast(dict[str, object], body)
        if set(fields) != {
            "connectorAuthorityKeyId",
            "appCarrierCredentialDigest",
            "gatewayCarrierCredentialDigest",
        }:
            return httpx.Response(400, json={"code": "invalid_request"})
        try:
            for value in fields.values():
                _decode_exact(value, 32)
        except ValueError:
            return httpx.Response(400, json={"code": "invalid_request"})
        key_id = str(fields["connectorAuthorityKeyId"])
        app_digest = str(fields["appCarrierCredentialDigest"])
        gateway_digest = str(fields["gatewayCarrierCredentialDigest"])
        if app_digest == gateway_digest:
            return httpx.Response(400, json={"code": "invalid_request"})
        self.requests.append(fields)
        existing = self.routes.get(key_id)
        if existing is not None and existing[1:] != (app_digest, gateway_digest):
            return httpx.Response(409, json={"code": "already_registered"})
        if existing is None:
            existing = (_encode(secrets.token_bytes(32)), app_digest, gateway_digest)
            self.routes[key_id] = existing
        return httpx.Response(
            200,
            json={
                "routingLocator": existing[0],
                "connectorRegion": _encode(b"region01"),
                "appWebsocketUrl": "wss://relay.example.invalid/v1/carrier/app",
                "gatewayControlWebsocketUrl": (
                    "wss://relay.example.invalid/v1/connector/control"
                ),
                "gatewayDataWebsocketUrl": (
                    "wss://relay.example.invalid/v1/connector/data"
                ),
            },
        )


def _service(
    tmp_path: Path,
) -> tuple[OperatorPairingService, OperatorRelayConfigurationRepository]:
    """Create one isolated gateway with relay identity files under ``tmp_path``."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "authority-key.bin")
    store = EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3")
    repository = OperatorRelayConfigurationRepository(
        store,
        certificate_path=tmp_path / "relay-cert.pem",
        private_key_path=tmp_path / "relay-key.pem",
    )
    service = OperatorPairingService(
        store,
        provider,
        relay_repository=repository,
        now=lambda: datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc),
    )
    return service, repository


def _registrar(
    relay: _ContractRelay,
) -> Callable[[str, RelayRegistrationMaterial], OperatorRelayProvisioning]:
    """Return a registration call bound to the in-process relay."""

    def register(
        origin: str, material: RelayRegistrationMaterial
    ) -> OperatorRelayProvisioning:
        return register_relay_route(
            origin,
            material,
            transport=httpx.MockTransport(relay.handle),
            sleep=lambda _seconds: None,
        )

    return register


def test_material_sends_only_the_key_identifier_and_credential_digests() -> None:
    """Secrets stay on the node; the body binds exactly the contract's fields."""

    material = RelayRegistrationMaterial.generate()
    body = material.registration_body()

    assert set(body) == {
        "connectorAuthorityKeyId",
        "appCarrierCredentialDigest",
        "gatewayCarrierCredentialDigest",
    }
    private_key = serialization.load_der_private_key(
        _decode_exact_any(material.connector_authority_private_key_pkcs8),
        password=None,
    )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert body["connectorAuthorityKeyId"] == _encode(
        hashlib.sha256(public_key).digest()
    )
    assert body["appCarrierCredentialDigest"] == _encode(
        hashlib.sha256(_decode_exact(material.app_carrier_credential, 32)).digest()
    )
    assert body["gatewayCarrierCredentialDigest"] == _encode(
        hashlib.sha256(_decode_exact(material.gateway_carrier_credential, 32)).digest()
    )
    assert material.app_carrier_credential != material.gateway_carrier_credential
    serialized = json.dumps(body)
    for secret in (
        material.connector_authority_private_key_pkcs8,
        material.app_carrier_credential,
        material.gateway_carrier_credential,
        material.connector_authority_epoch,
    ):
        assert secret not in serialized
    assert len(_decode_exact(material.connector_authority_epoch, 16)) == 16


def _decode_exact_any(value: str) -> bytes:
    """Decode canonical unpadded base64url of any length."""

    return base64.urlsafe_b64decode(f"{value}{'=' * (-len(value) % 4)}")


def test_registration_combines_the_relay_route_with_the_node_secrets() -> None:
    """The result is validated on-demand provisioning; a retry gets the same route."""

    relay = _ContractRelay()
    material = RelayRegistrationMaterial.generate()
    register = _registrar(relay)

    provisioning = register(_RELAY_ORIGIN, material)
    repeated = register(_RELAY_ORIGIN, material)

    assert provisioning.version == 2
    assert provisioning.routing_locator == repeated.routing_locator
    assert provisioning.connector_region == _encode(b"region01")
    assert provisioning.app_carrier_credential == material.app_carrier_credential
    assert (
        provisioning.gateway_carrier_credential == material.gateway_carrier_credential
    )
    assert (
        provisioning.connector_authority_key_id == material.connector_authority_key_id
    )
    assert provisioning.connector_authority_epoch == material.connector_authority_epoch
    assert len(relay.routes) == 1


def test_transient_failures_retry_with_the_same_registration() -> None:
    """Network errors and uncoded 5xx answers repeat the identical request."""

    relay = _ContractRelay()
    answers = iter(("connect-error", "tunnel-503", "relay"))
    bodies: list[bytes] = []
    delays: list[float] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        answer = next(answers)
        if answer == "connect-error":
            raise httpx.ConnectError("synthetic", request=request)
        if answer == "tunnel-503":
            return httpx.Response(503, text="origin unreachable")
        return relay.handle(request)

    provisioning = register_relay_route(
        _RELAY_ORIGIN,
        RelayRegistrationMaterial.generate(),
        transport=httpx.MockTransport(flaky),
        sleep=delays.append,
    )

    assert provisioning.version == 2
    assert len(bodies) == 3 and len(set(bodies)) == 1
    assert delays == [0.5, 1.5]


def test_persistent_unreachability_reports_after_bounded_attempts() -> None:
    """Three failed attempts end with a typed unreachable failure."""

    attempts: list[int] = []

    def unreachable(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectTimeout("synthetic", request=request)

    with pytest.raises(RelayRegistrationError) as raised:
        register_relay_route(
            _RELAY_ORIGIN,
            RelayRegistrationMaterial.generate(),
            transport=httpx.MockTransport(unreachable),
            sleep=lambda _seconds: None,
        )

    assert raised.value.failure == "unreachable"
    assert len(attempts) == 3


@pytest.mark.parametrize(
    ("status", "code", "failure"),
    [
        (400, "invalid_request", "invalid_request"),
        (409, "already_registered", "already_registered"),
        (429, "rate_limited", "rate_limited"),
        (503, "registration_paused", "registration_paused"),
        (503, "capacity_exhausted", "capacity_exhausted"),
        (400, None, "invalid_request"),
        (418, None, "invalid_response"),
    ],
)
def test_coded_refusals_are_reported_without_retrying(
    status: int,
    code: str | None,
    failure: str,
) -> None:
    """A refusal names its stable code; only a rate limit carries a delay."""

    attempts: list[int] = []

    def refuse(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        body: dict[str, object] = {} if code is None else {"code": code, "retry": False}
        return httpx.Response(status, json=body, headers={"Retry-After": "30"})

    with pytest.raises(RelayRegistrationError) as raised:
        register_relay_route(
            _RELAY_ORIGIN,
            RelayRegistrationMaterial.generate(),
            transport=httpx.MockTransport(refuse),
            sleep=lambda _seconds: None,
        )

    assert raised.value.failure == failure
    assert raised.value.retry_after_seconds == (
        30 if failure == "rate_limited" else None
    )
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"routingLocator": _encode(bytes(32))}),
        httpx.Response(
            200,
            json={
                "routingLocator": _encode(bytes(32)),
                "connectorRegion": _encode(bytes(8)),
                "appWebsocketUrl": "https://relay.example.invalid/v1/carrier/app",
                "gatewayControlWebsocketUrl": (
                    "wss://relay.example.invalid/v1/connector/control"
                ),
                "gatewayDataWebsocketUrl": "wss://relay.example.invalid/v1/connector/data",
            },
        ),
        httpx.Response(200, content=b"{" + b" " * 20_000 + b"}"),
        httpx.Response(302, headers={"Location": "https://elsewhere.invalid/"}),
    ],
)
def test_invalid_answers_are_rejected(response: httpx.Response) -> None:
    """Malformed, incomplete, oversized, unsafe, or redirected answers fail closed."""

    with pytest.raises(RelayRegistrationError) as raised:
        register_relay_route(
            _RELAY_ORIGIN,
            RelayRegistrationMaterial.generate(),
            transport=httpx.MockTransport(lambda _request: response),
            sleep=lambda _seconds: None,
        )

    assert raised.value.failure == "invalid_response"


def test_unknown_response_fields_are_ignored() -> None:
    """A relay may add response fields without breaking older gateways."""

    relay = _ContractRelay()

    def with_extra_field(request: httpx.Request) -> httpx.Response:
        answer = relay.handle(request)
        body = cast(dict[str, object], json.loads(answer.content))
        body["routeExpiresAt"] = "later"
        return httpx.Response(200, json=body)

    provisioning = register_relay_route(
        _RELAY_ORIGIN,
        RelayRegistrationMaterial.generate(),
        transport=httpx.MockTransport(with_extra_field),
        sleep=lambda _seconds: None,
    )

    assert provisioning.version == 2


def test_origin_resolution_honors_the_switch_offline_mode_and_default() -> None:
    """Disabled, offline, and unconfigured nodes refuse before any network call."""

    with pytest.raises(RelayRegistrationError) as disabled:
        resolve_registration_origin(
            enabled=False, configured_origin=_RELAY_ORIGIN, offline=False
        )
    with pytest.raises(RelayRegistrationError) as offline:
        resolve_registration_origin(
            enabled=True, configured_origin=_RELAY_ORIGIN, offline=True
        )
    with pytest.raises(RelayRegistrationError) as unconfigured:
        resolve_registration_origin(enabled=True, configured_origin=None, offline=False)

    assert (disabled.value.failure, offline.value.failure) == ("disabled", "offline")
    assert unconfigured.value.failure == "not_configured"
    assert (
        resolve_registration_origin(
            enabled=True,
            configured_origin="HTTPS://Relay.Example.Invalid/",
            offline=False,
        )
        == _RELAY_ORIGIN
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://relay.example.invalid",
        "https://relay.example.invalid/v1",
        "https://relay.example.invalid?x=1",
        "https://user@relay.example.invalid",
        "ftp://relay.example.invalid",
        "relay.example.invalid",
    ],
)
def test_unsafe_relay_origins_are_refused_by_configuration(origin: str) -> None:
    """Cleartext, paths, queries, and credentials never reach registration."""

    with pytest.raises(ValueError):
        RelayConnectivityConfig(registration_url=origin)


def test_loopback_development_relay_may_use_http() -> None:
    """A local development relay is the only cleartext exception."""

    assert (
        RelayConnectivityConfig(
            registration_url="http://127.0.0.1:8787"
        ).registration_url
        == "http://127.0.0.1:8787"
    )


def test_service_registers_forgets_and_registers_again(tmp_path: Path) -> None:
    """A forgotten route stops loading; a new registration replaces its tombstone."""

    relay = _ContractRelay()
    service, repository = _service(tmp_path)

    first = service.register_relay(
        registration_origin=_RELAY_ORIGIN,
        operator_api_port=52417,
        register=_registrar(relay),
    )
    assert service.relay_configuration() == first
    with pytest.raises(OperatorRelayAlreadyConfiguredError):
        service.register_relay(
            registration_origin=_RELAY_ORIGIN,
            operator_api_port=52417,
            register=_registrar(relay),
        )

    assert service.forget_relay() is True
    assert service.relay_configuration() is None
    assert not (tmp_path / "relay-cert.pem").exists()
    assert not (tmp_path / "relay-key.pem").exists()
    assert service.forget_relay() is False
    with pytest.raises(OperatorRelayUnavailableError):
        repository.reserve_connector_generation()

    second = service.register_relay(
        registration_origin=_RELAY_ORIGIN,
        operator_api_port=52417,
        register=_registrar(relay),
    )
    assert second.routing_locator != first.routing_locator
    assert second.connector_authority_key_id != first.connector_authority_key_id
    assert service.relay_configuration() == second
    assert repository.reserve_connector_generation() == 1


def test_identity_files_left_by_an_interrupted_forget_do_not_block_registration(
    tmp_path: Path,
) -> None:
    """Files surviving a crash after the tombstone belong to the forgotten route."""

    relay = _ContractRelay()
    service, _ = _service(tmp_path)
    service.register_relay(
        registration_origin=_RELAY_ORIGIN,
        operator_api_port=52417,
        register=_registrar(relay),
    )
    stale_certificate = (tmp_path / "relay-cert.pem").read_bytes()
    stale_key = (tmp_path / "relay-key.pem").read_bytes()
    service.forget_relay()
    (tmp_path / "relay-cert.pem").write_bytes(stale_certificate)
    (tmp_path / "relay-key.pem").write_bytes(stale_key)

    service.register_relay(
        registration_origin=_RELAY_ORIGIN,
        operator_api_port=52417,
        register=_registrar(relay),
    )

    assert (tmp_path / "relay-cert.pem").read_bytes() != stale_certificate


def test_a_never_configured_gateway_never_replaces_existing_identity_files(
    tmp_path: Path,
) -> None:
    """Without a forgotten route, unexpected files are protected, not overwritten."""

    relay = _ContractRelay()
    service, _ = _service(tmp_path)
    (tmp_path / "relay-key.pem").write_bytes(b"someone else's key")

    with pytest.raises(OperatorRelayUnavailableError):
        service.register_relay(
            registration_origin=_RELAY_ORIGIN,
            operator_api_port=52417,
            register=_registrar(relay),
        )

    assert (tmp_path / "relay-key.pem").read_bytes() == b"someone else's key"
    assert service.relay_configuration() is None


def _use_cli_service(
    monkeypatch: pytest.MonkeyPatch,
    service: OperatorPairingService,
    relay: _ContractRelay,
    config: SkulkConfig | None,
) -> list[str]:
    """Route the CLI to the isolated gateway, relay, and configuration."""

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
    monkeypatch.setattr(pairing_module, "register_relay_route", _registrar(relay))
    monkeypatch.setattr(operator_cli, "load_skulk_config", lambda: config)
    payloads: list[str] = []
    monkeypatch.setattr(operator_cli, "_print_pairing_qr", payloads.append)
    return payloads


def _relay_config(*, enabled: bool = True) -> SkulkConfig:
    """Return a configuration naming the in-process relay."""

    return SkulkConfig(
        connectivity=ConnectivityConfig(
            relay=RelayConnectivityConfig(
                enabled=enabled,
                registration_url=_RELAY_ORIGIN,
            )
        )
    )


def test_pair_registers_a_relay_route_when_none_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A headless gateway pairs with one command and no hand-copied file."""

    relay = _ContractRelay()
    service, _ = _service(tmp_path)
    payloads = _use_cli_service(monkeypatch, service, relay, _relay_config())

    assert operator_cli.main(["pair"]) == 0

    assert service.relay_configuration() is not None
    assert len(relay.routes) == 1
    assert len(payloads) == 1 and payloads[0].startswith("skulk://pair?z=")
    assert "Registered this gateway with the relay" in capsys.readouterr().out

    assert operator_cli.main(["pair"]) == 0
    assert len(relay.requests) == 1


def test_pair_explains_how_to_pair_when_registration_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A disabled or unconfigured relay names the direct alternative."""

    relay = _ContractRelay()
    service, _ = _service(tmp_path)
    _use_cli_service(monkeypatch, service, relay, _relay_config(enabled=False))

    with pytest.raises(SystemExit):
        operator_cli.main(["pair"])

    error = capsys.readouterr().err
    assert "connectivity.relay.enabled is false" in error
    assert "--exchange-url" in error
    assert relay.requests == []


def test_configure_relay_refuses_a_second_route_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Installing over an existing route names forget-relay instead of a traceback."""

    relay = _ContractRelay()
    service, _ = _service(tmp_path)
    _use_cli_service(monkeypatch, service, relay, _relay_config())
    service.register_relay(
        registration_origin=_RELAY_ORIGIN,
        operator_api_port=52417,
        register=_registrar(relay),
    )
    material = RelayRegistrationMaterial.generate()
    provisioning_file = tmp_path / "provisioning.json"
    provisioning_file.write_text(
        _registrar(relay)(_RELAY_ORIGIN, material).model_dump_json(by_alias=True),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        operator_cli.main(
            ["configure-relay", "--provisioning-file", str(provisioning_file)]
        )

    assert "skulk operator forget-relay" in capsys.readouterr().err


def test_forget_relay_command_turns_remote_access_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forgetting reports what happens to paired phones and is idempotent."""

    relay = _ContractRelay()
    service, _ = _service(tmp_path)
    _use_cli_service(monkeypatch, service, relay, _relay_config())
    service.register_relay(
        registration_origin=_RELAY_ORIGIN,
        operator_api_port=52417,
        register=_registrar(relay),
    )

    assert operator_cli.main(["forget-relay"]) == 0
    assert "Forgot this gateway's relay route" in capsys.readouterr().out
    assert service.relay_configuration() is None
    assert operator_cli.main(["forget-relay"]) == 0
    assert "has no relay route" in capsys.readouterr().out
