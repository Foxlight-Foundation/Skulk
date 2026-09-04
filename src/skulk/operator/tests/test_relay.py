"""Tests for designated-gateway relay provisioning and opaque lane bridging."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import socket
import ssl
import struct
import zlib
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast, final
from uuid import UUID

import aiohttp
import pytest
import qrcode
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID
from qrcode.constants import ERROR_CORRECT_L

import skulk.operator.cli as operator_cli
import skulk.operator.relay as relay_module
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import (
    OperatorPairingService,
    PairingChallengeRequest,
    PairingExchangeRequest,
    PairingInvitationPackage,
    PairingPackage,
    PairingPackageTooLargeError,
    pairing_signature_message,
)
from skulk.operator.relay import (
    OperatorGatewayConnector,
    OperatorRelayAlreadyConfiguredError,
    OperatorRelayConfigurationRepository,
    OperatorRelayProvisioning,
)


def _base64url(value: bytes) -> str:
    """Encode exact test carrier and device values."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _provisioning(origin: str = "ws://127.0.0.1:41000") -> OperatorRelayProvisioning:
    """Return one valid generated relay route."""

    return OperatorRelayProvisioning(
        version=1,
        app_websocket_url=f"{origin}/v1/carrier/app",
        gateway_websocket_url=f"{origin}/v1/carrier/gateway",
        routing_locator=_base64url(bytes(range(32))),
        app_carrier_credential=_base64url(bytes(range(32, 64))),
        gateway_carrier_credential=_base64url(bytes(range(64, 96))),
        lane_count=1,
    )


def _on_demand_provisioning(
    origin: str = "ws://127.0.0.1:41000",
) -> OperatorRelayProvisioning:
    """Return one valid generated on-demand relay route."""

    private_key = ec.generate_private_key(ec.SECP256R1())
    private_der = private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return OperatorRelayProvisioning(
        version=2,
        app_websocket_url=f"{origin}/v1/carrier/app",
        gateway_control_websocket_url=f"{origin}/v1/connector/control",
        gateway_data_websocket_url=f"{origin}/v1/connector/data",
        routing_locator=_base64url(bytes(range(32))),
        app_carrier_credential=_base64url(bytes(range(32, 64))),
        gateway_carrier_credential=_base64url(bytes(range(64, 96))),
        connector_authority_private_key_pkcs8=_base64url(private_der),
        connector_authority_key_id=_base64url(hashlib.sha256(public_der).digest()),
        connector_region=_base64url(bytes(range(8))),
        connector_authority_epoch=_base64url(bytes(range(16))),
    )


def _service(
    tmp_path: Path,
) -> tuple[OperatorPairingService, OperatorRelayConfigurationRepository]:
    """Create one isolated designated-gateway pairing and relay repository."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "authority-key.bin")
    store = EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3")
    repository = OperatorRelayConfigurationRepository(
        store,
        certificate_path=tmp_path / "relay-cert.pem",
        private_key_path=tmp_path / "relay-key.pem",
    )
    return (
        OperatorPairingService(
            store,
            provider,
            relay_repository=repository,
            now=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        ),
        repository,
    )


@final
class _RecordingWebSocket:
    """Record gateway-to-app binary frames without opening a relay socket."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    async def send_bytes(self, payload: bytes) -> None:
        """Retain one exact outbound binary frame."""

        self.frames.append(payload)


@pytest.mark.asyncio
async def test_gateway_default_frames_fit_native_client_buffer(tmp_path: Path) -> None:
    """Large TLS responses are split before reaching the 64 KiB native pump."""

    service, _ = _service(tmp_path)
    configuration = service.configure_relay(
        _provisioning(),
        operator_api_port=52416,
    )
    connector = OperatorGatewayConnector(configuration)
    reader = asyncio.StreamReader()
    reader.feed_data(b"x" * ((64 * 1024) + 1))
    reader.feed_eof()
    websocket = _RecordingWebSocket()

    await connector.forward_tls_to_websocket(
        reader,
        cast(aiohttp.ClientWebSocketResponse, cast(object, websocket)),
    )

    assert list(map(len, websocket.frames)) == [64 * 1024, 1]


def test_gateway_aiohttp_receive_limit_is_above_inclusive_frame_bound() -> None:
    """The aiohttp exclusive receive limit must admit an exact 64 KiB frame."""

    assert relay_module._aiohttp_receive_limit_bytes(64 * 1024) == (64 * 1024) + 1  # pyright: ignore[reportPrivateUsage]


def test_relay_configuration_is_encrypted_and_returned_once_with_pairing(
    tmp_path: Path,
) -> None:
    """Pairing returns app material while gateway credentials stay encrypted."""

    service, repository = _service(tmp_path)
    provisioning = _provisioning()
    configuration = service.configure_relay(
        provisioning,
        operator_api_port=52416,
        cluster_name="Fox Den",
    )
    assert repository.load() == configuration
    assert configuration.private_key_path.stat().st_mode & 0o077 == 0
    authority_bytes = (tmp_path / "authority.sqlite3").read_bytes()
    assert provisioning.app_carrier_credential.encode("ascii") not in authority_bytes
    assert provisioning.gateway_carrier_credential.encode("ascii") not in authority_bytes

    package = service.create_session()
    assert package.version == 2
    assert package.remote_access is not None
    assert package.remote_access.app_carrier_credential == (
        provisioning.app_carrier_credential
    )
    assert package.remote_access.routing_locator == provisioning.routing_locator
    assert str(package.exchange_url) == (
        f"https://{configuration.gateway_server_name}/"
    )
    assert len(package.as_url().encode("utf-8")) <= 1_170
    code = qrcode.QRCode(
        border=2,
        error_correction=ERROR_CORRECT_L,
    )
    code.add_data(package.as_url())
    code.make(fit=True)
    terminal_output = io.StringIO()
    code.print_ascii(out=terminal_output, invert=True)
    assert max(map(len, terminal_output.getvalue().splitlines())) <= 120
    encoded = package.as_url().split("?z=", maxsplit=1)[1]
    decoded_payload = cast(
        object,
        json.loads(
            zlib.decompress(
                base64.urlsafe_b64decode(f"{encoded}{'=' * (-len(encoded) % 4)}")
            )
        ),
    )
    assert isinstance(decoded_payload, dict)
    compact_payload = cast(dict[str, object], decoded_payload)
    assert compact_payload["v"] == 2
    assert compact_payload["i"] == str(package.cluster_id)
    remote_payload_value = compact_payload["r"]
    assert isinstance(remote_payload_value, dict)
    remote_payload = cast(dict[str, object], remote_payload_value)
    assert remote_payload["c"] == provisioning.app_carrier_credential
    assert provisioning.gateway_carrier_credential not in package.model_dump_json()

    direct_package = service.create_session(
        exchange_url="https://gateway.example.invalid"
    )
    assert direct_package.version == 1
    assert direct_package.remote_access is None
    device_private_key = Ed25519PrivateKey.generate()
    device_public_key = device_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    challenge = service.create_challenge(
        PairingChallengeRequest(
            nonce=package.nonce,
            device_name="Phone",
            device_public_key=_base64url(device_public_key),
        )
    )
    signature = device_private_key.sign(
        pairing_signature_message(
            cluster_id=UUID(str(package.cluster_id)),
            nonce=package.nonce,
            challenge=challenge.challenge,
        )
    )
    exchange = service.exchange(
        PairingExchangeRequest(
            nonce=package.nonce,
            signature=_base64url(signature),
        )
    )

    assert exchange.remote_access is not None
    assert exchange.remote_access.app_carrier_credential == (
        provisioning.app_carrier_credential
    )
    assert exchange.remote_access.routing_locator == provisioning.routing_locator
    assert "BEGIN CERTIFICATE" in exchange.remote_access.gateway_ca_certificate_pem
    certificate = x509.load_pem_x509_certificate(
        exchange.remote_access.gateway_ca_certificate_pem.encode("ascii")
    )
    assert not certificate.extensions.get_extension_for_class(
        x509.BasicConstraints
    ).value.ca
    assert certificate.extensions.get_extension_for_class(
        x509.ExtendedKeyUsage
    ).value == x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH])
    assert provisioning.gateway_carrier_credential not in exchange.model_dump_json()
    with pytest.raises(OperatorRelayAlreadyConfiguredError):
        service.configure_relay(provisioning, operator_api_port=52416)


def test_on_demand_configuration_reserves_generations_before_use(
    tmp_path: Path,
) -> None:
    """Every connector attempt advances durable encrypted fencing state."""

    service, repository = _service(tmp_path)
    provisioning = _on_demand_provisioning()
    configuration = service.configure_relay(
        provisioning,
        operator_api_port=52416,
    )

    assert configuration.version == 2
    assert configuration.connector_generation == 0
    assert service.reserve_relay_connector_generation() == 1
    assert service.reserve_relay_connector_generation() == 2
    persisted = repository.load()
    assert persisted is not None
    assert persisted.connector_generation == 2
    authority_bytes = (tmp_path / "authority.sqlite3").read_bytes()
    assert provisioning.connector_authority_private_key_pkcs8 is not None
    assert (
        provisioning.connector_authority_private_key_pkcs8.encode("ascii")
        not in authority_bytes
    )


def test_on_demand_provisioning_rejects_mismatched_connector_key() -> None:
    """A delegated signing key must match the relay-pinned public digest."""

    payload = _on_demand_provisioning().model_dump(mode="json")
    payload["connector_authority_key_id"] = _base64url(bytes(32))

    with pytest.raises(ValueError, match="connector authority material"):
        OperatorRelayProvisioning.model_validate(payload)


def test_on_demand_connector_requires_durable_generation_provider(
    tmp_path: Path,
) -> None:
    """Version 2 cannot start with process-local reusable generation state."""

    service, _ = _service(tmp_path)
    configuration = service.configure_relay(
        _on_demand_provisioning(),
        operator_api_port=52416,
    )

    with pytest.raises(ValueError, match="durable connector generations"):
        OperatorGatewayConnector(configuration)


def test_oversized_relay_package_is_rejected_before_session_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal-hostile relay QR must not leave a live nonce behind."""

    service, _ = _service(tmp_path)
    service.configure_relay(_provisioning(), operator_api_port=52416)

    def oversized_url(_package: PairingPackage) -> str:
        return "x" * 1_171

    monkeypatch.setattr(PairingPackage, "as_url", oversized_url)

    def unexpected_append(*_arguments: object) -> None:
        raise AssertionError("oversized pairing session was persisted")

    monkeypatch.setattr(OperatorPairingService, "_append_session", unexpected_append)

    with pytest.raises(PairingPackageTooLargeError, match="supported QR size"):
        service.create_session()


def test_oversized_invitation_is_rejected_before_invitation_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal-hostile reusable QR never creates invitation authority."""

    service, _ = _service(tmp_path)
    service.configure_relay(_provisioning(), operator_api_port=52416)

    def oversized_url(_package: PairingInvitationPackage) -> str:
        return "x" * 1_171

    monkeypatch.setattr(PairingInvitationPackage, "as_url", oversized_url)

    def unexpected_append(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("oversized pairing invitation was persisted")

    monkeypatch.setattr(
        OperatorPairingService,
        "_append_authority_record",
        unexpected_append,
    )

    with pytest.raises(PairingPackageTooLargeError, match="supported QR size"):
        service.create_invitation(lifetime=timedelta(days=90))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gateway_websocket_url", "ws://relay.example/v1/carrier/gateway"),
        ("app_websocket_url", "wss://relay.example/wrong"),
        ("routing_locator", "short"),
        ("gateway_carrier_credential", _base64url(bytes(range(32, 64)))),
    ),
)
def test_relay_provisioning_rejects_unsafe_or_ambiguous_material(
    field: str,
    value: str,
) -> None:
    """Provisioning fails closed before any secret is persisted."""

    payload = _provisioning().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError):
        OperatorRelayProvisioning.model_validate(payload)


def test_configure_relay_cli_never_echoes_rejected_secret_material(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid provisioning reports no credential-bearing validation detail."""

    secret = _base64url(bytes(range(32, 64)))
    provisioning_path = tmp_path / "invalid-provisioning.json"
    provisioning_path.write_text(
        json.dumps(
            {
                **_provisioning().model_dump(mode="json"),
                "app_carrier_credential": f"invalid-{secret}",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        operator_cli.main(
            ["configure-relay", "--provisioning-file", str(provisioning_path)]
        )

    captured = capsys.readouterr()
    assert "unreadable or invalid" in captured.err
    assert secret not in captured.err


def test_configure_relay_cli_uses_non_fabric_default_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator listener must not collide with Skulk's default fabric port."""

    service, repository = _service(tmp_path)
    provisioning_path = tmp_path / "relay-provisioning.json"
    provisioning_path.write_text(_provisioning().model_dump_json(), encoding="utf-8")

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

    assert operator_cli.DEFAULT_OPERATOR_API_PORT != 52416
    assert (
        operator_cli.main(
            ["configure-relay", "--provisioning-file", str(provisioning_path)]
        )
        == 0
    )
    configuration = repository.load()
    assert configuration is not None
    assert configuration.operator_api_port == operator_cli.DEFAULT_OPERATOR_API_PORT


def test_pair_cli_uses_configured_relay_without_direct_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal pairing needs no manually supplied phone-reachable URL."""

    service, _ = _service(tmp_path)
    service.configure_relay(_provisioning(), operator_api_port=52416)
    payloads: list[str] = []

    def service_from_default_paths(
        _service_type: type[OperatorPairingService],
    ) -> OperatorPairingService:
        """Return the isolated configured service for this CLI invocation."""

        return service

    monkeypatch.setattr(
        OperatorPairingService,
        "from_default_paths",
        classmethod(service_from_default_paths),
    )
    monkeypatch.setattr(operator_cli, "_print_pairing_qr", payloads.append)

    assert operator_cli.main(["pair"]) == 0
    assert len(payloads) == 1
    assert payloads[0].startswith("skulk://pair?z=")


def test_pair_cli_creates_protected_reusable_invitation_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit invitation flags preserve defaults and write a protected QR."""

    service, _ = _service(tmp_path)
    service.configure_relay(_provisioning(), operator_api_port=52416)
    payloads: list[str] = []

    def service_from_default_paths(
        _service_type: type[OperatorPairingService],
    ) -> OperatorPairingService:
        """Return the isolated configured service for this CLI invocation."""

        return service

    monkeypatch.setattr(
        OperatorPairingService,
        "from_default_paths",
        classmethod(service_from_default_paths),
    )
    monkeypatch.setattr(operator_cli, "_print_pairing_qr", payloads.append)
    qr_path = tmp_path / "review.png"

    assert (
        operator_cli.main(
            ["pair", "--valid-for", "90d", "--qr-output", str(qr_path)]
        )
        == 0
    )
    assert payloads[0].startswith("skulk://pair?z=")
    assert qr_path.read_bytes().startswith(b"\x89PNG")
    assert qr_path.stat().st_mode & 0o777 == 0o600
    invitation = service.invitations()[0]
    assert invitation.max_pairings == 10
    assert invitation.state == "active"
    output = capsys.readouterr().out
    assert "Treat the terminal QR" in output
    assert "secret bearer invitation" in output

    with pytest.raises(SystemExit):
        operator_cli.main(
            ["pair", "--valid-for", "90d", "--qr-output", str(qr_path)]
        )


def test_invitation_cli_lists_and_revokes_without_printing_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Host management exposes safe invitation status but never its capability."""

    service, _ = _service(tmp_path)
    package = service.create_invitation(
        lifetime=timedelta(days=1),
        exchange_url="https://example.invalid",
    )

    def service_from_default_paths(
        _service_type: type[OperatorPairingService],
    ) -> OperatorPairingService:
        """Return the isolated service for CLI management."""

        return service

    monkeypatch.setattr(
        OperatorPairingService,
        "from_default_paths",
        classmethod(service_from_default_paths),
    )

    assert operator_cli.main(["invitations", "list"]) == 0
    listed = capsys.readouterr().out
    assert str(package.invitation_id) in listed
    assert package.issued_at.isoformat() in listed
    assert package.expires_at.isoformat() in listed
    assert package.nonce not in listed
    assert "active" in listed

    assert (
        operator_cli.main(
            ["invitations", "revoke", str(package.invitation_id)]
        )
        == 0
    )
    capsys.readouterr()
    assert operator_cli.main(["invitations", "list"]) == 0
    assert "revoked" in capsys.readouterr().out


class _SyntheticRelay:
    """Minimal paired-WebSocket relay used only for connector integration."""

    def __init__(self, provisioning: OperatorRelayProvisioning) -> None:
        self._provisioning = provisioning
        self._gateway_queue: asyncio.Queue[
            tuple[web.WebSocketResponse, asyncio.Event]
        ] = asyncio.Queue()
        self.captured_app_bytes = bytearray()

    def use_provisioning(self, provisioning: OperatorRelayProvisioning) -> None:
        """Replace the placeholder with the route bound to the started port."""

        self._provisioning = provisioning

    async def gateway(self, request: web.Request) -> web.WebSocketResponse:
        """Admit one authenticated waiting gateway lane."""

        self._assert_headers(request, self._provisioning.gateway_carrier_credential)
        websocket = web.WebSocketResponse(max_msg_size=1024 * 1024)
        await websocket.prepare(request)
        completed = asyncio.Event()
        await self._gateway_queue.put((websocket, completed))
        await completed.wait()
        return websocket

    async def app(self, request: web.Request) -> web.WebSocketResponse:
        """Pair one authenticated app socket with the oldest gateway lane."""

        self._assert_headers(request, self._provisioning.app_carrier_credential)
        app = web.WebSocketResponse(max_msg_size=1024 * 1024)
        await app.prepare(request)
        gateway, completed = await asyncio.wait_for(
            self._gateway_queue.get(),
            timeout=2.0,
        )
        app_to_gateway = asyncio.create_task(
            self._copy(app, gateway, capture=self.captured_app_bytes)
        )
        gateway_to_app = asyncio.create_task(self._copy(gateway, app))
        tasks = (app_to_gateway, gateway_to_app)
        try:
            _, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            completed.set()
            await app.close()
            await gateway.close()
        return app

    def _assert_headers(self, request: web.Request, credential: str) -> None:
        """Require the generated locator and role bearer."""

        assert request.headers["x-skulk-relay-route"] == (
            self._provisioning.routing_locator
        )
        assert request.headers["Authorization"] == f"Bearer {credential}"

    @staticmethod
    async def _copy(
        source: web.WebSocketResponse,
        destination: web.WebSocketResponse,
        *,
        capture: bytearray | None = None,
    ) -> None:
        """Forward binary messages without interpreting their contents."""

        async for message in source:
            if message.type is not aiohttp.WSMsgType.BINARY:
                return
            payload = cast(bytes, message.data)
            if capture is not None:
                capture.extend(payload)
            await destination.send_bytes(payload)


def _test_control_fields(payload: bytes) -> tuple[int, dict[int, bytes]]:
    """Decode the small canonical subset needed by the independent test relay."""

    assert payload[:5] == b"SKRL\x01"
    kind = payload[5]
    assert payload[6:8] == b"\x00\x00"
    assert int.from_bytes(payload[8:12], "big") == len(payload) - 12
    fields: dict[int, bytes] = {}
    offset = 12
    while offset < len(payload):
        identifier, flags, reserved, length = struct.unpack(
            ">HBBI", payload[offset : offset + 8]
        )
        assert flags == 1 and reserved == 0
        offset += 8
        fields[identifier] = payload[offset : offset + length]
        offset += length
    assert offset == len(payload)
    return kind, fields


def _test_control_message(kind: int, *values: bytes) -> bytes:
    """Encode sequential required fields for the independent test relay."""

    payload = b"".join(
        struct.pack(">HBBI", index, 1, 0, len(value)) + value
        for index, value in enumerate(values, start=1)
    )
    return b"SKRL" + struct.pack(">BBHI", 1, kind, 0, len(payload)) + payload


@final
class _SyntheticOnDemandRelay:
    """Independent socket fixture for one signed control and requested data lane."""

    def __init__(self) -> None:
        self.provisioning: OperatorRelayProvisioning | None = None
        self.completed = asyncio.Event()
        self.control_connections = 0
        self.data_connections = 0
        self._connection_id = bytes([0x44]) * 16
        self._hello_proof = ""

    def _configuration(self) -> OperatorRelayProvisioning:
        """Return the fixture's installed protected route."""

        assert self.provisioning is not None
        return self.provisioning

    async def control(self, request: web.Request) -> web.WebSocketResponse:
        """Accept one hello and request one independent application connection."""

        provisioning = self._configuration()
        self._assert_gateway_headers(request, provisioning)
        websocket = web.WebSocketResponse(max_msg_size=65_537)
        await websocket.prepare(request)
        self.control_connections += 1
        hello_message = await websocket.receive(timeout=2.0)
        assert hello_message.type is aiohttp.WSMsgType.BINARY
        kind, fields = _test_control_fields(cast(bytes, hello_message.data))
        assert kind == 1
        assert fields[2] == base64.urlsafe_b64decode(
            f"{provisioning.routing_locator}="
        )
        connector_id = fields[4]
        epoch = fields[5]
        self._hello_proof = _base64url(fields[10])
        await websocket.send_bytes(
            _test_control_message(
                2,
                struct.pack(">HH", 1, 0),
                connector_id,
                epoch,
                fields[6],
                fields[7],
                fields[9],
                (5_000).to_bytes(4, "big"),
                (20_000).to_bytes(4, "big"),
                bytes(8),
            )
        )
        await websocket.send_bytes(
            _test_control_message(
                9,
                self._connection_id,
                (5_000).to_bytes(4, "big"),
            )
        )
        await asyncio.wait_for(self.completed.wait(), timeout=3.0)
        return websocket

    async def data(self, request: web.Request) -> web.WebSocketResponse:
        """Verify the control binding and exercise the opaque loopback bridge."""

        provisioning = self._configuration()
        self._assert_gateway_headers(request, provisioning)
        assert request.headers["x-skulk-relay-connection"] == _base64url(
            self._connection_id
        )
        assert request.headers["x-skulk-relay-admission"] == self._hello_proof
        websocket = web.WebSocketResponse(max_msg_size=65_537)
        await websocket.prepare(request)
        self.data_connections += 1
        accepted_message = await websocket.receive(timeout=2.0)
        assert accepted_message.type is aiohttp.WSMsgType.BINARY
        assert _test_control_fields(cast(bytes, accepted_message.data)) == (
            10,
            {1: self._connection_id},
        )
        await websocket.send_bytes(b"opaque-inner-tls")
        echoed = await websocket.receive(timeout=2.0)
        assert echoed.type is aiohttp.WSMsgType.BINARY
        assert cast(bytes, echoed.data) == b"opaque-inner-tls"
        self.completed.set()
        return websocket

    @staticmethod
    def _assert_gateway_headers(
        request: web.Request,
        provisioning: OperatorRelayProvisioning,
    ) -> None:
        """Require the unchanged route and gateway bearer contract."""

        assert request.headers["x-skulk-relay-route"] == provisioning.routing_locator
        assert request.headers["Authorization"] == (
            f"Bearer {provisioning.gateway_carrier_credential}"
        )


async def _start_on_demand_relay(
    relay: _SyntheticOnDemandRelay,
) -> tuple[web.AppRunner, int]:
    """Start the signed connector fixture on a pre-bound loopback socket."""

    application = web.Application()
    application.router.add_get("/v1/connector/control", relay.control)
    application.router.add_get("/v1/connector/data", relay.data)
    runner = web.AppRunner(application)
    await runner.setup()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = cast(tuple[str, int], listener.getsockname())[-1]
    await web.SockSite(runner, listener).start()
    return runner, port


@pytest.mark.asyncio
async def test_on_demand_connector_uses_control_plus_requested_data_socket(
    tmp_path: Path,
) -> None:
    """One control socket opens no warm lanes and bridges each explicit request."""

    async def echo(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        payload = await reader.read(64 * 1024)
        writer.write(payload)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    echo_server, echo_port = await _start_socket_server(echo)
    relay = _SyntheticOnDemandRelay()
    relay_runner, relay_port = await _start_on_demand_relay(relay)
    provisioning = _on_demand_provisioning(f"ws://127.0.0.1:{relay_port}")
    relay.provisioning = provisioning
    service, _ = _service(tmp_path)
    configuration = service.configure_relay(
        provisioning,
        operator_api_port=echo_port,
    )
    connector = OperatorGatewayConnector(
        configuration,
        next_connector_generation=service.reserve_relay_connector_generation,
        now_unix_millis=lambda: 1_000_000,
    )
    connector_task = asyncio.create_task(connector.run())
    try:
        await asyncio.wait_for(relay.completed.wait(), timeout=4.0)
        assert relay.control_connections == 1
        assert relay.data_connections == 1
    finally:
        connector_task.cancel()
        with suppress(asyncio.CancelledError):
            await connector_task
        await relay_runner.cleanup()
        echo_server.close()
        await echo_server.wait_closed()


async def _start_socket_server(
    handler: Callable[
        [asyncio.StreamReader, asyncio.StreamWriter],
        Awaitable[None],
    ],
    *,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[asyncio.AbstractServer, int]:
    """Start one loopback server on an operating-system-selected port."""

    server = await asyncio.start_server(
        handler,
        host="127.0.0.1",
        port=0,
        ssl=ssl_context,
    )
    sockets = server.sockets
    assert sockets is not None and len(sockets) == 1
    return server, cast(tuple[str, int], sockets[0].getsockname())[-1]


async def _start_relay(
    relay: _SyntheticRelay,
) -> tuple[web.AppRunner, int]:
    """Start the synthetic relay on a pre-bound loopback socket."""

    application = web.Application()
    application.router.add_get("/v1/carrier/app", relay.app)
    application.router.add_get("/v1/carrier/gateway", relay.gateway)
    runner = web.AppRunner(application)
    await runner.setup()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = cast(tuple[str, int], listener.getsockname())[-1]
    await web.SockSite(runner, listener).start()
    return runner, port


@pytest.mark.asyncio
async def test_gateway_connector_carries_real_inner_tls_without_plaintext_visibility(
    tmp_path: Path,
) -> None:
    """A canonical HTTP request crosses paired WebSockets under inner TLS."""

    service, _ = _service(tmp_path)
    initial = service.configure_relay(
        _provisioning(),
        operator_api_port=52416,
    )

    async def api_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2.0)
        assert b"GET /state HTTP/1.1" in request
        assert b"Authorization: Bearer inner-operator-token" in request
        body = b'{"cluster":"connector-proof"}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    operator_api, operator_api_port = await _start_socket_server(
        api_handler,
        ssl_context=initial.server_ssl_context(),
    )
    placeholder = _provisioning()
    synthetic_relay = _SyntheticRelay(placeholder)
    relay_runner, relay_port = await _start_relay(synthetic_relay)
    provisioning = _provisioning(f"ws://127.0.0.1:{relay_port}")
    synthetic_relay.use_provisioning(provisioning)
    configuration = initial.model_copy(
        update={
            "app_websocket_url": provisioning.app_websocket_url,
            "gateway_websocket_url": provisioning.gateway_websocket_url,
            "operator_api_port": operator_api_port,
        }
    )
    connector_task = asyncio.create_task(OperatorGatewayConnector(configuration).run())

    async with aiohttp.ClientSession() as app_session:

        async def app_adapter(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            headers = {
                "Authorization": f"Bearer {provisioning.app_carrier_credential}",
                "x-skulk-relay-route": provisioning.routing_locator,
            }
            async with app_session.ws_connect(
                provisioning.app_websocket_url,
                headers=headers,
                max_msg_size=1024 * 1024,
            ) as websocket:
                tcp_to_websocket = asyncio.create_task(
                    _tcp_to_websocket(reader, websocket)
                )
                websocket_to_tcp = asyncio.create_task(
                    _websocket_to_tcp(websocket, writer)
                )
                tasks = (tcp_to_websocket, websocket_to_tcp)
                _, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            await writer.wait_closed()

        adapter, adapter_port = await _start_socket_server(app_adapter)
        client_context = ssl.create_default_context(
            cadata=configuration.device_material().gateway_ca_certificate_pem
        )
        client_context.minimum_version = ssl.TLSVersion.TLSv1_3
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    "127.0.0.1",
                    adapter_port,
                    ssl=client_context,
                    server_hostname=configuration.gateway_server_name,
                ),
                timeout=5.0,
            )
            writer.write(
                b"GET /state HTTP/1.1\r\nHost: operator\r\n"
                b"Authorization: Bearer inner-operator-token\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), timeout=5.0)
            assert b"HTTP/1.1 200 OK" in response
            assert b"connector-proof" in response
            assert b"inner-operator-token" not in synthetic_relay.captured_app_bytes
            assert b"GET /state" not in synthetic_relay.captured_app_bytes
            writer.close()
            await writer.wait_closed()
        finally:
            adapter.close()
            await adapter.wait_closed()

    connector_task.cancel()
    with suppress(asyncio.CancelledError):
        await connector_task
    operator_api.close()
    await operator_api.wait_closed()
    await relay_runner.cleanup()


async def _tcp_to_websocket(
    reader: asyncio.StreamReader,
    websocket: aiohttp.ClientWebSocketResponse,
) -> None:
    """Copy local adapter bytes into app-role binary messages."""

    while payload := await reader.read(1024 * 1024):
        await websocket.send_bytes(payload)


async def _websocket_to_tcp(
    websocket: aiohttp.ClientWebSocketResponse,
    writer: asyncio.StreamWriter,
) -> None:
    """Copy app-role binary messages into the local TLS client stream."""

    async for message in websocket:
        if message.type is not aiohttp.WSMsgType.BINARY:
            return
        writer.write(cast(bytes, message.data))
        await writer.drain()
