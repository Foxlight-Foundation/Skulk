"""Designated-gateway material and outbound relay connector for operator access."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import ssl
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Literal, cast, final
from urllib.parse import urlsplit

import aiohttp
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from loguru import logger
from pydantic import Field, field_validator, model_validator

from skulk.operator.authority import (
    AuthorityNotInitializedError,
    EncryptedAuthorityStore,
)
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.utils.pydantic_ext import FrozenModel

_RELAY_RECORD_TYPE: Final = "operator_relay_configuration"
_RELAY_RECORD_ID: Final = "designated_gateway_v1"
_CARRIER_VALUE_BYTES: Final = 32
_ENCODED_CARRIER_VALUE_LENGTH: Final = 43
_DEFAULT_FRAME_BYTES: Final = 1024 * 1024
_MINIMUM_RECONNECT_SECONDS: Final = 0.25
_MAXIMUM_RECONNECT_SECONDS: Final = 5.0
_TLS_VALIDITY: Final = timedelta(days=3650)
_TLS_CLOCK_SKEW: Final = timedelta(minutes=5)
_ROUTE_HEADER: Final = "x-skulk-relay-route"


class OperatorRelayError(RuntimeError):
    """Base class for safe designated-gateway relay failures."""


class OperatorRelayAlreadyConfiguredError(OperatorRelayError):
    """Raised when v1 relay material already exists for this gateway."""


class OperatorRelayUnavailableError(OperatorRelayError):
    """Raised when persisted relay or TLS material cannot be used safely."""


class OperatorRelayProvisioning(FrozenModel):
    """One generated relay route delivered to the designated gateway."""

    version: Literal[1] = Field(description="Provisioning document format version.")
    app_websocket_url: str = Field(
        description="Outer WebSocket endpoint used by paired operator apps."
    )
    gateway_websocket_url: str = Field(
        description="Outer WebSocket endpoint used by gateway lanes."
    )
    routing_locator: str = Field(
        min_length=_ENCODED_CARRIER_VALUE_LENGTH,
        max_length=_ENCODED_CARRIER_VALUE_LENGTH,
        description="Opaque unpadded base64url relay route locator.",
    )
    app_carrier_credential: str = Field(
        min_length=_ENCODED_CARRIER_VALUE_LENGTH,
        max_length=_ENCODED_CARRIER_VALUE_LENGTH,
        description="Opaque app-role carrier bearer returned only during pairing.",
    )
    gateway_carrier_credential: str = Field(
        min_length=_ENCODED_CARRIER_VALUE_LENGTH,
        max_length=_ENCODED_CARRIER_VALUE_LENGTH,
        description="Opaque gateway-role carrier bearer retained only by Skulk.",
    )
    lane_count: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of independent waiting gateway WebSocket lanes.",
    )

    @field_validator("routing_locator", "app_carrier_credential", "gateway_carrier_credential")
    @classmethod
    def _carrier_values_are_canonical(cls, value: str) -> str:
        """Require exact 256-bit unpadded base64url relay values."""

        _decode_carrier_value(value)
        return value

    @field_validator("app_websocket_url")
    @classmethod
    def _app_url_is_safe(cls, value: str) -> str:
        """Require the fixed app carrier endpoint over safe transport."""

        return _validate_carrier_url(value, expected_path="/v1/carrier/app")

    @field_validator("gateway_websocket_url")
    @classmethod
    def _gateway_url_is_safe(cls, value: str) -> str:
        """Require the fixed gateway carrier endpoint over safe transport."""

        return _validate_carrier_url(value, expected_path="/v1/carrier/gateway")

    @model_validator(mode="after")
    def _roles_are_separate_on_one_relay(self) -> "OperatorRelayProvisioning":
        """Keep both role endpoints co-located and their credentials distinct."""

        app = urlsplit(self.app_websocket_url)
        gateway = urlsplit(self.gateway_websocket_url)
        if (app.scheme, app.hostname, app.port) != (
            gateway.scheme,
            gateway.hostname,
            gateway.port,
        ):
            raise ValueError("app and gateway carrier endpoints must share one relay origin")
        if self.app_carrier_credential == self.gateway_carrier_credential:
            raise ValueError("app and gateway carrier credentials must be distinct")
        return self


class OperatorRemoteAccessMaterial(FrozenModel):
    """Device-side relay and inner-TLS material returned once after pairing."""

    transport: Literal["paired_websocket_v1"] = Field(
        description="Selected remote carrier contract."
    )
    app_websocket_url: str = Field(
        description="Outer WebSocket endpoint used for each independent request."
    )
    routing_locator: str = Field(description="Opaque relay route locator.")
    app_carrier_credential: str = Field(
        description="Opaque app-role carrier bearer stored in platform secure storage."
    )
    gateway_server_name: str = Field(
        description="Inner TLS server name authenticated by the pinned certificate."
    )
    gateway_ca_certificate_pem: str = Field(
        description="PEM trust anchor for end-to-end TLS terminating at Skulk."
    )


class OperatorRelayConfiguration(FrozenModel):
    """Encrypted designated-gateway configuration plus protected TLS paths."""

    version: Literal[1] = Field(description="Persisted relay configuration version.")
    app_websocket_url: str = Field(description="Paired app carrier endpoint.")
    gateway_websocket_url: str = Field(description="Outbound gateway carrier endpoint.")
    routing_locator: str = Field(description="Opaque 256-bit relay locator.")
    app_carrier_credential: str = Field(description="App-role outer carrier bearer.")
    gateway_carrier_credential: str = Field(
        description="Gateway-role outer carrier bearer."
    )
    lane_count: int = Field(
        ge=1,
        le=32,
        description="Independent outbound gateway lanes to maintain.",
    )
    operator_api_port: int = Field(
        ge=1,
        le=65535,
        description="Loopback TLS port serving the scoped canonical API.",
    )
    gateway_server_name: str = Field(description="Pinned inner-TLS DNS name.")
    certificate_path: Path = Field(description="Protected gateway certificate path.")
    private_key_path: Path = Field(description="Owner-only gateway private-key path.")

    def device_material(self) -> OperatorRemoteAccessMaterial:
        """Load the public TLS certificate and project device-safe material.

        Returns:
            Relay app role, locator, and pinned inner-TLS trust material.

        Raises:
            OperatorRelayUnavailableError: The protected TLS files are absent
                or have unsafe private-key permissions.
        """

        _require_private_file(self.private_key_path)
        try:
            certificate = self.certificate_path.read_text(encoding="ascii")
        except (FileNotFoundError, OSError, UnicodeError) as exc:
            raise OperatorRelayUnavailableError(
                "operator gateway TLS certificate is unavailable"
            ) from exc
        try:
            x509.load_pem_x509_certificate(certificate.encode("ascii"))
        except ValueError as exc:
            raise OperatorRelayUnavailableError(
                "operator gateway TLS certificate is malformed"
            ) from exc
        return OperatorRemoteAccessMaterial(
            transport="paired_websocket_v1",
            app_websocket_url=self.app_websocket_url,
            routing_locator=self.routing_locator,
            app_carrier_credential=self.app_carrier_credential,
            gateway_server_name=self.gateway_server_name,
            gateway_ca_certificate_pem=certificate,
        )

    def server_ssl_context(self) -> ssl.SSLContext:
        """Build the TLS 1.3 server context for the relay-only API listener."""

        _require_private_file(self.private_key_path)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        try:
            context.load_cert_chain(self.certificate_path, self.private_key_path)
        except (OSError, ssl.SSLError) as exc:
            raise OperatorRelayUnavailableError(
                "operator gateway TLS identity is unavailable"
            ) from exc
        return context


@final
class OperatorRelayConfigurationRepository:
    """Persist one relay route in the encrypted local authority journal."""

    def __init__(
        self,
        store: EncryptedAuthorityStore,
        *,
        certificate_path: Path | None = None,
        private_key_path: Path | None = None,
    ) -> None:
        """Create a repository with injected authority and TLS file paths."""

        operator_directory = SKULK_CONFIG_HOME / "operator"
        self._store = store
        self._certificate_path = certificate_path or operator_directory / "relay-tls-v1.pem"
        self._private_key_path = private_key_path or operator_directory / "relay-tls-v1-key.pem"

    def configure(
        self,
        provisioning: OperatorRelayProvisioning,
        *,
        operator_api_port: int,
    ) -> OperatorRelayConfiguration:
        """Create protected TLS identity and persist one generated relay route.

        Args:
            provisioning: Generated route and role credentials from the relay.
            operator_api_port: Loopback-only TLS listener port used by this gateway.

        Returns:
            Persisted designated-gateway relay configuration.

        Raises:
            OperatorRelayAlreadyConfiguredError: A v1 route already exists.
            OperatorRelayUnavailableError: TLS files cannot be created safely.
        """

        if not 1 <= operator_api_port <= 65535:
            raise ValueError("operator_api_port must be between 1 and 65535")
        if self.load() is not None:
            raise OperatorRelayAlreadyConfiguredError(
                "operator relay is already configured; rotation is a later operation"
            )
        server_name = _gateway_server_name(provisioning.routing_locator)
        certificate_pem, private_key_pem = _generate_tls_identity(server_name)
        _write_create_only(self._private_key_path, private_key_pem, mode=0o600)
        try:
            _write_create_only(self._certificate_path, certificate_pem, mode=0o600)
            configuration = OperatorRelayConfiguration(
                version=1,
                app_websocket_url=provisioning.app_websocket_url,
                gateway_websocket_url=provisioning.gateway_websocket_url,
                routing_locator=provisioning.routing_locator,
                app_carrier_credential=provisioning.app_carrier_credential,
                gateway_carrier_credential=provisioning.gateway_carrier_credential,
                lane_count=provisioning.lane_count,
                operator_api_port=operator_api_port,
                gateway_server_name=server_name,
                certificate_path=self._certificate_path,
                private_key_path=self._private_key_path,
            )
            records = self._store.records()
            expected_commit_index = records[-1].commit_index
            self._store.append(
                expected_commit_index=expected_commit_index,
                expected_record_commit_index=0,
                authority_term=1,
                record_type=_RELAY_RECORD_TYPE,
                record_id=_RELAY_RECORD_ID,
                payload=cast(
                    Mapping[str, object],
                    configuration.model_dump(mode="json", by_alias=True),
                ),
            )
        except Exception:
            self._certificate_path.unlink(missing_ok=True)
            self._private_key_path.unlink(missing_ok=True)
            raise
        return configuration

    def load(self) -> OperatorRelayConfiguration | None:
        """Return the configured relay route, or ``None`` before provisioning."""

        try:
            payload = self._store.read_latest_payload(
                _RELAY_RECORD_TYPE,
                _RELAY_RECORD_ID,
            )
        except AuthorityNotInitializedError:
            return None
        try:
            return OperatorRelayConfiguration.model_validate_json(
                json.dumps(payload, separators=(",", ":"), allow_nan=False)
            )
        except ValueError as exc:
            raise OperatorRelayUnavailableError(
                "operator relay configuration is malformed"
            ) from exc


@final
class OperatorGatewayConnector:
    """Maintain bounded outbound WebSocket lanes to the content-blind relay."""

    def __init__(
        self,
        configuration: OperatorRelayConfiguration,
        *,
        frame_bytes: int = _DEFAULT_FRAME_BYTES,
    ) -> None:
        """Create one connector for a persisted designated-gateway route."""

        if not 1 <= frame_bytes <= _DEFAULT_FRAME_BYTES:
            raise ValueError("frame_bytes must be between 1 and 1048576")
        self._configuration = configuration
        self._frame_bytes = frame_bytes

    async def run(self) -> None:
        """Maintain the configured number of independent one-request lanes.

        Cancellation closes every lane and the shared HTTP client. Individual
        carrier or loopback failures reconnect with bounded exponential delay.
        """

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10.0, sock_read=None)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            asyncio.TaskGroup() as group,
        ):
            for lane_index in range(self._configuration.lane_count):
                group.create_task(
                    self._maintain_lane(session),
                    name=f"operator-relay-lane-{lane_index}",
                )

    async def _maintain_lane(self, session: aiohttp.ClientSession) -> None:
        """Reconnect one gateway lane until its owning task is cancelled."""

        delay = _MINIMUM_RECONNECT_SECONDS
        while True:
            try:
                await self._serve_lane(session)
                delay = _MINIMUM_RECONNECT_SECONDS
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, OSError, OperatorRelayError):
                logger.warning("Operator relay gateway lane disconnected; retrying")
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAXIMUM_RECONNECT_SECONDS)

    async def _serve_lane(self, session: aiohttp.ClientSession) -> None:
        """Bridge one relay WebSocket to the authenticated local TLS API."""

        headers = {
            "Authorization": f"Bearer {self._configuration.gateway_carrier_credential}",
            _ROUTE_HEADER: self._configuration.routing_locator,
        }
        async with session.ws_connect(
            self._configuration.gateway_websocket_url,
            headers=headers,
            heartbeat=20.0,
            autoping=True,
            autoclose=True,
            max_msg_size=self._frame_bytes,
        ) as websocket:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1",
                self._configuration.operator_api_port,
            )
            try:
                websocket_to_tls = asyncio.create_task(
                    self._websocket_to_tls(websocket, writer)
                )
                tls_to_websocket = asyncio.create_task(
                    self._tls_to_websocket(reader, websocket)
                )
                tasks = (websocket_to_tls, tls_to_websocket)
                _, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, asyncio.CancelledError):
                        continue
                    if isinstance(result, BaseException):
                        raise OperatorRelayError(
                            "operator relay lane forwarding failed"
                        ) from result
            finally:
                writer.close()
                await writer.wait_closed()

    async def _websocket_to_tls(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Copy opaque binary WebSocket messages into the inner TLS stream."""

        async for message in websocket:
            if message.type is aiohttp.WSMsgType.BINARY:
                payload = cast(bytes, message.data)
                if len(payload) > self._frame_bytes:
                    raise OperatorRelayError("operator relay frame exceeded its bound")
                writer.write(payload)
                await writer.drain()
                continue
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                return
            if message.type is aiohttp.WSMsgType.TEXT:
                raise OperatorRelayError("operator relay sent a non-binary frame")

    async def _tls_to_websocket(
        self,
        reader: asyncio.StreamReader,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        """Copy inner TLS records into bounded binary WebSocket messages."""

        while payload := await reader.read(self._frame_bytes):
            await websocket.send_bytes(payload)


def _decode_carrier_value(value: str) -> bytes:
    """Decode an exact unpadded 256-bit relay value."""

    if len(value) != _ENCODED_CARRIER_VALUE_LENGTH or "=" in value:
        raise ValueError("relay values must be unpadded base64url-encoded 32-byte values")
    try:
        decoded = base64.b64decode(
            f"{value}=",
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "relay values must be unpadded base64url-encoded 32-byte values"
        ) from exc
    if len(decoded) != _CARRIER_VALUE_BYTES:
        raise ValueError("relay values must decode to exactly 32 bytes")
    return decoded


def _validate_carrier_url(value: str, *, expected_path: str) -> str:
    """Require WSS except for loopback-only generated development relays."""

    parsed = urlsplit(value)
    if (
        parsed.path != expected_path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
    ):
        raise ValueError(f"relay URL must be an origin plus {expected_path}")
    if parsed.scheme == "wss":
        return value
    if parsed.scheme != "ws":
        raise ValueError("relay URL must use wss")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if not loopback:
        raise ValueError("cleartext relay URL is allowed only on loopback")
    return value


def _gateway_server_name(routing_locator: str) -> str:
    """Derive a stable private inner-TLS name without exposing the locator."""

    digest = hashlib.sha256(_decode_carrier_value(routing_locator)).hexdigest()
    return f"skulk-{digest[:32]}.remote"


def _generate_tls_identity(server_name: str) -> tuple[bytes, bytes]:
    """Generate one self-signed pinned TLS 1.3 gateway identity."""

    private_key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(tz=timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, server_name)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _TLS_CLOCK_SKEW)
        .not_valid_after(now + _TLS_VALIDITY)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(server_name)]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return certificate_pem, private_key_pem


def _write_create_only(path: Path, payload: bytes, *, mode: int) -> None:
    """Create one protected file without replacing existing material."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except OSError as exc:
        raise OperatorRelayUnavailableError(
            "operator gateway TLS identity already exists or cannot be created"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if os.name == "posix":
        path.chmod(mode)


def _require_private_file(path: Path) -> None:
    """Require a present TLS private key with owner-only POSIX permissions."""

    try:
        metadata = path.stat()
    except OSError as exc:
        raise OperatorRelayUnavailableError(
            "operator gateway TLS private key is unavailable"
        ) from exc
    if os.name == "posix" and metadata.st_mode & 0o077:
        raise OperatorRelayUnavailableError(
            "operator gateway TLS private key permissions are unsafe"
        )
