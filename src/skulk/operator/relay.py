"""Designated-gateway material and outbound relay connector for operator access."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import ssl
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Final, Literal, cast, final
from urllib.parse import urlsplit

import aiohttp
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from loguru import logger
from pydantic import Field, field_validator, model_validator

from skulk.operator.authority import (
    AuthorityCommitConflictError,
    AuthorityNotInitializedError,
    EncryptedAuthorityStore,
)
from skulk.operator.relay_protocol import (
    ConnectorAccepted,
    DrainRequest,
    HeartbeatAcknowledgement,
    OpenConnection,
    RelayProtocolError,
    build_connector_hello,
    build_lease_renewal,
    decode_server_message,
    encode_connection_accepted,
    encode_drain_ack,
    encode_heartbeat,
    load_connector_private_key,
)
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.utils.pydantic_ext import FrozenModel

_RELAY_RECORD_TYPE: Final = "operator_relay_configuration"
_RELAY_RECORD_ID: Final = "designated_gateway_v1"
_CARRIER_VALUE_BYTES: Final = 32
_ENCODED_CARRIER_VALUE_LENGTH: Final = 43
# The native Rust carrier accepts one opaque ingress frame into a bounded
# 64 KiB session buffer. Larger HTTP responses remain streaming-safe because
# the gateway splits the TLS byte stream before the relay forwards it.
_DEFAULT_FRAME_BYTES: Final = 64 * 1024
_MAXIMUM_FRAME_BYTES: Final = 1024 * 1024
_MINIMUM_RECONNECT_SECONDS: Final = 0.25
_MAXIMUM_RECONNECT_SECONDS: Final = 5.0
_TLS_VALIDITY: Final = timedelta(days=3650)
_TLS_CLOCK_SKEW: Final = timedelta(minutes=5)
_ROUTE_HEADER: Final = "x-skulk-relay-route"
_CONNECTION_HEADER: Final = "x-skulk-relay-connection"
_ADMISSION_HEADER: Final = "x-skulk-relay-admission"
_CONNECTOR_AUTHORITY_TERM: Final = 1
_CONNECTOR_RENEWAL_SECONDS: Final = 120.0
_CONTROL_HELLO_TIMEOUT_SECONDS: Final = 5.0
_GENERATION_RESERVATION_RETRIES: Final = 3
# Each admitted lane consumes one relay WebSocket and one loopback TCP socket.
# Keep the gateway below ordinary per-process descriptor limits while allowing
# many paired devices to be active without the version-one warm-lane ceiling.
_MAXIMUM_ON_DEMAND_DATA_LANES: Final = 64


def _aiohttp_receive_limit_bytes(inclusive_frame_bytes: int) -> int:
    """Translate the inclusive carrier frame bound to aiohttp's exclusive limit."""

    return inclusive_frame_bytes + 1


class OperatorRelayError(RuntimeError):
    """Base class for safe designated-gateway relay failures."""


class OperatorRelayAlreadyConfiguredError(OperatorRelayError):
    """Raised when v1 relay material already exists for this gateway."""


class OperatorRelayUnavailableError(OperatorRelayError):
    """Raised when persisted relay or TLS material cannot be used safely."""


@final
class _OperatorRelayDrainRequestedError(OperatorRelayError):
    """Carry one authenticated relay drain deadline into reconnect policy."""

    def __init__(self, deadline_unix_millis: int) -> None:
        """Create a safe drain signal without retaining relay-provided text."""

        super().__init__("operator relay requested connector drain")
        self.deadline_unix_millis = deadline_unix_millis


@final
class _ConnectorAdmissionProof:
    """Hold the latest signed lease proof for subsequent data sockets."""

    def __init__(self, proof: bytes) -> None:
        """Create the shared control/data admission boundary."""

        self._header_value = _encode_base64url(proof)

    @property
    def header_value(self) -> str:
        """Return the current portable admission header value."""

        return self._header_value

    def replace(self, proof: bytes) -> None:
        """Publish one successfully transmitted renewal proof."""

        self._header_value = _encode_base64url(proof)


@final
class _OnDemandDataLaneCapacity:
    """Reserve a bounded number of event-loop-confined data lanes."""

    def __init__(self, limit: int) -> None:
        """Create one non-blocking admission counter."""

        if limit < 1:
            raise ValueError("on-demand data lane limit must be positive")
        self._limit = limit
        self._active = 0

    def try_reserve(self) -> bool:
        """Reserve one lane without queuing when capacity is exhausted."""

        if self._active >= self._limit:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        """Release one previously reserved lane."""

        if self._active < 1:
            raise RuntimeError("on-demand data lane capacity underflow")
        self._active -= 1


class OperatorRelayProvisioning(FrozenModel):
    """One generated relay route delivered to the designated gateway."""

    version: Literal[1, 2] = Field(description="Provisioning document format version.")
    app_websocket_url: str = Field(
        description="Outer WebSocket endpoint used by paired operator apps."
    )
    gateway_websocket_url: str | None = Field(
        default=None,
        description="Outer WebSocket endpoint used by gateway lanes."
    )
    gateway_control_websocket_url: str | None = Field(
        default=None,
        description="Signed on-demand connector control endpoint.",
    )
    gateway_data_websocket_url: str | None = Field(
        default=None,
        description="Independent on-demand connector data endpoint.",
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
    lane_count: int | None = Field(
        default=None,
        ge=1,
        le=32,
        description="Number of independent waiting gateway WebSocket lanes.",
    )
    connector_authority_private_key_pkcs8: str | None = Field(
        default=None,
        description="Unpadded base64url delegated P-256 PKCS8 private key.",
    )
    connector_authority_key_id: str | None = Field(
        default=None,
        description="Unpadded base64url SHA-256 connector public-key digest.",
    )
    connector_region: str | None = Field(
        default=None,
        description="Opaque unpadded base64url eight-byte relay region.",
    )
    connector_authority_epoch: str | None = Field(
        default=None,
        description="Opaque unpadded base64url sixteen-byte authority epoch.",
    )

    @model_validator(mode="before")
    @classmethod
    def _preserve_legacy_lane_default(cls, data: object) -> object:
        """Apply the historical four-lane default only to version one."""

        if not isinstance(data, dict):
            return data
        values = cast(dict[str, object], data)
        if (
            values.get("version") == 1
            and "lane_count" not in values
            and "laneCount" not in values
        ):
            return {**values, "lane_count": 4}
        return cast(object, values)

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
    def _gateway_url_is_safe(cls, value: str | None) -> str | None:
        """Require the fixed gateway carrier endpoint over safe transport."""

        if value is None:
            return None
        return _validate_carrier_url(value, expected_path="/v1/carrier/gateway")

    @field_validator("gateway_control_websocket_url")
    @classmethod
    def _control_url_is_safe(cls, value: str | None) -> str | None:
        """Require the fixed connector control endpoint over safe transport."""

        if value is None:
            return None
        return _validate_carrier_url(value, expected_path="/v1/connector/control")

    @field_validator("gateway_data_websocket_url")
    @classmethod
    def _data_url_is_safe(cls, value: str | None) -> str | None:
        """Require the fixed connector data endpoint over safe transport."""

        if value is None:
            return None
        return _validate_carrier_url(value, expected_path="/v1/connector/data")

    @model_validator(mode="after")
    def _roles_are_separate_on_one_relay(self) -> "OperatorRelayProvisioning":
        """Require one complete version-specific route on a single relay."""

        app = urlsplit(self.app_websocket_url)
        if self.app_carrier_credential == self.gateway_carrier_credential:
            raise ValueError("app and gateway carrier credentials must be distinct")
        if self.version == 1:
            if (
                self.gateway_websocket_url is None
                or self.lane_count is None
                or any(
                    value is not None
                    for value in (
                        self.gateway_control_websocket_url,
                        self.gateway_data_websocket_url,
                        self.connector_authority_private_key_pkcs8,
                        self.connector_authority_key_id,
                        self.connector_region,
                        self.connector_authority_epoch,
                    )
                )
            ):
                raise ValueError("version 1 provisioning fields are incomplete")
            gateway_urls = (self.gateway_websocket_url,)
        else:
            if (
                self.gateway_websocket_url is not None
                or self.lane_count is not None
                or any(
                    value is None
                    for value in (
                        self.gateway_control_websocket_url,
                        self.gateway_data_websocket_url,
                        self.connector_authority_private_key_pkcs8,
                        self.connector_authority_key_id,
                        self.connector_region,
                        self.connector_authority_epoch,
                    )
                )
            ):
                raise ValueError("version 2 provisioning fields are incomplete")
            _validate_connector_authority(self)
            gateway_urls = cast(
                tuple[str, str],
                (self.gateway_control_websocket_url, self.gateway_data_websocket_url),
            )
        if any(
            (urlsplit(url).scheme, urlsplit(url).hostname, urlsplit(url).port)
            != (app.scheme, app.hostname, app.port)
            for url in gateway_urls
        ):
            raise ValueError("app and gateway endpoints must share one relay origin")
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

    version: Literal[1, 2] = Field(description="Persisted relay configuration version.")
    app_websocket_url: str = Field(description="Paired app carrier endpoint.")
    gateway_websocket_url: str | None = Field(
        default=None,
        description="Legacy outbound gateway carrier endpoint.",
    )
    gateway_control_websocket_url: str | None = Field(
        default=None,
        description="Signed connector control endpoint.",
    )
    gateway_data_websocket_url: str | None = Field(
        default=None,
        description="Independent connector data endpoint.",
    )
    routing_locator: str = Field(description="Opaque 256-bit relay locator.")
    app_carrier_credential: str = Field(description="App-role outer carrier bearer.")
    gateway_carrier_credential: str = Field(
        description="Gateway-role outer carrier bearer."
    )
    lane_count: int | None = Field(
        default=None,
        ge=1,
        le=32,
        description="Independent outbound gateway lanes to maintain.",
    )
    connector_authority_private_key_pkcs8: str | None = Field(
        default=None,
        description="Encrypted-at-rest delegated P-256 connector signing key.",
    )
    connector_authority_key_id: str | None = Field(
        default=None,
        description="SHA-256 identifier pinned by the relay.",
    )
    connector_region: str | None = Field(
        default=None,
        description="Opaque relay region assigned to this connector.",
    )
    connector_authority_epoch: str | None = Field(
        default=None,
        description="Opaque fencing epoch assigned to this connector authority.",
    )
    connector_generation: int = Field(
        default=0,
        ge=0,
        le=(2**64) - 1,
        description="Greatest connector generation durably reserved by Skulk.",
    )
    operator_api_port: int = Field(
        ge=1,
        le=65535,
        description="Loopback TLS port serving the scoped canonical API.",
    )
    gateway_server_name: str = Field(description="Pinned inner-TLS DNS name.")
    certificate_path: Path = Field(description="Protected gateway certificate path.")
    private_key_path: Path = Field(description="Owner-only gateway private-key path.")

    @model_validator(mode="after")
    def _transport_is_complete(self) -> "OperatorRelayConfiguration":
        """Reject mixed legacy and on-demand persisted transport fields."""

        if self.version == 1:
            if (
                self.gateway_websocket_url is None
                or self.lane_count is None
                or self.connector_generation != 0
                or any(
                    value is not None
                    for value in (
                        self.gateway_control_websocket_url,
                        self.gateway_data_websocket_url,
                        self.connector_authority_private_key_pkcs8,
                        self.connector_authority_key_id,
                        self.connector_region,
                        self.connector_authority_epoch,
                    )
                )
            ):
                raise ValueError("version 1 relay configuration is incomplete")
        elif (
            self.gateway_websocket_url is not None
            or self.lane_count is not None
            or any(
                value is None
                for value in (
                    self.gateway_control_websocket_url,
                    self.gateway_data_websocket_url,
                    self.connector_authority_private_key_pkcs8,
                    self.connector_authority_key_id,
                    self.connector_region,
                    self.connector_authority_epoch,
                )
            )
        ):
            raise ValueError("version 2 relay configuration is incomplete")
        return self

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
                version=provisioning.version,
                app_websocket_url=provisioning.app_websocket_url,
                gateway_websocket_url=provisioning.gateway_websocket_url,
                gateway_control_websocket_url=(
                    provisioning.gateway_control_websocket_url
                ),
                gateway_data_websocket_url=provisioning.gateway_data_websocket_url,
                routing_locator=provisioning.routing_locator,
                app_carrier_credential=provisioning.app_carrier_credential,
                gateway_carrier_credential=provisioning.gateway_carrier_credential,
                lane_count=provisioning.lane_count,
                connector_authority_private_key_pkcs8=(
                    provisioning.connector_authority_private_key_pkcs8
                ),
                connector_authority_key_id=provisioning.connector_authority_key_id,
                connector_region=provisioning.connector_region,
                connector_authority_epoch=provisioning.connector_authority_epoch,
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

    def reserve_connector_generation(self) -> int:
        """Durably reserve and return the next on-demand connector generation.

        The journal update completes before a signed hello can use the value, so
        process crashes may skip generations but can never reuse one.

        Returns:
            Strictly increasing generation for one connector control attempt.

        Raises:
            OperatorRelayUnavailableError: Configuration is absent, legacy, at
                the generation ceiling, or repeatedly changed concurrently.
        """

        for _ in range(_GENERATION_RESERVATION_RETRIES):
            try:
                record, payload = self._store.read_latest_record_payload(
                    _RELAY_RECORD_TYPE,
                    _RELAY_RECORD_ID,
                )
                configuration = OperatorRelayConfiguration.model_validate_json(
                    json.dumps(payload, separators=(",", ":"), allow_nan=False)
                )
            except (AuthorityNotInitializedError, ValueError) as exc:
                raise OperatorRelayUnavailableError(
                    "on-demand operator relay is not configured"
                ) from exc
            if configuration.version != 2:
                raise OperatorRelayUnavailableError(
                    "on-demand operator relay is not configured"
                )
            if configuration.connector_generation == (2**64) - 1:
                raise OperatorRelayUnavailableError(
                    "operator relay connector generation is exhausted"
                )
            records = self._store.records()
            expected_commit_index = records[-1].commit_index
            next_generation = configuration.connector_generation + 1
            next_configuration = configuration.model_copy(
                update={"connector_generation": next_generation}
            )
            try:
                self._store.append(
                    expected_commit_index=expected_commit_index,
                    expected_record_commit_index=record.commit_index,
                    authority_term=_CONNECTOR_AUTHORITY_TERM,
                    record_type=_RELAY_RECORD_TYPE,
                    record_id=_RELAY_RECORD_ID,
                    payload=cast(
                        Mapping[str, object],
                        next_configuration.model_dump(mode="json", by_alias=True),
                    ),
                )
            except AuthorityCommitConflictError:
                continue
            return next_generation
        raise OperatorRelayUnavailableError(
            "operator relay connector generation changed concurrently"
        )


@final
class OperatorGatewayConnector:
    """Maintain legacy warm lanes or one signed on-demand relay connector."""

    def __init__(
        self,
        configuration: OperatorRelayConfiguration,
        *,
        frame_bytes: int = _DEFAULT_FRAME_BYTES,
        next_connector_generation: Callable[[], int] | None = None,
        now_unix_millis: Callable[[], int] | None = None,
    ) -> None:
        """Create one connector for a persisted designated-gateway route.

        Args:
            configuration: Encrypted-at-rest relay route configuration.
            frame_bytes: Maximum application carrier frame size.
            next_connector_generation: Durable generation reservation used only
                by version 2 before each signed control attempt.
            now_unix_millis: Injectable wall clock for signed lease timestamps.
        """

        if not 1 <= frame_bytes <= _MAXIMUM_FRAME_BYTES:
            raise ValueError("frame_bytes must be between 1 and 1048576")
        if configuration.version == 2 and next_connector_generation is None:
            raise ValueError("version 2 requires durable connector generations")
        self._configuration = configuration
        self._frame_bytes = frame_bytes
        self._next_connector_generation = next_connector_generation
        self._now_unix_millis = now_unix_millis or (
            lambda: time.time_ns() // 1_000_000
        )
        self._on_demand_data_lanes = _OnDemandDataLaneCapacity(
            _MAXIMUM_ON_DEMAND_DATA_LANES
        )

    async def run(self) -> None:
        """Maintain the configured legacy lanes or signed on-demand control.

        Cancellation closes every control/data socket and the shared HTTP
        client. Individual carrier failures reconnect with bounded exponential
        delay; version-2 generations are durably advanced before each attempt.
        """

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10.0, sock_read=None)
        if self._configuration.version == 2:
            await self._run_on_demand(timeout)
            return
        lane_count = self._configuration.lane_count
        if lane_count is None:
            raise OperatorRelayUnavailableError("legacy relay lane count is unavailable")
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            asyncio.TaskGroup() as group,
        ):
            for lane_index in range(lane_count):
                group.create_task(
                    self._maintain_legacy_lane(session),
                    name=f"operator-relay-lane-{lane_index}",
                )

    async def _maintain_legacy_lane(self, session: aiohttp.ClientSession) -> None:
        """Reconnect one gateway lane until its owning task is cancelled."""

        delay = _MINIMUM_RECONNECT_SECONDS
        while True:
            try:
                await self._serve_legacy_lane(session)
                delay = _MINIMUM_RECONNECT_SECONDS
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, OSError, OperatorRelayError):
                logger.warning("Operator relay gateway lane disconnected; retrying")
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAXIMUM_RECONNECT_SECONDS)

    async def _serve_legacy_lane(self, session: aiohttp.ClientSession) -> None:
        """Bridge one relay WebSocket to the authenticated local TLS API."""

        gateway_websocket_url = self._configuration.gateway_websocket_url
        if gateway_websocket_url is None:
            raise OperatorRelayUnavailableError("legacy gateway URL is unavailable")
        headers = {
            "Authorization": f"Bearer {self._configuration.gateway_carrier_credential}",
            _ROUTE_HEADER: self._configuration.routing_locator,
        }
        async with session.ws_connect(
            gateway_websocket_url,
            headers=headers,
            heartbeat=20.0,
            autoping=True,
            autoclose=True,
            max_msg_size=_aiohttp_receive_limit_bytes(self._frame_bytes),
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
                    self.forward_tls_to_websocket(reader, websocket)
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

    async def _run_on_demand(self, timeout: aiohttp.ClientTimeout) -> None:
        """Maintain one signed control socket and independent requested data lanes."""

        delay = _MINIMUM_RECONNECT_SECONDS
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            asyncio.TaskGroup() as data_tasks,
        ):
            while True:
                try:
                    generation_provider = self._next_connector_generation
                    if generation_provider is None:
                        raise OperatorRelayUnavailableError(
                            "connector generation provider is unavailable"
                        )
                    connector_generation = generation_provider()
                    await self._serve_control_connection(
                        session,
                        data_tasks,
                        connector_generation,
                    )
                    raise OperatorRelayError(
                        "operator relay control connection ended"
                    )
                except asyncio.CancelledError:
                    raise
                except _OperatorRelayDrainRequestedError as exc:
                    drain_seconds = max(
                        _MINIMUM_RECONNECT_SECONDS,
                        min(
                            300.0,
                            (
                                exc.deadline_unix_millis
                                - self._now_unix_millis()
                            )
                            / 1_000,
                        ),
                    )
                    await asyncio.sleep(drain_seconds)
                    delay = _MINIMUM_RECONNECT_SECONDS
                except (
                    aiohttp.ClientError,
                    OSError,
                    OperatorRelayError,
                    RelayProtocolError,
                ):
                    logger.warning(
                        "Operator relay control connection disconnected; retrying"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _MAXIMUM_RECONNECT_SECONDS)

    async def _serve_control_connection(
        self,
        session: aiohttp.ClientSession,
        data_tasks: asyncio.TaskGroup,
        connector_generation: int,
    ) -> None:
        """Authenticate one control socket and dispatch its data-lane requests."""

        control_url = self._configuration.gateway_control_websocket_url
        if control_url is None:
            raise OperatorRelayUnavailableError("connector control URL is unavailable")
        private_key, key_id, locator, region, epoch = self._connector_authority()
        connector_id = os.urandom(16)
        hello, lease = build_connector_hello(
            private_key=private_key,
            authority_key_id=key_id,
            routing_locator=locator,
            region_id=region,
            connector_id=connector_id,
            authority_epoch=epoch,
            authority_term=_CONNECTOR_AUTHORITY_TERM,
            connector_generation=connector_generation,
            now_unix_millis=self._now_unix_millis(),
        )
        headers = self._gateway_headers()
        async with session.ws_connect(
            control_url,
            headers=headers,
            heartbeat=20.0,
            autoping=True,
            autoclose=True,
            max_msg_size=_aiohttp_receive_limit_bytes(65_536),
        ) as websocket:
            await websocket.send_bytes(hello)
            first_message = await websocket.receive(
                timeout=_CONTROL_HELLO_TIMEOUT_SECONDS
            )
            if first_message.type is not aiohttp.WSMsgType.BINARY:
                raise OperatorRelayError("operator relay rejected connector control")
            accepted = decode_server_message(
                cast(bytes, first_message.data),
                connector_id=connector_id,
                authority_epoch=epoch,
                authority_term=_CONNECTOR_AUTHORITY_TERM,
                connector_generation=connector_generation,
            )
            if not isinstance(accepted, ConnectorAccepted) or (
                accepted.lease_expires_at_unix_millis
                != lease.expires_at_unix_millis
            ):
                raise OperatorRelayError("operator relay returned invalid acceptance")
            admission = _ConnectorAdmissionProof(lease.proof)
            receiver = asyncio.create_task(
                self._receive_control_messages(
                    session,
                    data_tasks,
                    websocket,
                    connector_id=connector_id,
                    authority_epoch=epoch,
                    connector_generation=connector_generation,
                    admission=admission,
                ),
                name="operator-relay-control-receiver",
            )
            heartbeat = asyncio.create_task(
                self._maintain_control_authority(
                    websocket,
                    heartbeat_seconds=accepted.heartbeat_millis / 1_000,
                    private_key=private_key,
                    key_id=key_id,
                    locator=locator,
                    region=region,
                    connector_id=connector_id,
                    epoch=epoch,
                    connector_generation=connector_generation,
                    admission=admission,
                ),
                name="operator-relay-control-heartbeat",
            )
            tasks = (receiver, heartbeat)
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, _OperatorRelayDrainRequestedError):
                    raise result
                if isinstance(result, BaseException):
                    raise OperatorRelayError(
                        "operator relay control task failed"
                    ) from result

    async def _receive_control_messages(
        self,
        session: aiohttp.ClientSession,
        data_tasks: asyncio.TaskGroup,
        websocket: aiohttp.ClientWebSocketResponse,
        *,
        connector_id: bytes,
        authority_epoch: bytes,
        connector_generation: int,
        admission: _ConnectorAdmissionProof,
    ) -> None:
        """Receive bounded control messages and start requested data sockets."""

        async for message in websocket:
            if message.type is aiohttp.WSMsgType.BINARY:
                decoded = decode_server_message(
                    cast(bytes, message.data),
                    connector_id=connector_id,
                    authority_epoch=authority_epoch,
                    authority_term=_CONNECTOR_AUTHORITY_TERM,
                    connector_generation=connector_generation,
                )
                if isinstance(decoded, OpenConnection):
                    if not self._on_demand_data_lanes.try_reserve():
                        logger.warning(
                            "Operator relay data-lane capacity reached; "
                            "declining connection"
                        )
                        continue
                    data_coroutine = self._serve_reserved_on_demand_data(
                        session,
                        decoded,
                        admission.header_value,
                    )
                    try:
                        data_tasks.create_task(
                            data_coroutine,
                            name="operator-relay-data-lane",
                        )
                    except BaseException:
                        data_coroutine.close()
                        self._on_demand_data_lanes.release()
                        raise
                    continue
                if isinstance(decoded, HeartbeatAcknowledgement):
                    continue
                if isinstance(decoded, DrainRequest):
                    await websocket.send_bytes(
                        encode_drain_ack(decoded.deadline_unix_millis)
                    )
                    raise _OperatorRelayDrainRequestedError(
                        decoded.deadline_unix_millis
                    )
                raise OperatorRelayError("operator relay sent unexpected control state")
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                return
            if message.type is aiohttp.WSMsgType.TEXT:
                raise OperatorRelayError("operator relay sent text control data")

    async def _maintain_control_authority(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        *,
        heartbeat_seconds: float,
        private_key: ec.EllipticCurvePrivateKey,
        key_id: bytes,
        locator: bytes,
        region: bytes,
        connector_id: bytes,
        epoch: bytes,
        connector_generation: int,
        admission: _ConnectorAdmissionProof,
    ) -> None:
        """Send canonical heartbeats and renew the signed lease before expiry."""

        sequence = 0
        next_renewal = time.monotonic() + _CONNECTOR_RENEWAL_SECONDS
        while True:
            await asyncio.sleep(heartbeat_seconds)
            sequence += 1
            await websocket.send_bytes(
                encode_heartbeat(sequence, self._now_unix_millis())
            )
            if time.monotonic() < next_renewal:
                continue
            renewal, lease = build_lease_renewal(
                private_key=private_key,
                authority_key_id=key_id,
                routing_locator=locator,
                region_id=region,
                connector_id=connector_id,
                authority_epoch=epoch,
                authority_term=_CONNECTOR_AUTHORITY_TERM,
                connector_generation=connector_generation,
                now_unix_millis=self._now_unix_millis(),
            )
            await websocket.send_bytes(renewal)
            admission.replace(lease.proof)
            next_renewal = time.monotonic() + _CONNECTOR_RENEWAL_SECONDS

    async def _serve_reserved_on_demand_data(
        self,
        session: aiohttp.ClientSession,
        request: OpenConnection,
        admission: str,
    ) -> None:
        """Serve one admitted data lane and always release its reservation."""

        try:
            await self._serve_on_demand_data(session, request, admission)
        finally:
            self._on_demand_data_lanes.release()

    async def _serve_on_demand_data(
        self,
        session: aiohttp.ClientSession,
        request: OpenConnection,
        admission: str,
    ) -> None:
        """Claim one relay connection and bridge it to the loopback TLS API."""

        data_url = self._configuration.gateway_data_websocket_url
        if data_url is None:
            raise OperatorRelayUnavailableError("connector data URL is unavailable")
        headers = {
            **self._gateway_headers(),
            _CONNECTION_HEADER: _encode_base64url(request.connection_id),
            _ADMISSION_HEADER: admission,
        }
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (request.timeout_millis / 1_000)
        websocket: aiohttp.ClientWebSocketResponse | None = None
        writer: asyncio.StreamWriter | None = None
        try:
            websocket = await asyncio.wait_for(
                session.ws_connect(
                    data_url,
                    headers=headers,
                    heartbeat=20.0,
                    autoping=True,
                    autoclose=True,
                    max_msg_size=_aiohttp_receive_limit_bytes(self._frame_bytes),
                ),
                timeout=max(0.001, deadline - loop.time()),
            )
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    "127.0.0.1", self._configuration.operator_api_port
                ),
                timeout=max(0.001, deadline - loop.time()),
            )
            await asyncio.wait_for(
                websocket.send_bytes(
                    encode_connection_accepted(request.connection_id)
                ),
                timeout=max(0.001, deadline - loop.time()),
            )
            websocket_to_tls = asyncio.create_task(
                self._websocket_to_tls(websocket, writer)
            )
            tls_to_websocket = asyncio.create_task(
                self.forward_tls_to_websocket(reader, websocket)
            )
            tasks = (websocket_to_tls, tls_to_websocket)
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except (TimeoutError, aiohttp.ClientError, OSError, RelayProtocolError):
            logger.warning("Operator relay data connection failed")
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    logger.debug(
                        "Operator relay loopback connection reset during cleanup"
                    )
            if websocket is not None:
                try:
                    await websocket.close()
                except (aiohttp.ClientError, OSError):
                    logger.debug("Operator relay data socket failed during cleanup")

    def _gateway_headers(self) -> dict[str, str]:
        """Return role-separated outer carrier authentication headers."""

        return {
            "Authorization": f"Bearer {self._configuration.gateway_carrier_credential}",
            _ROUTE_HEADER: self._configuration.routing_locator,
        }

    def _connector_authority(
        self,
    ) -> tuple[ec.EllipticCurvePrivateKey, bytes, bytes, bytes, bytes]:
        """Load and cross-check protected version-2 authority material."""

        private_key_text = self._configuration.connector_authority_private_key_pkcs8
        key_id_text = self._configuration.connector_authority_key_id
        region_text = self._configuration.connector_region
        epoch_text = self._configuration.connector_authority_epoch
        if any(
            value is None
            for value in (private_key_text, key_id_text, region_text, epoch_text)
        ):
            raise OperatorRelayUnavailableError(
                "connector authority material is unavailable"
            )
        key_id = _decode_base64url_exact(cast(str, key_id_text), 32)
        private_key = load_connector_private_key(
            _decode_base64url(cast(str, private_key_text)), key_id
        )
        return (
            private_key,
            key_id,
            _decode_carrier_value(self._configuration.routing_locator),
            _decode_base64url_exact(cast(str, region_text), 8),
            _decode_base64url_exact(cast(str, epoch_text), 16),
        )

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

    async def forward_tls_to_websocket(
        self,
        reader: asyncio.StreamReader,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        """Copy inner TLS bytes into native-client-sized WebSocket messages."""

        while payload := await reader.read(self._frame_bytes):
            await websocket.send_bytes(payload)


def _decode_carrier_value(value: str) -> bytes:
    """Decode an exact unpadded 256-bit relay value."""

    if len(value) != _ENCODED_CARRIER_VALUE_LENGTH:
        raise ValueError("relay values must be unpadded base64url-encoded 32-byte values")
    return _decode_base64url_exact(value, _CARRIER_VALUE_BYTES)


def _decode_base64url_exact(value: str, expected_bytes: int) -> bytes:
    """Decode one canonical unpadded base64url value of an exact size."""

    decoded = _decode_base64url(value)
    if len(decoded) != expected_bytes:
        raise ValueError("relay value has invalid decoded length")
    return decoded


def _decode_base64url(value: str) -> bytes:
    """Decode one bounded canonical unpadded base64url value."""

    if not value or len(value) > 4_096 or "=" in value:
        raise ValueError("relay value must be canonical unpadded base64url")
    try:
        decoded = base64.b64decode(
            f"{value}{'=' * (-len(value) % 4)}",
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError("relay value must be canonical unpadded base64url") from exc
    if _encode_base64url(decoded) != value:
        raise ValueError("relay value must use canonical base64url encoding")
    return decoded


def _encode_base64url(value: bytes) -> str:
    """Encode canonical unpadded base64url bytes."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _validate_connector_authority(provisioning: OperatorRelayProvisioning) -> None:
    """Cross-check all delegated version-2 authority material."""

    private_key_text = provisioning.connector_authority_private_key_pkcs8
    key_id_text = provisioning.connector_authority_key_id
    region_text = provisioning.connector_region
    epoch_text = provisioning.connector_authority_epoch
    if any(
        value is None
        for value in (private_key_text, key_id_text, region_text, epoch_text)
    ):
        raise ValueError("connector authority material is incomplete")
    key_id = _decode_base64url_exact(cast(str, key_id_text), 32)
    _decode_base64url_exact(cast(str, region_text), 8)
    _decode_base64url_exact(cast(str, epoch_text), 16)
    try:
        load_connector_private_key(
            _decode_base64url(cast(str, private_key_text)),
            key_id,
        )
    except RelayProtocolError as exc:
        raise ValueError("connector authority material is invalid") from exc


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
        # The phone pins this exact leaf certificate as its trust anchor. Marking
        # the served leaf as a CA works with OpenSSL but Rustls correctly rejects
        # it as `CaUsedAsEndEntity`, so the gateway must remain a server leaf.
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
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
