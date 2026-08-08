"""Tests for designated-gateway relay provisioning and opaque lane bridging."""

from __future__ import annotations

import asyncio
import base64
import socket
import ssl
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

import aiohttp
import pytest
from aiohttp import web
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import (
    OperatorPairingService,
    PairingChallengeRequest,
    PairingExchangeRequest,
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

    package = service.create_session(exchange_url="https://example.invalid")
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
    assert provisioning.gateway_carrier_credential not in exchange.model_dump_json()
    with pytest.raises(OperatorRelayAlreadyConfiguredError):
        service.configure_relay(provisioning, operator_api_port=52416)


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
