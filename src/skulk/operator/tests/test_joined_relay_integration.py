# Copyright 2026 Foxlight Foundation
"""Opt-in joined qualification against the real paired-WebSocket relay binary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import ssl
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

import aiohttp
import anyio
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.api.main import API
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import OperatorPairingService, pairing_signature_message
from skulk.operator.relay import (
    OperatorRelayConfigurationRepository,
    OperatorRelayProvisioning,
    OperatorRemoteAccessMaterial,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel

_RELAY_BINARY_ENVIRONMENT_VARIABLE = "SKULK_PAIRED_RELAY_BINARY"
_RELAY_LOOPBACK_HOST = "127.0.0.1"
_RELAY_LOOPBACK_PORT = 8787


@dataclass(frozen=True, slots=True)
class _HttpResponse:
    """Parsed status and JSON object returned by the inner operator API."""

    status: int
    body: dict[str, object]


def _base64url(value: bytes) -> str:
    """Encode exact device-key and proof bytes without padding."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _available_loopback_port() -> int:
    """Reserve and release one currently available loopback TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((_RELAY_LOOPBACK_HOST, 0))
        return cast(tuple[str, int], listener.getsockname())[1]


def _require_relay_port_available() -> None:
    """Fail before provisioning if the relay's fixed dev listener is occupied."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((_RELAY_LOOPBACK_HOST, _RELAY_LOOPBACK_PORT))
    except OSError as error:
        pytest.fail(
            f"loopback relay port {_RELAY_LOOPBACK_PORT} is unavailable: {error}"
        )


def _require_relay_binary() -> Path:
    """Return the explicitly selected external relay binary or skip the test."""

    configured = os.environ.get(_RELAY_BINARY_ENVIRONMENT_VARIABLE)
    if configured is None:
        pytest.skip(
            f"set {_RELAY_BINARY_ENVIRONMENT_VARIABLE} to run joined relay qualification"
        )
    relay_binary = Path(configured).expanduser().resolve()
    if not relay_binary.is_file() or not os.access(relay_binary, os.X_OK):
        pytest.fail(
            f"{_RELAY_BINARY_ENVIRONMENT_VARIABLE} must name an executable file"
        )
    return relay_binary


def _build_api(service: OperatorPairingService) -> API:
    """Build the smallest real Skulk API needed for relay qualification."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("joined-relay-qualification"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        operator_pairing_service=service,
    )


async def _wait_for_relay_health() -> None:
    """Wait until the external relay process serves its health endpoint."""

    url = f"http://{_RELAY_LOOPBACK_HOST}:{_RELAY_LOOPBACK_PORT}/healthz"
    async with aiohttp.ClientSession() as session:
        for _ in range(100):
            try:
                async with session.get(url) as response:
                    if response.status == 204:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.05)
    raise AssertionError("paired-WebSocket relay did not become healthy")


async def _wait_for_operator_listener(port: int) -> None:
    """Wait until Skulk exposes its relay-only inner-TLS listener."""

    for _ in range(100):
        try:
            _, writer = await asyncio.open_connection(_RELAY_LOOPBACK_HOST, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise AssertionError("Skulk operator listener did not become available")


async def _copy_tcp_to_websocket(
    reader: asyncio.StreamReader,
    websocket: aiohttp.ClientWebSocketResponse,
) -> None:
    """Copy opaque inner-TLS bytes into app-role relay frames."""

    while payload := await reader.read(1024 * 1024):
        await websocket.send_bytes(payload)


async def _copy_websocket_to_tcp(
    websocket: aiohttp.ClientWebSocketResponse,
    writer: asyncio.StreamWriter,
) -> None:
    """Copy opaque app-role relay frames into the inner-TLS socket."""

    async for message in websocket:
        if message.type is aiohttp.WSMsgType.BINARY:
            writer.write(cast(bytes, message.data))
            await writer.drain()
            continue
        if message.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        }:
            return
        raise AssertionError("relay emitted a non-binary carrier frame")


async def _bridge_app_lane(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    remote_access: OperatorRemoteAccessMaterial,
) -> None:
    """Adapt one local inner-TLS socket to one real app-role WebSocket."""

    headers = {
        "Authorization": f"Bearer {remote_access.app_carrier_credential}",
        "x-skulk-relay-route": remote_access.routing_locator,
    }
    async with aiohttp.ClientSession() as session:
        websocket: aiohttp.ClientWebSocketResponse | None = None
        for _ in range(100):
            try:
                websocket = await session.ws_connect(
                    remote_access.app_websocket_url,
                    headers=headers,
                    max_msg_size=1024 * 1024,
                )
                break
            except aiohttp.ClientResponseError as error:
                if error.status != 503:
                    raise
                await asyncio.sleep(0.05)
        if websocket is None:
            raise AssertionError("gateway lane did not become available")

        tasks = (
            asyncio.create_task(_copy_tcp_to_websocket(reader, websocket)),
            asyncio.create_task(_copy_websocket_to_tcp(websocket, writer)),
        )
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await websocket.close()
    writer.close()
    await writer.wait_closed()


async def _request(
    remote_access: OperatorRemoteAccessMaterial,
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
    bearer: str | None = None,
) -> _HttpResponse:
    """Send one canonical HTTP request through the real paired carrier."""

    server = await asyncio.start_server(
        lambda reader, writer: _bridge_app_lane(
            reader,
            writer,
            remote_access,
        ),
        _RELAY_LOOPBACK_HOST,
        0,
    )
    sockets = server.sockets
    assert sockets is not None
    adapter_port = cast(tuple[str, int], sockets[0].getsockname())[1]
    context = ssl.create_default_context(
        cadata=remote_access.gateway_ca_certificate_pem
    )
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    try:
        reader, writer = await asyncio.open_connection(
            _RELAY_LOOPBACK_HOST,
            adapter_port,
            ssl=context,
            server_hostname=remote_access.gateway_server_name,
        )
        encoded_body = (
            b""
            if body is None
            else json.dumps(body, separators=(",", ":")).encode("utf-8")
        )
        headers = [
            f"{method} {path} HTTP/1.1",
            f"Host: {remote_access.gateway_server_name}",
            "Accept: application/json",
            "Connection: close",
        ]
        if encoded_body:
            headers.extend(
                (
                    "Content-Type: application/json",
                    f"Content-Length: {len(encoded_body)}",
                )
            )
        if bearer is not None:
            headers.append(f"Authorization: Bearer {bearer}")
        writer.write("\r\n".join(headers).encode("ascii") + b"\r\n\r\n" + encoded_body)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=10.0)
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    response_headers, separator, response_body = response.partition(b"\r\n\r\n")
    assert separator
    status = int(response_headers.split(b" ", maxsplit=2)[1])
    decoded_value: object = (
        cast(object, json.loads(response_body))
        if response_body
        else dict[str, object]()
    )
    assert isinstance(decoded_value, dict)
    return _HttpResponse(
        status=status,
        body=cast(dict[str, object], decoded_value),
    )


async def _stop_gateway_task(
    task: asyncio.Task[None] | None,
    shutdown: anyio.Event | None,
) -> None:
    """Stop the optional Skulk gateway task without masking test failures."""

    if task is None:
        return
    if shutdown is not None:
        shutdown.set()
    try:
        await asyncio.wait_for(task, timeout=10.0)
    except TimeoutError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _stop_relay_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the external relay and escalate only after a bounded wait."""

    process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5.0)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5.0)


@pytest.mark.asyncio
async def test_joined_relay_pairing_refresh_and_revocation(tmp_path: Path) -> None:
    """Prove real relay pairing and canonical authorization without a fleet."""

    relay_binary = _require_relay_binary()
    _require_relay_port_available()
    relay_configuration_path = tmp_path / "relay.json"
    skulk_provisioning_path = tmp_path / "skulk-provisioning.json"
    relay_origin = f"ws://{_RELAY_LOOPBACK_HOST}:{_RELAY_LOOPBACK_PORT}"
    subprocess.run(
        (
            relay_binary,
            "provision",
            relay_origin,
            relay_configuration_path,
            skulk_provisioning_path,
        ),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    relay_process = subprocess.Popen(
        (relay_binary, "serve", relay_configuration_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gateway_task: asyncio.Task[None] | None = None
    shutdown: anyio.Event | None = None
    try:
        await _wait_for_relay_health()
        key_provider = LocalFileAuthorityKeyProvider(tmp_path / "authority-key.bin")
        authority_store = EncryptedAuthorityStore(
            key_provider,
            tmp_path / "authority.sqlite3",
        )
        relay_repository = OperatorRelayConfigurationRepository(
            authority_store,
            certificate_path=tmp_path / "operator-tls.pem",
            private_key_path=tmp_path / "operator-tls-key.pem",
        )
        pairing_service = OperatorPairingService(
            authority_store,
            key_provider,
            relay_repository=relay_repository,
        )
        provisioning = OperatorRelayProvisioning.model_validate_json(
            skulk_provisioning_path.read_text(encoding="utf-8")
        )
        configuration = pairing_service.configure_relay(
            provisioning,
            operator_api_port=_available_loopback_port(),
            cluster_name="Joined Relay Qualification",
        )
        api = _build_api(pairing_service)
        shutdown = anyio.Event()
        gateway_task = asyncio.create_task(api.run_operator_remote_access(shutdown))
        await _wait_for_operator_listener(configuration.operator_api_port)

        pairing_package = pairing_service.create_session()
        assert pairing_package.as_url().startswith("skulk://pair?z=")
        remote_access = pairing_package.remote_access
        assert remote_access is not None

        device_private_key = Ed25519PrivateKey.generate()
        device_public_key = device_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        challenge = await _request(
            remote_access,
            "POST",
            "/v1/auth/pairing-sessions/challenge",
            body={
                "nonce": pairing_package.nonce,
                "deviceName": "Joined qualification device",
                "devicePublicKey": _base64url(device_public_key),
            },
        )
        assert challenge.status == 200
        signature = device_private_key.sign(
            pairing_signature_message(
                cluster_id=UUID(str(pairing_package.cluster_id)),
                nonce=pairing_package.nonce,
                challenge=str(challenge.body["challenge"]),
            )
        )
        exchange = await _request(
            remote_access,
            "POST",
            "/v1/auth/pairing-sessions/exchange",
            body={
                "nonce": pairing_package.nonce,
                "signature": _base64url(signature),
            },
        )
        assert exchange.status == 200
        access_token = str(exchange.body["accessToken"])
        refresh_token = str(exchange.body["refreshToken"])
        device_id = str(exchange.body["deviceId"])

        state = await _request(remote_access, "GET", "/state", bearer=access_token)
        assert state.status == 200
        assert "topology" in state.body

        refreshed = await _request(
            remote_access,
            "POST",
            "/v1/auth/token",
            body={"deviceId": device_id, "refreshToken": refresh_token},
        )
        assert refreshed.status == 200
        next_access_token = str(refreshed.body["accessToken"])
        stale_access = await _request(
            remote_access,
            "GET",
            "/state",
            bearer=access_token,
        )
        assert stale_access.status == 401

        devices = await _request(
            remote_access,
            "GET",
            "/v1/auth/devices",
            bearer=next_access_token,
        )
        assert devices.status == 200
        paired_devices = devices.body["devices"]
        assert isinstance(paired_devices, list)
        assert len(cast(list[object], paired_devices)) == 1

        revoked = await _request(
            remote_access,
            "DELETE",
            f"/v1/auth/devices/{device_id}",
            bearer=next_access_token,
        )
        assert revoked.status == 204
        revoked_access = await _request(
            remote_access,
            "GET",
            "/state",
            bearer=next_access_token,
        )
        assert revoked_access.status == 401
    finally:
        await _stop_gateway_task(gateway_task, shutdown)
        _stop_relay_process(relay_process)
