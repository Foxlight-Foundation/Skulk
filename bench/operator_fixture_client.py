"""Loopback-only, no-retry TLS-over-WebSocket fixture contract test client."""

import asyncio
import json
import ssl
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

import aiohttp
import h11

from skulk.operator.relay import OperatorRemoteAccessMaterial


@dataclass(frozen=True)
class FixtureResponse:
    """Bounded HTTP response used only by generated-data contract tests."""

    status: int
    body: dict[str, object]


@dataclass(frozen=True)
class FixtureByteResponse:
    """Bounded decoded HTTP body from a generated fixture, never a live user."""

    status: int
    body: bytes


async def request_fixture(
    remote: OperatorRemoteAccessMaterial,
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
    bearer: str | None = None,
    close_after_response_body: bool = False,
) -> FixtureResponse:
    """Send one no-retry generated request and require a JSON object response."""
    response = await request_fixture_bytes(
        remote,
        method,
        path,
        body=body,
        bearer=bearer,
        close_after_response_body=close_after_response_body,
    )
    decoded: object = cast(object, json.loads(response.body)) if response.body else {}
    if not isinstance(decoded, dict):
        raise RuntimeError("fixture response must be a JSON object")
    return FixtureResponse(response.status, cast(dict[str, object], decoded))


async def request_fixture_bytes(
    remote: OperatorRemoteAccessMaterial,
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
    bearer: str | None = None,
    close_after_response_body: bool = False,
) -> FixtureByteResponse:
    """Send one request with real inner TLS and no retries to a local fixture.

    The caller must supply generated credentials. This is protocol smoke
    evidence, not a released-client workload implementation or load generator.
    With `close_after_response_body`, close on HTTP framing completion instead
    of waiting for server EOF, exercising finite and chunked response cleanup.
    """
    url = urlsplit(remote.app_websocket_url)
    if (
        url.scheme != "ws"
        or url.hostname != "127.0.0.1"
        or url.path != "/v1/carrier/app"
    ):
        raise ValueError("fixture client accepts only generated loopback relay URLs")
    encoded = b"" if body is None else json.dumps(body).encode()
    if len(encoded) > 65536 or any(character in method + path for character in "\r\n"):
        raise ValueError("invalid fixture request")
    async with asyncio.timeout(10), aiohttp.ClientSession() as session:
        carrier_headers = {
            "Authorization": f"Bearer {remote.app_carrier_credential}",
            "x-skulk-relay-route": remote.routing_locator,
        }
        async with session.ws_connect(
            remote.app_websocket_url,
            headers=carrier_headers,
            max_msg_size=1048576,
        ) as websocket:

            async def to_websocket(reader: asyncio.StreamReader) -> None:
                while payload := await reader.read(65536):
                    await websocket.send_bytes(payload)

            async def from_websocket(writer: asyncio.StreamWriter) -> None:
                async for message in websocket:
                    if message.type is not aiohttp.WSMsgType.BINARY:
                        raise RuntimeError("fixture received a non-binary frame")
                    writer.write(cast(bytes, message.data))
                    await writer.drain()

            async def bridge(
                reader: asyncio.StreamReader, writer: asyncio.StreamWriter
            ) -> None:
                outgoing = asyncio.create_task(to_websocket(reader))
                incoming = asyncio.create_task(from_websocket(writer))
                try:
                    _, pending = await asyncio.wait(
                        (outgoing, incoming), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    for result in await asyncio.gather(
                        outgoing, incoming, return_exceptions=True
                    ):
                        if isinstance(result, BaseException) and not isinstance(
                            result, asyncio.CancelledError
                        ):
                            raise result
                finally:
                    outgoing.cancel()
                    incoming.cancel()
                    await asyncio.gather(outgoing, incoming, return_exceptions=True)
                    writer.close()
                    await writer.wait_closed()

            async with asyncio.TaskGroup() as group:

                def connected(
                    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
                ) -> None:
                    group.create_task(bridge(reader, writer))

                server = await asyncio.start_server(connected, "127.0.0.1", 0)
                writer: asyncio.StreamWriter | None = None
                try:
                    port = cast(tuple[str, int], server.sockets[0].getsockname())[1]
                    context = ssl.create_default_context(
                        cadata=remote.gateway_ca_certificate_pem
                    )
                    context.minimum_version = ssl.TLSVersion.TLSv1_3
                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1",
                        port,
                        ssl=context,
                        server_hostname=remote.gateway_server_name,
                    )
                    headers = [
                        ("Host", remote.gateway_server_name),
                        ("Connection", "close"),
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(encoded))),
                    ]
                    if bearer is not None:
                        headers.append(("Authorization", f"Bearer {bearer}"))
                    parser = h11.Connection(h11.CLIENT, max_incomplete_event_size=65536)
                    for outgoing_event in (
                        h11.Request(method=method, target=path, headers=headers),
                        h11.Data(data=encoded),
                        h11.EndOfMessage(),
                    ):
                        writer.write(parser.send(outgoing_event) or b"")
                    await writer.drain()
                    response = bytearray()
                    status = 0
                    complete = False
                    received_bytes = 0
                    while True:
                        chunk = await reader.read(65536)
                        received_bytes += len(chunk)
                        if received_bytes > 1048576:
                            raise RuntimeError("fixture response limit")
                        parser.receive_data(chunk)
                        while True:
                            event = parser.next_event()
                            if event is h11.NEED_DATA or event is h11.PAUSED:
                                break
                            if isinstance(event, h11.Response):
                                status = event.status_code
                            elif isinstance(event, h11.Data):
                                response.extend(event.data)
                            elif isinstance(event, h11.EndOfMessage):
                                complete = True
                                break
                            elif isinstance(event, h11.ConnectionClosed):
                                break
                        if not chunk or (complete and close_after_response_body):
                            break
                    if not complete or not status:
                        raise RuntimeError("fixture response incomplete")
                finally:
                    if writer is not None:
                        writer.close()
                        await writer.wait_closed()
                    server.close()
                    await server.wait_closed()
    return FixtureByteResponse(status, bytes(response))
