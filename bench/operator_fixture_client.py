"""Loopback-only, no-retry TLS-over-WebSocket fixture contract test client."""

import asyncio
import json
import ssl
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

import aiohttp

from skulk.operator.relay import OperatorRemoteAccessMaterial


@dataclass(frozen=True)
class FixtureResponse:
    """Bounded HTTP response used only by generated-data contract tests."""

    status: int
    body: dict[str, object]


async def request_fixture(
    remote: OperatorRemoteAccessMaterial,
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
    bearer: str | None = None,
) -> FixtureResponse:
    """Send one request with real inner TLS and no retries to a local fixture.

    The caller must supply generated credentials. This is protocol smoke
    evidence, not a released-client workload implementation or load generator.
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
                        f"{method} {path} HTTP/1.1",
                        f"Host: {remote.gateway_server_name}",
                        "Connection: close",
                        "Content-Type: application/json",
                        f"Content-Length: {len(encoded)}",
                    ]
                    if bearer is not None:
                        headers.append(f"Authorization: Bearer {bearer}")
                    writer.write("\r\n".join(headers).encode() + b"\r\n\r\n" + encoded)
                    await writer.drain()
                    response = bytearray()
                    while chunk := await reader.read(65536):
                        if len(response) + len(chunk) > 1048576:
                            raise RuntimeError("fixture response limit")
                        response.extend(chunk)
                finally:
                    if writer is not None:
                        writer.close()
                        await writer.wait_closed()
                    server.close()
                    await server.wait_closed()
    headers, separator, payload = response.partition(b"\r\n\r\n")
    if not separator:
        raise RuntimeError("fixture response missing HTTP headers")
    status = int(headers.split(b" ", 2)[1])
    decoded: object = cast(object, json.loads(payload)) if payload else {}
    if not isinstance(decoded, dict):
        raise RuntimeError("fixture response must be a JSON object")
    return FixtureResponse(status, cast(dict[str, object], decoded))
