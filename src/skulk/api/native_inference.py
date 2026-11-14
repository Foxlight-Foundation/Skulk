"""Exact native inference API with real original-caller authentication."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Final

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from starlette.websockets import WebSocketState

from skulk.api.inference_access import authenticate_inference_request
from skulk.api.native_calls import OwnedNativeResponse
from skulk.extensions.native_host_protocol import Received
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_hosts import InferenceListing
from skulk.extensions.native_inference_streams import NativeInferenceStream
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.shared.types.common import NodeId

MAX_NATIVE_INFERENCE_JSON_BYTES: Final = 65_536
"""Frozen portable/2 JSON value bound; larger inputs use declared stream media."""


def create_native_inference_router(
    node_id: NodeId,
    *,
    connection_path: Path = SKULK_CONFIG_HOME / "managed-service" / "connection.json",
) -> APIRouter:
    """Bind protected host custody; no request selects an issuer, profile or state path."""
    broker: NativeInferenceCalls | None = None

    async def host() -> NativeInferenceCalls:
        nonlocal broker
        try:
            connection = ServiceConnection.model_validate_json(
                read_private(connection_path)
            )
            if broker is not None and broker.connection != connection:
                await broker.close()
                broker = None
            if broker is None:
                broker = NativeInferenceCalls(connection, str(node_id))
            return broker
        except (OSError, ValueError):
            raise HTTPException(503, "Native inference host is unavailable.") from None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if broker is not None:
                await finish_runtime_work(asyncio.create_task(broker.close()))

    router = APIRouter(
        prefix="/v1/native-inference", tags=["Native inference"], lifespan=lifespan
    )

    async def directory(request: Request) -> tuple[InferenceListing, ...]:
        """Expose signed observations; clients see only their explicitly enrolled targets."""
        service = await host()
        caller = await authenticate_inference_request(request, service.callers, node_id)
        try:
            listings = await service.directory.directory(streaming=True)
            if caller.principal.kind == "operator":
                return listings
            permitted: list[InferenceListing] = []
            for listing in listings:
                try:
                    service.callers.authorize(caller, listing.grant())
                except ValueError:
                    continue
                permitted.append(listing)
            return tuple(permitted)
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(
                503, "Native inference directory is unavailable."
            ) from None

    async def invoke(
        request: Request,
        plugin_id: str,
        capability_id: str,
        version: str,
        timeout_seconds: Annotated[
            float | None,
            Query(
                gt=0,
                allow_inf_nan=False,
                description="Optional request duration ceiling; omission adds no generation timeout.",
            ),
        ] = None,
    ) -> Response:
        """Invoke exact JSON input under current caller, effect/domain and signed target."""
        service = await host()
        caller = await authenticate_inference_request(request, service.callers, node_id)
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            raise HTTPException(415, "Native inference input must be application/json.")
        try:
            reservation = service.hosts.budget.reserve(MAX_NATIVE_INFERENCE_JSON_BYTES)
        except ValueError:
            raise HTTPException(
                503, "Native inference capacity is unavailable."
            ) from None
        result: Received | None = None
        transferred = False
        try:
            try:
                async with asyncio.timeout(10):
                    body = bytearray()
                    async for chunk in request.stream():
                        if len(chunk) > MAX_NATIVE_INFERENCE_JSON_BYTES - len(body):
                            raise HTTPException(
                                413, "Native inference input exceeds its byte bound."
                            )
                        body.extend(chunk)
            except TimeoutError:
                raise HTTPException(408, "Native inference input timed out.") from None
            payload = bytes(body)
            del body
            reservation.shrink(len(payload))
            finished = asyncio.Event()

            async def disconnected() -> None:
                while not finished.is_set():
                    if await request.is_disconnected():
                        return
                    await asyncio.sleep(0.05)

            call = asyncio.create_task(
                service.invoke(
                    caller, plugin_id, capability_id, version, payload, timeout_seconds
                )
            )
            watch = asyncio.create_task(disconnected())
            try:
                try:
                    done, _ = await asyncio.wait(
                        {call, watch}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if watch in done:
                        raise HTTPException(
                            499, "Native inference caller disconnected."
                        )
                    if call.cancelled():
                        raise HTTPException(
                            503,
                            "Native inference authority was withdrawn; do not replay automatically.",
                        )
                    result = call.result()
                finally:
                    finished.set()
                    if not call.done():
                        call.cancel()
                    if not watch.done():
                        watch.cancel()
                    await finish_runtime_work(asyncio.create_task(_drain(call, watch)))
                response = OwnedNativeResponse(result)
                transferred = True
                return response
            finally:
                if (
                    result is None
                    and call.done()
                    and not call.cancelled()
                    and call.exception() is None
                ):
                    result = call.result()
                if result is not None and not transferred:
                    result.release()
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(
                503,
                "Native inference outcome is unavailable; do not replay automatically.",
            ) from None
        finally:
            reservation.release()

    async def stream(
        socket: WebSocket,
        plugin_id: str,
        capability_id: str,
        version: str,
        timeout_seconds: Annotated[
            float | None,
            Query(
                gt=0,
                allow_inf_nan=False,
                description="Optional generation ceiling; omission adds none.",
            ),
        ] = None,
    ) -> None:
        """Carry exact protocol-six packets under an enrolled original client identity.

        The first text message is bounded raw JSON input. A ready event binds the
        server-assigned call ID; binary caller packets follow in input modes.
        Disconnect cancels this call. There are no query/subprotocol credentials,
        ambient owner fallback, aggregate output allowance or automatic replay.
        """
        call: asyncio.Task[None] | None = None
        reservation = None
        try:
            service = await host()
            caller = await authenticate_inference_request(
                socket, service.callers, node_id
            )
            reservation = service.hosts.budget.reserve(MAX_NATIVE_INFERENCE_JSON_BYTES)
            await socket.accept(headers=[(b"cache-control", b"no-store")])
            async with asyncio.timeout(10):
                payload = (await socket.receive_text()).encode()
            if len(payload) > MAX_NATIVE_INFERENCE_JSON_BYTES:
                await socket.close(
                    1009, "Native inference input exceeds its byte bound."
                )
                return
            reservation.shrink(len(payload))

            async def consume(port: NativeInferenceStream) -> None:
                async with asyncio.timeout(port.write_seconds):
                    await socket.send_json(
                        {
                            "type": "ready",
                            "call_id": port.call_id,
                            "mode": port.mode,
                        }
                    )

                async def ingress() -> None:
                    while True:
                        wire = await socket.receive_bytes()
                        if len(wire) > 1048576 + 65540:
                            raise ValueError("native stream frame exceeds bound")
                        await port.send_input(wire)

                async def egress() -> None:
                    while (value := await port.next_output()) is not None:
                        try:
                            async with asyncio.timeout(port.write_seconds):
                                await socket.send_bytes(value.packet)
                        finally:
                            value.release()
                    end = service.kernel.encode_transport_v2_packet(
                        '{"kind":"end","call_id":"' + port.call_id + '"}', b""
                    )
                    async with asyncio.timeout(port.write_seconds):
                        await socket.send_bytes(end)

                input_task, output_task = (
                    asyncio.create_task(ingress()),
                    asyncio.create_task(egress()),
                )
                try:
                    done, _ = await asyncio.wait(
                        (input_task, output_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if input_task in done:
                        input_task.result()
                        raise ValueError("native stream input ended without disconnect")
                    output_task.result()
                finally:
                    for task in (input_task, output_task):
                        if not task.done() and task.cancelling() == 0:
                            task.cancel()
                    await finish_runtime_work(
                        asyncio.create_task(
                            _drain_stream_tasks(input_task, output_task)
                        )
                    )

            call = asyncio.create_task(
                service.invoke_stream(
                    caller,
                    plugin_id,
                    capability_id,
                    version,
                    payload,
                    consume,
                    timeout_seconds,
                )
            )
            try:
                await asyncio.shield(call)
            except asyncio.CancelledError:
                if call.done() and call.cancelled():
                    await socket.close(
                        1008,
                        "Native inference authority was withdrawn; do not replay.",
                    )
                    return
                raise
            await socket.close(1000)
        except WebSocketDisconnect:
            pass
        except HTTPException:
            await socket.close(
                1008, "Native inference authentication or host unavailable."
            )
        except (OSError, ValueError, KeyError, TimeoutError):
            if socket.application_state != WebSocketState.DISCONNECTED:
                await socket.close(
                    1011, "Native inference outcome unavailable; do not replay."
                )
        finally:

            async def cleanup() -> None:
                try:
                    if call is not None:
                        if not call.done() and call.cancelling() == 0:
                            call.cancel()
                        await _drain_stream_call(call)
                finally:
                    if reservation is not None:
                        reservation.release()

            await finish_runtime_work(asyncio.create_task(cleanup()))

    router.websocket(
        "/{plugin_id}/{capability_id}/{version}/stream", name="native_inference_stream"
    )(stream)
    router.get(
        "",
        summary="List signed native inference destinations",
        description="A direct owner can inspect signed ready destinations for enrollment. Enrolled clients authenticate with a standard bearer and see only exact permitted destinations. Discovery grants no authority. Unary calls use the JSON call route; qualified stream modes use the documented bearer WebSocket. Governed effects, unsupported modes, unqualified and offline owners supply no executable capacity.",
    )(directory)
    router.post(
        "/{plugin_id}/{capability_id}/{version}/call",
        summary="Invoke an exact enrolled native inference capability",
        description="Authenticate a direct owner or enrolled client over a direct protected transport. Send exact bounded raw JSON input to a signed, selected, ready implementation. Apply per-caller effect/domain/runtime/mode grants and recheck at the protected owner write. timeout_seconds is optional; null adds no generation ceiling. Request upload, health and partial-I/O bounds are independent resources. Disconnect or authority withdrawal cancels active work. No automatic replay is performed; an unavailable outcome must not be retried automatically. This endpoint does not choose a model, route work, or widen the existing read/chat/operator APIs.",
        response_class=Response,
        responses={
            200: {"content": {"application/json": {"schema": {"type": "object"}}}},
            503: {"description": "Outcome unavailable; do not replay automatically."},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            }
        },
    )(invoke)
    return router


async def _drain(call: asyncio.Task[Received], watch: asyncio.Task[None]) -> None:
    await asyncio.gather(call, watch, return_exceptions=True)


async def _drain_stream_tasks(
    first: asyncio.Task[None], second: asyncio.Task[None]
) -> None:
    await asyncio.gather(first, second, return_exceptions=True)


async def _drain_stream_call(call: asyncio.Task[None]) -> None:
    await asyncio.gather(call, return_exceptions=True)
