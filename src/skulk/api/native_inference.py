"""Exact native inference API with real original-caller authentication."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response

from skulk.api.inference_access import authenticate_inference_request
from skulk.api.native_calls import OwnedNativeResponse
from skulk.extensions.native_host_protocol import MAXIMUM_PAYLOAD, Received
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_hosts import InferenceListing
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.shared.types.common import NodeId


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
            listings = await service.directory.directory()
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
            reservation = service.hosts.budget.reserve(MAXIMUM_PAYLOAD)
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
                        if len(chunk) > MAXIMUM_PAYLOAD - len(body):
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

    router.get(
        "",
        summary="List signed native inference destinations",
        description="A direct owner can inspect signed ready destinations for enrollment. Enrolled clients authenticate with a standard bearer and see only exact permitted destinations. Discovery grants no authority. Only the currently admitted native unary adapter is advertised; governed effects, unsupported modes, unqualified and offline owners supply no executable capacity.",
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
