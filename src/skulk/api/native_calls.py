"""Direct-owner API for explicitly enrolled installed native unary reads."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, final

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response
from starlette.types import Receive, Scope, Send

from skulk.api.operator_auth import authenticate_capability_owner
from skulk.extensions.native_calls import NativeCalls, NativeListing, PolicyUpdate
from skulk.extensions.native_host_protocol import MAXIMUM_PAYLOAD, ReadPolicy, Received
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.shared.types.common import NodeId


@final
class _OwnedResponse(Response):
    def __init__(self, received: Received) -> None:
        super().__init__(received.payload, media_type="application/json")
        self.received = received

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.received.release()


def create_native_calls_router(
    node_id: NodeId,
    *,
    connection_path: Path = SKULK_CONFIG_HOME / "managed-service" / "connection.json",
) -> APIRouter:
    """Bind the actual Core issuer and host-configured local service connection.

    Args:
        node_id: The serving API's transport identity, never a caller parameter.
        connection_path: Host-owned service configuration; not exposed over HTTP.

    Returns:
        Documented direct-owner routes with their owned peer lifetime.
    """
    broker: NativeCalls | None = None

    async def host() -> NativeCalls:
        nonlocal broker
        try:
            connection = ServiceConnection.model_validate_json(
                read_private(connection_path)
            )
            if broker is not None and broker.connection != connection:
                await broker.close()
                broker = None
            if broker is None:
                broker = NativeCalls(connection, str(node_id))
            return broker
        except (OSError, ValueError):
            raise HTTPException(503, "Native capability host is unavailable.") from None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if broker is not None:
                await finish_runtime_work(asyncio.create_task(broker.close()))

    router = APIRouter(
        prefix="/v1/native-capabilities",
        tags=["Native capabilities"],
        lifespan=lifespan,
    )

    async def directory(request: Request) -> tuple[NativeListing, ...]:
        """Expose only signed installed reads on actual ready native channels."""
        await authenticate_capability_owner(request, node_id)
        try:
            return await (await host()).directory()
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(
                503, "Native capability directory is unavailable."
            ) from None

    async def policy(request: Request) -> ReadPolicy:
        """Read direct-owner enrollment without conferring execution authority."""
        await authenticate_capability_owner(request, node_id)
        try:
            return (await host()).policy()
        except (OSError, ValueError):
            raise HTTPException(503, "Native read policy is unavailable.") from None

    async def update_policy(request: Request, update: PolicyUpdate) -> ReadPolicy:
        await authenticate_capability_owner(request, node_id)
        try:
            return await (await host()).update_policy(update)
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(
                409, "Native read enrollment changed or is unavailable."
            ) from None

    async def invoke(
        request: Request,
        plugin_id: str,
        capability_id: str,
        version: str,
        timeout_seconds: Annotated[
            float,
            Query(
                gt=0,
                le=30,
                description="Unary time budget in seconds; not an inference output cap.",
            ),
        ] = 10,
    ) -> Response:
        """Invoke exact raw input under host-minted caller and current read policy."""
        owner = await authenticate_capability_owner(request, node_id)
        service = await host()
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            raise HTTPException(415, "Native call input must be application/json.")
        try:
            reservation = service.budget.reserve(MAXIMUM_PAYLOAD)
        except ValueError:
            raise HTTPException(
                503, "Native call resource capacity is unavailable."
            ) from None
        try:
            async with asyncio.timeout(10):
                body = bytearray()
                async for chunk in request.stream():
                    if len(chunk) > MAXIMUM_PAYLOAD - len(body):
                        raise HTTPException(
                            413, "Native call input exceeds its byte bound."
                        )
                    body.extend(chunk)
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
                    owner, plugin_id, capability_id, version, payload, timeout_seconds
                )
            )
            watch = asyncio.create_task(disconnected())
            transferred = False
            try:
                try:
                    done, _ = await asyncio.wait(
                        {call, watch}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if watch in done:
                        raise HTTPException(499, "Native caller disconnected.")
                    if call.cancelled():
                        raise HTTPException(
                            503,
                            "Native call authority was withdrawn; do not replay automatically.",
                        )
                    result = call.result()
                finally:
                    # Request.is_disconnected uses a cancellation scope internally.
                    # It may consume a concurrent task cancellation; the explicit
                    # terminal event still stops its loop before response delivery.
                    finished.set()
                    watch.cancel()
                    if not call.done():
                        call.cancel()
                    await finish_runtime_work(asyncio.create_task(_drain(call, watch)))
                # No await follows custody transfer. Cancellation during watcher
                # cleanup must release a completed result that was never delivered.
                response = _OwnedResponse(result)
                transferred = True
                return response
            finally:
                if (
                    call.done()
                    and not call.cancelled()
                    and call.exception() is None
                    and not transferred
                ):
                    call.result().release()
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(
                503,
                "Native execution outcome is unavailable; do not replay automatically.",
            ) from None
        finally:
            reservation.release()

    router.get(
        "",
        summary="List live installed native reads",
        description="Direct owner access. Match ready native child channels against signed installed contracts. Independent owners share a three-second discovery deadline and bounded concurrency; invalid, unavailable or unfinished owners are omitted. Cancellation cleanup may add up to three seconds. Discovery grants no execution authority and caches no stale capacity.",
    )(directory)
    router.get(
        "/policy",
        summary="Read native read enrollment",
        description="Direct owner access. Return the current revision and exact enrolled local reads. Execution is disabled by default; this is independent of inference/provider allowances.",
    )(policy)
    router.put(
        "/policy",
        summary="Update native read enrollment",
        description="Direct owner access. Atomically replace local read enrollment using expected_revision. Every change fences existing queued contexts; stale revisions or unavailable/unqualified targets return 409.",
    )(update_policy)
    router.post(
        "/{plugin_id}/{capability_id}/{version}/call",
        summary="Invoke an enrolled installed native read",
        description="Direct owner access. Send the exact raw JSON object as input to an enrolled local unary read. Core supplies caller identity and delegation context. No automatic retry is performed; unavailable outcomes must not be replayed automatically. Disconnect cancels the active call tree. Paired-client, governed/write, streaming and remote Fabric calls are not admitted by this endpoint.",
        response_class=Response,
        responses={
            200: {"content": {"application/json": {"schema": {"type": "object"}}}},
            503: {
                "description": "Execution outcome unavailable; do not replay automatically."
            },
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
