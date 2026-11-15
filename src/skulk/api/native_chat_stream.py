"""Owned HTTP delivery over the protected native stream broker.

Application message serialization is shared with capability chat. The native
broker still owns identity, current policy, exact dispatch, terminal admission
and cancellation; no detached output queue can outlive its original caller.
"""

import asyncio
import json
import math
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from typing import final

from fastapi import HTTPException, Response
from pydantic import JsonValue, TypeAdapter
from starlette.types import Receive, Scope, Send

from skulk.api.inference_continuation import ContinuationProof, ReceiptProof
from skulk.api.inference_gateway import (
    chat_response,
    continuation_receipt_response,
    send_chat_body,
)
from skulk.api.native_chat_catalog import NativeApplicationTarget
from skulk.extensions.calls import CapabilityResult
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_streams import (
    NativeInferenceOutput,
    NativeInferenceStream,
)
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.extensions.streams import (
    CapabilityStreamError,
    CapabilityStreamFrame,
    CapabilityStreamInput,
    CapabilityStreamSession,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])


@final
class NativeChatDeliveryUncertainError(RuntimeError):
    """Abort an already-started collected response instead of acknowledging success."""


@final
class NativeChatSession:
    """Adapt admitted native messages while retaining actual delivery leases."""

    def __init__(
        self,
        port: NativeInferenceStream,
        target: NativeApplicationTarget,
        *,
        receipt: bool,
    ) -> None:
        """Bind one admitted stream; receipt collection retains its two output frames."""
        self.port, self.target, self.receipt = port, target, receipt
        self.retained: list[NativeInferenceOutput] = []

    async def _input(self, frame: CapabilityStreamFrame) -> None:
        fields: dict[str, JsonValue] = {
            "call_id": self.port.call_id,
            "direction": "caller_to_provider",
            "sequence": frame.sequence,
            "kind": frame.kind,
        }
        media = b""
        if frame.payload is not None:
            fields["payload"] = _OBJECT.validate_python(frame.payload, strict=True)
        if frame.media is not None:
            if frame.media.kind != "inline":
                raise ValueError("native chat requires inline request media")
            fields["media"] = _OBJECT.validate_json(
                frame.media.model_dump_json(exclude={"data"}, exclude_none=True)
            )
            media = frame.media.data
        if frame.error is not None:
            fields["error"] = frame.error.code
        await self.port.send_input(
            self.port.kernel.encode_transport_v2_packet(json.dumps(fields), media)
        )

    async def _frames(self) -> AsyncGenerator[CapabilityStreamFrame]:
        # The native transport starts at the first chunk. This opener belongs
        # only to the shared semantic serializer; native admission is unchanged.
        yield CapabilityStreamFrame(
            call_id=self.port.call_id,
            direction="provider_to_caller",
            sequence=0,
            kind="started",
        )
        sequence = 1
        while (value := await self.port.next_output()) is not None:
            retained = False
            try:
                fields = _OBJECT.validate_json(value.projection)
                if value.media or fields.get("media") is not None:
                    raise ValueError(
                        "native chat output must be structured application data"
                    )
                kind = value.kind
                if kind == "chunk":
                    payload = self.target.application.unwrap(
                        _OBJECT.validate_python(fields["payload"], strict=True)
                    )
                    frame = CapabilityStreamFrame(
                        call_id=self.port.call_id,
                        direction="provider_to_caller",
                        sequence=sequence,
                        kind="chunk",
                        payload=dict[str, object](payload),
                    )
                elif kind == "completed":
                    if fields.get("payload") not in (None, {}):
                        raise ValueError("native chat result must be empty")
                    frame = CapabilityStreamFrame(
                        call_id=self.port.call_id,
                        direction="provider_to_caller",
                        sequence=sequence,
                        kind="completed",
                        payload={},
                    )
                elif kind in {"failed", "cancelled"}:
                    frame = CapabilityStreamFrame(
                        call_id=self.port.call_id,
                        direction="provider_to_caller",
                        sequence=sequence,
                        kind="cancelled" if kind == "cancelled" else "failed",
                        error=CapabilityStreamError(
                            code="cancelled"
                            if kind == "cancelled"
                            else "provider_error",
                            message="Native capability did not complete.",
                        ),
                    )
                else:
                    raise ValueError("native chat output kind refused")
                if self.receipt:
                    if len(self.retained) >= 2:
                        raise ValueError("native receipt output window exceeded")
                    self.retained.append(value)
                    retained = True
                yield frame
                sequence += 1
            finally:
                if not retained:
                    value.release()

    async def open(self) -> CapabilityStreamSession:
        """Create the shared serializer session without adding a generation ceiling."""
        caller_input = None
        if self.port.mode == "bidirectional":
            caller_input = CapabilityStreamInput(
                call_id=self.port.call_id, deadline_at=math.inf, send_frame=self._input
            )
            await caller_input.start()
        return CapabilityStreamSession(
            CapabilityResult(call_id=self.port.call_id, ok=True, result={}),
            self._frames(),
            caller_input,
        )

    def release(self) -> None:
        """Return receipt output capacity only after physical HTTP delivery/discard."""
        for value in self.retained:
            value.release()
        self.retained.clear()


@final
class NativeChatResponse(Response):
    """Run HTTP writes inside the revocable native invocation's owning task."""

    def __init__(
        self,
        broker: NativeInferenceCalls,
        caller: AuthenticatedInferenceCaller,
        target: NativeApplicationTarget,
        payload: dict[str, JsonValue],
        *,
        model: str,
        streamed: bool,
        include_usage: bool = False,
        body: bytes | None = None,
        continuation: ContinuationProof | None = None,
        receipt: ReceiptProof | None = None,
    ) -> None:
        """Retain an exact trusted selection; dispatch still rechecks current authority."""
        super().__init__(
            media_type="text/event-stream" if streamed else "application/json",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
        del self.headers["content-length"]
        self.broker, self.caller, self.target, self.payload = (
            broker,
            caller,
            target,
            payload,
        )
        self.model, self.streamed, self.include_usage = model, streamed, include_usage
        self.request_body, self.continuation, self.receipt = body, continuation, receipt

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Own disconnect observation, guarded delivery and complete native retirement."""
        disconnected = False
        started = False
        done_marker = False

        async def watch_disconnect() -> None:
            nonlocal disconnected
            while True:
                if (await receive())["type"] == "http.disconnect":
                    disconnected = True
                    return

        def connected() -> None:
            if disconnected:
                raise ValueError("native HTTP caller disconnected")

        async def is_disconnected() -> bool:
            return disconnected

        async def start(status: int = 200) -> None:
            nonlocal started
            if not started:
                started = True
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": self.raw_headers,
                    }
                )

        async def consume(port: NativeInferenceStream) -> None:
            nonlocal done_marker
            adapter = NativeChatSession(
                port, self.target, receipt=self.receipt is not None
            )
            try:
                session = await adapter.open()
                if self.receipt is not None:
                    outcome = await continuation_receipt_response(
                        session, self.receipt, is_disconnected
                    )
                    port.authorize()
                    async with asyncio.timeout(port.write_seconds):
                        await start()
                        await send(
                            {
                                "type": "http.response.body",
                                "body": outcome.model_dump_json().encode(),
                                "more_body": True,
                            }
                        )
                else:
                    if self.request_body is not None:
                        await send_chat_body(
                            session, self.request_body, is_disconnected
                        )
                    async with aclosing(
                        chat_response(
                            session,
                            model=self.model,
                            stream=self.streamed,
                            include_usage=self.include_usage,
                            continuation=self.continuation,
                            drain_refusals=True,
                        )
                    ) as messages:
                        async for message in messages:
                            port.authorize()
                            if message == "data: [DONE]\n\n":
                                done_marker = True
                                continue
                            if not self.streamed and not port.finished:
                                raise ValueError(
                                    "native collected chat omitted transport completion"
                                )
                            async with asyncio.timeout(port.write_seconds):
                                await start()
                                await send(
                                    {
                                        "type": "http.response.body",
                                        "body": message.encode(),
                                        "more_body": True,
                                    }
                                )
            finally:
                adapter.release()

        listing = self.target.listing

        async def complete_delivery(check: Callable[[], None]) -> None:
            # Native control/descendant retirement has already drained. The
            # broker still owns the original caller and can cancel this task
            # throughout the final physical HTTP acknowledgment.
            async with asyncio.timeout(listing.stream_seconds):
                check()
                await start()
                check()
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"data: [DONE]\n\n" if done_marker else b"",
                        "more_body": False,
                    }
                )

        call = asyncio.create_task(
            self.broker.invoke_stream(
                self.caller,
                listing.plugin_id,
                listing.target.capability_id,
                listing.target.version,
                json.dumps(self.target.application.wrap(self.payload)).encode(),
                consume,
                caller_guard=connected,
                expected_listing=listing,
                complete_delivery=complete_delivery,
            )
        )
        watch = asyncio.create_task(watch_disconnect())
        try:
            done, _ = await asyncio.wait(
                (call, watch), return_when=asyncio.FIRST_COMPLETED
            )
            if watch in done:
                watch.result()
                return
            try:
                call.result()
            except asyncio.CancelledError:
                # Broker cancellation means current caller authority withdrew;
                # parent ASGI cancellation is handled by the outer finally.
                raise ValueError("native HTTP authority withdrew") from None
        except (HTTPException, OSError, ValueError, RuntimeError, TimeoutError):
            if started and not self.streamed:
                # A valid-looking JSON body is not a completed invocation.
                # Once headers/payload escaped, only an aborted transport can
                # preserve uncertainty; a clean terminator would assert success.
                raise NativeChatDeliveryUncertainError(
                    "Native capability delivery is uncertain; do not replay."
                ) from None
            if not disconnected:
                # Fixed diagnostics contain no provider output. An uncertain
                # result never becomes permission to submit another request.
                failure = b'{"error":{"type":"capability_error","code":"native_outcome_unavailable","message":"Native capability outcome unavailable; do not replay automatically."}}'
                async with asyncio.timeout(listing.stream_seconds):
                    if not started:
                        await start(503)
                        body = (
                            b"data: " + failure + b"\n\n" if self.streamed else failure
                        )
                    else:
                        body = b"data: " + failure + b"\n\n" if self.streamed else b""
                    await send(
                        {"type": "http.response.body", "body": body, "more_body": False}
                    )
        finally:

            async def cleanup() -> None:
                for task in (call, watch):
                    if not task.done() and task.cancelling() == 0:
                        task.cancel()
                await asyncio.gather(call, watch, return_exceptions=True)

            await finish_runtime_work(asyncio.create_task(cleanup()))
