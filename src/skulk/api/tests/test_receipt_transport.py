"""Receipt transport failures must withhold success and drain owned resources."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextvars import ContextVar

import pytest
from fastapi import HTTPException

from skulk.api.inference_continuation import ReceiptProof
from skulk.api.inference_gateway import continuation_receipt_response
from skulk.extensions.calls import CapabilityResult
from skulk.extensions.streams import (
    CapabilityStreamFrame,
    CapabilityStreamInput,
    CapabilityStreamSession,
)


def receipt_proof() -> ReceiptProof:
    """Supply a synthetic already-admitted public lookup, without signing keys."""
    return ReceiptProof.model_validate(
        {
            "caller_id": "b" * 32,
            "request": {
                "contract": "chat.continuation.receipt@0.1.0",
                "conversation_id": "c" * 64,
                "target": {
                    "target_node": "fixture",
                    "callable_id": "fixture.continuation",
                    "descriptor_revision": "a" * 16,
                    "model": "text",
                },
                "request_nonce": "d" * 64,
                "nonce": "e" * 64,
                "expires_at": 2000000000.0,
            },
            "signature": "0" * 128,
        }
    )


@pytest.mark.parametrize("stage", ["half_close", "collection", "close", "cancel"])
@pytest.mark.parametrize("failure", [RuntimeError, OSError, TimeoutError])
async def test_receipt_transport_refuses_success_and_cancels(
    stage: str, failure: type[Exception]
) -> None:
    """A semantic outcome cannot escape before all owned cleanup succeeds."""
    cancelled = 0

    async def send(frame: CapabilityStreamFrame) -> None:
        if stage == "half_close" and frame.kind == "completed":
            raise failure("private transport details")

    async def cancel() -> None:
        nonlocal cancelled
        cancelled += 1
        if stage == "cancel":
            raise failure("private cleanup details")

    async def disconnected() -> bool:
        return False

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        try:
            if stage == "collection":
                raise failure("private receiver details")
            yield CapabilityStreamFrame(
                call_id="receipt",
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"error": "handle_unavailable"},
            )
            yield CapabilityStreamFrame(
                call_id="receipt",
                direction="provider_to_caller",
                sequence=2,
                kind="completed",
            )
        finally:
            if stage == "close":
                raise failure("private iterator details")

    input_stream = CapabilityStreamInput(
        call_id="receipt", deadline_at=time.monotonic() + 30, send_frame=send
    )
    await input_stream.start()
    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="receipt", ok=True, result={}),
        frames=frames(),
        input=input_stream,
        cancel_output=cancel,
    )
    with pytest.raises(HTTPException) as raised:
        await continuation_receipt_response(session, receipt_proof(), disconnected)
    assert raised.value.status_code == 503
    assert raised.value.detail == "Continuation receipt delivery is uncertain"
    assert cancelled > 0


async def test_receipt_disconnect_remains_cancellation() -> None:
    """Caller task cancellation must still retire the provider without a receipt."""
    cancelled = asyncio.Event()
    collecting = asyncio.Event()

    async def send(frame: CapabilityStreamFrame) -> None:
        pass

    async def cancel() -> None:
        cancelled.set()

    async def disconnected() -> bool:
        return False

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        collecting.set()
        await asyncio.Future[None]()
        yield CapabilityStreamFrame(
            call_id="receipt",
            direction="provider_to_caller",
            sequence=1,
            kind="completed",
        )

    input_stream = CapabilityStreamInput(
        call_id="receipt", deadline_at=time.monotonic() + 30, send_frame=send
    )
    await input_stream.start()
    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="receipt", ok=True, result={}),
        frames=frames(),
        input=input_stream,
        cancel_output=cancel,
    )
    task = asyncio.create_task(
        continuation_receipt_response(session, receipt_proof(), disconnected)
    )
    await collecting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_receipt_preserves_iterator_context_during_cleanup() -> None:
    """Iterator cleanup must preserve the consuming task's context-variable token."""
    cancelled = False
    context: ContextVar[bool] = ContextVar("receipt_context", default=False)

    async def send(frame: CapabilityStreamFrame) -> None:
        pass

    async def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    async def disconnected() -> bool:
        return False

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        token = context.set(True)
        try:
            yield CapabilityStreamFrame(
                call_id="receipt",
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"error": "handle_unavailable"},
            )
            yield CapabilityStreamFrame(
                call_id="receipt",
                direction="provider_to_caller",
                sequence=2,
                kind="completed",
            )
        finally:
            context.reset(token)

    input_stream = CapabilityStreamInput(
        call_id="receipt", deadline_at=time.monotonic() + 30, send_frame=send
    )
    await input_stream.start()
    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="receipt", ok=True, result={}),
        frames=frames(),
        input=input_stream,
        cancel_output=cancel,
    )
    outcome = await continuation_receipt_response(
        session, receipt_proof(), disconnected
    )
    assert outcome.error == "handle_unavailable" and cancelled
    assert not context.get()


async def test_receipt_cleanup_deadline_still_cancels_the_provider() -> None:
    """A stalled iterator close cannot block receipt recovery or skip cancellation."""
    cancelled = False

    async def send(frame: CapabilityStreamFrame) -> None:
        pass

    async def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    async def disconnected() -> bool:
        return False

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        try:
            yield CapabilityStreamFrame(
                call_id="receipt",
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"error": "handle_unavailable"},
            )
            yield CapabilityStreamFrame(
                call_id="receipt",
                direction="provider_to_caller",
                sequence=2,
                kind="completed",
            )
        finally:
            await asyncio.Future[None]()

    input_stream = CapabilityStreamInput(
        call_id="receipt", deadline_at=time.monotonic() + 30, send_frame=send
    )
    await input_stream.start()
    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="receipt", ok=True, result={}),
        frames=frames(),
        input=input_stream,
        cancel_output=cancel,
    )
    with pytest.raises(HTTPException) as raised:
        await asyncio.wait_for(
            continuation_receipt_response(session, receipt_proof(), disconnected), 5
        )
    assert raised.value.status_code == 503 and cancelled
