"""Bounded bidirectional RPC on one authenticated Core/owner stream.

Mirrored in the owner. Only the protected peer constructs these requests. Author
frames never enter this dispatcher, and losing a reply never permits a retry.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, final

from pydantic import JsonValue

from .native_host_protocol import (
    AuthenticatedStream,
    CancelFrame,
    Operation,
    Received,
    RequestFrame,
    ResponseFrame,
)


@final
@dataclass(frozen=True, slots=True)
class Reply:
    """Handler-owned result whose body lease survives actual reply delivery."""

    outcome: Literal["complete", "pre_dispatch", "unknown"]
    parameters: dict[str, JsonValue]
    payload: bytes = b""
    release: Callable[[], None] = lambda: None
    check: Callable[[], None] = lambda: None


type Handler = Callable[["Peer", RequestFrame, bytes], Awaitable[Reply]]


class _DeliveryWithdrawnError(Exception):
    """A per-call delivery guard refused before any authenticated frame write."""


@final
class Peer:
    """Own the sole reader, bounded pending requests and cancellation cleanup."""

    def __init__(
        self,
        stream: AuthenticatedStream,
        handler: Handler,
        operations: frozenset[str],
        current: Callable[[], None],
    ) -> None:
        """Start only after mutual host authentication and bind allowed directions."""
        self.stream, self.handler = stream, handler
        self.operations, self.current = operations, current
        self._sequence = self._incoming = 0
        self._pending: dict[int, asyncio.Future[Received]] = {}
        self._handlers: dict[int, asyncio.Task[None]] = {}
        self._closed = False
        self.pump = asyncio.create_task(self._read())

    def check(self) -> None:
        """Recheck lifetime admission without trusting cached discovery claims."""
        if self._closed or self.stream.closed:
            raise ValueError("native host peer unavailable")
        self.current()

    async def request(
        self,
        operation: Operation,
        parameters: dict[str, JsonValue],
        payload: bytes = b"",
        *,
        check: Callable[[], None] = lambda: None,
        dispatched: Callable[[], None] = lambda: None,
        cleanup_seconds: float = 3.0,
    ) -> Received:
        """Transfer an owned response; interrupted dispatch is never replayed."""
        self.check()
        if not math.isfinite(cleanup_seconds) or cleanup_seconds <= 0:
            raise ValueError("native host cleanup resource bound refused")
        if len(self._pending) >= 32 or self._sequence >= 2**53 - 1:
            raise ValueError("native host request capacity unavailable")
        self._sequence += 1
        identifier = self._sequence
        future: asyncio.Future[Received] = asyncio.get_running_loop().create_future()
        self._pending[identifier] = future
        written = False
        transferred = False

        def authorize() -> None:
            self.check()
            check()

        def mark() -> None:
            nonlocal written
            dispatched()
            written = True

        try:
            await self.stream.send(
                RequestFrame(
                    request_id=identifier,
                    operation=operation,
                    parameters=parameters,
                    payload_size=len(payload),
                ),
                payload,
                check=authorize,
                dispatched=mark,
            )
            response = await asyncio.shield(future)
            transferred = True
            return response
        finally:
            if written and not future.done() and not self._closed:
                # A cancelled caller cannot leave a live remote invocation. Keep
                # its correlation slot until cleanup acknowledges or fence the
                # whole peer; late results can never match a different request.
                cleanup = asyncio.create_task(
                    self._cancel(identifier, future, cleanup_seconds)
                )
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
            self._pending.pop(identifier, None)
            if future.done() and not future.cancelled():
                error = future.exception()
                if error is None and not transferred:
                    future.result().release()
            elif not future.done():
                future.cancel()

    async def _cancel(
        self, identifier: int, future: asyncio.Future[Received], seconds: float
    ) -> None:
        try:
            async with asyncio.timeout(seconds):
                await self.stream.send(
                    CancelFrame(request_id=identifier), b"", check=self.check
                )
                value = await asyncio.shield(future)
                value.release()
        except (OSError, ValueError, TimeoutError):
            self._fence()

    def _fence(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stream.poison()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ValueError("native host outcome unavailable"))
        for task in self._handlers.values():
            if task is not asyncio.current_task():
                task.cancel()

    async def _handle(self, value: Received, frame: RequestFrame) -> None:
        reply: Reply | None = None
        try:
            try:
                self.check()
                reply = await self.handler(self, frame, value.payload)
            except asyncio.CancelledError:
                reply = Reply("unknown", {})
            except Exception:
                # The generic transport cannot prove that a handler never wrote.
                reply = Reply("unknown", {})
            result = reply

            def permitted() -> None:
                self.check()
                try:
                    result.check()
                except Exception:
                    # AuthenticatedStream calls this guard before advancing its
                    # sequence or writing. Only this response is withdrawn; the
                    # actual role/lifetime check above still fences the peer.
                    raise _DeliveryWithdrawnError from None

            try:
                await self.stream.send(
                    ResponseFrame(
                        request_id=frame.request_id,
                        outcome=reply.outcome,
                        parameters=reply.parameters,
                        payload_size=len(reply.payload),
                    ),
                    reply.payload,
                    check=permitted,
                )
            except _DeliveryWithdrawnError:
                await self.stream.send(
                    ResponseFrame(
                        request_id=frame.request_id,
                        outcome="unknown",
                        parameters={},
                        payload_size=0,
                    ),
                    b"",
                    check=self.check,
                )
        except (Exception, asyncio.CancelledError):
            self._fence()
        finally:
            if reply is not None:
                reply.release()
            value.release()
            self._handlers.pop(frame.request_id, None)

    async def _read(self) -> None:
        try:
            while True:
                value = await self.stream.receive()
                transferred = False
                try:
                    self.check()
                    frame = value.frame
                    if isinstance(frame, RequestFrame):
                        if (
                            frame.request_id <= self._incoming
                            or frame.operation not in self.operations
                            or len(self._handlers) >= 32
                        ):
                            raise ValueError(
                                "native host request direction or sequence refused"
                            )
                        self._incoming = frame.request_id
                        self._handlers[frame.request_id] = asyncio.create_task(
                            self._handle(value, frame)
                        )
                        task = self._handlers[frame.request_id]
                        task.add_done_callback(
                            partial(self._settled, frame.request_id, value)
                        )
                        transferred = True
                        await asyncio.sleep(0)
                    elif isinstance(frame, ResponseFrame):
                        future = self._pending.get(frame.request_id)
                        if future is None or future.done():
                            raise ValueError("native host reply correlation refused")
                        future.set_result(value)
                        transferred = True
                    else:
                        task = self._handlers.get(frame.request_id)
                        if task is None:
                            if frame.request_id > self._incoming:
                                raise ValueError(
                                    "native host cancellation correlation refused"
                                )
                            continue  # A terminal/cancel race grants no new work.
                        if not task.cancelling():
                            task.cancel()
                finally:
                    if not transferred:
                        value.release()
        except (Exception, asyncio.CancelledError):
            self._fence()

    def _settled(
        self, identifier: int, value: Received, task: asyncio.Task[None]
    ) -> None:
        # Cancellation before first coroutine entry cannot run its finally block.
        # The task owner still holds and retires the exact ingress reservation.
        value.release()
        if self._handlers.get(identifier) is task:
            self._handlers.pop(identifier, None)
        if not task.cancelled() and task.exception() is not None:
            self._fence()

    async def close(self) -> None:
        """Withdraw role/call authority first, then drain all owned handler tasks."""
        self._fence()
        self.pump.cancel()
        await asyncio.gather(
            self.pump, *list(self._handlers.values()), return_exceptions=True
        )
        await self.stream.close()
