"""Bounded owned filesystem work which never blocks the node event loop."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import cast, final

from skulk.store.preservation_journal import PreservationHostError


@final
class PreservationIO:
    """Two active filesystem calls, retained through cancellation and shutdown.

    Availability reads reuse one outstanding probe per location and return an
    unavailable observation after a short wait. Effects drain before returning
    cancellation, so their caller retains its setup lock and journal ownership.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="preservation-storage"
        )
        self._slots = asyncio.Semaphore(2)
        self._pending: set[asyncio.Future[object]] = set()
        self._probes: dict[str, asyncio.Future[object]] = {}
        self._leases: set[asyncio.Event] = set()
        self._closed = False

    def _submit[T](self, operation: Callable[[], T]) -> asyncio.Future[T]:
        future = asyncio.get_running_loop().run_in_executor(self._executor, operation)
        owned = cast(asyncio.Future[object], future)
        self._pending.add(owned)

        def finished(result: asyncio.Future[T]) -> None:
            self._pending.discard(owned)
            self._slots.release()
            if not result.cancelled():
                result.exception()  # Observe failures even after a probe timed out.

        future.add_done_callback(finished)
        return future

    async def run[T](
        self,
        operation: Callable[[], T],
        *,
        discard: Callable[[T], None] | None = None,
        _cleanup: bool = False,
    ) -> T:
        """Run one owned operation; cancelled callers wait for its actual end.

        A returned directory descriptor can supply a discard callback so a
        cancellation during open closes the handle rather than leaking it.
        No filesystem operation is abandoned or replayed after cancellation.
        """
        if self._closed and not _cleanup:
            raise PreservationHostError("preservation_host_closed")
        await self._slots.acquire()
        if self._closed and not _cleanup:
            self._slots.release()
            raise PreservationHostError("preservation_host_closed")
        future = self._submit(operation)
        cancelled = False
        try:
            while True:
                try:
                    result = await asyncio.shield(future)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if future.cancelled():
                        raise
            if cancelled:
                if discard is not None:
                    await self.run(lambda: discard(result), _cleanup=True)
                raise asyncio.CancelledError
            return result
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise

    @asynccontextmanager
    async def directory(self, operation: Callable[[], int]) -> AsyncIterator[int]:
        """Retain a handle lease until its off-loop close actually finishes."""
        finished = asyncio.Event()
        self._leases.add(finished)
        descriptor: int | None = None
        try:
            descriptor = await self.run(operation, discard=os.close)
            yield descriptor
        finally:
            try:
                if descriptor is not None:
                    opened = descriptor
                    await self.run(lambda: os.close(opened), _cleanup=True)
            finally:
                self._leases.discard(finished)
                finished.set()

    async def probe[T](self, key: str, operation: Callable[[], T], fallback: T) -> T:
        """Observe without an unbounded queue or restarting a stalled probe."""
        if self._closed:
            return fallback
        previous = self._probes.get(key)
        if previous is None:
            if self._slots.locked():
                return fallback
            # With a free slot this acquire does not suspend on the event loop.
            await self._slots.acquire()
            future = self._submit(operation)
            self._probes[key] = cast(asyncio.Future[object], future)
            future.add_done_callback(lambda _: self._probes.pop(key, None))
        else:
            future = cast(asyncio.Future[T], previous)
        try:
            return await asyncio.wait_for(asyncio.shield(future), 0.25)
        except TimeoutError:
            return fallback

    async def close(self) -> None:
        """Drain actual effects before the owner releases its journal lock.

        A stalled operating-system mount may delay shutdown; unrelated node work
        remains schedulable. Cancellation never makes continuing writes ownerless.
        """
        self._closed = True
        while self._pending or self._leases:
            try:
                if self._pending:
                    await asyncio.gather(
                        *(asyncio.shield(future) for future in tuple(self._pending)),
                        return_exceptions=True,
                    )
                elif self._leases:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue
        self._executor.shutdown(wait=False)
