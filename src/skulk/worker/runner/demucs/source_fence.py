"""Transfer an existing source lifetime lock through local process spawning.

Duplicated descriptors share the original open file description and its flock.
The fence therefore survives abrupt worker/runner death while native inference
still reads the source. No path or descriptor appears in replicated metadata.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import threading
import time
from collections.abc import Callable
from multiprocessing import reduction
from pathlib import Path
from typing import Protocol, final

from skulk.utils.runtime_files import RuntimeLock


class TransferredDescriptor(Protocol):
    """One-shot descriptor transfer supplied by multiprocessing's spawn protocol."""

    def detach(self) -> int:
        """Adopt the transferred descriptor in the receiving process."""
        ...


def _restore_source_fence(descriptor: TransferredDescriptor) -> SourceReadFence:
    """Adopt a local spawn transfer without opening another independent lock."""
    return SourceReadFence(descriptor.detach(), adopt=True)


@final
class SourceReadFence:
    """Hold the source owner's existing lock until the last reader exits.

    The parent owner closes its copy only after its runner has exited. The
    runner also passes its copy to the native subprocess. Closing one duplicate
    never explicitly unlocks the shared flock held by the remaining readers.
    """

    def __init__(self, descriptor: int, *, adopt: bool = False) -> None:
        """Duplicate an owned descriptor, or adopt a one-shot spawn transfer."""
        self.descriptor = descriptor if adopt else os.dup(descriptor)

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[TransferredDescriptor], SourceReadFence],
        tuple[TransferredDescriptor],
    ]:
        """Transfer the fence through multiprocessing without serializing its number."""
        if self.descriptor < 0:
            raise RuntimeError("source reader fence is closed")
        return _restore_source_fence, (reduction.DupFd(self.descriptor),)

    def close(self) -> None:
        """Close only this process's duplicate after its reader has drained."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


async def acquire_source_read_fence(
    root: Path, *, timeout_seconds: float = 30.0
) -> SourceReadFence:
    """Acquire a private per-model reader fence without abandoning file effects.

    Wait at most the supplied finite budget for prior native readers. Cancellation
    joins each owned lock attempt and closes any acquired descriptor before it
    escapes, including cancellation immediately after the thread acquired it.
    """
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise ValueError(
            "source reader acquisition requires a finite budget within one minute"
        )
    deadline = time.monotonic() + timeout_seconds
    while True:
        stopped = threading.Event()

        def acquire(stopped: threading.Event = stopped) -> RuntimeLock:
            if stopped.is_set() or time.monotonic() >= deadline:
                raise TimeoutError(
                    "source reader lock request expired before file effects"
                )
            owner = RuntimeLock(root, name="attachment.lock")
            if stopped.is_set() or time.monotonic() >= deadline:
                owner.close()
                raise TimeoutError("source reader lock request expired")
            return owner

        attempt = asyncio.create_task(asyncio.to_thread(acquire))
        try:
            owner = await asyncio.shield(attempt)
        except asyncio.CancelledError:
            stopped.set()
            while not attempt.done():
                try:
                    await asyncio.shield(attempt)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            with contextlib.suppress(Exception):
                attempt.result().close()
            raise
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "prior native source readers have not drained"
                ) from None
            await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            continue
        try:
            return SourceReadFence(owner.descriptor)
        finally:
            owner.close()
