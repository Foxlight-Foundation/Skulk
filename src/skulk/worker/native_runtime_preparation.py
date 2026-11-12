"""Bounded asynchronous ownership of explicit preparation and cached recovery.

This controller runs the existing byte/device-verifying manager off the worker
loop. It starts no work on construction or inventory reads and registers no
command/event handler, backend, model or HTTP route by itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from loguru import logger

from skulk.provisioning.demucs_runtime import (
    DemucsPreparationCancelledError,
    PreparationPhase,
    RuntimePackage,
)
from skulk.shared.types.common import CommandId


class PreparationManager[Package: RuntimePackage, Device: str, Prepared](Protocol):
    """Inject preparation effects; production supplies its persistent verified manager."""

    def cached_lanes(self) -> tuple[tuple[Package, Device], ...]:
        """Read fixed eligible original-cache hints; no hint establishes readiness."""
        ...

    def prepare(
        self,
        wheel: Package,
        *,
        device: Device,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
    ) -> Prepared:
        """Verify exact package/device within a complete finite queue/work budget."""
        ...

    def engine_builds(self) -> dict[str, str]:
        """Snapshot successfully verified libraries, never loaded-model readiness."""
        ...

    def close(self) -> None:
        """Withdraw observations and drain owned installation, retaining failed cleanup."""
        ...


async def _join[Value](task: asyncio.Task[Value]) -> Value:
    """Join owned native work despite repeated cancellation of its coroutine owner."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    return task.result()


class NativePreparationController[Package: RuntimePackage, Device: str, Prepared]:
    """Fence preparation, cached recovery and shutdown around one worker manager.

    All effect methods run on the worker's asyncio loop. Native progress callbacks
    run on the preparation thread and must be thread-safe. At most four queued or
    active requests exist; a bounded recent-ID fence suppresses duplicate/reordered
    cancellation without retaining an unbounded completed-ID history.
    """

    def __init__(
        self,
        manager: PreparationManager[Package, Device, Prepared],
        *,
        offline: bool,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Bind the worker policy without disk, native process or network effects."""
        self.manager, self.offline, self.clock = manager, offline, clock
        self._active: dict[
            CommandId, tuple[threading.Event, asyncio.Task[Prepared]]
        ] = {}
        self._recent: OrderedDict[CommandId, None] = OrderedDict()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._restore_task: asyncio.Task[None] | None = None

    def engine_builds(self) -> dict[str, str]:
        """Publish only current verified libraries; shutdown withdraws immediately."""
        return {} if self._closed else self.manager.engine_builds()

    def _remember(self, request_id: CommandId) -> None:
        self._recent[request_id] = None
        self._recent.move_to_end(request_id)
        while len(self._recent) > 64:
            self._recent.popitem(last=False)

    def cancel(self, request_id: CommandId) -> None:
        """Cancel this request, including one whose ordered prepare arrives later."""
        current = self._active.get(request_id)
        if current is not None:
            current[0].set()
        self._remember(request_id)

    async def prepare(
        self,
        request_id: CommandId,
        wheel: Package,
        *,
        device: Device,
        expires_at: float,
        progress: Callable[[PreparationPhase], None],
    ) -> Prepared:
        """Prepare once off-loop, joining native cleanup on cancellation or shutdown.

        The deadline includes manager queue time, archive verification, install
        and device probing. The manager validates the immutable host package and
        actual explicit device; this controller supplies no fallback or download
        authority beyond the worker's fixed offline policy.
        """
        return await self._prepare(
            request_id, wheel, device=device, expires_at=expires_at,
            progress=progress, offline=self.offline,
        )

    async def _prepare(
        self, request_id: CommandId, wheel: Package, *, device: Device,
        expires_at: float, progress: Callable[[PreparationPhase], None], offline: bool,
    ) -> Prepared:
        if self._closed or request_id in self._recent:
            raise DemucsPreparationCancelledError(
                "native preparation is no longer active"
            )
        if request_id in self._active:
            raise ValueError("native preparation request is already in progress")
        budget = expires_at - self.clock()
        if not math.isfinite(budget) or not 0 < budget <= 2 * 60 * 60:
            raise TimeoutError(
                "native preparation requires a live deadline within two hours"
            )
        if len(self._active) >= 4:
            raise RuntimeError("native preparation request capacity is full")
        stopped = threading.Event()

        def prepare_in_thread() -> Prepared:
            # asyncio's shared executor can delay dispatch after coroutine
            # admission. That queue consumes the original absolute deadline too;
            # expired work must not begin installation with a refreshed budget.
            remaining = expires_at - self.clock()
            if stopped.is_set():
                raise DemucsPreparationCancelledError(
                    "native preparation was cancelled"
                )
            if not math.isfinite(remaining) or not 0 < remaining <= 2 * 60 * 60:
                raise TimeoutError(
                    "native preparation expired while waiting for executor capacity"
                )
            return self.manager.prepare(
                wheel,
                device=device,
                offline=offline,
                is_cancelled=stopped.is_set,
                progress=progress,
                timeout_seconds=remaining,
            )

        task = asyncio.create_task(asyncio.to_thread(prepare_in_thread))
        self._active[request_id] = stopped, task
        try:
            try:
                prepared = await asyncio.shield(task)
            except asyncio.CancelledError:
                stopped.set()
                # Preserve the owner's cancellation after native drain; a
                # separate native failure cannot replace that cancellation.
                with contextlib.suppress(Exception):
                    await _join(task)
                raise
            if self._closed or stopped.is_set():
                raise DemucsPreparationCancelledError(
                    "native preparation was cancelled"
                )
            return prepared
        finally:
            self._active.pop(request_id, None)
            self._remember(request_id)

    async def restore_cached(self, *, timeout_seconds: float = 120) -> None:
        """Recheck retained native libraries once, with no startup download authority.

        Cache discovery and normal byte/device verification run off-loop. Failures
        leave unavailable lanes unadvertised and are logged without stopping the
        node. Shutdown/cancellation joins discovery and native cleanup; a second
        caller observes the original bounded task rather than starting it again.
        """
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
            raise ValueError("native cache recovery requires a finite budget up to 120 seconds")
        if self._closed:
            return
        if self._restore_task is None:
            self._restore_task = asyncio.create_task(
                self._restore_cached(self.clock() + timeout_seconds)
            )
        task = self._restore_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(Exception):
                await _join(task)
            raise

    async def _restore_cached(self, expires_at: float) -> None:
        discovery = asyncio.create_task(asyncio.to_thread(self.manager.cached_lanes))
        try:
            try:
                candidates = await asyncio.shield(discovery)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await _join(discovery)
                raise
        except Exception as error:
            logger.warning("Native engine cached inventory could not be read: {}", error)
            return
        for package, device in candidates:
            if self._closed:
                return
            if self.clock() >= expires_at:
                logger.warning("Native engine cached-library recovery exceeded its startup budget")
                return
            try:
                await self._prepare(
                    CommandId("native-cache-" + uuid4().hex), package,
                    device=device, expires_at=expires_at, offline=True,
                    progress=lambda _phase: None,
                )
            except Exception as error:
                logger.warning(
                    "Cached native engine {} on {} is not ready: {}",
                    package.build_identity, device, error,
                )

    async def close(self) -> None:
        """Cancel and join all work before releasing the persistent manager's fences.

        An unsuccessful close retains the original manager for retry. Coroutine
        cancellation cannot abandon the owned close thread or active preparations.
        """
        self._closed = True
        if self._restore_task is not None:
            self._restore_task.cancel()
        for stopped, _ in self._active.values():
            stopped.set()
        pending = tuple(task for _, task in self._active.values())
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain_and_close(pending))
        close_task = self._close_task
        cancelled = False
        try:
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                cancelled = True
                await _join(close_task)
        except BaseException:
            self._close_task = None
            raise
        if cancelled:
            raise asyncio.CancelledError

    async def _drain_and_close(
        self, pending: tuple[asyncio.Task[Prepared], ...]
    ) -> None:
        if self._restore_task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await _join(self._restore_task)
        # The cache deliberately refuses close during installation: its retained
        # cleanup handle cannot be released until the preparation thread exits.
        # Keep this join inside the shielded close task so cancellation of the
        # worker's close coroutine cannot reverse that ownership order.
        for task in pending:
            # Each request owner reports its own cancelled/failure result.
            with contextlib.suppress(Exception):
                await _join(task)
        await asyncio.to_thread(self.manager.close)
