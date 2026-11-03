"""Real async/thread ownership and cancellation over injected preparation effects."""

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES
from skulk.provisioning.demucs_runtime import (
    DemucsDevice,
    DemucsPreparationCancelledError,
    DemucsWheel,
    PreparationPhase,
    PreparedDemucsRuntime,
)
from skulk.shared.types.common import CommandId
from skulk.worker.demucs_preparation import DemucsPreparationController


class Manager:
    """Bounded blocking effect, exposing actual thread completion and close retry."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stopped = threading.Event()
        self.calls: list[tuple[DemucsDevice, bool, float, int]] = []
        self.builds: dict[str, str] = {}
        self.active = 0
        self.closed = 0
        self.fail_close = False
        self.ignore_cancel = False
        self.lock = threading.Lock()

    def prepare(
        self, wheel: DemucsWheel, *, device: DemucsDevice, offline: bool,
        is_cancelled: Callable[[], bool], progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
    ) -> PreparedDemucsRuntime:
        """Block under a finite test budget and report genuine completion to the owner."""
        with self.lock:
            self.calls.append((device, offline, timeout_seconds, threading.get_ident()))
            self.active += 1
        self.entered.set()
        try:
            deadline = time.monotonic() + 3
            while not self.release.wait(0.005):
                if is_cancelled() and not self.ignore_cancel:
                    self.stopped.set()
                    raise DemucsPreparationCancelledError("fixture cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixture work did not finish")
            progress("prepared")
            with self.lock:
                self.builds = {"demucs-cpu": wheel.build_identity}
            return PreparedDemucsRuntime(
                wheel=wheel, generation=self.root,
                interpreter=self.root / "python/bin/python3", device=device,
            )
        finally:
            with self.lock:
                self.active -= 1

    def engine_builds(self) -> dict[str, str]:
        """Return separate library observations, not model or task readiness."""
        with self.lock:
            return dict(self.builds)

    def close(self) -> None:
        """Withdraw observations and permit an injected cleanup retry on this handle."""
        with self.lock:
            if self.active:
                raise RuntimeError("cancel and join native preparation before closing its cache")
            self.closed += 1
            self.builds.clear()
        if self.fail_close:
            self.fail_close = False
            raise RuntimeError("injected close failure")


async def entered(manager: Manager, *, calls: int = 1) -> None:
    """Observe worker-thread admission without blocking the event loop under test."""
    async with asyncio.timeout(2):
        while True:
            with manager.lock:
                count = len(manager.calls)
            if count >= calls:
                return
            await asyncio.sleep(0.005)


def request(controller: DemucsPreparationController, identifier: str) -> asyncio.Task[PreparedDemucsRuntime]:
    """Start one bounded prepare using the actual controller coroutine."""
    return asyncio.create_task(controller.prepare(
        CommandId(identifier), DEMUCS_PACKAGES[0], device="cpu", expires_at=120.0,
        progress=lambda _: None,
    ))


@pytest.mark.parametrize("offline", [False, True])
async def test_prepare_runs_off_loop_with_worker_policy_and_finite_budget(tmp_path: Path, offline: bool) -> None:
    """Waiting for native work leaves the worker loop responsive and forwards policy."""
    manager = Manager(tmp_path)
    controller = DemucsPreparationController(manager, offline=offline, clock=lambda: 100.0)
    assert controller.engine_builds() == {} and not manager.calls
    task = request(controller, "prepare")
    await entered(manager)
    assert not task.done() and manager.calls == [("cpu", offline, 20.0, manager.calls[0][3])]
    assert manager.calls[0][3] != threading.get_ident()
    manager.release.set()
    assert (await task).wheel == DEMUCS_PACKAGES[0]
    assert controller.engine_builds() == {"demucs-cpu": DEMUCS_PACKAGES[0].build_identity}
    await controller.close()
    assert controller.engine_builds() == {} and manager.active == 0


async def test_cancelling_coroutine_joins_owned_thread_before_return(tmp_path: Path) -> None:
    """No prepare thread is abandoned when its API/job owner is cancelled."""
    manager = Manager(tmp_path)
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    task = request(controller, "cancel")
    await entered(manager)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.stopped.is_set() and manager.active == 0
    assert controller.engine_builds() == {}
    await controller.close()


async def test_cancellation_before_request_and_duplicates_start_no_extra_effect(tmp_path: Path) -> None:
    """Ordered cancellation and replay cannot cause a second preparation."""
    manager = Manager(tmp_path)
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    controller.cancel(CommandId("pre-cancelled"))
    with pytest.raises(DemucsPreparationCancelledError):
        await request(controller, "pre-cancelled")
    assert not manager.calls
    task = request(controller, "one")
    await entered(manager)
    with pytest.raises(ValueError, match="already in progress"):
        await request(controller, "one")
    manager.release.set()
    await task
    with pytest.raises(DemucsPreparationCancelledError):
        await request(controller, "one")
    assert len(manager.calls) == 1
    await controller.close()


async def test_four_request_bound_keeps_fifth_out_of_native_queue(tmp_path: Path) -> None:
    """One worker cannot accumulate an unbounded set of native queue waiters."""
    manager = Manager(tmp_path)
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    tasks = [request(controller, f"job-{index}") for index in range(4)]
    await entered(manager, calls=4)
    with pytest.raises(RuntimeError, match="capacity"):
        await request(controller, "fifth")
    await controller.close()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, DemucsPreparationCancelledError) for result in results)
    assert manager.active == 0 and len(manager.calls) == 4


async def test_close_failure_retains_original_manager_for_retry(tmp_path: Path) -> None:
    """Failed cleanup never replaces the handles still responsible for native work."""
    manager = Manager(tmp_path)
    manager.fail_close = True
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    with pytest.raises(RuntimeError, match="close failure"):
        await controller.close()
    await controller.close()
    await controller.close()
    assert manager.closed == 2 and controller.engine_builds() == {}


async def test_late_native_success_cannot_reopen_closed_resource_observations(tmp_path: Path) -> None:
    """A result racing shutdown cannot advertise this worker as ready afterward."""
    manager = Manager(tmp_path)
    manager.ignore_cancel = True
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    prepared = request(controller, "late")
    await entered(manager)
    closing = asyncio.create_task(controller.close())
    await asyncio.sleep(0)
    assert controller.engine_builds() == {} and not closing.done()
    manager.release.set()
    await closing
    with pytest.raises(DemucsPreparationCancelledError):
        await prepared
    assert controller.engine_builds() == {} and manager.active == 0


async def test_cancelling_close_owner_still_joins_before_cache_close(tmp_path: Path) -> None:
    """Repeated cancellation cannot abandon cleanup or close an active cache."""
    manager = Manager(tmp_path)
    manager.ignore_cancel = True
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    prepared = request(controller, "draining")
    await entered(manager)
    closing = asyncio.create_task(controller.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and manager.closed == 0 and manager.active == 1
    assert controller.engine_builds() == {}
    manager.release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(DemucsPreparationCancelledError):
        await prepared
    assert manager.active == 0 and manager.closed == 1
    await controller.close()
    assert manager.closed == 1


@pytest.mark.parametrize("expires", [99.0, 100.0, float("nan"), float("inf"), 8000.0])
async def test_expired_or_unbounded_prepare_never_executes(tmp_path: Path, expires: float) -> None:
    """A finite live complete budget is required before any native effect."""
    manager = Manager(tmp_path)
    controller = DemucsPreparationController(manager, offline=True, clock=lambda: 100.0)
    with pytest.raises(TimeoutError):
        await controller.prepare(CommandId("expired"), DEMUCS_PACKAGES[0], device="cpu", expires_at=expires, progress=lambda _: None)
    assert not manager.calls
    await controller.close()
