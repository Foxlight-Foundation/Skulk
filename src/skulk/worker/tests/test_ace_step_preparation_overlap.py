# pyright: reportPrivateUsage=false
"""Real cache revalidation cannot invalidate another runner's unfinished handoff."""

import threading
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest

from skulk.provisioning.tests import test_ace_step_runtime as fixtures
from skulk.shared.types.events import TaskStatusUpdated
from skulk.shared.types.tasks import TaskStatus
from skulk.worker.tests.test_ace_step_activation import request, runner_fixture

interpreter_fixture = fixtures.interpreter_fixture
installed = fixtures.installed
pytestmark = pytest.mark.parametrize("variant", ("cpu", "cuda", "rocm"))


@pytest.mark.parametrize("cached_recovery", [False, True])
async def test_competing_prepare_cannot_withdraw_library_during_runner_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
    *, cached_recovery: bool,
) -> None:
    """The real supervisor is registered before a competing verification can start."""
    worker, task, output = runner_fixture(tmp_path, monkeypatch, installed)
    if cached_recovery:
        # The CPU package also supports Metal; this fixture executes only the
        # selected lane's probe while preserving its real cache verification.
        monkeypatch.setattr(
            worker._ace_step_manager, "cached_lanes",
            lambda: ((installed.wheel, installed.device),),
        )
    handoff = anyio.Event()
    second_entered = threading.Event()
    release_second = threading.Event()
    supervisor_present_at_second_probe: list[bool] = []
    give_up = AsyncMock()
    monkeypatch.setattr(worker, "_give_up_on_instance", give_up)
    original_probe = installed.probe
    publish = worker._publish_fresh_node_resources
    first_publication = True

    def probe(
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        if installed.probes == 1:
            supervisor_present_at_second_probe.append(bool(worker.runners))
            second_entered.set()
            assert release_second.wait(5), "release the controlled competing probe"
        return original_probe(
            arguments,
            environment=environment,
            timeout_seconds=timeout_seconds,
            capture=capture,
        )

    async def publish_during_handoff() -> None:
        nonlocal first_publication
        if first_publication:
            first_publication = False
            handoff.set()
            with anyio.fail_after(5):
                while not second_entered.is_set():
                    capacity = getattr(worker, "_native_music_preparation_capacity", None)
                    if isinstance(capacity, anyio.CapacityLimiter) and capacity.borrowed_tokens == 2:
                        break
                    await anyio.sleep(0.001)
        await publish()

    async def competing_prepare() -> None:
        await handoff.wait()
        if cached_recovery:
            await worker._restore_cached_native_runtimes()
        else:
            await worker._prepare_ace_step_engine(request(worker, installed))

    monkeypatch.setattr(installed, "probe", probe)
    monkeypatch.setattr(worker, "_publish_fresh_node_resources", publish_during_handoff)
    try:
        with anyio.fail_after(10):
            async with worker._tg:
                worker._tg.start_soon(competing_prepare)
                await worker._execute_create_runner(task)
                try:
                    event = await output.receive()
                    assert isinstance(event, TaskStatusUpdated)
                    assert event.task_status == TaskStatus.Complete
                finally:
                    release_second.set()
        assert supervisor_present_at_second_probe == [True]
        assert installed.probes == 2
        give_up.assert_not_awaited()
    finally:
        release_second.set()
        for supervisor in worker.runners.values():
            context = supervisor._native_music_context
            if context is not None:
                context.source_fence.close()
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


async def test_preparation_waiters_are_bounded_and_cancelled_slots_are_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """One handoff plus three waiters cannot admit an unbounded preparation queue."""
    worker, _, _ = runner_fixture(tmp_path, monkeypatch, installed)

    async def wait_for_slot() -> None:
        async with worker._native_music_preparation_scope(time.time() + 10):
            pytest.fail("the owner has not released its handoff")

    try:
        with anyio.fail_after(5):
            async with worker._native_music_preparation_scope(time.time() + 10):
                async with anyio.create_task_group() as tasks:
                    for _ in range(3):
                        tasks.start_soon(wait_for_slot)
                    while worker._native_music_preparation_capacity.borrowed_tokens < 4:
                        await anyio.sleep(0)
                    with pytest.raises(RuntimeError, match="capacity is full"):
                        async with worker._native_music_preparation_scope(time.time() + 10):
                            pytest.fail("the fifth owner must be refused")
                    tasks.cancel_scope.cancel()
                assert worker._native_music_preparation_capacity.borrowed_tokens == 1
            assert worker._native_music_preparation_capacity.borrowed_tokens == 0
            async with worker._native_music_preparation_scope(time.time() + 10):
                assert worker._native_music_preparation_capacity.borrowed_tokens == 1
        assert installed.probes == 0
    finally:
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


async def test_queued_preparation_deadline_expires_without_starting_native_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """Queue delay cannot renew the caller's budget or leak its admission slot."""
    worker, _, _ = runner_fixture(tmp_path, monkeypatch, installed)
    expired = anyio.Event()

    async def queue() -> None:
        with pytest.raises(TimeoutError):
            async with worker._native_music_preparation_scope(time.time() + 0.05):
                pytest.fail("the held handoff cannot admit this expired owner")
        expired.set()

    try:
        with anyio.fail_after(5):
            async with worker._native_music_preparation_scope(time.time() + 10):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(queue)
                    await expired.wait()
                assert worker._native_music_preparation_capacity.borrowed_tokens == 1
            async with worker._native_music_preparation_scope(time.time() + 10):
                assert worker._native_music_preparation_capacity.borrowed_tokens == 1
        assert installed.probes == 0
    finally:
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()
