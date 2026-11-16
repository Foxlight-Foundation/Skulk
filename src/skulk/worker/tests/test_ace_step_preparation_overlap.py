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
from skulk.shared.types.events import (
    AceStepPreparationCompleted,
    IndexedEvent,
    InstanceDeleted,
    StateSnapshotHydrated,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import CreateRunner, TaskId, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import channel
from skulk.worker import main as worker_module
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


@pytest.mark.parametrize("snapshot", [False, True])
async def test_withdrawn_queued_creations_release_slots_before_active_owner_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall, *, snapshot: bool,
) -> None:
    """Actual indexed withdrawal frees all queued slots without waiting for the owner."""
    worker, original, outgoing = runner_fixture(tmp_path, monkeypatch, installed)
    indexed, incoming = channel[IndexedEvent]()
    worker.event_receiver = incoming
    def preserve_config(*_arguments: object) -> None:
        """Snapshot tests never mutate the operator's model-trust configuration."""
    monkeypatch.setattr(worker_module, "persist_model_trust_config", preserve_config)
    tasks: list[CreateRunner] = []
    instances = dict(worker.state.instances)
    for index in range(3):
        identifier = InstanceId(f"queued-{index}")
        instance = original.bound_instance.instance.model_copy(update={"instance_id": identifier})
        instances[identifier] = instance
        bound = original.bound_instance.model_copy(update={"instance": instance})
        tasks.append(original.model_copy(update={"task_id": TaskId(f"create-{index}"), "instance_id": identifier, "bound_instance": bound}))
    worker.state = worker.state.model_copy(update={"instances": instances, "tasks": {task.task_id: task for task in tasks}})
    try:
        with anyio.fail_after(5):
            async with worker._tg as group:
                group.start_soon(worker._event_applier)
                async with worker._native_music_preparation_scope(time.time() + 30):
                    for task in tasks:
                        await worker._execute_create_runner(task)
                    while worker._native_music_preparation_capacity.borrowed_tokens < 4:
                        await anyio.sleep(0)
                    if snapshot:
                        retained = worker.state.model_copy(update={"last_event_applied_idx": 0})
                        await indexed.send(IndexedEvent(idx=0, event=StateSnapshotHydrated(state=retained)))
                        while worker.state.last_event_applied_idx != 0:
                            await anyio.sleep(0)
                        assert len(worker._native_music_creations) == 3
                        assert worker._native_music_preparation_capacity.borrowed_tokens == 4
                        current = worker.state.model_copy(update={"instances": {}, "last_event_applied_idx": 1})
                        await indexed.send(IndexedEvent(idx=1, event=StateSnapshotHydrated(state=current)))
                    else:
                        for index, task in enumerate(tasks):
                            await indexed.send(IndexedEvent(idx=index, event=InstanceDeleted(instance_id=task.instance_id)))
                    while worker._native_music_creations:
                        await anyio.sleep(0)
                    assert worker._native_music_preparation_capacity.borrowed_tokens == 1
                    assert not worker._native_music_creation_requests
                    assert not worker._native_music_creation_scopes
                    cancelled = [await outgoing.receive() for _ in tasks]
                    assert all(isinstance(event, TaskStatusUpdated) and event.task_status == TaskStatus.Cancelled for event in cancelled)
                    assert {event.task_id for event in cancelled if isinstance(event, TaskStatusUpdated)} == {task.task_id for task in tasks}
                    assert installed.probes == installed.downloads == 0
                async with worker._native_music_preparation_scope(time.time() + 30):
                    assert worker._native_music_preparation_capacity.borrowed_tokens == 1
                group.cancel_scope.cancel()
    finally:
        indexed.close()
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


async def test_withdrawn_active_creation_drains_native_work_before_releasing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """Withdrawal cancels the real creation but retains ownership until native drain."""
    worker, task, outgoing = runner_fixture(tmp_path, monkeypatch, installed)
    worker.state = worker.state.model_copy(update={"tasks": {task.task_id: task}})
    indexed, incoming = channel[IndexedEvent]()
    worker.event_receiver = incoming
    entered, release = threading.Event(), threading.Event()
    original_probe = installed.probe
    def probe(arguments: Sequence[str], *, environment: dict[str, str], timeout_seconds: int, capture: bool) -> str:
        if installed.probes == 0:
            entered.set()
            assert release.wait(5)
        return original_probe(arguments, environment=environment, timeout_seconds=timeout_seconds, capture=capture)
    monkeypatch.setattr(installed, "probe", probe)
    try:
        with anyio.fail_after(10):
            async with worker._tg as group:
                group.start_soon(worker._event_applier)
                await worker._execute_create_runner(task)
                while not entered.is_set():
                    await anyio.sleep(0.001)
                group.start_soon(worker._prepare_ace_step_engine, request(worker, installed))
                while worker._native_music_preparation_capacity.borrowed_tokens < 2:
                    await anyio.sleep(0)
                await indexed.send(IndexedEvent(idx=0, event=InstanceDeleted(instance_id=task.instance_id)))
                while task.instance_id in worker.state.instances:
                    await anyio.sleep(0)
                await anyio.sleep(0.05)
                assert worker._native_music_preparation_capacity.borrowed_tokens == 2
                assert task.instance_id in worker._native_music_creation_requests
                release.set()
                events = [await outgoing.receive() for _ in range(2)]
                assert any(isinstance(event, AceStepPreparationCompleted) and event.success for event in events)
                assert any(isinstance(event, TaskStatusUpdated) and event.task_id == task.task_id and event.task_status == TaskStatus.Cancelled for event in events)
                assert not worker._native_music_creations and not worker.runners
                group.cancel_scope.cancel()
    finally:
        release.set()
        indexed.close()
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()
