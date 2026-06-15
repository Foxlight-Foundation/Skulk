import multiprocessing as mp
from typing import cast

import anyio
import pytest

from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import ErrorChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.diagnostics import (
    RunnerDiagnosticContext,
    RunnerDiagnosticUpdate,
)
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import Task, TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerRunning
from skulk.utils.channels import channel, mp_channel
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class _DeadProcess:
    pid = 123
    exitcode = -6

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def join(self, _timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


class _ReapTrackingProcess:
    """A live process stub that records whether it was reaped.

    Stays alive until ``join``/``terminate``/``kill`` is called, so the
    supervisor's watcher/forwarder subtasks block (keeping ``run()`` alive)
    until the test cancels it externally — mirroring a worker shutdown on a
    master-election transition.
    """

    pid = 123
    exitcode = 0

    def __init__(self) -> None:
        self.joins = 0
        self.closed = False
        self._alive = True

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, _timeout: float | None = None) -> None:
        self.joins += 1
        self._alive = False

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_teardown_reaps_runner_process_under_cancellation() -> None:
    """The runner process must be reaped even when run() is cancelled.

    Regression: on a master-election transition the worker is torn down via
    ``worker.shutdown()``, which cancels the worker task group and so cancels
    each ``RunnerSupervisor.run()``. The teardown ``finally`` reaps the runner
    process (Metal reclaims its wired GPU memory on exit). Without shielding,
    the first ``await`` in that ``finally`` (the process join) re-raised
    CancelledError immediately, so the process was never reaped — it lingered
    holding GPU memory. The replacement worker then planned CreateRunner for the
    same shard, the pre-load memory guard saw the not-yet-reclaimed memory,
    falsely refused, and #290 deleted the carried instance (master failover
    silently killed a healthy serving instance). The teardown is now shielded,
    so the process is joined/closed before run() returns.
    """
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = _ReapTrackingProcess()
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, proc)),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(supervisor.run)
            # let run() start the process + its watcher/forwarder subtasks
            await anyio.sleep(0.2)
            assert proc.is_alive()
            # simulate worker.shutdown() cancelling the worker task group
            tg.cancel_scope.cancel()

    # despite the cancellation, the teardown must have reaped the process so its
    # GPU memory is reclaimed before the replacement worker admits a new runner.
    assert proc.joins >= 1, "process was not joined during cancelled teardown"
    assert not proc.is_alive()
    assert proc.closed, "process.close() was skipped during cancelled teardown"

    event_sender.close()


@pytest.mark.asyncio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_cancelled_task_event_clears_in_progress_and_emits_delete() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerRunning()

    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-a"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    forwarded_cancel: Event | None = None
    forwarded_delete: Event | None = None

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        ev_send.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Cancelled))
        forwarded_cancel = await event_receiver.receive()
        forwarded_delete = await event_receiver.receive()
        tg.cancel_scope.cancel()

    ev_send.close()

    assert isinstance(forwarded_cancel, TaskStatusUpdated)
    assert forwarded_cancel.task_status == TaskStatus.Cancelled
    assert isinstance(forwarded_delete, TaskDeleted)
    assert forwarded_delete.task_id == task.task_id
    assert task.task_id not in supervisor.in_progress
    assert task.task_id in supervisor.cancelled

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_runner_flight_recorder_retains_newest_entries() -> None:
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    context = RunnerDiagnosticContext(
        node_id="node-a",
        runner_id="runner-a",
        pid=123,
        instance_id="instance-a",
        model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        rank=0,
        world_size=1,
        start_layer=0,
        end_layer=30,
        n_layers=30,
    )

    for idx in range(140):
        supervisor._apply_diagnostic_update(  # pyright: ignore[reportPrivateUsage]
            RunnerDiagnosticUpdate(
                at=f"2026-04-23T00:00:{idx:02d}+00:00",
                phase="decode_stream",
                event=f"token_{idx}",
                context=context,
                task_id="task-a",
            )
        )

    diagnostics = supervisor.diagnostics()
    assert diagnostics.phase == "decode_stream"
    assert len(diagnostics.flight_recorder) == 128
    assert diagnostics.flight_recorder[0].event == "token_12"
    assert diagnostics.flight_recorder[-1].event == "token_139"


@pytest.mark.asyncio
async def test_duplicate_terminal_status_is_forwarded_exactly_once() -> None:
    """A re-reported terminal status must not mint another event pair (#278).

    An idle SequentialGenerator used to re-report every ever-cancelled task
    on every step; the supervisor converted each report into a fresh
    TaskStatusUpdated(Cancelled) + TaskDeleted pair, flooding the cluster
    log. The supervisor now forwards a terminal status at most once.
    """
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerRunning()

    task_id = TaskId("task-a")
    forwarded: list[Event] = []

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        ev_send.send(
            TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Cancelled)
        )
        # first report: status + explicit delete
        forwarded.append(await event_receiver.receive())
        forwarded.append(await event_receiver.receive())
        # duplicate report: must be suppressed entirely
        ev_send.send(
            TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Cancelled)
        )
        with anyio.move_on_after(0.3):
            forwarded.append(await event_receiver.receive())
        tg.cancel_scope.cancel()

    ev_send.close()

    assert len(forwarded) == 2
    assert isinstance(forwarded[0], TaskStatusUpdated)
    assert isinstance(forwarded[1], TaskDeleted)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()
