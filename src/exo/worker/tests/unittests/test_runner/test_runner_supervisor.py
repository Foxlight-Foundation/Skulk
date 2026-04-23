import multiprocessing as mp
from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.diagnostics import RunnerDiagnosticContext, RunnerDiagnosticUpdate
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerRunning
from exo.utils.channels import channel, mp_channel
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


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
