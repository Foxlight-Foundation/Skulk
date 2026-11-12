"""Addressed edit output and terminal lifecycle over the actual supervisor loop."""

import multiprocessing as mp
from pathlib import Path

import anyio
import pytest
from anyio import to_thread

from skulk.shared.types.chunks import DataChunk, MusicEditingChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.music_edits import MusicEditingOutput
from skulk.shared.types.tasks import MusicEditing, Task, TaskId, TaskStatus
from skulk.shared.types.worker.runners import RunnerRunning
from skulk.utils.channels import channel, mp_channel
from skulk.worker.runner.ace_step.tests.test_model import edit_task, fixture
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


@pytest.mark.parametrize(
    "ending",
    [TaskStatus.Complete, TaskStatus.Cancelled, TaskStatus.Failed, TaskStatus.TimedOut],
)
async def test_music_edit_output_and_terminal_frames_retain_the_api_owner(
    tmp_path: Path, ending: TaskStatus
) -> None:
    """All edit outcomes close DATA once; complete WAV metadata bypasses event logs."""
    model, _factory, _generation = fixture(tmp_path)
    owner = NodeId("native-edit-api")
    original = edit_task(model, "repaint")
    task = MusicEditing.model_validate_json(
        original.model_copy(update={"owner_node": owner}).model_dump_json()
    )
    completion: MusicEditingChunk | None = None
    if ending == TaskStatus.Complete:
        model.start(is_cancelled=lambda: False)
        generated = model.run(
            task, progress=lambda _update: None, is_cancelled=lambda: False
        )
        assert generated is not None
        completion = MusicEditingChunk(
            model=model.bound.bound_shard.model_card.model_id,
            finish_reason="stop",
            output=MusicEditingOutput(
                source=task.task_params.source, result=generated.result
            ),
        )
    controls, control_receiver = channel[Event](20)
    data, data_receiver = channel[DataChunk](20)
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    runner_events, event_receiver = mp_channel[Event]()
    diagnostics, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    deliveries: list[tuple[CommandId, NodeId | None, MusicEditingChunk]] = []

    def deliver(
        command: CommandId, target: NodeId | None, chunk: MusicEditingChunk
    ) -> None:
        deliveries.append((command, target, chunk))

    process = mp.Process()
    supervisor = RunnerSupervisor(
        shard_metadata=model.bound.bound_shard,
        bound_instance=model.bound,
        runner_process=process,
        initialize_timeout=400,
        _ev_recv=event_receiver,
        _diag_recv=diagnostic_receiver,
        _task_sender=task_sender,
        _event_sender=controls,
        _cancel_sender=cancel_sender,
        _data_sender=data,
        _on_music_edit_output=deliver,
    )
    supervisor.status = RunnerRunning()
    try:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(
                    supervisor._forward_events  # pyright: ignore[reportPrivateUsage]
                )
                group.start_soon(supervisor.start_task, task)
                dispatched = await to_thread.run_sync(task_receiver.receive_timeout, 5)
                assert dispatched == task
                runner_events.send(TaskAcknowledged(task_id=task.task_id))
                if completion is not None:
                    runner_events.send(
                        ChunkGenerated(command_id=task.command_id, chunk=completion)
                    )
                terminal = TaskStatusUpdated(task_id=task.task_id, task_status=ending)
                runner_events.send(terminal)
                frames = await data_receiver.receive_at_least(2)
                assert await control_receiver.receive() == terminal
                assert len(frames) == 2
                assert [frame.kind for frame in frames] == [
                    "started",
                    "completed"
                    if ending == TaskStatus.Complete
                    else "cancelled"
                    if ending == TaskStatus.Cancelled
                    else "failed",
                ]
                assert [frame.sequence for frame in frames] == [0, 1]
                assert all(
                    frame.command_id == task.command_id and frame.owner_node == owner
                    for frame in frames
                )
                assert data_receiver.collect() == []
                assert not any(
                    isinstance(event, ChunkGenerated)
                    for event in control_receiver.collect()
                )
                assert deliveries == (
                    [(task.command_id, owner, completion)]
                    if completion is not None
                    else []
                )
                assert frames[1].chunk == completion if completion is not None else True
                assert task.task_id not in supervisor.in_progress
                group.cancel_scope.cancel()
    finally:
        model.close()
        process.close()
        runner_events.close()
        event_receiver.close()
        diagnostics.close()
        diagnostic_receiver.close()
        task_sender.close()
        task_receiver.close()
        cancel_sender.close()
        cancel_receiver.close()
        controls.close()
        control_receiver.close()
        data.close()
        data_receiver.close()
