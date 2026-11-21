# pyright: reportPrivateUsage=false
"""Real HTTP jobs retain bounded native failure reasons across both planes."""

import contextlib
import multiprocessing as mp
from pathlib import Path

import anyio
import pytest
from anyio import to_thread

from skulk.api.main import API
from skulk.api.tests.test_music_edit_api import create as create_edit
from skulk.api.tests.test_music_edit_api import fixture as edit_fixture
from skulk.api.tests.test_music_edit_api import terminal as edit_terminal
from skulk.api.tests.test_separation_api import create as create_separation
from skulk.api.tests.test_separation_api import fixture as separation_fixture
from skulk.api.types.separation import SeparationResource
from skulk.shared.election import ElectionMessage
from skulk.shared.types.chunks import DataChunk, ErrorChunk
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    TaskAcknowledged,
    TaskFailed,
    TaskStatusUpdated,
)
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    AudioSeparation,
    MusicEditing,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import BoundInstance, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerRunning
from skulk.utils.channels import Sender, channel, mp_channel
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


def event_api(owner: NodeId) -> tuple[API, Sender[IndexedEvent]]:
    """Construct the real state applier without starting a server or providers."""
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return (
        API(
            owner,
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=False,
            telemetry_view=TelemetryView(),
            enable_builtin_providers=False,
        ),
        event_sender,
    )


@pytest.mark.parametrize(
    "message",
    ["Native allocation ceiling exceeded.", "x" * 2048],
    ids=["native-reason", "overlong-reason"],
)
async def test_separator_stream_retains_bounded_worker_reason(
    tmp_path: Path, message: str
) -> None:
    """A worker error reaches the HTTP receipt, bounded to its existing limit."""
    async with separation_fixture(tmp_path) as (harness, client):
        identifier = await create_separation(client)
        task = await harness.task(identifier)
        await harness.queues[identifier].send(
            ErrorChunk(model=ModelId(task.task_params.model), error_message=message)
        )
        with anyio.fail_after(10):
            while (await harness.api.job(identifier)).status != "failed":
                await anyio.sleep(0.01)
        response = await client.get(f"/v1/audio/separations/{identifier}")
        assert response.status_code == 200
        resource = SeparationResource.model_validate_json(response.content)
        assert resource.error == message[:1024]
        assert resource.output is None


@pytest.mark.parametrize(
    "message",
    ["Native allocation ceiling exceeded.", "x" * 2048],
    ids=["native-reason", "overlong-reason"],
)
async def test_music_edit_stream_retains_bounded_worker_reason(
    tmp_path: Path, message: str
) -> None:
    """Source-conditioned editing retains the same bounded worker diagnostics."""
    async with edit_fixture(tmp_path) as (harness, client):
        accepted = await create_edit(client)
        identifier = CommandId(accepted.id)
        task = await harness.task(identifier)
        await harness.queues[identifier].send(
            ErrorChunk(model=ModelId(task.task_params.model), error_message=message)
        )
        resource = await edit_terminal(client, accepted.id)
        assert resource.status == "failed"
        assert resource.error == message[:1024]
        assert resource.output is None


async def test_separator_ordered_failure_retains_busy_reason(tmp_path: Path) -> None:
    """The real state applier must preserve admission failure before stream cleanup."""
    async with separation_fixture(tmp_path) as (harness, client):
        identifier = await create_separation(client)
        task = await harness.task(identifier)
        api, events = event_api(task.owner_node)
        api.state = harness.snapshot
        api._separation_service = harness.api
        message = "No idle ready single-host source-separation instance serves the selected model"
        async with anyio.create_task_group() as group:
            group.start_soon(api._apply_state)
            await events.send(
                IndexedEvent(
                    idx=0,
                    event=TaskFailed(
                        task_id=task.task_id,
                        error_type="audio_separation_unavailable",
                        error_message=message,
                    ),
                )
            )
            with anyio.fail_after(10):
                while (await harness.api.job(identifier)).status != "failed":
                    await anyio.sleep(0.01)
            response = await client.get(f"/v1/audio/separations/{identifier}")
            assert (
                SeparationResource.model_validate_json(response.content).error
                == message
            )
            await events.send(
                IndexedEvent(
                    idx=1,
                    event=TaskStatusUpdated(
                        task_id=task.task_id,
                        task_status=TaskStatus.Cancelled,
                    ),
                )
            )
            with anyio.fail_after(10):
                while api.state.tasks[task.task_id].task_status != TaskStatus.Cancelled:
                    await anyio.sleep(0.01)
            assert (await harness.api.job(identifier)).error == message
            group.cancel_scope.cancel()


async def test_music_edit_ordered_failure_retains_busy_reason(tmp_path: Path) -> None:
    """Native edit admission carries its reason through ordered terminal handling."""
    async with edit_fixture(tmp_path) as (harness, client):
        accepted = await create_edit(client)
        identifier = CommandId(accepted.id)
        task = await harness.task(identifier)
        api, events = event_api(task.owner_node)
        api.state = harness.snapshot
        api._music_edit_service = harness.api
        message = "No idle ready native music instance serves this operation with the current verified engine build."
        async with anyio.create_task_group() as group:
            group.start_soon(api._apply_state)
            await events.send(
                IndexedEvent(
                    idx=0,
                    event=TaskFailed(
                        task_id=task.task_id,
                        error_type="music_edit_unavailable",
                        error_message=message,
                    ),
                )
            )
            resource = await edit_terminal(client, accepted.id)
            assert resource.error == message
            await events.send(
                IndexedEvent(
                    idx=1,
                    event=TaskStatusUpdated(
                        task_id=task.task_id,
                        task_status=TaskStatus.Cancelled,
                    ),
                )
            )
            with anyio.fail_after(10):
                while api.state.tasks[task.task_id].task_status != TaskStatus.Cancelled:
                    await anyio.sleep(0.01)
            assert (await edit_terminal(client, accepted.id)).error == message
            group.cancel_scope.cancel()


async def test_separator_without_main_queue_retains_failure(tmp_path: Path) -> None:
    """Failure after queue retirement still records the actual supplied reason."""
    async with separation_fixture(tmp_path) as (harness, client):
        identifier = await create_separation(client)
        task = await harness.task(identifier)
        api, _ = event_api(task.owner_node)
        api.state = harness.snapshot
        api._separation_service = harness.api
        message = "Separator process exceeded its admitted device allocation ceiling."
        await api._terminate_command_stream(task.task_id, message)
        assert (await harness.api.job(identifier)).error == message


async def test_music_edit_without_main_queue_retains_failure(tmp_path: Path) -> None:
    """Native editing's post-stream failure path must retain its actual cause."""
    async with edit_fixture(tmp_path) as (harness, client):
        accepted = await create_edit(client)
        identifier = CommandId(accepted.id)
        task = await harness.task(identifier)
        api, _ = event_api(task.owner_node)
        api.state = harness.snapshot
        api._music_edit_service = harness.api
        message = "Native edit process exceeded its admitted device allocation ceiling."
        await api._terminate_command_stream(task.task_id, message)
        assert (await edit_terminal(client, accepted.id)).error == message


async def supervised_native_failure(
    task: MusicEditing | AudioSeparation, snapshot: State, message: str
) -> tuple[Event, list[DataChunk]]:
    """Run the real supervisor while retaining its DATA frames off-network."""
    instance = snapshot.instances[task.instance_id]
    assert isinstance(instance, MlxRingInstance)
    runner = next(iter(instance.shard_assignments.runner_to_shard))
    node = next(
        node
        for node, selected in instance.shard_assignments.node_to_runner.items()
        if selected == runner
    )
    bound = BoundInstance(instance=instance, bound_runner_id=runner, bound_node_id=node)
    controls, control_receiver = channel[Event](20)
    data, data_receiver = channel[DataChunk](20)
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    runner_events, event_receiver = mp_channel[Event]()
    diagnostics, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    process = mp.Process()
    supervisor = RunnerSupervisor(
        shard_metadata=bound.bound_shard,
        bound_instance=bound,
        runner_process=process,
        initialize_timeout=400,
        _ev_recv=event_receiver,
        _diag_recv=diagnostic_receiver,
        _task_sender=task_sender,
        _event_sender=controls,
        _cancel_sender=cancel_sender,
        _data_sender=data,
    )
    supervisor.status = RunnerRunning()
    terminal: Event | None = None
    held_data: list[DataChunk] = []
    try:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(supervisor._forward_events)
                group.start_soon(supervisor.start_task, task)
                assert (
                    await to_thread.run_sync(task_receiver.receive_timeout, 5) == task
                )
                runner_events.send(TaskAcknowledged(task_id=task.task_id))
                runner_events.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=ErrorChunk(
                            model=ModelId(task.task_params.model),
                            error_message=message,
                        ),
                    )
                )
                runner_events.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Failed
                    )
                )
                terminal = await control_receiver.receive()
                held_data = await data_receiver.receive_at_least(2)
                assert [frame.kind for frame in held_data] == ["started", "failed"]
                assert task.task_id not in supervisor.in_progress
                group.cancel_scope.cancel()
        assert terminal is not None
        return terminal, held_data
    finally:
        process.close()
        for sender in (runner_events, diagnostics, task_sender, cancel_sender):
            sender.close()
        for receiver in (
            event_receiver,
            diagnostic_receiver,
            task_receiver,
            cancel_receiver,
        ):
            receiver.close()
        controls.close()
        control_receiver.close()
        data.close()
        data_receiver.close()


@pytest.mark.parametrize(
    "message",
    ["Native separator allocation guard refused the render.", "x" * 2048],
    ids=["native-reason", "overlong-reason"],
)
async def test_separator_status_before_data_retains_native_reason(
    tmp_path: Path, message: str
) -> None:
    """The supervisor's ordered terminal is sufficient before delayed DATA."""
    async with separation_fixture(tmp_path) as (harness, client):
        identifier = await create_separation(client)
        task = await harness.task(identifier)
        late_sender = harness.queues[identifier]
        terminal, held_data = await supervised_native_failure(
            task, harness.snapshot, message
        )
        api, events = event_api(task.owner_node)
        api.state = harness.snapshot
        api._separation_service = harness.api
        async with anyio.create_task_group() as group:
            group.start_soon(api._apply_state)
            await events.send(IndexedEvent(idx=0, event=terminal))
            with anyio.fail_after(10):
                while (await harness.api.job(identifier)).status != "failed":
                    await anyio.sleep(0.01)
            # DATA has been captured, so the control path wins deterministically.
            # A terminal owner may close its input; that is a valid late-DATA drop.
            late = held_data[-1].chunk
            assert isinstance(late, ErrorChunk)
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError
            ):
                await late_sender.send(late)
            response = await client.get(f"/v1/audio/separations/{identifier}")
            resource = SeparationResource.model_validate_json(response.content)
            assert resource.error == message[:1024]
            assert resource.output is None
            group.cancel_scope.cancel()


@pytest.mark.parametrize(
    "message",
    ["Native edit allocation guard refused the render.", "x" * 2048],
    ids=["native-reason", "overlong-reason"],
)
async def test_music_edit_status_before_data_retains_native_reason(
    tmp_path: Path, message: str
) -> None:
    """Native edit failure metadata survives the same real reordered planes."""
    async with edit_fixture(tmp_path) as (harness, client):
        accepted = await create_edit(client)
        identifier = CommandId(accepted.id)
        task = await harness.task(identifier)
        late_sender = harness.queues[identifier]
        terminal, held_data = await supervised_native_failure(
            task, harness.snapshot, message
        )
        api, events = event_api(task.owner_node)
        api.state = harness.snapshot
        api._music_edit_service = harness.api
        async with anyio.create_task_group() as group:
            group.start_soon(api._apply_state)
            await events.send(IndexedEvent(idx=0, event=terminal))
            resource = await edit_terminal(client, accepted.id)
            late = held_data[-1].chunk
            assert isinstance(late, ErrorChunk)
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError
            ):
                await late_sender.send(late)
            assert resource.error == message[:1024]
            assert resource.output is None
            group.cancel_scope.cancel()
