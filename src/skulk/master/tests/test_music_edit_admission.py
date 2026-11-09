# pyright: reportPrivateUsage=false
"""Native edits/generation share one live verified resident execution slot."""

import anyio
import pytest

from skulk.master.main import (
    Master,
    music_editing_events,
    music_editing_instances,
    music_generation_events,
    music_generation_instances,
)
from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    MusicEditing,
    MusicGeneration,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    TaskCreated,
    TaskFailed,
)
from skulk.shared.types.music import MusicGenerationTaskParams
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import MusicEditing as MusicEditingTask
from skulk.shared.types.tasks import MusicGeneration as MusicGenerationTask
from skulk.shared.types.tasks import TaskStatus
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def state(
    *, replicas: int = 1, status: RunnerStatus | None = None,
) -> tuple[State, dict[NodeId, NodeResources]]:
    """Use complete placed native cards; no library probe masquerades as load."""
    card, _ = native_music_fixture()
    instances: dict[InstanceId, MlxRingInstance] = {}
    runners: dict[RunnerId, RunnerStatus] = {}
    resources: dict[NodeId, NodeResources] = {}
    for number in range(replicas):
        node, runner = NodeId(f"node-{number}"), RunnerId(f"runner-{number}")
        instance = MlxRingInstance(
            instance_id=InstanceId(f"instance-{number}"), hosts_by_node={},
            ephemeral_port=50000,
            shard_assignments=ShardAssignments(
                model_id=card.model_id, node_to_runner={node: runner},
                runner_to_shard={runner: PipelineShardMetadata(
                    model_card=card, device_rank=0, world_size=1,
                    start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers,
                    resolved_backend="ace_step-cpu",
                    resolved_engine_build="ace_step@sha256:" + "d" * 64,
                )},
            ),
        )
        instances[instance.instance_id] = instance
        runners[runner] = status if status is not None else RunnerReady()
        resources[node] = NodeResources(
            backends=frozenset({"ace_step-cpu"}),
            engine_builds={"ace_step-cpu": "ace_step@sha256:" + "d" * 64},
        )
    return State(instances=instances, runners=runners), resources


def edit() -> MusicEditing:
    """Create a fixed-seed edit carrying measured source facts, never a path."""
    job, _, _, _ = music_edit_fixture()
    return MusicEditing(owner_node=NodeId("owner"), task_params=job.params)


def generation() -> MusicGeneration:
    """Create an ordinary generation command targeting the same native model."""
    card, _ = native_music_fixture()
    return MusicGeneration(
        owner_node=NodeId("owner"),
        task_params=MusicGenerationTaskParams(
            model=str(card.model_id), prompt="Warm acoustic folk", seconds=5, seed=0,
        ),
    )


@pytest.mark.parametrize("first_edit", [True, False])
def test_generation_and_edit_reserve_each_other_before_state_echo(*, first_edit: bool) -> None:
    """Both command orders enforce one active native operation in the same batch."""
    snapshot, resources = state()
    editing, generating = edit(), generation()
    if first_edit:
        events, _ = music_editing_events(snapshot, editing, node_resources=resources)
        refused, _ = music_generation_events(snapshot, generating, events, node_resources=resources)
    else:
        events, _ = music_generation_events(snapshot, generating, node_resources=resources)
        refused, _ = music_editing_events(snapshot, editing, events, node_resources=resources)
    assert isinstance(events[0], TaskCreated) and events[0].task.task_status == TaskStatus.Pending
    assert isinstance(refused[0], TaskCreated) and refused[0].task.task_status == TaskStatus.Failed
    assert isinstance(refused[1], TaskFailed)


def test_another_loaded_replica_is_usable_while_first_is_reserved() -> None:
    """Serial admission is per resident instance and does not serialize a cluster."""
    snapshot, resources = state(replicas=2)
    first, _ = music_editing_events(snapshot, edit(), node_resources=resources)
    second, _ = music_generation_events(snapshot, generation(), first, node_resources=resources)
    assert isinstance(first[0], TaskCreated) and isinstance(second[0], TaskCreated)
    assert first[0].task.instance_id != second[0].task.instance_id
    assert second[0].task.task_status == TaskStatus.Pending


@pytest.mark.parametrize("status", [RunnerLoading(), RunnerRunning()])
def test_loading_or_busy_runner_is_not_executable(status: RunnerStatus) -> None:
    """Package readiness cannot replace an idle successfully loaded native model."""
    snapshot, resources = state(status=status)
    assert not music_editing_instances(snapshot, edit().task_params, node_resources=resources)
    assert not music_generation_instances(snapshot, ModelId(edit().task_params.model), node_resources=resources)


@pytest.mark.parametrize("change", ["missing", "build", "backend", "installable_only"])
def test_live_resources_must_match_the_placed_verified_build(change: str) -> None:
    """Stale/absent resources and merely installable packages cannot admit native work."""
    snapshot, resources = state()
    node = NodeId("node-0")
    if change == "missing":
        resources.clear()
    elif change == "build":
        resources[node] = resources[node].model_copy(update={"engine_builds": {"ace_step-cpu": "ace_step@sha256:" + "f" * 64}})
    elif change == "backend":
        resources[node] = resources[node].model_copy(update={"backends": frozenset({"mlx"})})
    else:
        resources[node] = resources[node].model_copy(update={
            "engine_builds": {}, "backends": frozenset(),
            "installable_engine_builds": resources[node].engine_builds,
        })
    assert not music_editing_instances(snapshot, edit().task_params, node_resources=resources)
    assert not music_generation_instances(snapshot, ModelId(edit().task_params.model), node_resources=resources)


@pytest.mark.parametrize("field", ["world_size", "device_rank", "start_layer", "end_layer", "n_layers"])
def test_partial_or_distributed_native_placement_is_not_executable(field: str) -> None:
    """A Ready label never makes a partial model executable as a native editor."""
    snapshot, resources = state()
    instance = snapshot.instances[InstanceId("instance-0")]
    runner = RunnerId("runner-0")
    assignments = instance.shard_assignments
    shard = assignments.runner_to_shard[runner].model_copy(update={field: 2})
    changed = instance.model_copy(update={"shard_assignments": assignments.model_copy(update={"runner_to_shard": {runner: shard}})})
    snapshot = snapshot.model_copy(update={"instances": {instance.instance_id: changed}})
    assert not music_editing_instances(snapshot, edit().task_params, node_resources=resources)


@pytest.mark.parametrize("active", [TaskStatus.Pending, TaskStatus.Running, TaskStatus.Complete, TaskStatus.Cancelled])
def test_applied_tasks_and_terminal_history_share_the_same_admission_rule(active: TaskStatus) -> None:
    """Recovered idle capacity is usable, while pending/running work remains exclusive."""
    snapshot, resources = state()
    events, identifier = music_editing_events(snapshot, edit(), node_resources=resources)
    assert isinstance(events[0], TaskCreated)
    task = events[0].task.model_copy(update={"task_status": active})
    snapshot = snapshot.model_copy(update={"tasks": {identifier: task}})
    eligible = music_generation_instances(snapshot, ModelId(edit().task_params.model), node_resources=resources)
    assert bool(eligible) is (active in {TaskStatus.Complete, TaskStatus.Cancelled})


@pytest.mark.parametrize("first_edit", [True, False])
async def test_master_reservation_survives_delayed_indexed_event(*, first_edit: bool) -> None:
    """The actual command processor cannot overbook between separate input messages."""
    node = NodeId("master")
    commands, receiver = channel[ForwarderCommand]()
    events, output = channel[Event]()
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, local = channel[LocalForwarderEvent]()
    sync_sender, sync_receiver = channel[StateSyncMessage]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    master = Master(
        node, SessionId(master_node_id=node, election_clock=0), event_sender=events,
        global_event_sender=global_sender, local_event_receiver=local,
        command_receiver=receiver, state_sync_receiver=sync_receiver,
        state_sync_sender=sync_sender, download_command_sender=downloads,
    )
    master.state, resources = state()
    master._telemetry_view.node_resources.update(resources)
    first, second = (edit(), generation()) if first_edit else (generation(), edit())
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(master._command_processor)
            with anyio.fail_after(3):
                await commands.send(ForwarderCommand(origin=SystemId("API"), command=first))
                admitted = await output.receive()
                assert isinstance(admitted, TaskCreated)
                assert isinstance(admitted.task, (MusicEditingTask, MusicGenerationTask))
                assert admitted.task.task_status == TaskStatus.Pending
                assert not master.state.tasks and len(master._pending_native_audio_reservations) == 1
                await commands.send(ForwarderCommand(origin=SystemId("API"), command=second))
                refused = await output.receive()
                assert isinstance(refused, TaskCreated) and refused.task.task_status == TaskStatus.Failed
                assert isinstance(await output.receive(), TaskFailed)
                master._apply_indexed_event(IndexedEvent(idx=0, event=admitted))
                assert not master._pending_native_audio_reservations
                assert master.state.tasks[admitted.task_id] == admitted.task
            group.cancel_scope.cancel()
    finally:
        master._event_log.close()
