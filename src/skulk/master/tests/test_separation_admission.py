"""Real task/event admission cannot overbook a resident separator within a batch."""

import tomllib
from pathlib import Path

import pytest

from skulk.master.main import (
    separation_generation_events,
    separation_generation_instances,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.separation_runtime import resolve_demucs_policy
from skulk.shared.types.audio_separation import (
    AudioSeparationTaskParams,
    SeparationSource,
)
from skulk.shared.types.commands import AudioSeparation
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import TaskCreated, TaskFailed
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation as SeparationTask
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


def state(status: RunnerStatus, replicas: int = 1) -> State:
    """Create exact native single-host placements without any qualification claims."""
    card = ModelCard.model_validate(
        tomllib.loads(
            (
                Path(__file__).parents[2] / "shared/models/tests/htdemucs-card.toml"
            ).read_text()
        )
    )
    instances: dict[InstanceId, MlxRingInstance] = {}
    runners: dict[RunnerId, RunnerStatus] = {}
    for number in range(replicas):
        runner = RunnerId(f"runner-{number}")
        instance = MlxRingInstance(
            instance_id=InstanceId(f"instance-{number}"),
            shard_assignments=ShardAssignments(
                model_id=card.model_id,
                node_to_runner={NodeId(f"node-{number}"): runner},
                runner_to_shard={
                    runner: PipelineShardMetadata(
                        model_card=card,
                        device_rank=0,
                        world_size=1,
                        start_layer=0,
                        end_layer=5,
                        n_layers=5,
                        resolved_backend="demucs-cpu",
                        resolved_engine_build="demucs@sha256:" + "d" * 64,
                        separation_policy=resolve_demucs_policy(card, "demucs-cpu"),
                    )
                },
            ),
            hosts_by_node={},
            ephemeral_port=50000,
        )
        instances[instance.instance_id] = instance
        runners[runner] = status
    return State(instances=instances, runners=runners)


def command() -> AudioSeparation:
    """Metadata identifies a verified source; its WAV never enters the command."""
    return AudioSeparation(
        owner_node=NodeId("owning-api"),
        task_params=AudioSeparationTaskParams(
            model="adefossez/HTDemucs",
            total_input_chunks=1,
            source=SeparationSource(
                sha256="c" * 64,
                size_bytes=16044,
                frames=8000,
                sample_rate=8000,
                channels=1,
                bit_depth=16,
                encoding="pcm",
            ),
        ),
    )


def test_same_batch_reserves_capacity_before_events_apply() -> None:
    """A second command cannot see the first batch reservation as idle capacity."""
    snapshot = state(RunnerReady())
    first = command()
    admitted, identifier = separation_generation_events(snapshot, first)
    assert len(admitted) == 1 and isinstance(admitted[0], TaskCreated)
    assert admitted[0].task_id == identifier
    assert isinstance(admitted[0].task, SeparationTask)
    assert admitted[0].task.command_id == first.command_id
    assert admitted[0].task.owner_node == first.owner_node
    assert admitted[0].task.task_status == TaskStatus.Pending
    refused, _ = separation_generation_events(snapshot, command(), admitted)
    assert len(refused) == 2 and isinstance(refused[1], TaskFailed)
    assert isinstance(refused[0], TaskCreated)
    assert refused[0].task.task_status == TaskStatus.Failed
    assert refused[1].error_type == "audio_separation_unavailable"


def test_same_batch_uses_another_ready_replica() -> None:
    """Independent ready models remain usable while the first model is reserved."""
    snapshot = state(RunnerReady(), 2)
    first, _ = separation_generation_events(snapshot, command())
    second, _ = separation_generation_events(snapshot, command(), first)
    assert isinstance(first[0], TaskCreated) and isinstance(second[0], TaskCreated)
    assert first[0].task.instance_id != second[0].task.instance_id
    assert second[0].task.task_status == TaskStatus.Pending


@pytest.mark.parametrize("status", [RunnerLoading(), RunnerRunning()])
def test_nonready_model_is_not_executable_capacity(status: RunnerStatus) -> None:
    """Prepared libraries, loading and an occupied native model do not admit work."""
    assert not separation_generation_instances(
        state(status), ModelId("adefossez/HTDemucs")
    )


@pytest.mark.parametrize("status", [TaskStatus.Pending, TaskStatus.Running])
def test_persisted_active_task_removes_instance_capacity(status: TaskStatus) -> None:
    """Applied admission and in-flight native work enforce the same capacity rule."""
    snapshot = state(RunnerReady())
    events, identifier = separation_generation_events(snapshot, command())
    assert isinstance(events[0], TaskCreated)
    active = events[0].task.model_copy(update={"task_status": status})
    snapshot = snapshot.model_copy(update={"tasks": {identifier: active}})
    assert not separation_generation_instances(snapshot, ModelId("adefossez/HTDemucs"))


def test_terminal_history_does_not_consume_recovered_capacity() -> None:
    """After normal model reload, terminal history cannot permanently disable it."""
    snapshot = state(RunnerReady())
    events, identifier = separation_generation_events(snapshot, command())
    assert isinstance(events[0], TaskCreated)
    finished = events[0].task.model_copy(update={"task_status": TaskStatus.Cancelled})
    snapshot = snapshot.model_copy(update={"tasks": {identifier: finished}})
    assert separation_generation_instances(snapshot, ModelId("adefossez/HTDemucs")) == [
        InstanceId("instance-0")
    ]


def test_different_model_is_not_a_fallback() -> None:
    """A model alias cannot borrow another loaded separator's execution."""
    assert not separation_generation_instances(
        state(RunnerReady()), ModelId("a/different-model")
    )


@pytest.mark.parametrize("field", ["world_size", "start_layer", "end_layer", "n_layers", "separation_policy"])
def test_old_or_partial_native_model_has_no_generation_capacity(field: str) -> None:
    """A nominal Ready state cannot bypass complete geometry and bound policy."""
    snapshot = state(RunnerReady())
    instance = snapshot.instances[InstanceId("instance-0")]
    assignments = instance.shard_assignments
    runner = RunnerId("runner-0")
    shard = assignments.runner_to_shard[runner].model_copy(update={field: None if field == "separation_policy" else 2})
    changed = instance.model_copy(update={"shard_assignments": assignments.model_copy(update={"runner_to_shard": {runner: shard}})})
    snapshot = snapshot.model_copy(update={"instances": {instance.instance_id: changed}})
    assert not separation_generation_instances(snapshot, ModelId("adefossez/HTDemucs"))
