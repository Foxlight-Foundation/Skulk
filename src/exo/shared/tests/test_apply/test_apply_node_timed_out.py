from datetime import datetime

from exo.shared.apply import apply_node_timed_out
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import StartWarmup, TaskId, TaskStatus
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerWarmingUp,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_pipeline_shard(model_id: ModelId, device_rank: int, world_size: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100000),
            n_layers=32,
            hidden_size=2048,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


def test_apply_node_timed_out_removes_affected_instances_runners_and_tasks() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_c = NodeId("node-c")

    affected_instance_id = InstanceId("affected-instance")
    unaffected_instance_id = InstanceId("unaffected-instance")

    affected_runner_a = RunnerId("affected-runner-a")
    affected_runner_b = RunnerId("affected-runner-b")
    unaffected_runner = RunnerId("unaffected-runner")

    model_id = ModelId("mlx-community/gemma-4-26b-a4b-it-4bit")

    affected_instance = MlxRingInstance(
        instance_id=affected_instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={
                node_a: affected_runner_a,
                node_b: affected_runner_b,
            },
            runner_to_shard={
                affected_runner_a: _make_pipeline_shard(model_id, device_rank=0, world_size=2),
                affected_runner_b: _make_pipeline_shard(model_id, device_rank=1, world_size=2),
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )

    unaffected_instance = MlxRingInstance(
        instance_id=unaffected_instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={node_c: unaffected_runner},
            runner_to_shard={
                unaffected_runner: _make_pipeline_shard(model_id, device_rank=0, world_size=1),
            },
        ),
        hosts_by_node={},
        ephemeral_port=50001,
    )

    affected_task_id = TaskId("affected-task")
    unaffected_task_id = TaskId("unaffected-task")
    state = State(
        instances={
            affected_instance_id: affected_instance,
            unaffected_instance_id: unaffected_instance,
        },
        runners={
            affected_runner_a: RunnerWarmingUp(),
            affected_runner_b: RunnerReady(),
            unaffected_runner: RunnerIdle(),
        },
        tasks={
            affected_task_id: StartWarmup(
                task_id=affected_task_id,
                instance_id=affected_instance_id,
                task_status=TaskStatus.Pending,
            ),
            unaffected_task_id: StartWarmup(
                task_id=unaffected_task_id,
                instance_id=unaffected_instance_id,
                task_status=TaskStatus.Pending,
            ),
        },
        last_seen={
            node_a: datetime.now(),
            node_b: datetime.now(),
            node_c: datetime.now(),
        },
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=node_a), state)

    assert affected_instance_id not in new_state.instances
    assert unaffected_instance_id in new_state.instances

    assert affected_runner_a not in new_state.runners
    assert affected_runner_b not in new_state.runners
    assert unaffected_runner in new_state.runners

    assert affected_task_id not in new_state.tasks
    assert unaffected_task_id in new_state.tasks

    assert node_a not in new_state.last_seen
    assert node_b in new_state.last_seen
    assert node_c in new_state.last_seen
