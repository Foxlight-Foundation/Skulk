from skulk.shared.apply import apply_runner_status_updated
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerShutdown,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata


def _instance_owning(runner_id: RunnerId) -> MlxRingInstance:
    model_id = ModelId("mlx-community/test-model")
    return MlxRingInstance(
        instance_id=InstanceId("inst"),
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={NodeId("node-a"): runner_id},
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=ModelCard(
                        model_id=model_id,
                        storage_size=Memory.from_mb(1000),
                        n_layers=4,
                        hidden_size=128,
                        supports_tensor=False,
                        tasks=[ModelTask.TextGeneration],
                    ),
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=4,
                    n_layers=4,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_apply_runner_shutdown_removes_runner():
    runner_id = RunnerId()
    state = State(runners={runner_id: RunnerIdle()})

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()), state
    )

    assert runner_id not in new_state.runners


def test_apply_runner_status_updated_adds_runner_owned_by_an_instance():
    runner_id = RunnerId()
    state = State(instances={InstanceId("inst"): _instance_owning(runner_id)})

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()), state
    )

    assert runner_id in new_state.runners


def test_apply_runner_status_updated_ignores_orphan_runner():
    # A status update for a runner that belongs to no instance (its instance was
    # deleted) must not add a record — that was the unbounded State.runners leak.
    runner_id = RunnerId()
    state = State()

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()), state
    )

    assert runner_id not in new_state.runners
