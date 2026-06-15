from skulk.shared.apply import apply_instance_deleted, apply_runner_status_updated
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import InstanceDeleted, RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerShuttingDown,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata


def _make_pipeline_shard(
    model_id: ModelId, device_rank: int, world_size: int
) -> PipelineShardMetadata:
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


def _make_instance(
    instance_id: InstanceId,
    model_id: ModelId,
    node_to_runner: dict[NodeId, RunnerId],
) -> MlxRingInstance:
    runner_to_shard = {
        rid: _make_pipeline_shard(model_id, device_rank=i, world_size=len(node_to_runner))
        for i, rid in enumerate(node_to_runner.values())
    }
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner=node_to_runner,
            runner_to_shard=runner_to_shard,
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_apply_instance_deleted_removes_its_runner_records() -> None:
    """Deleting an instance must also drop its runner records (state leak fix).

    Runner records are otherwise only removed by a terminal
    RunnerStatusUpdated(RunnerShutdown), which is unreliably delivered (the
    worker cancels the supervisor's event forwarder on shutdown, often before
    that final status is forwarded, and on a master-failover teardown the
    forwarder is torn down outright). Every instance delete therefore leaked one
    RunnerShuttingDown record per rank, growing State.runners without bound.
    apply_instance_deleted now prunes the deleted instance's runners directly.
    """
    model_id = ModelId("mlx-community/gemma-4-26b-a4b-it-4bit")
    node_a, node_b = NodeId("node-a"), NodeId("node-b")
    node_c = NodeId("node-c")
    doomed_id = InstanceId("doomed")
    survivor_id = InstanceId("survivor")
    runner_a, runner_b = RunnerId("runner-a"), RunnerId("runner-b")
    survivor_runner = RunnerId("survivor-runner")

    doomed = _make_instance(
        doomed_id, model_id, {node_a: runner_a, node_b: runner_b}
    )
    survivor = _make_instance(survivor_id, model_id, {node_c: survivor_runner})

    state = State(
        instances={doomed_id: doomed, survivor_id: survivor},
        runners={
            runner_a: RunnerReady(),
            runner_b: RunnerShuttingDown(),
            survivor_runner: RunnerReady(),
        },
    )

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=doomed_id), state)

    # instance and BOTH its runner records gone; survivor untouched
    assert doomed_id not in new_state.instances
    assert survivor_id in new_state.instances
    assert runner_a not in new_state.runners
    assert runner_b not in new_state.runners
    assert survivor_runner in new_state.runners
    assert len(new_state.runners) == 1


def test_runner_status_update_does_not_resurrect_a_deleted_instances_runner() -> None:
    """A late RunnerShuttingDown after InstanceDeleted must not re-add the record.

    This is the other half of the leak: even after apply_instance_deleted drops
    the runner records, the worker's teardown emits a RunnerShuttingDown status
    that races behind the deletion. Without the guard it re-adds a record that
    never gets a terminal RunnerShutdown to clear it (that final status is
    routinely lost when the supervisor's forwarder is cancelled), so the record
    leaks. apply_runner_status_updated now ignores updates for a runner that no
    longer belongs to any instance.
    """
    model_id = ModelId("mlx-community/gemma-4-26b-a4b-it-4bit")
    runner_a = RunnerId("runner-a")
    doomed = _make_instance(
        InstanceId("doomed"), model_id, {NodeId("node-a"): runner_a}
    )
    state = State(instances={InstanceId("doomed"): doomed}, runners={})

    # delete the instance (drops any runner records + the instance)
    state = apply_instance_deleted(
        InstanceDeleted(instance_id=InstanceId("doomed")), state
    )
    assert runner_a not in state.runners

    # the straggler RunnerShuttingDown arrives afterward — must be ignored
    state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_a, runner_status=RunnerShuttingDown()),
        state,
    )
    assert runner_a not in state.runners
    assert len(state.runners) == 0


def test_apply_instance_deleted_unknown_instance_is_a_noop() -> None:
    # Deleting an instance that isn't in state must not raise or touch runners.
    model_id = ModelId("mlx-community/gemma-4-26b-a4b-it-4bit")
    survivor_runner = RunnerId("survivor-runner")
    survivor = _make_instance(
        InstanceId("survivor"), model_id, {NodeId("node-c"): survivor_runner}
    )
    state = State(
        instances={InstanceId("survivor"): survivor},
        runners={survivor_runner: RunnerReady()},
    )

    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=InstanceId("ghost")), state
    )

    assert InstanceId("survivor") in new_state.instances
    assert survivor_runner in new_state.runners
