"""What an engine package said a load takes is kept with its placement and no longer."""

from skulk.shared.apply import apply
from skulk.shared.models.engine_memory_settings import engine_package_memory_settings
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    EnginePackageNeedStated,
    Event,
    IndexedEvent,
    InstanceDeleted,
    NodeTimedOut,
    is_persistable_control_event,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata

PACKAGE_TAG = "tensorfold-rocm"
NEED = 30 * 1024**3


def _instance(node: NodeId, runner: RunnerId, backend: str) -> MlxRingInstance:
    card = ModelCard(
        model_id=ModelId("test/text"),
        storage_size=Memory.from_gb(8),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=32,
                    n_layers=32,
                    resolved_backend=backend,
                    memory_settings=engine_package_memory_settings(2 * 1024**3)
                    if backend == PACKAGE_TAG
                    else None,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def _placed(backend: str = PACKAGE_TAG) -> tuple[State, MlxRingInstance, RunnerId, NodeId]:
    node, runner = NodeId(), RunnerId()
    instance = _instance(node, runner, backend)
    return State(instances={instance.instance_id: instance}), instance, runner, node


def _after(state: State, event: Event) -> State:
    return apply(
        state, IndexedEvent(idx=state.last_event_applied_idx + 1, event=event)
    )


def test_the_figure_is_a_durable_control_event() -> None:
    """The master must order it; an unlisted event would be skipped unseen."""
    assert is_persistable_control_event(
        EnginePackageNeedStated(runner_id=RunnerId(), need_bytes=NEED)
    )


def test_a_package_runners_figure_is_recorded_and_a_later_one_replaces_it() -> None:
    state, _, runner, _ = _placed()

    first = _after(state, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED))
    second = _after(
        first, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED + 1)
    )

    assert first.engine_package_needs == {runner: NEED}
    assert second.engine_package_needs == {runner: NEED + 1}


def test_a_figure_for_a_built_in_engines_runner_is_dropped() -> None:
    """Memory accounting for built-in engines can never be changed this way."""
    state, _, runner, _ = _placed(backend="llama_server-rocm")

    after = _after(state, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED))

    assert after.engine_package_needs == {}


def test_a_figure_for_a_runner_of_no_placement_is_dropped() -> None:
    """A figure can arrive after its placement was removed."""
    state, _, _, _ = _placed()

    after = _after(
        state, EnginePackageNeedStated(runner_id=RunnerId(), need_bytes=NEED)
    )

    assert after.engine_package_needs == {}


def test_removing_the_placement_removes_its_figure_and_no_other() -> None:
    state, instance, runner, _ = _placed()
    other_node, other_runner = NodeId(), RunnerId()
    other = _instance(other_node, other_runner, PACKAGE_TAG)
    state = state.model_copy(
        update={"instances": {**state.instances, other.instance_id: other}}
    )
    state = _after(state, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED))
    state = _after(
        state, EnginePackageNeedStated(runner_id=other_runner, need_bytes=NEED + 7)
    )

    after = _after(state, InstanceDeleted(instance_id=instance.instance_id))

    assert after.engine_package_needs == {other_runner: NEED + 7}


def test_a_node_that_times_out_takes_its_placements_figures_with_it() -> None:
    state, _, runner, node = _placed()
    state = _after(state, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED))

    after = _after(state, NodeTimedOut(node_id=node))

    assert after.instances == {}
    assert after.engine_package_needs == {}


def test_a_new_master_session_keeps_the_figures_of_the_placements_it_keeps() -> None:
    """Placements survive failover with their runners, so their reservations do."""
    state, instance, runner, _ = _placed()
    state = _after(state, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED))

    seeded = seed_state_for_new_session(state)

    assert seeded.instances == {instance.instance_id: instance}
    assert seeded.engine_package_needs == {runner: NEED}


def test_figures_survive_the_snapshot_wire_format() -> None:
    """A follower that bootstraps from a snapshot reserves what the master does."""
    state, _, runner, _ = _placed()
    state = _after(state, EnginePackageNeedStated(runner_id=runner, need_bytes=NEED))

    restored = State.model_validate_json(state.model_dump_json())

    assert restored.engine_package_needs == {runner: NEED}
    assert State().engine_package_needs == {}
