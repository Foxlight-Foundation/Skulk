from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    LarqlRunnerReadinessUpdated,
    RunnerStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.larql import LarqlRunnerReadiness
from exo.shared.types.worker.runners import RunnerShutdown
from exo.worker.tests.constants import RUNNER_1_ID
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_larql_readiness_updates_state() -> None:
    readiness = LarqlRunnerReadiness(
        runner_id=RUNNER_1_ID,
        vindex_uri="hf://skulk/gemma-4-26b-a4b-expert-server-q4-k-vindex",
        preset="expert-server",
        start_layer=4,
        end_layer=12,
        expert_range=None,
        units_manifest_path=None,
        host="127.0.0.1",
        port=49152,
        status="ready",
        ram_footprint=Memory.from_mb(128),
    )

    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=LarqlRunnerReadinessUpdated(readiness=readiness),
        ),
    )

    assert state.larql_runner_readiness[RUNNER_1_ID] == readiness


def test_larql_readiness_is_removed_on_runner_shutdown() -> None:
    readiness = LarqlRunnerReadiness(
        runner_id=RUNNER_1_ID,
        vindex_uri="hf://skulk/gemma",
        preset="full",
        start_layer=0,
        end_layer=1,
        host="127.0.0.1",
        port=49152,
        status="ready",
    )
    state = State(larql_runner_readiness={RUNNER_1_ID: readiness})

    updated = apply(
        state,
        IndexedEvent(
            idx=0,
            event=RunnerStatusUpdated(
                runner_id=RUNNER_1_ID,
                runner_status=RunnerShutdown(),
            ),
        ),
    )

    assert RUNNER_1_ID not in updated.larql_runner_readiness


def test_larql_readiness_is_removed_on_instance_delete() -> None:
    instance = get_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        node_to_runner={NodeId("node-a"): RUNNER_1_ID},
        runner_to_shard={
            RUNNER_1_ID: get_pipeline_shard_metadata(
                ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"), 0
            )
        },
    )
    readiness = LarqlRunnerReadiness(
        runner_id=RUNNER_1_ID,
        vindex_uri="hf://skulk/gemma",
        preset="full",
        start_layer=0,
        end_layer=1,
        host="127.0.0.1",
        port=49152,
        status="ready",
    )
    state = apply(State(), IndexedEvent(idx=0, event=InstanceCreated(instance=instance)))
    state = state.model_copy(
        update={"larql_runner_readiness": {RUNNER_1_ID: readiness}}
    )

    updated = apply(
        state,
        IndexedEvent(idx=1, event=InstanceDeleted(instance_id=instance.instance_id)),
    )

    assert RUNNER_1_ID not in updated.larql_runner_readiness
