"""Native placement needs verified resources and complete workload accounting."""

import tomllib
from pathlib import Path

import pytest

from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    add_instance_to_placements,
    place_instance,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.separation_runtime import (
    estimate_demucs_peak,
    resolve_demucs_policy,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance, PlaceInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding

NODE = NodeId("native-host")
RUNNER = RunnerId("native-runner")
BUILD = "demucs@sha256:" + "a" * 64


def card() -> ModelCard:
    """Use an immutable tensor fixture without any qualification permission list."""
    return ModelCard.model_validate(tomllib.loads(
        (Path(__file__).parents[2] / "shared/models/tests/htdemucs-card.toml").read_text()
    ))


def resources() -> NodeResources:
    """Observe a usable CPU library and its exact native payload identity."""
    return NodeResources(
        backends=frozenset({"demucs-cpu"}),
        engine_builds={"demucs-cpu": BUILD},
    )


def memory(available: int) -> dict[NodeId, MemoryUsage]:
    """Keep total memory fixed while varying the live workload headroom."""
    return {NODE: MemoryUsage.from_bytes(
        ram_total=4 * 1024**3, ram_available=available,
        swap_total=0, swap_available=0,
    )}


def place(observed: NodeResources | None, available: int) -> Instance:
    """Exercise ordinary placement without replacing its backend or fit logic."""
    topology = Topology()
    topology.add_node(NODE)
    result = place_instance(
        PlaceInstance(model_card=card(), sharding=Sharding.Pipeline,
                      instance_meta=InstanceMeta.MlxRing, min_nodes=1),
        topology, {}, memory(available), {},
        node_resources={NODE: observed} if observed is not None else {},
        # A CPU engine must not be admitted against an unrelated discrete GPU.
        node_vram={NODE: Memory.from_mb(128)},
    )
    return next(iter(result.values()))


def exact(*, backend: str | None = None, build: str | None = None) -> CreateInstance:
    """Construct an exact complete native model assignment for the other entry."""
    model = card()
    return CreateInstance(instance=MlxRingInstance(
        instance_id=InstanceId("native-instance"),
        shard_assignments=ShardAssignments(
            model_id=model.model_id,
            node_to_runner={NODE: RUNNER},
            runner_to_shard={RUNNER: PipelineShardMetadata(
                model_card=model, device_rank=0, world_size=1,
                start_layer=0, end_layer=model.n_layers, n_layers=model.n_layers,
                resolved_backend=backend, resolved_engine_build=build,
            )},
        ), hosts_by_node={}, ephemeral_port=50000,
    ))


@pytest.mark.parametrize("observed", [
    None,
    NodeResources(backends=frozenset()),
    NodeResources(backends=frozenset({"demucs-cpu"})),
    NodeResources(backends=frozenset({"demucs-cpu"}), engine_builds={"demucs-cpu": "unverified"}),
    NodeResources(backends=frozenset({"demucs-cpu"}),
                  on_demand_backends=frozenset({"demucs-cpu"}),
                  installable_engine_builds={"demucs-cpu": BUILD}),
])
def test_installable_or_unverified_runtime_cannot_place(observed: NodeResources | None) -> None:
    """Neither route may turn a package candidate or caller stamp into capacity."""
    with pytest.raises(PlacementInfoPendingError):
        place(observed, 4 * 1024**3)
    with pytest.raises(PlacementInfoPendingError):
        add_instance_to_placements(exact(backend="demucs-cpu", build=BUILD), Topology(), {},
                                   memory(4 * 1024**3),
                                   node_resources={NODE: observed} if observed is not None else {})


def test_verified_untested_native_runtime_stamps_both_placement_entries() -> None:
    """No support claim is needed; actual compatible runtime identity is recorded."""
    available = estimate_demucs_peak(card(), backend="demucs-cpu").in_bytes
    ordinary = place(resources(), available)
    shard = next(iter(ordinary.shard_assignments.runner_to_shard.values()))
    assert (shard.resolved_backend, shard.resolved_engine_build) == ("demucs-cpu", BUILD)
    assert shard.separation_policy == resolve_demucs_policy(card(), "demucs-cpu")
    assert shard.world_size == 1 and shard.start_layer == 0 and shard.end_layer == card().n_layers
    created = exact()
    result = add_instance_to_placements(created, Topology(), {}, memory(available),
                                       node_resources={NODE: resources()})
    direct = result[created.instance.instance_id].shard_assignments.runner_to_shard[RUNNER]
    assert (direct.resolved_backend, direct.resolved_engine_build) == ("demucs-cpu", BUILD)
    assert direct.separation_policy == shard.separation_policy


def test_small_weight_file_cannot_bypass_complete_workload_reservation() -> None:
    """The native 84 MB tensor is not its several-GB execution footprint."""
    available = estimate_demucs_peak(card(), backend="demucs-cpu").in_bytes - 1
    with pytest.raises(PlacementError, match="fits"):
        place(resources(), available)
    with pytest.raises(PlacementError, match="Insufficient system memory"):
        add_instance_to_placements(exact(), Topology(), {}, memory(available),
                                   node_resources={NODE: resources()})


def test_exact_stale_native_build_cannot_substitute_live_package() -> None:
    """Replanning is required if an explicitly requested build has changed."""
    with pytest.raises(PlacementError, match="differs from live"):
        add_instance_to_placements(exact(backend="demucs-cpu", build="demucs@sha256:" + "b" * 64),
                                   Topology(), {}, memory(4 * 1024**3),
                                   node_resources={NODE: resources()})


@pytest.mark.parametrize("partial", ["world", "start", "end", "layers"])
def test_exact_native_assignment_cannot_load_only_part_of_model(partial: str) -> None:
    """Internal exact commands enforce complete model geometry as ordinary ones do."""
    created = exact()
    assignment = created.instance.shard_assignments
    shard = assignment.runner_to_shard[RUNNER]
    field = {"world": "world_size", "start": "start_layer", "end": "end_layer", "layers": "n_layers"}[partial]
    altered = shard.model_copy(update={field: 2})
    instance = created.instance.model_copy(update={"shard_assignments": assignment.model_copy(
        update={"runner_to_shard": {RUNNER: altered}})})
    with pytest.raises(PlacementError, match="partial"):
        add_instance_to_placements(CreateInstance(instance=instance), Topology(), {}, memory(4 * 1024**3),
                                   node_resources={NODE: resources()})
