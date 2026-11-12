"""Actual native placement/reservations use real platform filtering and fixed pins."""

from collections.abc import Mapping

import pytest

import skulk.master.placement as placement
from skulk.master.main import Master
from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    add_instance_to_placements,
    place_instance,
)
from skulk.master.placement_utils import (
    reserve_instance_system_ram,
    reserve_instance_vram,
)
from skulk.shared.models.native_music_memory import estimate_native_music_peak
from skulk.shared.models.tests.test_native_music_admission import (
    GIB,
    NODE,
    instance,
    resources,
)
from skulk.shared.models.tests.test_native_music_runtime import card_with_layouts
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance, PlaceInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.native_music import (
    NativeMusicDevice,
    native_music_admission_capacity,
)
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.steward_actions import StewardActionProposalId
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.shards import Sharding


@pytest.fixture
def native_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the observation clock is injected; platform and admission are real."""
    monkeypatch.setattr(placement.time, "time", lambda: 1.0)


def memory(free: int = 64 * GIB) -> dict[NodeId, MemoryUsage]:
    """Supply an unrelated generic pool to detect accidental legacy fitting."""
    return {
        NODE: MemoryUsage.from_bytes(
            ram_total=64 * GIB,
            ram_available=free,
            swap_total=0,
            swap_available=0,
        )
    }


def ordinary(
    observed: NodeResources, commitments: Mapping[InstanceId, Instance] | None = None
) -> Instance:
    """Call production automatic placement against raw native and tiny generic pools."""
    topology = Topology()
    topology.add_node(NODE)
    result = place_instance(
        PlaceInstance(
            model_card=card_with_layouts(),
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        {NODE: memory(1)[NODE]},
        {},
        node_resources={NODE: observed},
        node_vram={NODE: Memory.from_bytes(1)},
        native_music_commitments=commitments,
    )
    return next(iter(result.values()))


@pytest.mark.usefixtures("native_clock")
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "rocm"])
def test_ordinary_and_exact_native_use_same_raw_actual_three_pool_policy(
    device: NativeMusicDevice,
) -> None:
    """Already-reserved generic pools cannot re-charge a completed native assessment."""
    observed = resources(device)
    placed = ordinary(observed)
    shard = next(iter(placed.shard_assignments.runner_to_shard.values()))
    assert shard.native_music_policy is not None
    assert shard.resolved_backend == shard.native_music_policy.backend
    assert (
        shard.world_size == 1
        and shard.start_layer == 0
        and shard.end_layer == shard.model_card.n_layers
    )
    direct = add_instance_to_placements(
        CreateInstance(instance=instance(observed)),
        Topology(),
        {},
        {NODE: memory(1)[NODE]},
        node_resources={NODE: observed},
        node_vram={NODE: Memory.from_bytes(1)},
    )
    exact_shard = next(
        iter(next(iter(direct.values())).shard_assignments.runner_to_shard.values())
    )
    assert exact_shard.native_music_policy == shard.native_music_policy
    assert exact_shard.resolved_engine_build == shard.resolved_engine_build


@pytest.mark.usefixtures("native_clock")
@pytest.mark.parametrize("exact", [False, True])
def test_pending_native_owner_cannot_be_omitted_from_both_master_entry_points(
    exact: bool,
) -> None:
    """Pending plus indexed identities are supplied independently of the returned State map."""
    observed = resources(free=50 * GIB)
    held = instance(observed, "pending")
    commitments = {held.instance_id: held}
    with pytest.raises(PlacementError, match="host memory"):
        if exact:
            add_instance_to_placements(
                CreateInstance(instance=instance(observed, "new")),
                Topology(),
                {},
                {NODE: memory()[NODE]},
                node_resources={NODE: observed},
                native_music_commitments=commitments,
            )
        else:
            ordinary(observed, commitments)


@pytest.mark.usefixtures("native_clock")
def test_exact_native_cannot_supply_cheaper_tensor_policy_or_partial_owner() -> None:
    """Exact callers do not author memory facts or split an unshardable native model."""
    observed = resources()
    original = instance(observed)
    assignments = original.shard_assignments
    runner, shard = next(iter(assignments.runner_to_shard.items()))
    assert shard.native_music_policy is not None
    for updates, message in (
        (
            {
                "native_music_policy": shard.native_music_policy.model_copy(
                    update={"max_target_frames": 48000}
                )
            },
            "policy differs",
        ),
        ({"end_layer": shard.end_layer - 1}, "complete model shard"),
        ({"resolved_engine_build": "ace_step@sha256:" + "a" * 64}, "build differs"),
    ):
        changed = original.model_copy(
            update={
                "shard_assignments": assignments.model_copy(
                    update={
                        "runner_to_shard": {runner: shard.model_copy(update=updates)},
                    }
                )
            }
        )
        with pytest.raises(PlacementError, match=message):
            add_instance_to_placements(
                CreateInstance(instance=changed),
                Topology(),
                {},
                {NODE: memory()[NODE]},
                node_resources={NODE: observed},
            )


@pytest.mark.usefixtures("native_clock")
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "rocm"])
def test_idle_native_owners_hold_full_host_peak_for_other_engines(
    device: NativeMusicDevice,
) -> None:
    """Idle status and generic discrete-VRAM ownership never erase native host costs."""
    observed = resources(device)
    held = instance(observed)
    shard = next(iter(held.shard_assignments.runner_to_shard.values()))
    assert shard.native_music_policy is not None
    estimate = estimate_native_music_peak(shard.model_card, shard.native_music_policy)
    host = (
        estimate.unified_peak_bytes
        if device in {"mps", "rocm"}
        else estimate.host_peak_bytes
    )
    reserved = reserve_instance_system_ram(
        {NODE: memory()[NODE]},
        {held.instance_id: held},
        {NODE: Memory.from_bytes(256 * GIB)},
        node_resources={NODE: observed},
    )
    assert reserved[NODE].in_bytes <= 64 * GIB - host
    assert (
        reserve_instance_system_ram(
            {NODE: memory()[NODE]},
            {held.instance_id: held},
            node_resources={NODE: observed},
            unreflected=frozenset({held.instance_id}),
        )
        == reserved
    )


@pytest.mark.usefixtures("native_clock")
@pytest.mark.parametrize("device", ["mps", "cuda", "rocm"])
def test_ordinary_gpu_cannot_borrow_another_gpu_or_additive_carve_capacity(
    device: NativeMusicDevice,
) -> None:
    """An unrelated 256 GiB generic pool is capped by actual device zero after commitments."""
    observed = resources(device)
    held = instance(observed)
    shard = next(iter(held.shard_assignments.runner_to_shard.values()))
    assert shard.native_music_policy is not None
    cost = estimate_native_music_peak(shard.model_card, shard.native_music_policy)
    capacity = next(iter(observed.native_music_probes.values())).memory_capacity
    assert capacity is not None
    normalized = native_music_admission_capacity(capacity)
    assert normalized.device_available_bytes is not None
    result = reserve_instance_vram(
        {NODE: Memory.from_bytes(256 * GIB)},
        {},
        {held.instance_id: held},
        node_resources={NODE: observed},
    )
    maximum = normalized.device_available_bytes - cost.device_peak_bytes
    if normalized.shared_available_bytes is not None:
        maximum = min(
            maximum, normalized.shared_available_bytes - cost.unified_peak_bytes
        )
    assert result[NODE].in_bytes == maximum


@pytest.mark.usefixtures("native_clock")
def test_restored_native_without_policy_refuses_coexistence_instead_of_crashing_master() -> (
    None
):
    """A missing policy cannot fall back to a serialized-size generic estimate."""
    observed = resources("cuda")
    held = instance(observed)
    assignments = held.shard_assignments
    runner, shard = next(iter(assignments.runner_to_shard.items()))
    held = held.model_copy(
        update={
            "shard_assignments": assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner: shard.model_copy(update={"native_music_policy": None})
                    },
                }
            )
        }
    )
    placements = {held.instance_id: held}
    assert (
        reserve_instance_system_ram({NODE: memory()[NODE]}, placements)[NODE].in_bytes
        == 0
    )
    assert (
        reserve_instance_vram({NODE: Memory.from_bytes(256 * GIB)}, {}, placements)[
            NODE
        ].in_bytes
        == 0
    )


def test_cold_placement_or_repair_requires_preparation_instead_of_installable_readiness() -> (
    None
):
    """A cold repair returns actionable pending rather than inventing a ready engine."""
    observed = resources()
    cold = observed.model_copy(
        update={
            "backends": frozenset(),
            "engine_builds": {},
            "native_music_probes": {},
            "installable_engine_builds": observed.engine_builds,
        }
    )
    with pytest.raises(
        PlacementInfoPendingError, match="installable package is not ready"
    ):
        ordinary(cold)


def test_stale_prepared_native_resources_request_refresh_for_placement_and_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle cached library cannot authorize work with an expired device-zero sample."""
    monkeypatch.setattr(placement.time, "time", lambda: 123.0)
    with pytest.raises(
        PlacementInfoPendingError, match="refresh through mount preparation"
    ):
        ordinary(resources())


@pytest.mark.parametrize("exact", [False, True])
def test_actual_master_native_commitment_map_deduplicates_pending_steward_and_indexed(
    exact: bool,
) -> None:
    """Both production command owners pass complete identities without duplicating State."""
    observed = resources()
    indexed = instance(observed, "indexed")
    pending = instance(observed, "pending")
    steward = instance(observed, "steward")
    master = object.__new__(Master)
    master.state = State().model_copy(
        update={"instances": {indexed.instance_id: indexed}}
    )
    master._pending_instance_reservations = {  # pyright: ignore[reportPrivateUsage]
        indexed.instance_id: indexed.model_copy(update={"ephemeral_port": 24001}),
        pending.instance_id: pending,
    }
    master._steward_reserved_placements = {  # pyright: ignore[reportPrivateUsage]
        StewardActionProposalId("steward-reservation"): {steward.instance_id: steward},
    }
    command = (
        CreateInstance(instance=instance(observed, "new"))
        if exact
        else PlaceInstance(
            model_card=card_with_layouts(),
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        )
    )
    result = master._native_music_placement_commitments(command, master.state.instances)  # pyright: ignore[reportPrivateUsage]
    assert result == {item.instance_id: item for item in (indexed, pending, steward)}
    assert result is not None and result[indexed.instance_id] is indexed
