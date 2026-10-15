# pyright: reportPrivateUsage=false
"""Real placement and worker guards agree on staged native video reservations."""

import asyncio

import pytest

import skulk.worker.main as worker_module
from skulk.master.placement import PlacementError, place_instance
from skulk.master.tests.conftest import create_node_network
from skulk.provisioning.mlx_video import PUBLISHED_WHEELS
from skulk.shared.models.model_cards import ModelCard, RuntimeCapabilityCardConfig
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding
from skulk.worker.main import Worker


def _card() -> ModelCard:
    card = asyncio.run(load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4")))
    assert card is not None
    return card


def _memory(total_gib: int) -> MemoryUsage:
    return MemoryUsage.from_bytes(ram_total=total_gib * 1024**3, ram_available=(total_gib - 2) * 1024**3, swap_total=0, swap_available=0)


def _resources(installed: bool) -> NodeResources:
    backend = "mlx_video-metal"
    build = PUBLISHED_WHEELS[0].build_identity
    return NodeResources(
        backends=frozenset({backend}),
        on_demand_backends=frozenset() if installed else frozenset({backend}),
        engine_builds={backend: build} if installed else {},
        installable_engine_builds={} if installed else {backend: build},
        hardware_classes=frozenset({"platform:darwin", "accelerator:previously-untested"}),
        participation="full",
    )


@pytest.mark.parametrize("installed", [True, False])
def test_ordinary_24_gib_placement_and_worker_admit_untested_native_hardware(monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    """Both ready and on-demand capacity survive every ordinary placement guard."""
    node = NodeId("native-video-resource-node")
    topology = Topology()
    topology.add_node(node)
    card = _card()
    command = PlaceInstance(model_card=card, sharding=Sharding.Pipeline, instance_meta=InstanceMeta.MlxRing, min_nodes=1)
    placed = place_instance(command, topology, {}, {node: _memory(24)}, {node: create_node_network()}, node_resources={node: _resources(installed)})
    assert len(placed) == 1
    instance = next(iter(placed.values()))
    assignments = instance.shard_assignments
    assert tuple(assignments.node_to_runner) == (node,)
    shard = next(iter(assignments.runner_to_shard.values()))
    assert (shard.world_size, shard.start_layer, shard.end_layer) == (1, 0, card.n_layers)
    assert shard.resolved_backend == "mlx_video-metal"
    assert shard.resolved_engine_build == PUBLISHED_WHEELS[0].build_identity
    monkeypatch.setattr(worker_module, "_local_usable_vram", lambda: None)

    def observed(_cls: type[MemoryUsage]) -> MemoryUsage:
        return _memory(24)

    monkeypatch.setattr(MemoryUsage, "from_local_gpu_wireable", classmethod(observed))
    worker = object.__new__(Worker)
    worker.node_id = node
    assert worker._local_shard_fit_error(shard) is None


@pytest.mark.parametrize("total_gib,budget", [(16, None), (24, 32 * 1024**3)])
def test_native_placement_keeps_real_capacity_limits(total_gib: int, budget: int | None) -> None:
    """Staging and a larger explicit reservation never bypass the live memory ceiling."""
    node = NodeId("native-video-small-node")
    topology = Topology()
    topology.add_node(node)
    card = _card()
    if budget is not None:
        card = card.model_copy(update={"runtime": RuntimeCapabilityCardConfig(video_memory_budget_bytes=budget)})
    command = PlaceInstance(model_card=card, sharding=Sharding.Pipeline, instance_meta=InstanceMeta.MlxRing, min_nodes=1)
    with pytest.raises(PlacementError):
        place_instance(command, topology, {}, {node: _memory(total_gib)}, {node: create_node_network()}, node_resources={node: _resources(True)})
