"""Music placements remain single-host through both master command paths."""

import tomllib
from pathlib import Path

import pytest

import skulk.master.placement as placement_module
from skulk.master.placement import (
    PlacementError,
    add_instance_to_placements,
    place_instance,
)
from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance, PlaceInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.worker.instances import (
    InstanceId,
    InstanceMeta,
    LlamaRpcInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding


def _music_card() -> ModelCard:
    """Use the shipped immutable MiniMax card as placement truth."""
    path = Path(RESOURCES_DIR) / "music_model_cards" / "audio-cpp--MiniMax-Music3-GGUF-Q4.toml"
    return ModelCard.model_validate(tomllib.loads(path.read_text()))


def test_music_place_command_rejects_requested_multi_node_width() -> None:
    """An ordinary placement must not mint a two-node music instance."""
    command = PlaceInstance(
        model_card=_music_card(), sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing, min_nodes=2,
    )
    with pytest.raises(PlacementError, match="exactly one node"):
        place_instance(command, Topology(), {}, {}, {})


def test_music_exact_command_rejects_two_node_instance() -> None:
    """Internal exact commands enforce the same host count as the API."""
    card = _music_card()
    nodes = (NodeId("music-a"), NodeId("music-b"))
    runners = (RunnerId("runner-a"), RunnerId("runner-b"))
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card, device_rank=rank, world_size=2,
                    start_layer=rank * 18, end_layer=(rank + 1) * 18,
                    n_layers=36,
                )
                for rank, runner in enumerate(runners)
            },
            node_to_runner=dict(zip(nodes, runners, strict=True)),
        ),
        hosts_by_node={node: [] for node in nodes},
        ephemeral_port=52415,
    )
    with pytest.raises(PlacementError, match="exactly one node"):
        add_instance_to_placements(CreateInstance(instance=instance), Topology(), {}, {})


def test_music_exact_command_rejects_two_runners_on_one_node() -> None:
    """One host must still contain only one complete music model runner."""
    card = _music_card()
    node = NodeId("music-node")
    runners = (RunnerId("runner-a"), RunnerId("runner-b"))
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card, device_rank=rank, world_size=2,
                    start_layer=rank * 18, end_layer=(rank + 1) * 18,
                    n_layers=36,
                )
                for rank, runner in enumerate(runners)
            },
            node_to_runner={node: runners[0]},
        ),
        hosts_by_node={node: []},
        ephemeral_port=52415,
    )
    with pytest.raises(PlacementError, match="exactly one runner shard"):
        add_instance_to_placements(CreateInstance(instance=instance), Topology(), {}, {})


def test_music_exact_command_rejects_stamped_unclaimed_build() -> None:
    """An internal caller cannot bypass preparation with a client backend stamp."""
    card = _music_card()
    node = NodeId("music-node")
    runner = RunnerId("music-runner")
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=36, n_layers=36,
                resolved_backend="audio_cpp-cpu",
            )},
            node_to_runner={node: runner},
        ),
        hosts_by_node={node: []},
        ephemeral_port=52415,
    )
    resources = NodeResources(
        backends=frozenset({"audio_cpp", "audio_cpp-cpu"}),
        architecture="arm64",
        hardware_classes=frozenset({"platform:darwin"}),
        engine_builds={"audio_cpp-cpu": "unqualified-build"},
    )
    with pytest.raises(PlacementError, match="matching signed support claim"):
        add_instance_to_placements(
            CreateInstance(instance=instance), Topology(), {}, {},
            node_resources={node: resources},
        )


def test_music_exact_command_rejects_rpc_instance() -> None:
    """A one-host RPC shape cannot skip the music engine readiness check."""
    card = _music_card()
    node = NodeId("music-node")
    runner = RunnerId("music-runner")
    instance = LlamaRpcInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=36, n_layers=36,
            )},
            node_to_runner={node: runner},
        ),
        driver_node=node,
        donor_endpoints={},
    )
    with pytest.raises(PlacementError, match="cannot use llama.cpp RPC"):
        add_instance_to_placements(
            CreateInstance(instance=instance), Topology(), {}, {},
        )


def test_music_exact_placement_stamps_verified_engine_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller supplied build cannot replace the node's measured identity."""
    card = _music_card()
    node = NodeId("music-node")
    runner = RunnerId("music-runner")
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=36, n_layers=36,
                resolved_backend="audio_cpp-cpu",
                resolved_engine_build="caller-build",
            )},
            node_to_runner={node: runner},
        ),
        hosts_by_node={node: []},
        ephemeral_port=52415,
    )
    resources = NodeResources(
        backends=frozenset({"audio_cpp", "audio_cpp-cpu"}),
        engine_builds={"audio_cpp-cpu": "verified-build"},
    )
    def supported(_card: ModelCard, _resources: NodeResources) -> frozenset[str]:
        return frozenset({"audio_cpp-cpu"})

    monkeypatch.setattr(placement_module, "_card_platform_backends", supported)
    memory = MemoryUsage.from_bytes(
        ram_total=1 << 40, ram_available=1 << 40,
        swap_total=0, swap_available=0,
    )
    placed = add_instance_to_placements(
        CreateInstance(instance=instance), Topology(), {}, {node: memory},
        node_resources={node: resources},
    )
    stamped = placed[instance.instance_id].shard_assignments.runner_to_shard[runner]
    assert stamped.resolved_engine_build == "verified-build"


@pytest.mark.parametrize("lane", ["audio_cpp-cpu", "audio_cpp-metal"])
def test_music_exact_placement_checks_full_system_memory_footprint(
    monkeypatch: pytest.MonkeyPatch, lane: str,
) -> None:
    """Weights alone cannot admit a CPU or unified-memory music server."""
    card = _music_card()
    node = NodeId("music-node")
    runner = RunnerId("music-runner")
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=36, n_layers=36,
                resolved_backend=lane,
            )},
            node_to_runner={node: runner},
        ),
        hosts_by_node={node: []},
        ephemeral_port=52415,
    )
    resources = NodeResources(
        backends=frozenset({"audio_cpp", lane}),
        engine_builds={lane: "verified-build"},
    )
    def supported(_card: ModelCard, _resources: NodeResources) -> frozenset[str]:
        return frozenset({lane})

    monkeypatch.setattr(placement_module, "_card_platform_backends", supported)
    memory = MemoryUsage.from_bytes(
        ram_total=Memory.from_gb(64).in_bytes,
        ram_available=card.storage_size.in_bytes + Memory.from_mb(100).in_bytes,
        swap_total=0, swap_available=0,
    )
    with pytest.raises(PlacementError, match="Insufficient system memory"):
        add_instance_to_placements(
            CreateInstance(instance=instance), Topology(), {}, {node: memory},
            node_resources={node: resources},
        )
