"""The actual cache policy must agree across placement and worker fit guards."""

import pytest

from skulk.master.placement import (
    PlacementError,
    add_instance_to_placements,
    place_instance,
)
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.shared.models.engine_memory_settings import EngineMemorySettings
from skulk.shared.models.memory_estimate import estimate_shard_footprint
from skulk.shared.models.model_cards import PlacementCardConfig
from skulk.shared.models.tests.test_engine_memory_settings import memory_card
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources


def test_cache_compression_changes_fit_without_changing_card() -> None:
    card = memory_card().model_copy(
        update={
            "placement": PlacementCardConfig(compatible_backends=frozenset({"mlx"}))
        }
    )
    compressed = EngineMemorySettings(
        cache_backend="mlx_quantized", key_bits=4, value_bits=4
    )
    plain_bytes = estimate_shard_footprint(
        card, 1.0, resolved_backend="mlx-metal"
    ).in_bytes
    compressed_bytes = estimate_shard_footprint(
        card, 1.0, resolved_backend="mlx-metal", memory_settings=compressed
    ).in_bytes
    capacity = (plain_bytes + compressed_bytes) // 2
    node = NodeId("cache-node")
    topology = Topology()
    topology.add_node(node)
    resources = NodeResources(
        backends=frozenset({"mlx", "mlx-metal"}),
        engine_memory_settings={"mlx": compressed},
    )
    memory = {node: create_node_memory(capacity, ram_total=Memory.from_gb(32).in_bytes)}
    command = place_instance_command(card)
    instances = place_instance(
        command,
        topology,
        {},
        memory,
        {node: create_node_network()},
        node_resources={node: resources},
    )
    shard = next(
        iter(next(iter(instances.values())).shard_assignments.runner_to_shard.values())
    )
    assert shard.memory_settings == compressed
    assert (
        estimate_shard_footprint(
            card,
            1.0,
            resolved_backend=shard.resolved_backend,
            memory_settings=shard.memory_settings,
            cache_layer_count=shard.end_layer - shard.start_layer,
        ).in_bytes
        < capacity
    )
    with pytest.raises(PlacementError):
        place_instance(
            command,
            topology,
            {},
            memory,
            {node: create_node_network()},
            node_resources={
                node: resources.model_copy(
                    update={"engine_memory_settings": {"mlx": EngineMemorySettings()}}
                )
            },
        )


def test_exact_placement_cannot_silently_replace_cache_policy() -> None:
    """An exact request retries when its captured policy differs from live controls."""
    from skulk.master.tests.test_recurrent_memory_admission import hybrid_instance
    node=NodeId("policy-node")
    instance=hybrid_instance(node)
    shards={runner:shard.model_copy(update={"memory_settings":EngineMemorySettings(cache_backend="mlx_quantized",key_bits=4,value_bits=4)}) for runner,shard in instance.shard_assignments.runner_to_shard.items()}
    instance=instance.model_copy(update={"shard_assignments":instance.shard_assignments.model_copy(update={"runner_to_shard":shards})})
    topology=Topology()
    topology.add_node(node)
    resources=NodeResources(backends=frozenset({"llama_server-cuda"}),engine_memory_settings={"llama_server":EngineMemorySettings()})
    with pytest.raises(PlacementError,match="settings changed"):
        add_instance_to_placements(CreateInstance(instance=instance),topology,{},
            {node:create_node_memory(Memory.from_gb(128).in_bytes)},node_resources={node:resources})
