"""Managed vLLM placement binds the exact installable recipe without a GPU allowlist."""

import pytest

from skulk.master.placement import place_instance
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.provisioning.vllm_runtime import PUBLISHED_RECIPES
from skulk.shared.models.model_cards import ModelCard, ModelTask, PlacementCardConfig
from skulk.shared.topology import Topology
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources


@pytest.mark.parametrize("on_demand", [True, False])
def test_vllm_placement_stamps_the_recipe_for_install_and_reload(
    on_demand: bool,
) -> None:
    """Installed and first-use placements retain the same immutable runtime identity."""
    recipe = PUBLISHED_RECIPES[0]
    backend = "vllm-cuda"
    card = ModelCard(
        model_id=ModelId("test/text"),
        source_revision="a" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(compatible_backends=frozenset({backend})),
    )
    node = NodeId("compatible-untested-node")
    topology = Topology()
    topology.add_node(node)
    resources = NodeResources(
        backends=frozenset({backend}),
        hardware_classes=frozenset({"nvidia", "nvidia:sm-8.6"}),
        on_demand_backends=frozenset({backend}) if on_demand else frozenset(),
        installable_engine_builds={backend: recipe.build_identity} if on_demand else {},
        engine_builds={} if on_demand else {backend: recipe.build_identity},
    )
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {node: create_node_memory(Memory.from_gb(16).in_bytes)},
        {node: create_node_network()},
        node_resources={node: resources},
        node_vram={node: Memory.from_gb(8)},
    )
    shard = next(
        iter(next(iter(placements.values())).shard_assignments.runner_to_shard.values())
    )
    assert shard.resolved_backend == backend
    assert shard.resolved_engine_build == recipe.build_identity
