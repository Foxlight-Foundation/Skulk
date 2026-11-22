"""Placement on an installed engine package: stamped exactly, text only, and absent when no node offers one."""

import pytest

from skulk.master.placement import PlacementError, place_instance
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PlacementCardConfig,
)
from skulk.shared.topology import Topology
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.shards import ShardMetadata

NODE = NodeId("node-a")
PACKAGE_TAG = "tensorfold-rocm"
PACKAGE_BUILD = "c" * 64


def _card(*backends: str, task: ModelTask = ModelTask.TextGeneration) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/text"),
        source_revision="a" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[task],
        placement=PlacementCardConfig(compatible_backends=frozenset(backends)),
    )


def _place(card: ModelCard, resources: NodeResources) -> ShardMetadata:
    topology = Topology()
    topology.add_node(NODE)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(16).in_bytes)},
        {NODE: create_node_network()},
        node_resources={NODE: resources},
    )
    instance = next(iter(placements.values()))
    return next(iter(instance.shard_assignments.runner_to_shard.values()))


@pytest.mark.parametrize("declared", ["tensorfold", PACKAGE_TAG])
def test_a_package_placement_is_stamped_with_its_tag_and_exact_build(
    declared: str,
) -> None:
    """A card may name the engine or one lane; the stamp is always the lane held."""
    shard = _place(
        _card(declared),
        NodeResources(
            backends=frozenset({PACKAGE_TAG}),
            engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
        ),
    )

    assert shard.resolved_backend == PACKAGE_TAG
    assert shard.resolved_engine_build == PACKAGE_BUILD


def test_a_package_placement_without_its_build_is_refused() -> None:
    with pytest.raises(PlacementError, match="exact package build"):
        _place(_card("tensorfold"), NodeResources(backends=frozenset({PACKAGE_TAG})))


def test_a_card_that_also_allows_a_package_is_placed_as_before_without_one() -> None:
    """Naming a package engine changes nothing on a node that holds none."""
    resources = NodeResources(backends=frozenset({"llama_cpp", "llama_cpp-vulkan"}))

    with_package_named = _place(_card("llama_cpp", "tensorfold"), resources)
    plain = _place(_card("llama_cpp"), resources)

    assert with_package_named.resolved_backend == plain.resolved_backend
    assert with_package_named.resolved_backend == "llama_cpp-vulkan"
    assert with_package_named.resolved_engine_build == plain.resolved_engine_build


def test_no_node_offering_the_package_means_no_placement() -> None:
    with pytest.raises(PlacementError):
        _place(
            _card("tensorfold"),
            NodeResources(backends=frozenset({"llama_cpp", "llama_cpp-vulkan"})),
        )


def test_a_package_is_not_offered_to_a_card_that_is_not_text_generation() -> None:
    """An embedding card naming the engine finds no eligible node."""
    with pytest.raises(PlacementError):
        _place(
            _card("tensorfold", task=ModelTask.TextEmbedding),
            NodeResources(
                backends=frozenset({PACKAGE_TAG}),
                engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
            ),
        )
