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


BUILT_IN_NODE = NodeId("node-built-in")
PACKAGE_NODE = NodeId("node-package")


def _chosen_node(
    card: ModelCard, *, built_in_memory_gb: float, package_memory_gb: float = 64
) -> NodeId:
    """Place on a cluster of one built-in node and one package-only node."""
    topology = Topology()
    topology.add_node(BUILT_IN_NODE)
    topology.add_node(PACKAGE_NODE)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {
            BUILT_IN_NODE: create_node_memory(Memory.from_gb(built_in_memory_gb).in_bytes),
            PACKAGE_NODE: create_node_memory(Memory.from_gb(package_memory_gb).in_bytes),
        },
        {BUILT_IN_NODE: create_node_network(), PACKAGE_NODE: create_node_network()},
        node_resources={
            BUILT_IN_NODE: NodeResources(
                backends=frozenset({"llama_cpp", "llama_cpp-vulkan"})
            ),
            PACKAGE_NODE: NodeResources(
                backends=frozenset({PACKAGE_TAG}),
                engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
            ),
        },
    )
    instance = next(iter(placements.values()))
    return next(iter(instance.shard_assignments.node_to_runner))


def _sized_card(preference: tuple[str, ...] = ()) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/text"),
        source_revision="a" * 40,
        storage_size=Memory.from_gb(4),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"llama_cpp", "tensorfold"}),
            backend_preference=preference,
        ),
    )


def test_a_package_node_does_not_draw_a_model_from_a_built_in_node() -> None:
    """More free memory on the package node is not a choice of its engine."""
    assert _chosen_node(_sized_card(), built_in_memory_gb=16) == BUILT_IN_NODE


@pytest.mark.parametrize("preferred", [PACKAGE_TAG, "tensorfold"])
def test_a_card_preference_for_the_package_chooses_its_node(preferred: str) -> None:
    """By lane or by engine name, and against a built-in node with more free memory."""
    assert (
        _chosen_node(
            _sized_card((preferred,)), built_in_memory_gb=64, package_memory_gb=16
        )
        == PACKAGE_NODE
    )


def test_a_card_preference_for_a_built_in_engine_is_ranked_as_before() -> None:
    assert (
        _chosen_node(
            _sized_card(("llama_cpp-vulkan",)), built_in_memory_gb=16
        )
        == BUILT_IN_NODE
    )


def test_a_package_node_is_used_when_no_built_in_node_can_take_the_model() -> None:
    assert _chosen_node(_sized_card(), built_in_memory_gb=1) == PACKAGE_NODE


OFFERED_BUILD = "d" * 64


def _installable(memory_gb: float = 64) -> tuple[NodeResources, int]:
    """A node that does not hold the package but offers to install it."""
    return (
        NodeResources(
            backends=frozenset({PACKAGE_TAG}),
            on_demand_backends=frozenset({PACKAGE_TAG}),
            installable_engine_builds={PACKAGE_TAG: OFFERED_BUILD},
        ),
        Memory.from_gb(memory_gb).in_bytes,
    )


@pytest.mark.parametrize("declared", ["tensorfold", PACKAGE_TAG])
def test_a_package_the_node_will_install_is_stamped_with_the_offered_build(
    declared: str,
) -> None:
    resources, _ = _installable()

    shard = _place(_card(declared), resources)

    assert shard.resolved_backend == PACKAGE_TAG
    assert shard.resolved_engine_build == OFFERED_BUILD


def test_an_offer_without_its_build_is_refused() -> None:
    with pytest.raises(PlacementError, match="exact package build"):
        _place(
            _card("tensorfold"),
            NodeResources(
                backends=frozenset({PACKAGE_TAG}),
                on_demand_backends=frozenset({PACKAGE_TAG}),
            ),
        )


def _between_holder_and_installer(
    card: ModelCard, *, holder_memory_gb: float, excluded: bool = False
) -> tuple[NodeId, ShardMetadata]:
    """Place on a cluster of one node holding the package and one offering it."""
    holder, installer = NodeId("node-holder"), NodeId("node-installer")
    topology = Topology()
    topology.add_node(holder)
    topology.add_node(installer)
    resources, installer_memory = _installable(64)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {
            holder: create_node_memory(Memory.from_gb(holder_memory_gb).in_bytes),
            installer: create_node_memory(installer_memory),
        },
        {holder: create_node_network(), installer: create_node_network()},
        excluded_nodes={holder} if excluded else None,
        node_resources={
            holder: NodeResources(
                backends=frozenset({PACKAGE_TAG}),
                engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
            ),
            installer: resources,
        },
    )
    instance = next(iter(placements.values()))
    node = next(iter(instance.shard_assignments.node_to_runner))
    return node, next(iter(instance.shard_assignments.runner_to_shard.values()))


@pytest.mark.parametrize("declared", ["tensorfold", PACKAGE_TAG])
def test_a_node_that_holds_the_package_is_preferred_to_one_that_would_install_it(
    declared: str,
) -> None:
    """Even against more free memory on the node that would have to install."""
    node, shard = _between_holder_and_installer(_sized_card_for(declared), holder_memory_gb=16)

    assert node == NodeId("node-holder")
    assert shard.resolved_engine_build == PACKAGE_BUILD


def test_a_person_who_rules_out_the_holder_gets_the_install_on_the_node_they_left() -> None:
    node, shard = _between_holder_and_installer(
        _sized_card_for("tensorfold"), holder_memory_gb=16, excluded=True
    )

    assert node == NodeId("node-installer")
    assert shard.resolved_engine_build == OFFERED_BUILD


def test_the_installer_is_used_when_the_holder_cannot_take_the_model() -> None:
    node, _ = _between_holder_and_installer(
        _sized_card_for("tensorfold"), holder_memory_gb=1
    )

    assert node == NodeId("node-installer")


def _sized_card_for(declared: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/text"),
        source_revision="a" * 40,
        storage_size=Memory.from_gb(4),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(compatible_backends=frozenset({declared})),
    )
