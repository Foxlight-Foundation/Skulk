"""Placement on an installed engine package: stamped exactly, text only, and absent when no node offers one."""

import pytest

from skulk.master.placement import PlacementError, place_instance
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.shared.models.engine_memory_settings import (
    EngineMemorySettings,
    engine_package_memory_settings,
    node_engine_memory_settings,
)
from skulk.shared.models.memory_estimate import (
    backend_offloads_to_vram,
    estimate_weight_runtime_bytes,
    shard_preallocates_kv_upfront,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PlacementCardConfig,
    VisionCardConfig,
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


def test_a_text_card_with_a_vision_section_is_still_offered_the_package() -> None:
    """Vision on a chat card does not keep it off a package.

    The package serves the card's text and its runner refuses an image request
    when one arrives. Most signed chat cards have a vision section, so treating
    one as a non-text card left them with no add-on engine at all.
    """
    card = _card("tensorfold").model_copy(
        update={"vision": VisionCardConfig(image_token_id=7, model_type="qwen3_5")}
    )

    shard = _place(
        card,
        NodeResources(
            backends=frozenset({PACKAGE_TAG}),
            engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
        ),
    )

    assert shard.resolved_backend == PACKAGE_TAG
    assert shard.resolved_engine_build == PACKAGE_BUILD


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


def _package_node(overhead_bytes: int) -> NodeResources:
    """A node holding the package, advertising the memory its build declares."""
    return NodeResources(
        backends=frozenset({PACKAGE_TAG}),
        engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
        engine_memory_settings={
            PACKAGE_TAG: engine_package_memory_settings(overhead_bytes)
        },
    )


def _place_sized(resources: NodeResources, *, memory_gb: float) -> ShardMetadata:
    topology = Topology()
    topology.add_node(NODE)
    placements = place_instance(
        place_instance_command(_sized_card((PACKAGE_TAG,))),
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(memory_gb).in_bytes)},
        {NODE: create_node_network()},
        node_resources={NODE: resources},
    )
    instance = next(iter(placements.values()))
    return next(iter(instance.shard_assignments.runner_to_shard.values()))


def test_a_package_placement_is_stamped_with_the_memory_its_build_declares() -> None:
    """Reservations and the worker's load guard read the figure from the shard."""
    overhead = Memory.from_gb(2).in_bytes

    shard = _place_sized(_package_node(overhead), memory_gb=16)

    assert shard.memory_settings == EngineMemorySettings(
        package_overhead_bytes=overhead
    )
    assert estimate_weight_runtime_bytes(
        shard.model_card,
        1.0,
        resolved_backend=shard.resolved_backend,
        memory_settings=shard.memory_settings,
    ) == Memory.from_gb(4) + Memory.from_bytes(overhead)


def test_a_launch_is_refused_when_weights_plus_the_declared_memory_do_not_fit() -> None:
    """Four gigabytes of weights fit eight; with six more declared they do not."""
    _place_sized(_package_node(Memory.from_gb(1).in_bytes), memory_gb=8)

    with pytest.raises(PlacementError, match="node-a needs"):
        _place_sized(_package_node(Memory.from_gb(6).in_bytes), memory_gb=8)


def test_a_small_declared_memory_admits_what_the_built_in_multiplier_refused() -> None:
    """The declared figure replaces the multiplier tuned for built-in engines.

    Four gigabytes of weights need 5.45 under the built-in rule (1.3 times
    the weights plus the floor) and 4.5 with half a gigabyte declared.
    """
    undeclared = NodeResources(
        backends=frozenset({PACKAGE_TAG}),
        engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
    )
    with pytest.raises(PlacementError):
        _place_sized(undeclared, memory_gb=5)

    shard = _place_sized(_package_node(Memory.from_gb(0.5).in_bytes), memory_gb=5)

    assert shard.resolved_backend == PACKAGE_TAG


def test_a_package_window_is_committed_at_load_and_its_lane_names_the_pool() -> None:
    shard = _place_sized(_package_node(Memory.from_gb(1).in_bytes), memory_gb=16)

    assert shard_preallocates_kv_upfront(shard)
    assert backend_offloads_to_vram("tensorfold-rocm")
    assert backend_offloads_to_vram("tensorfold-cuda")
    assert backend_offloads_to_vram("tensorfold-vulkan")
    assert not backend_offloads_to_vram("tensorfold-cpu")
    assert not backend_offloads_to_vram("tensorfold-metal")
    assert not backend_offloads_to_vram("tensorfold")


def test_built_in_engines_keep_their_memory_rule_and_their_report() -> None:
    """No package, no difference: same estimate, same serialized settings."""
    card = _sized_card()
    plain = EngineMemorySettings()

    assert estimate_weight_runtime_bytes(
        card, 1.0, resolved_backend="llama_cpp-vulkan", memory_settings=plain
    ) == estimate_weight_runtime_bytes(card, 1.0, resolved_backend="llama_cpp-vulkan")
    assert "packageOverheadBytes" not in plain.model_dump(by_alias=True)
    assert "package_overhead_bytes" not in plain.model_dump()
    assert EngineMemorySettings.model_validate(plain.model_dump()) == plain
    declared = engine_package_memory_settings(7)
    assert EngineMemorySettings.model_validate(declared.model_dump()) == declared
    assert node_engine_memory_settings(
        ["llama_cpp", "llama_cpp-vulkan", "mlx"], {}, {}
    ) == {"llama_cpp": plain, "mlx": plain}
    assert node_engine_memory_settings(
        ["llama_cpp-vulkan", PACKAGE_TAG], {PACKAGE_TAG: 7}, {}
    ) == {"llama_cpp": plain, PACKAGE_TAG: declared}
