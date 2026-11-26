"""Placement under a person's engine order: it outranks the card's default and changes nothing when unset."""

import ast
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

import skulk
from skulk.master.placement import (
    add_instance_to_placements,
    fallback_command_for_refused_instance,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
)
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PlacementCardConfig,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance, PlaceInstance
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.instances import Instance
from skulk.shared.types.worker.shards import ShardMetadata

MODEL = ModelId("test/text")
NODE = NodeId("node-a")
PACKAGE_TAG = "tensorfold-rocm"
PACKAGE_BUILD = "c" * 64
BOTH_ENGINES = NodeResources(
    backends=frozenset({"llama_cpp", "llama_cpp-vulkan", "llama_cpp-rocm", PACKAGE_TAG}),
    engine_builds={PACKAGE_TAG: PACKAGE_BUILD},
)


def _card(preference: tuple[str, ...] = (), *, gigabytes: float = 1) -> ModelCard:
    return ModelCard(
        model_id=MODEL,
        source_revision="a" * 40,
        storage_size=Memory.from_gb(gigabytes),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"llama_cpp", "tensorfold"}),
            backend_preference=preference,
        ),
    )


def _place(
    card: ModelCard,
    resources: NodeResources = BOTH_ENGINES,
    *,
    orders: dict[ModelId, tuple[str, ...]] | None = None,
) -> Instance:
    topology = Topology()
    topology.add_node(NODE)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(32).in_bytes)},
        {NODE: create_node_network()},
        node_resources={NODE: resources},
        model_engine_orders=orders,
    )
    return next(iter(placements.values()))


def _shard(instance: Instance) -> ShardMetadata:
    return next(iter(instance.shard_assignments.runner_to_shard.values()))


def _backend(
    card: ModelCard,
    resources: NodeResources = BOTH_ENGINES,
    *,
    orders: dict[ModelId, tuple[str, ...]] | None = None,
) -> str | None:
    return _shard(_place(card, resources, orders=orders)).resolved_backend


@pytest.mark.parametrize("orders", [None, {}, {ModelId("test/other"): (PACKAGE_TAG,)}])
@pytest.mark.parametrize("preference", [(), ("llama_cpp-rocm",), (PACKAGE_TAG,)])
def test_a_model_with_no_order_of_its_own_is_placed_as_before(
    orders: dict[ModelId, tuple[str, ...]] | None, preference: tuple[str, ...]
) -> None:
    """No order, an empty set, or another model's order: the same shard either way."""
    card = _card(preference)

    with_orders = _shard(_place(card, orders=orders))
    without = _shard(_place(card))

    assert with_orders == without


def test_the_persons_order_outranks_the_cards_default() -> None:
    card = _card(("llama_cpp-vulkan",))

    assert _backend(card) == "llama_cpp-vulkan"
    assert _backend(card, orders={MODEL: ("llama_cpp-rocm",)}) == "llama_cpp-rocm"


def test_the_persons_order_may_choose_an_installed_package() -> None:
    card = _card()
    assert _backend(card) != PACKAGE_TAG

    shard = _shard(_place(card, orders={MODEL: ("tensorfold",)}))

    assert shard.resolved_backend == PACKAGE_TAG
    assert shard.resolved_engine_build == PACKAGE_BUILD


def test_the_persons_order_may_keep_a_model_off_a_package_the_card_prefers() -> None:
    card = _card((PACKAGE_TAG,))
    assert _backend(card) == PACKAGE_TAG

    assert _backend(card, orders={MODEL: ("llama_cpp-vulkan",)}) == "llama_cpp-vulkan"


def test_where_the_persons_order_names_nothing_the_node_has_the_card_decides() -> None:
    """An order never blocks a launch: the next level answers."""
    card = _card(("llama_cpp-rocm",))

    assert _backend(card, orders={MODEL: ("vllm-cuda", "mlx")}) == "llama_cpp-rocm"


def test_where_no_list_applies_the_built_in_fallback_decides() -> None:
    card = _card()

    assert _backend(card, orders={MODEL: ("vllm-cuda",)}) == _backend(card)


def test_the_card_on_the_shard_is_never_rewritten() -> None:
    """The order ranks; the card the workers verify stays the card that was launched."""
    card = _card(("llama_cpp-vulkan",))

    shard = _shard(_place(card, orders={MODEL: ("llama_cpp-rocm",)}))

    assert shard.model_card == card


BUILT_IN_NODE = NodeId("node-built-in")
PACKAGE_NODE = NodeId("node-package")


def _chosen_node(card: ModelCard, orders: dict[ModelId, tuple[str, ...]] | None) -> NodeId:
    """Place on one built-in node with more free memory and one package-only node."""
    topology = Topology()
    topology.add_node(BUILT_IN_NODE)
    topology.add_node(PACKAGE_NODE)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {
            BUILT_IN_NODE: create_node_memory(Memory.from_gb(64).in_bytes),
            PACKAGE_NODE: create_node_memory(Memory.from_gb(16).in_bytes),
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
        model_engine_orders=orders,
    )
    instance = next(iter(placements.values()))
    return next(iter(instance.shard_assignments.node_to_runner))


def test_the_persons_order_also_chooses_the_node() -> None:
    card = _card(("llama_cpp-vulkan",), gigabytes=4)

    assert _chosen_node(card, None) == BUILT_IN_NODE
    assert _chosen_node(card, {MODEL: ("tensorfold",)}) == PACKAGE_NODE
    assert _chosen_node(card, {MODEL: ("llama_cpp",)}) == BUILT_IN_NODE


def test_an_exact_placement_that_names_no_engine_follows_the_persons_order() -> None:
    card = _card(("llama_cpp-vulkan",))
    planned = _place(card)
    unstamped = planned.model_copy(
        update={
            "shard_assignments": planned.shard_assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner_id: shard.model_copy(
                            update={
                                "resolved_backend": None,
                                "resolved_engine_build": None,
                            }
                        )
                        for runner_id, shard in planned.shard_assignments.runner_to_shard.items()
                    }
                }
            )
        }
    )
    topology = Topology()
    topology.add_node(NODE)

    def create(orders: dict[ModelId, tuple[str, ...]] | None) -> str | None:
        placements = add_instance_to_placements(
            CreateInstance(instance=unstamped),
            topology,
            {},
            {NODE: create_node_memory(Memory.from_gb(32).in_bytes)},
            node_vram={NODE: Memory.from_gb(32)},
            node_resources={NODE: BOTH_ENGINES},
            model_engine_orders=orders,
        )
        return _shard(placements[unstamped.instance_id]).resolved_backend

    assert create(None) == "llama_cpp-vulkan"
    assert create({MODEL: ("llama_cpp-rocm",)}) == "llama_cpp-rocm"


def _launch(
    card: ModelCard,
    launch_order: list[str] | None,
    orders: dict[ModelId, tuple[str, ...]] | None = None,
) -> Instance:
    topology = Topology()
    topology.add_node(NODE)
    command = place_instance_command(card)
    command.engine_order = launch_order
    placements = place_instance(
        command,
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(32).in_bytes)},
        {NODE: create_node_network()},
        node_resources={NODE: BOTH_ENGINES},
        model_engine_orders=orders,
    )
    return next(iter(placements.values()))


def test_an_order_given_for_one_launch_outranks_every_standing_order() -> None:
    card = _card(("llama_cpp-vulkan",))
    standing = {MODEL: ("llama_cpp-rocm",)}

    trial = _launch(card, ["tensorfold"], standing)

    assert _shard(trial).resolved_backend == PACKAGE_TAG
    assert trial.engine_order == ["tensorfold"]
    # The standing order is what an ordinary launch still follows.
    assert _shard(_launch(card, None, standing)).resolved_backend == "llama_cpp-rocm"


def test_an_ordinary_launch_records_no_launch_order() -> None:
    assert _launch(_card(), None).engine_order == []


def _fallback(instance: Instance) -> PlaceInstance:
    return fallback_command_for_refused_instance(instance, NodeId("x"))


def _after_download_failure(instance: Instance) -> PlaceInstance:
    return replacement_command_for_download_failed_instance(instance, {NodeId("x")})


@pytest.mark.parametrize(
    "build",
    [replacement_command_for_refused_instance, _fallback, _after_download_failure],
)
def test_a_repair_keeps_the_engines_a_launch_asked_for(
    build: Callable[[Instance], PlaceInstance],
) -> None:
    trial = _launch(_card(), ["tensorfold"])
    ordinary = _launch(_card(), None)

    assert build(trial).engine_order == ["tensorfold"]
    assert build(ordinary).engine_order is None


def test_an_exact_placement_carrying_a_launch_order_follows_it() -> None:
    card = _card(("llama_cpp-vulkan",))
    planned = _place(card)
    unstamped = planned.model_copy(
        update={
            "engine_order": ["llama_cpp-rocm"],
            "shard_assignments": planned.shard_assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner_id: shard.model_copy(
                            update={
                                "resolved_backend": None,
                                "resolved_engine_build": None,
                            }
                        )
                        for runner_id, shard in planned.shard_assignments.runner_to_shard.items()
                    }
                }
            ),
        }
    )
    topology = Topology()
    topology.add_node(NODE)

    placements = add_instance_to_placements(
        CreateInstance(instance=unstamped),
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(32).in_bytes)},
        node_vram={NODE: Memory.from_gb(32)},
        node_resources={NODE: BOTH_ENGINES},
        model_engine_orders={MODEL: ("tensorfold",)},
    )

    assert _shard(placements[unstamped.instance_id]).resolved_backend == "llama_cpp-rocm"


def test_an_instance_refuses_a_malformed_launch_order() -> None:
    placed = _place(_card())

    with pytest.raises(ValidationError):
        type(placed).model_validate(
            {**placed.model_dump(), "engine_order": ["Not A Tag"]}
        )


_PLACEMENT_ENTRY_POINTS = frozenset(
    {"place_instance", "get_instance_placements", "add_instance_to_placements"}
)


def _source_files() -> list[Path]:
    root = Path(skulk.__file__).parent
    return [
        path
        for path in root.rglob("*.py")
        if "tests" not in path.parts and path.name != "conftest.py"
    ]


def test_every_placement_call_passes_the_engine_orders() -> None:
    """A caller that forgot them would preview or repair on a different engine."""
    missing: list[str] = []
    seen = 0
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            name = (
                called.id
                if isinstance(called, ast.Name)
                else called.attr
                if isinstance(called, ast.Attribute)
                else None
            )
            if name not in _PLACEMENT_ENTRY_POINTS:
                continue
            seen += 1
            if all(keyword.arg != "model_engine_orders" for keyword in node.keywords):
                missing.append(f"{path.name}:{node.lineno}")

    assert seen >= 12
    assert missing == []


def test_placement_reads_the_cards_preference_in_one_place() -> None:
    """Every other read would rank engines without the person's order."""
    source = (Path(skulk.__file__).parent / "master" / "placement.py").read_text()

    assert source.count("placement.backend_preference") == 1
