# pyright: reportPrivateUsage=false
"""A model an engine package refuses to load is placed on the next engine."""

from collections.abc import Mapping

import anyio
import pytest

from skulk.master.main import (
    Master,
    engine_package_builds_of,
    engine_refusal_failure_message,
)
from skulk.master.placement import (
    PlacementError,
    add_instance_to_placements,
    engine_builds_unsupporting,
    place_instance,
    replacement_command_for_engine_refused_instance,
    without_engine_builds,
)
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PlacementCardConfig,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import (
    CreateInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    RefuseInstancePlacement,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.engine_packages import (
    NodeEngineLoadResult,
    NodeEnginePackage,
)
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import Instance
from skulk.shared.types.worker.shards import ShardMetadata
from skulk.utils.channels import channel

MODEL = ModelId("test/text")
OTHER_MODEL = ModelId("test/other")
NODE = NodeId("node-a")
OTHER_NODE = NodeId("node-b")
TAG = "tensorfold-rocm"
SECOND_TAG = "otherfold-rocm"
BUILT_IN = "llama_cpp-vulkan"
BUILD = "c" * 64
NEWER_BUILD = "d" * 64
SECOND_BUILD = "e" * 64


def _card(
    model_id: ModelId = MODEL,
    *,
    engines: tuple[str, ...] = ("llama_cpp", "tensorfold"),
    preference: tuple[str, ...] = ("tensorfold",),
) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        source_revision="a" * 40,
        storage_size=Memory.from_gb(4),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset(engines), backend_preference=preference
        ),
    )


def _refused(model_id: ModelId, build: str) -> NodeEngineLoadResult:
    return NodeEngineLoadResult(
        build=build,
        outcome="refused",
        code="unsupported_model",
        model_id=str(model_id),
        at_unix_ms=1000,
    )


def _resources(
    *,
    build: str = BUILD,
    built_in: bool = True,
    installed: bool = True,
    unsupported: tuple[tuple[ModelId, str], ...] = (),
    more: Mapping[str, str] | None = None,
) -> NodeResources:
    """A node with one TensorFold build, held or installable, and maybe a built-in engine."""
    packages = {TAG: build, **(more or {})}
    return NodeResources(
        backends=frozenset(
            {*packages, *(("llama_cpp", BUILT_IN) if built_in else ())}
        ),
        on_demand_backends=frozenset() if installed else frozenset(packages),
        engine_builds=packages if installed else {},
        installable_engine_builds={} if installed else packages,
        engine_packages=(
            NodeEnginePackage(
                engine_id="foxlight.tensorfold",
                lane="rocm",
                backend_tag=TAG,
                status="active" if installed else "offered",
                build=build,
                installed_builds=(build,) if installed else (),
                unsupported_models=tuple(
                    _refused(model_id, refused_build)
                    for model_id, refused_build in unsupported
                ),
            ),
        ),
    )


def _placed(
    card: ModelCard, node_resources: Mapping[NodeId, NodeResources]
) -> tuple[NodeId, ShardMetadata]:
    topology = Topology()
    for node_id in node_resources:
        topology.add_node(node_id)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {node_id: create_node_memory(Memory.from_gb(64).in_bytes) for node_id in node_resources},
        {node_id: create_node_network() for node_id in node_resources},
        node_resources=node_resources,
    )
    instance = next(iter(placements.values()))
    (node_id,) = instance.shard_assignments.node_to_runner
    (shard,) = instance.shard_assignments.runner_to_shard.values()
    return node_id, shard


def test_the_preferred_package_serves_a_model_nothing_refused() -> None:
    _, shard = _placed(_card(), {NODE: _resources()})

    assert (shard.resolved_backend, shard.resolved_engine_build) == (TAG, BUILD)


def test_a_build_that_refused_the_model_is_passed_over_for_the_next_engine() -> None:
    resources = {NODE: _resources(unsupported=((MODEL, BUILD),))}

    _, shard = _placed(_card(), resources)
    assert shard.resolved_backend == BUILT_IN

    # Another model is still served by the package, on the same node.
    _, other = _placed(_card(OTHER_MODEL), resources)
    assert (other.resolved_backend, other.resolved_engine_build) == (TAG, BUILD)


def test_one_nodes_report_counts_for_every_node_with_the_same_build() -> None:
    """A build is the same bytes everywhere, so its answer is too."""
    resources = {
        NODE: _resources(unsupported=((MODEL, BUILD),), built_in=False),
        OTHER_NODE: _resources(),
    }

    node_id, shard = _placed(_card(), resources)

    assert (node_id, shard.resolved_backend) == (OTHER_NODE, BUILT_IN)


def test_a_different_build_of_the_same_engine_is_still_tried() -> None:
    """An update, or another accelerator's build, has not answered yet."""
    resources = {
        NODE: _resources(unsupported=((MODEL, BUILD),), built_in=False),
        OTHER_NODE: _resources(build=NEWER_BUILD),
    }

    node_id, shard = _placed(_card(), resources)

    assert (node_id, shard.resolved_backend, shard.resolved_engine_build) == (
        OTHER_NODE,
        TAG,
        NEWER_BUILD,
    )


def test_a_build_a_node_would_install_is_not_installed_for_a_model_it_refuses() -> None:
    resources = {
        NODE: _resources(unsupported=((MODEL, BUILD),), built_in=False),
        OTHER_NODE: _resources(installed=False),
    }

    _, shard = _placed(_card(), resources)

    assert shard.resolved_backend == BUILT_IN


def test_a_model_with_no_other_engine_is_refused_before_anything_is_fetched() -> None:
    """And the refusal names the engine's answer, not missing hardware."""
    with pytest.raises(PlacementError, match="refused to load this model as unsupported"):
        _placed(
            _card(engines=("tensorfold",)),
            {NODE: _resources(unsupported=((MODEL, BUILD),), built_in=False)},
        )


def test_a_cluster_where_nothing_was_refused_keeps_the_resources_it_passed() -> None:
    resources = {NODE: _resources(), OTHER_NODE: _resources(build=NEWER_BUILD)}

    assert engine_builds_unsupporting(resources, MODEL) == frozenset()
    assert without_engine_builds(resources, frozenset()) is resources
    # A build no node holds or offers changes nothing either.
    assert without_engine_builds(resources, frozenset({SECOND_BUILD})) is resources


def test_taking_a_build_off_offer_leaves_built_in_engines_and_other_builds() -> None:
    resources = {
        NODE: _resources(more={SECOND_TAG: SECOND_BUILD}),
        OTHER_NODE: _resources(build=NEWER_BUILD, installed=False),
    }

    offered = without_engine_builds(resources, frozenset({BUILD, NEWER_BUILD}))

    assert offered[NODE].backends == frozenset({"llama_cpp", BUILT_IN, SECOND_TAG})
    assert offered[OTHER_NODE].backends == frozenset({"llama_cpp", BUILT_IN})
    assert offered[OTHER_NODE].on_demand_backends == frozenset()
    # The caller's mapping is not changed.
    assert TAG in resources[NODE].backends


def _instance(card: ModelCard, resources: NodeResources) -> Instance:
    topology = Topology()
    topology.add_node(NODE)
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(64).in_bytes)},
        {NODE: create_node_network()},
        node_resources={NODE: resources},
    )
    return next(iter(placements.values()))


def test_an_exact_placement_that_names_no_engine_passes_over_the_build_too() -> None:
    card = _card()
    planned = _instance(card, _resources())
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

    def create(resources: NodeResources) -> str | None:
        placements = add_instance_to_placements(
            CreateInstance(instance=unstamped),
            topology,
            {},
            {NODE: create_node_memory(Memory.from_gb(64).in_bytes)},
            node_vram={NODE: Memory.from_gb(64)},
            node_resources={NODE: resources},
        )
        (shard,) = placements[unstamped.instance_id].shard_assignments.runner_to_shard.values()
        return shard.resolved_backend

    assert create(_resources()) == TAG
    assert create(_resources(unsupported=((MODEL, BUILD),))) == BUILT_IN


def test_the_replacement_keeps_the_width_and_the_launchs_own_choices() -> None:
    refused = _instance(_card(), _resources()).model_copy(
        update={"excluded_nodes": [OTHER_NODE], "engine_order": ["tensorfold"]}
    )

    command = replacement_command_for_engine_refused_instance(refused)

    assert command.min_nodes == 1
    assert command.excluded_nodes == [OTHER_NODE]
    assert command.engine_order == ["tensorfold"]
    assert engine_package_builds_of(refused) == frozenset({BUILD})


def test_the_record_says_what_skulk_did_and_keeps_only_the_code() -> None:
    moved = engine_refusal_failure_message(MODEL, "unsupported_model", None)
    stuck = engine_refusal_failure_message(MODEL, "unsupported_model", "no node fits")

    assert "placed it again on the next engine" in moved
    assert "no other engine could take it: no node fits" in stuck
    assert all("unsupported_model" in message for message in (moved, stuck))


async def _refuse(
    resources: Mapping[NodeId, NodeResources],
    instance: Instance,
    *,
    code: str = "unsupported_model",
    deliveries: int = 1,
    lineage: frozenset[str] = frozenset(),
) -> Master:
    """Run a master holding ``instance`` and deliver one engine refusal for it."""
    keypair = get_node_id_keypair()
    master_node = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=master_node, election_clock=0)
    topology = Topology()
    telemetry = TelemetryView()
    for node_id, node_resources in resources.items():
        topology.add_node(node_id)
        telemetry.node_memory[node_id] = create_node_memory(Memory.from_gb(64).in_bytes)
        telemetry.node_resources[node_id] = node_resources
    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()
    ev_send, ev_recv = channel[Event]()

    async def mock_event_router() -> None:
        index = 0
        origin = SystemId()
        with ev_recv as master_events:
            async for event in master_events:
                await local_event_sender.send(
                    LocalForwarderEvent(
                        origin=origin, origin_idx=index, session=session_id, event=event
                    )
                )
                index += 1

    master = Master(
        master_node,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=fcds,
        initial_state=State(
            topology=topology,
            instances={instance.instance_id: instance},
            node_network={node_id: create_node_network() for node_id in resources},
        ),
        telemetry_view=telemetry,
    )
    if lineage:
        master._engine_refusal_lineage[instance.instance_id] = lineage

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        tg.start_soon(mock_event_router)
        with anyio.fail_after(5):
            while instance.instance_id not in master.state.instances:
                await anyio.sleep(0.005)
        for _ in range(deliveries):
            await command_sender.send(
                ForwarderCommand(
                    origin=SystemId("Worker"),
                    command=RefuseInstancePlacement(
                        instance_id=instance.instance_id,
                        node_id=NODE,
                        reason="the engine package refused to load the model",
                        engine_refusal_code=code,
                    ),
                )
            )
        with anyio.fail_after(5):
            while (
                instance.instance_id in master.state.instances
                or not master.state.instance_failures
            ):
                await anyio.sleep(0.005)
        # Let any further delivery be handled before the master is read.
        await anyio.sleep(0.05)
        global_event_receiver.collect()
        tg.cancel_scope.cancel()
    return master


def _only_shard(master: Master) -> tuple[Instance, NodeId, ShardMetadata]:
    (instance,) = master.state.instances.values()
    (node_id,) = instance.shard_assignments.node_to_runner
    (shard,) = instance.shard_assignments.runner_to_shard.values()
    return instance, node_id, shard


@pytest.mark.parametrize("code", ["unsupported_model", "load_refused"])
async def test_a_refused_load_moves_to_the_next_engine_on_the_same_node(
    code: str,
) -> None:
    """The node has not reported the refusal yet; the command alone is enough."""
    resources = {NODE: _resources(), OTHER_NODE: _resources(build=NEWER_BUILD)}
    refused = _instance(_card(), resources[NODE])

    master = await _refuse(resources, refused, code=code)

    replacement, node_id, shard = _only_shard(master)
    # The other node's newer build ranks first in the model's order, but the
    # model's files are on this node, so its next engine is used.
    assert (node_id, shard.resolved_backend) == (NODE, BUILT_IN)
    (failure,) = master.state.instance_failures
    assert failure.instance_id == refused.instance_id
    assert failure.error_code == "placement_failed"
    assert f"({code})" in failure.error_message
    assert "placed it again on the next engine" in failure.error_message
    assert master._engine_refusal_lineage == {
        replacement.instance_id: frozenset({BUILD})
    }


async def test_a_refusal_delivered_twice_is_handled_once() -> None:
    resources = {NODE: _resources()}

    master = await _refuse(
        resources, _instance(_card(), resources[NODE]), deliveries=3
    )

    assert len(master.state.instances) == 1
    assert len(master.state.instance_failures) == 1


async def test_the_rest_of_the_cluster_is_searched_when_the_node_has_no_other_engine() -> None:
    resources = {
        NODE: _resources(built_in=False),
        OTHER_NODE: _resources(build=NEWER_BUILD),
    }

    master = await _refuse(resources, _instance(_card(), resources[NODE]))

    _, node_id, shard = _only_shard(master)
    assert (node_id, shard.resolved_backend, shard.resolved_engine_build) == (
        OTHER_NODE,
        TAG,
        NEWER_BUILD,
    )


async def test_a_model_no_other_engine_can_take_is_torn_down_with_the_reason() -> None:
    resources = {NODE: _resources(built_in=False)}
    refused = _instance(_card(engines=("tensorfold",)), resources[NODE])

    master = await _refuse(resources, refused)

    assert master.state.instances == {}
    (failure,) = master.state.instance_failures
    assert "no other engine could take it" in failure.error_message
    assert master._engine_refusal_lineage == {}


async def test_builds_that_refused_earlier_in_the_chain_stay_off_offer() -> None:
    """Two packages that both refuse must not hand the model back and forth."""
    resources = {NODE: _resources(more={SECOND_TAG: SECOND_BUILD})}
    card = _card(
        engines=("llama_cpp", "tensorfold", "otherfold"),
        preference=("otherfold", "tensorfold"),
    )
    # The first package already refused and the model moved to the second.
    refused = _instance(card, resources[NODE])
    (shard,) = refused.shard_assignments.runner_to_shard.values()
    assert shard.resolved_backend == SECOND_TAG

    master = await _refuse(
        resources, refused, code="load_refused", lineage=frozenset({BUILD})
    )

    replacement, _, shard = _only_shard(master)
    assert shard.resolved_backend == BUILT_IN
    assert master._engine_refusal_lineage == {
        replacement.instance_id: frozenset({BUILD, SECOND_BUILD})
    }
