"""The master's side of an engine package change: ordering the request and reading the answer."""

import time

import pytest
from pydantic import ValidationError

from skulk.master.main import Master
from skulk.shared.apply import event_apply
from skulk.shared.topology import Topology
from skulk.shared.types.commands import (
    ChangeEnginePackage,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from skulk.shared.types.common import CommandId, NodeId, SessionId
from skulk.shared.types.engine_packages import NodeEnginePackage
from skulk.shared.types.events import (
    EnginePackageChangeCompleted,
    EnginePackageChangeRequested,
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    is_persistable_control_event,
)
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import channel

NODE = NodeId("node-a")
API_NODE = NodeId("api-node")
BUILD = "c" * 64
HELD = NodeEnginePackage(
    engine_id="foxlight.tensorfold",
    lane="rocm",
    backend_tag="tensorfold-rocm",
    status="active",
    build=BUILD,
    installed_builds=(BUILD,),
)


def _master(*live: NodeId) -> Master:
    topology = Topology()
    for node in live:
        topology.add_node(node)
    master = Master(
        NodeId(),
        SessionId(master_node_id=NodeId(), election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )
    # A new session begins with no members; these have since been seen.
    master.state = master.state.model_copy(update={"topology": topology})
    return master


def _command(**changes: object) -> ChangeEnginePackage:
    return ChangeEnginePackage(
        target_node=NODE,
        owner_node=API_NODE,
        engine_id="foxlight.tensorfold",
        lane="rocm",
        action="activate",
        build=BUILD,
    ).model_copy(update=changes)


def _order(master: Master, command: ChangeEnginePackage) -> Event:
    return master._order_engine_package_change(command)  # pyright: ignore[reportPrivateUsage]


def _completed(resources: NodeResources | None, *, node: NodeId = NODE) -> EnginePackageChangeCompleted:
    return EnginePackageChangeCompleted(
        request_id=CommandId("change"),
        target_node=node,
        owner_node=API_NODE,
        success=resources is not None,
        code=None if resources is not None else "generation_missing",
        resources=resources,
    )


def test_a_change_for_a_live_node_is_ordered_with_a_deadline() -> None:
    master = _master(NODE)
    command = _command()
    before = time.time()

    event = _order(master, command)

    assert isinstance(event, EnginePackageChangeRequested)
    assert event.request_id == command.command_id
    assert (event.target_node, event.owner_node) == (NODE, API_NODE)
    assert (event.engine_id, event.lane, event.action, event.build) == (
        "foxlight.tensorfold",
        "rocm",
        "activate",
        BUILD,
    )
    assert before < event.expires_at <= time.time() + 120


def test_a_change_for_a_node_that_left_is_answered_at_once() -> None:
    """The API that asked must not wait for a worker that will never read it."""
    master = _master()
    command = _command()

    event = _order(master, command)

    assert isinstance(event, EnginePackageChangeCompleted)
    assert event.request_id == command.command_id
    assert (event.success, event.code, event.resources) == (False, "node_unavailable", None)


def test_placement_sees_the_changed_node_before_the_answer_is_broadcast() -> None:
    master = _master(NODE)
    fresh = NodeResources(engine_packages=(HELD,))

    master._record_engine_package_change(_completed(fresh))  # pyright: ignore[reportPrivateUsage]

    assert master._telemetry_view.node_resources[NODE] is fresh  # pyright: ignore[reportPrivateUsage]


def test_a_refusal_or_an_answer_from_a_node_that_left_changes_nothing() -> None:
    master = _master(NODE)
    gone = NodeId("node-gone")

    master._record_engine_package_change(_completed(None))  # pyright: ignore[reportPrivateUsage]
    master._record_engine_package_change(  # pyright: ignore[reportPrivateUsage]
        _completed(NodeResources(engine_packages=(HELD,)), node=gone)
    )

    assert master._telemetry_view.node_resources == {}  # pyright: ignore[reportPrivateUsage]


def test_both_events_are_ordered_control_events_that_leave_state_alone() -> None:
    master = _master(NODE)
    requested = _order(master, _command())
    completed = _completed(NodeResources(engine_packages=(HELD,)))
    state = State()

    for event in (requested, completed):
        assert is_persistable_control_event(event)
        assert event_apply(event, state) == state


@pytest.mark.parametrize(
    "changes",
    [
        {"success": True, "code": None, "resources": None},
        {"success": True, "code": "generation_missing", "resources": NodeResources()},
        {"success": False, "code": None, "resources": None},
        {"success": False, "code": "generation_missing", "resources": NodeResources()},
        {"success": False, "code": "Cannot read /private/x", "resources": None},
    ],
)
def test_an_answer_is_a_change_with_facts_or_a_refusal_with_a_code(
    changes: dict[str, object],
) -> None:
    """A refusal's code is a short word, so a node's own text never reaches the log."""
    with pytest.raises(ValidationError):
        EnginePackageChangeCompleted.model_validate(
            {
                "request_id": CommandId("change"),
                "target_node": NODE,
                "owner_node": API_NODE,
                **changes,
            }
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"engine_id": "../foxlight.tensorfold"},
        {"engine_id": "foxlight"},
        {"lane": "gpu"},
        {"action": "install"},
        {"build": "abc"},
        {"build": "../" + "c" * 61},
    ],
)
def test_a_malformed_change_is_refused_where_the_command_is_built(
    changes: dict[str, object],
) -> None:
    fields = _command().model_dump() | changes
    with pytest.raises(ValidationError):
        ChangeEnginePackage.model_validate(fields)
