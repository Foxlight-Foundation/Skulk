from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSnapshot
from skulk.shared.types.topology import Connection, SocketConnection


def test_state_serialization_roundtrip() -> None:
    """Verify that State → JSON → State round-trip preserves topology."""

    # --- build a simple state ------------------------------------------------
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    connection = Connection(
        source=node_a,
        sink=node_b,
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/10001"),
        ),
    )

    state = State()
    state.topology.add_connection(connection)

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert (
        state.topology.to_snapshot().nodes
        == restored_state.topology.to_snapshot().nodes
    )
    assert set(state.topology.to_snapshot().connections) == set(
        restored_state.topology.to_snapshot().connections
    )
    assert restored_state.model_dump_json() == json_repr


def test_state_snapshot_serialization_roundtrip() -> None:
    node_id = NodeId("node-a")
    state = State(last_event_applied_idx=3)
    state.topology.add_node(node_id)

    snapshot = StateSnapshot(
        session_id=SessionId(master_node_id=node_id, election_clock=1),
        last_event_applied_idx=3,
        state=state,
    )

    restored = StateSnapshot.model_validate_json(snapshot.model_dump_json())

    assert restored.session_id == snapshot.session_id
    assert restored.last_event_applied_idx == snapshot.last_event_applied_idx
    assert restored.state.last_event_applied_idx == snapshot.state.last_event_applied_idx
    assert restored.state.topology.to_snapshot() == snapshot.state.topology.to_snapshot()


def test_state_sync_message_round_trips_with_last_seen() -> None:
    """State-sync snapshot must survive the wire codec WITH last_seen entries.

    Regression for the bug that broke follower bootstrap: a model_validator on
    State (added for cross-version key-stripping) forced strict PYTHON-mode
    validation, where ISO datetime *strings* — last_seen, serialized over the
    wire — were rejected. Followers then dropped every state-sync snapshot
    ("schema-incompatible") and livelocked requesting the event log from 0.
    The earlier snapshot test missed it because its fixture had no last_seen.
    """
    from datetime import datetime, timezone

    from skulk.routing.topics import STATE_SYNC_MESSAGES
    from skulk.shared.topology import Topology
    from skulk.shared.types.common import SystemId
    from skulk.shared.types.state_sync import StateSyncMessage

    topo = Topology()
    nodes = [NodeId(), NodeId(), NodeId()]
    for n in nodes:
        topo.add_node(n)
    state = State(
        topology=topo,
        last_seen={n: datetime.now(tz=timezone.utc) for n in nodes},
        last_event_applied_idx=184,
    )
    session = SessionId(master_node_id=nodes[0], election_clock=1)
    msg = StateSyncMessage(
        kind="response",
        requester=SystemId(),
        session_id=session,
        snapshot=StateSnapshot(
            session_id=session, last_event_applied_idx=184, state=state
        ),
    )
    # The exact path that runs on the wire (TypedTopic uses model_validate_json).
    restored = STATE_SYNC_MESSAGES.deserialize(STATE_SYNC_MESSAGES.serialize(msg))
    assert restored.snapshot is not None
    assert len(restored.snapshot.state.last_seen) == 3
    assert restored.snapshot.state.topology.to_snapshot() == topo.to_snapshot()


def test_state_forbids_genuinely_unknown_fields() -> None:
    """A truly unknown field must still raise (extra='forbid'), so a stale
    binary fails loudly instead of silently dropping state it can't model."""
    import pytest
    from pydantic import ValidationError

    payload = State().model_dump(mode="json", by_alias=True)
    payload["someFutureField"] = 123

    with pytest.raises(ValidationError):
        State.model_validate(payload)
