from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.state_sync import StateSnapshot
from exo.shared.types.topology import Connection, SocketConnection


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
