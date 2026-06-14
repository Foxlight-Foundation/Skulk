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


def test_state_ignores_legacy_node_resources_from_old_snapshot() -> None:
    """A pre-#279 snapshot carrying ``nodeResources`` must still hydrate.

    ``node_resources`` moved to the telemetry plane (#279), but ``State`` keeps
    ``extra="forbid"``. Without the legacy-key strip, an upgraded follower
    rejects an old master's state-sync snapshot, falls back to replay from
    index 0, and loses instances/topology that lived only in an already-
    compacted log prefix (the #273 outage class). This pins that an inbound
    payload containing the removed key validates instead of raising.
    """
    node_id = NodeId("node-a")
    state = State(last_event_applied_idx=7)
    state.topology.add_node(node_id)

    # Simulate the wire payload an older binary would emit: the current JSON
    # plus the removed camelCase field.
    payload = state.model_dump(mode="json", by_alias=True)
    payload["nodeResources"] = {
        str(node_id): {"backends": ["mlx"], "participation": "management"}
    }

    restored = State.model_validate(payload)

    assert restored.last_event_applied_idx == 7
    assert restored.topology.to_snapshot().nodes == state.topology.to_snapshot().nodes
    # The legacy field is dropped, not surfaced anywhere on the model.
    assert not hasattr(restored, "node_resources")


def test_state_still_forbids_genuinely_unknown_fields() -> None:
    """The legacy strip must not weaken the forbid-extra guard otherwise.

    Only the known-removed ``node_resources`` key is tolerated; a truly
    unknown field (e.g. one a NEWER binary added) must still raise so a stale
    binary fails loudly instead of silently dropping state it can't model.
    """
    import pytest
    from pydantic import ValidationError

    payload = State().model_dump(mode="json", by_alias=True)
    payload["someFutureField"] = 123

    with pytest.raises(ValidationError):
        State.model_validate(payload)
