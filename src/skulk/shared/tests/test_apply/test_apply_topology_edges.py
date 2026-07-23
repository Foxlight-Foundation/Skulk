"""Edge apply reaps never-member topology nodes when their last edge goes (#671)."""

from datetime import datetime, timezone

from skulk.shared.apply import (
    apply_topology_edge_created,
    apply_topology_edge_deleted,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import TopologyEdgeCreated, TopologyEdgeDeleted
from skulk.shared.types.state import State
from skulk.shared.types.topology import Connection, Multiaddr, SocketConnection

MEMBER = NodeId("member-node")
PHANTOM = NodeId("phantom-node")
OTHER = NodeId("other-member")


def _session_edge() -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/203.0.113.9/tcp/41234"),
        session=True,
    )


def _connection(source: NodeId, sink: NodeId) -> Connection:
    return Connection(source=source, sink=sink, edge=_session_edge())


def _member_state() -> State:
    return State(last_seen={MEMBER: datetime.now(timezone.utc)})


def test_last_edge_deletion_reaps_never_member_node() -> None:
    """A peer known only through a session edge disappears with the edge.

    The edge minted the topology node before the peer ever published
    NodeGatheredInfo; without a last_seen stamp the timeout loop can never
    reap it, so the deletion of its last edge must, or it lingers as a
    floating phantom member.
    """
    state = _member_state()
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(MEMBER, PHANTOM)), state
    )
    assert state.topology.contains_node(PHANTOM)

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(MEMBER, PHANTOM)), state
    )
    assert not state.topology.contains_node(PHANTOM)
    # The member endpoint carries last_seen and is never touched here.
    assert state.topology.contains_node(MEMBER)


def test_member_node_survives_losing_its_last_edge() -> None:
    """Nodes with membership state are NodeTimedOut's to reap, never this path."""
    state = State(
        last_seen={
            MEMBER: datetime.now(timezone.utc),
            OTHER: datetime.now(timezone.utc),
        }
    )
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(MEMBER, OTHER)), state
    )
    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(MEMBER, OTHER)), state
    )
    assert state.topology.contains_node(OTHER)
    assert state.topology.contains_node(MEMBER)


def test_never_member_with_remaining_edges_survives() -> None:
    """Only a fully isolated never-member is reaped.

    While another observer's edge still references the peer (its own
    disconnect has not arrived yet), the node stays; the LAST deletion
    removes it.
    """
    state = State(
        last_seen={
            MEMBER: datetime.now(timezone.utc),
            OTHER: datetime.now(timezone.utc),
        }
    )
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(MEMBER, PHANTOM)), state
    )
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(OTHER, PHANTOM)), state
    )

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(MEMBER, PHANTOM)), state
    )
    assert state.topology.contains_node(PHANTOM)

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(OTHER, PHANTOM)), state
    )
    assert not state.topology.contains_node(PHANTOM)
