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


def test_reap_ignores_the_phantom_dangling_own_edges() -> None:
    """A dead peer's own emitted edges cannot pin it in the graph.

    A crash-looping peer can publish its own session edge (peer to member)
    before its first NodeGatheredInfo and then die; nobody remains to emit
    that edge's deletion. Reaping keys on IN-edges (what live observers
    maintain toward the peer), so the member's own deletion still removes
    the phantom, dangling out-edge and all (PR #674 review).
    """
    state = _member_state()
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(MEMBER, PHANTOM)), state
    )
    # The phantom's own emission, indexed before it died.
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(PHANTOM, MEMBER)), state
    )

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(MEMBER, PHANTOM)), state
    )
    assert not state.topology.contains_node(PHANTOM)
    assert state.topology.contains_node(MEMBER)
    # The dangling reverse edge went with the node.
    assert not list(state.topology.get_all_connections_between(PHANTOM, MEMBER))


def test_reap_cascades_through_the_phantom_chain() -> None:
    """Reaping a phantom frees nodes its dangling edges were pinning.

    A phantom's own emitted edge can be the only in-edge of ANOTHER
    never-member (a peer that also died pre-publish); clearing it with the
    first removal must cascade, even though that second node is not an
    endpoint of the deletion event at all (PR #674 review).
    """
    second = NodeId("phantom-two")
    state = _member_state()
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(MEMBER, PHANTOM)), state
    )
    # The phantom's own emission toward another dead peer, indexed before
    # both died; it is the only edge referencing phantom-two.
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(PHANTOM, second)), state
    )

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(MEMBER, PHANTOM)), state
    )
    assert not state.topology.contains_node(PHANTOM)
    assert not state.topology.contains_node(second)
    assert state.topology.contains_node(MEMBER)


def test_non_member_source_reaps_with_its_own_deletion() -> None:
    """An isolated no-membership source is reaped alongside its deletion.

    Source immunity was dropped once emission became membership-gated: a
    node without last_seen has no legitimate live edges in state to lose,
    a genuinely live one is re-added by its NodeGatheredInfo and re-edged
    by self-heal sweeps, and a dead one whose deletion was indexed
    posthumously must not linger (PR #674 review rounds).
    """
    early = NodeId("not-yet-member")
    state = _member_state()
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(early, MEMBER)), state
    )

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(early, MEMBER)), state
    )
    assert not state.topology.contains_node(early)
    assert state.topology.contains_node(MEMBER)


def test_mutual_pair_reaps_on_a_single_posthumous_deletion() -> None:
    """One observed disconnect clears a dead mutual never-member pair.

    With edges A to B and B to A in the log and only A's deletion of its
    side indexed (posthumously), reaping B clears the stale reverse edge
    and the cascade must then take A too (PR #674 review).
    """
    second = NodeId("phantom-two")
    state = State()
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(PHANTOM, second)), state
    )
    state = apply_topology_edge_created(
        TopologyEdgeCreated(conn=_connection(second, PHANTOM)), state
    )

    state = apply_topology_edge_deleted(
        TopologyEdgeDeleted(conn=_connection(PHANTOM, second)), state
    )
    assert not state.topology.contains_node(PHANTOM)
    assert not state.topology.contains_node(second)


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
