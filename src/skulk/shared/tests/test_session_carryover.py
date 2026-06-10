"""Tests for failover state carry-over (#273).

Pins what a promoted master inherits from the node's prior replicated state
(instances, downloads, node info, tracing) and what it must NOT inherit
(in-flight tasks, dead runner statuses, stale topology/liveness, the old
session's event index).
"""

from datetime import datetime, timezone

from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import MemoryUsage, NodeIdentity
from skulk.shared.types.state import State


def _prior_state() -> tuple[State, NodeId]:
    node = NodeId()
    topology = Topology()
    topology.add_node(node)
    return State(
        # A stand-in mapping is enough: the seed must carry the field
        # through untouched, not interpret it.
        instances={},
        runners={},
        tasks={},
        downloads={node: []},
        last_seen={node: datetime.now(tz=timezone.utc)},
        topology=topology,
        tracing_enabled=True,
        last_event_applied_idx=4242,
        node_identities={node: NodeIdentity(friendly_name="kite-test")},
        node_memory={
            node: MemoryUsage.from_bytes(
                ram_total=16 * 2**30,
                ram_available=8 * 2**30,
                swap_total=0,
                swap_available=0,
            )
        },
    ), node


def test_carries_durable_facts():
    prior, node = _prior_state()
    seed = seed_state_for_new_session(prior)
    assert seed.instances == prior.instances
    assert seed.downloads == prior.downloads
    assert seed.tracing_enabled is True
    assert seed.node_identities[node].friendly_name == "kite-test"
    assert seed.node_memory[node].ram_available.in_bytes == 8 * 2**30


def test_drops_session_scoped_state():
    prior, node = _prior_state()
    seed = seed_state_for_new_session(prior)
    # In-flight tasks died with the old session's plumbing.
    assert seed.tasks == {}
    # Runner statuses describe processes the transition tears down.
    assert seed.runners == {}
    # Liveness must come from live gossip — a carried topology would keep a
    # dead master's out-edges forever (only their source node deletes them).
    assert seed.topology.list_nodes() == []
    assert seed.last_seen == {}
    # The new session's event log starts at the beginning.
    assert seed.last_event_applied_idx == -1
    assert node not in seed.last_seen
