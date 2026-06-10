"""Seeding a new master session from the prior session's replicated state.

Until #273, a newly-promoted master always started from an empty ``State()``:
the empty state propagated to every follower via snapshot bootstrap, each
worker's plan loop saw no instances and shut down its healthy runners, and
every placement silently became a 404 until an operator re-placed it — a full
serving outage from a single master restart.

The promoted node is not amnesiac: as a follower it held the entire
replicated state. ``seed_state_for_new_session`` turns that last view into a
safe starting state for the new session.
"""

from skulk.shared.types.state import State


def seed_state_for_new_session(prior: State) -> State:
    """Build the new session's initial state from the prior replicated view.

    Carried (the durable facts the new session should preserve):
    - ``instances`` — the point of the exercise: placements survive failover.
      Workers re-create runners for them through the ordinary plan loop
      (``_create_runner`` keys on instances + local supervisors only), so
      models reload and serving resumes without operator action. Instances
      whose ranks lived on the dead master are pruned by the master plan
      loop's existing dead-node cleanup once live topology shows the node
      gone.
    - ``downloads`` — completed-download knowledge; avoids re-downloading.
    - ``node_*`` info maps — memory/identity/network facts; carrying them
      avoids an artificial ``PlacementInfoPendingError`` window after
      failover. Gossip refreshes them within seconds either way.
    - ``thunderbolt_bridge_cycles`` and ``tracing_enabled`` — cluster
      configuration facts that have no session affinity.

    Deliberately dropped (session-scoped or liveness-derived):
    - ``tasks`` — in-flight commands/streams died with the old session's
      router plumbing; carrying them would spawn cancel loops against
      runners that no longer exist.
    - ``runners`` — statuses describe processes that the session transition
      tears down (each node's worker is re-created and shuts its supervisors
      down); stale ``RunnerReady`` entries would confuse the ConnectToGroup
      readiness gates. Fresh CreateRunner cycles repopulate them.
    - ``topology`` and ``last_seen`` — liveness must come from the live
      router's connection gossip, not a snapshot that still shows the dead
      master as connected; a stale edge here would delay dead-node instance
      cleanup or admit placements onto a corpse.
    - ``last_event_applied_idx`` — the new session's event log starts at the
      beginning; followers hydrate this seed via the ordinary snapshot
      bootstrap and replay from index 0.
    """
    return State(
        instances=prior.instances,
        downloads=prior.downloads,
        tracing_enabled=prior.tracing_enabled,
        node_identities=prior.node_identities,
        node_memory=prior.node_memory,
        node_disk=prior.node_disk,
        node_system=prior.node_system,
        node_network=prior.node_network,
        node_thunderbolt=prior.node_thunderbolt,
        node_thunderbolt_bridge=prior.node_thunderbolt_bridge,
        node_rdma_ctl=prior.node_rdma_ctl,
        thunderbolt_bridge_cycles=prior.thunderbolt_bridge_cycles,
    )
