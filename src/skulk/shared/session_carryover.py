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

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from skulk.shared.types.common import NodeId
from skulk.shared.types.state import State
from skulk.shared.types.worker.downloads import DownloadCompleted, DownloadProgress


def _completed_downloads_only(
    downloads: Mapping[NodeId, Sequence[DownloadProgress]],
) -> dict[NodeId, list[DownloadProgress]]:
    """Keep only completed download knowledge across the session boundary.

    Pending/ongoing/failed entries describe work owned by the OLD session's
    download coordinator, which the promotion path restarts — carrying a
    ``DownloadOngoing`` would make the new planner treat the download as
    already in hand and never re-issue it, stranding a mid-download
    placement forever (review catch on #274). Completed entries are durable
    facts about bytes on disk and carry; everything else is re-planned
    fresh.
    """
    return {
        node_id: [p for p in progress if isinstance(p, DownloadCompleted)]
        for node_id, progress in downloads.items()
    }


def seed_state_for_new_session(prior: State, now: datetime | None = None) -> State:
    """Build the new session's initial state from the prior replicated view.

    Carried (the durable facts the new session should preserve):
    - ``instances`` — the point of the exercise: placements survive failover.
      Workers re-create runners for them through the ordinary plan loop
      (``_create_runner`` keys on instances + local supervisors only), so
      models reload and serving resumes without operator action. Instances
      whose ranks lived on the dead master are pruned by the master plan
      loop's existing dead-node cleanup once live topology shows the node
      gone.
    - ``downloads`` — COMPLETED entries only (durable bytes-on-disk facts;
      avoids re-downloading). Pending/ongoing/failed entries are dropped:
      they describe work owned by the old session's restarted coordinator,
      and a carried ``DownloadOngoing`` would stop the planner from ever
      re-issuing the download.
    - ``node_*`` info maps — memory/identity/network facts; carrying them
      avoids an artificial ``PlacementInfoPendingError`` window after
      failover. Gossip refreshes them within seconds either way.
    - ``thunderbolt_bridge_cycles`` and ``tracing_enabled`` — cluster
      configuration facts that have no session affinity.

    - ``last_seen`` — carried, but re-stamped to ``now`` for every node the
      seed still knows about (the union of the carried ``node_*`` maps) rather
      than copied from the prior snapshot. This arms the master's 30s
      liveness clock for each carried identity: a node that re-gossips (under
      the same id) keeps refreshing normally, while a carried identity that
      never returns is reaped by the ordinary ``NodeTimedOut`` prune after the
      settle grace. Dropping ``last_seen`` entirely (the pre-fix behavior)
      left carried identities with NO liveness anchor, and the prune loop only
      reaps via ``last_seen`` — so a node that restarted under a new id around
      a session transition leaked its prior identity into ``node_identities``/
      ``node_memory`` permanently, surfacing as a phantom node in ``/state``
      (#218 family). Re-stamping to ``now`` (not the stale prior timestamp)
      avoids admitting a long-dead node onto the live clock as already-fresh.

    Deliberately dropped (session-scoped or liveness-derived):
    - ``tasks`` — in-flight commands/streams died with the old session's
      router plumbing; carrying them would spawn cancel loops against
      runners that no longer exist.
    - ``runners`` — statuses describe processes that the session transition
      tears down (each node's worker is re-created and shuts its supervisors
      down); stale ``RunnerReady`` entries would confuse the ConnectToGroup
      readiness gates. Fresh CreateRunner cycles repopulate them.
    - ``topology`` — liveness must come from the live router's connection
      gossip, not a snapshot that still shows the dead master as connected; a
      stale edge here would delay dead-node instance cleanup or admit
      placements onto a corpse. (Placement eligibility and instance cleanup
      both key off ``topology``, not ``last_seen``, so re-stamping the latter
      above does neither.)
    - ``last_event_applied_idx`` — the master indexes this seed as the FIRST
      EVENT of the new session (a logged ``StateSnapshotHydrated``), setting
      the index at that point; followers receive it inside the snapshot if
      they bootstrap later, or as live event 0 if they bootstrapped against
      the momentarily-empty pre-seed state (the promotion race).
    """
    stamp = now if now is not None else datetime.now(tz=timezone.utc)
    # Arm the liveness clock for every node the seed still carries info about.
    # The prune loop only reaps via last_seen, so a carried identity with no
    # last_seen entry is immortal (the phantom-node leak, #218 family). This
    # MUST cover every node_* map copied into the seed below — a node present
    # only in node_thunderbolt/node_thunderbolt_bridge/node_rdma_ctl (not in
    # identities) would otherwise still leak (review catch on #291). node_memory
    # is no longer carried — it moved to the telemetry plane (#279 slice 2).
    carried_node_ids = (
        prior.node_identities.keys()
        | prior.node_disk.keys()
        | prior.node_network.keys()
        | prior.node_thunderbolt.keys()
        | prior.node_thunderbolt_bridge.keys()
        | prior.node_rdma_ctl.keys()
    )
    last_seen = {node_id: stamp for node_id in carried_node_ids}
    return State(
        instances=prior.instances,
        downloads=_completed_downloads_only(prior.downloads),
        tracing_enabled=prior.tracing_enabled,
        last_seen=last_seen,
        node_identities=prior.node_identities,
        node_disk=prior.node_disk,
        node_network=prior.node_network,
        node_thunderbolt=prior.node_thunderbolt,
        node_thunderbolt_bridge=prior.node_thunderbolt_bridge,
        node_rdma_ctl=prior.node_rdma_ctl,
        thunderbolt_bridge_cycles=prior.thunderbolt_bridge_cycles,
    )
