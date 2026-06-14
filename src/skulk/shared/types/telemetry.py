"""The telemetry plane (#279): per-node live readings, off the event log.

Telemetry is gossiped last-write-wins on its own topic and kept in a
``TelemetryView`` that lives outside the event-sourced ``State``. It is never
indexed by the master and never persisted: a stale reading coalesces under the
next one instead of inflating the log. This is the plane a management/edge node
subscribes to so it can render the dashboard and route placement without
joining the inference data plane.

Phase 1 slice 1 carried only ``NodeResources`` (single reader: the planner).
Slice 2 adds node memory and the system performance profile — the highest-volume
``NodeGatheredInfo`` readings — and the placement memory-fit check, dashboard,
and power sampler now read them here instead of off the event log.
"""

from __future__ import annotations

from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event, NodeGatheredInfo, NodeTimedOut
from skulk.shared.types.profiling import (
    MemoryUsage,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MacmonMetrics,
    MactopMetrics,
)
from skulk.utils.pydantic_ext import CamelCaseModel

# GatheredInfo variants that live on the telemetry plane (#279): gossiped
# last-write-wins, never indexed or persisted. NodeResources (slice 1) plus
# memory + the system profile (slice 2). Workers fork these onto the TELEMETRY
# topic instead of emitting NodeGatheredInfo events.
TELEMETRY_PLANE_INFO = (NodeResources, MemoryUsage, MactopMetrics, MacmonMetrics)


class NodeTelemetry(CamelCaseModel):
    """A single node's latest telemetry reading, published on the TELEMETRY
    topic. Mirrors ``NodeGatheredInfo`` but is a plain message, not a log
    event: it is gossiped node -> all and never enters the event log."""

    node_id: NodeId
    info: GatheredInfo


class TelemetryView:
    """Latest-per-node telemetry, maintained outside event-sourced ``State``.

    Owned by the ``Node`` so it survives master re-election (a freshly elected
    master keeps the cluster's current readings instead of starting blind and
    risking a placement on a management node during the repopulation window).
    Updated only by the telemetry subscriber; readers (planner, API previews)
    hold a reference and read the maps directly.
    """

    def __init__(self) -> None:
        self.node_resources: dict[NodeId, NodeResources] = {}
        self.node_memory: dict[NodeId, MemoryUsage] = {}
        self.node_system: dict[NodeId, SystemPerformanceProfile] = {}

    def prune(self, node_id: NodeId) -> None:
        """Drop all telemetry for a node that left the cluster.

        The telemetry plane is last-write-wins with no natural expiry, so a
        node that times out (``NodeTimedOut``) would otherwise keep its last
        readings forever — surfacing as a ghost node in the dashboard and
        skewing capacity/energy aggregates. Callers invoke this where
        ``NodeTimedOut`` is applied so the view tracks live membership; readers
        then never see a dead node regardless of whether they also filter.
        """
        self.node_resources.pop(node_id, None)
        self.node_memory.pop(node_id, None)
        self.node_system.pop(node_id, None)

    def apply(self, message: NodeTelemetry) -> None:
        """Coalesce one telemetry message into the latest-value view."""
        info = message.info
        node_id = message.node_id
        if isinstance(info, NodeResources):
            self.node_resources[node_id] = info
        elif isinstance(info, MemoryUsage):
            self.node_memory[node_id] = info
        elif isinstance(info, (MactopMetrics, MacmonMetrics)):
            # MacmonMetrics is a decode-only rolling-upgrade shim with the same
            # normalized memory/system_profile shape as MactopMetrics.
            self.node_memory[node_id] = info.memory
            self.node_system[node_id] = info.system_profile
        # Remaining GatheredInfo variants (disk, network, thunderbolt,
        # identities) still travel the event log; they migrate in later slices.


def record_membership_from_event(view: TelemetryView, event: Event) -> None:
    """Maintain telemetry-plane membership from the control event stream.

    Called from EVERY node's event applier(s) — worker and API both — so the
    Node-shared view tracks live membership regardless of which long-running
    components a node runs (``--no-api`` skips the API applier, ``--no-worker``
    skips the worker applier; a node runs at least one). Two duties:

    - Prune a node's readings when it times out (``NodeTimedOut``) — the plane
      has no natural expiry, so without this a dead node lingers as a ghost.
    - Bridge a legacy/un-upgraded worker's telemetry-plane ``NodeGatheredInfo``
      into the view during a rolling upgrade. ``apply`` no-ops those events, and
      an upgraded node reads memory only from the view, so without the bridge it
      would report "memory not gathered" and refuse placement on those live
      nodes until every worker restarts. New workers gossip these directly and
      never as events, so only legacy senders are bridged.
    """
    if isinstance(event, NodeTimedOut):
        view.prune(event.node_id)
    elif isinstance(event, NodeGatheredInfo) and isinstance(
        event.info, TELEMETRY_PLANE_INFO
    ):
        view.apply(NodeTelemetry(node_id=event.node_id, info=event.info))
