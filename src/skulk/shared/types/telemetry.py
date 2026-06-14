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
