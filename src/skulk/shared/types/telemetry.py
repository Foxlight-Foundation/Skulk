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
from skulk.shared.types.events import Event, NodeTimedOut
from skulk.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeIdentity,
    NodeRdmaCtlStatus,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    LinuxGpuMetrics,
    MacmonMetrics,
    MactopMetrics,
    MiscData,
    NodeDiskUsage,
    RdmaCtlStatus,
    StaticNodeInformation,
)
from skulk.utils.pydantic_ext import CamelCaseModel

# GatheredInfo variants that live on the telemetry plane (#279): gossiped
# last-write-wins, never indexed or persisted. NodeResources (slice 1); memory +
# the system profile (slice 2); the observational node readings disk/identity/
# rdma-ctl (slice 3). Workers fork these onto the TELEMETRY topic instead of
# emitting NodeGatheredInfo events.
#
# Deliberately NOT here: the connectivity readings (NodeNetworkInterfaces,
# MacThunderboltIdentifiers/Connections, ThunderboltBridgeInfo). Those define
# the topology graph — apply() builds RDMA edges and thunderbolt-bridge cycles
# from them, and the placement planner reads node_network for host selection —
# so they stay on the ordered control plane rather than an unordered
# last-write-wins plane (#279 slice 3 scoping).
TELEMETRY_PLANE_INFO = (
    NodeResources,
    MemoryUsage,
    MactopMetrics,
    MacmonMetrics,
    LinuxGpuMetrics,
    NodeDiskUsage,
    MiscData,
    StaticNodeInformation,
    RdmaCtlStatus,
)


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
        # Slice 3 observational readings (disk/identity/rdma-ctl).
        self.node_disk: dict[NodeId, DiskUsage] = {}
        self.node_identities: dict[NodeId, NodeIdentity] = {}
        self.node_rdma_ctl: dict[NodeId, NodeRdmaCtlStatus] = {}

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
        self.node_disk.pop(node_id, None)
        self.node_identities.pop(node_id, None)
        self.node_rdma_ctl.pop(node_id, None)

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
        elif isinstance(info, LinuxGpuMetrics):
            # AMD/Linux GPU collector: carries only the system profile (with its
            # accelerator block); node memory arrives separately via MemoryUsage.
            self.node_system[node_id] = info.system_profile
        elif isinstance(info, NodeDiskUsage):
            self.node_disk[node_id] = info.disk_usage
        elif isinstance(info, RdmaCtlStatus):
            self.node_rdma_ctl[node_id] = NodeRdmaCtlStatus(
                enabled=info.enabled,
                interfaces_present=info.interfaces_present,
            )
        elif isinstance(info, MiscData):
            # Identity is assembled from two readings (friendly name +
            # static info), so merge into the existing entry rather than
            # overwrite — mirrors the accumulation the event applier did.
            current = self.node_identities.get(node_id, NodeIdentity())
            self.node_identities[node_id] = current.model_copy(
                update={"friendly_name": info.friendly_name}
            )
        elif isinstance(info, StaticNodeInformation):
            current = self.node_identities.get(node_id, NodeIdentity())
            self.node_identities[node_id] = current.model_copy(
                update={
                    "model_id": info.model,
                    "chip_id": info.chip,
                    "os_version": info.os_version,
                    "os_build_version": info.os_build_version,
                    "skulk_version": info.skulk_version,
                    "skulk_commit": info.skulk_commit,
                }
            )
        # Connectivity readings (network, thunderbolt, thunderbolt-bridge)
        # deliberately stay on the control plane — they define the topology
        # graph (see TELEMETRY_PLANE_INFO note above).


def record_membership_from_event(view: TelemetryView, event: Event) -> None:
    """Prune a node's telemetry when it leaves the cluster.

    Called from EVERY node's event applier(s) — worker and API both — so the
    Node-shared view tracks live membership regardless of which long-running
    components a node runs (``--no-api`` skips the API applier, ``--no-worker``
    skips the worker applier; a node runs at least one). The telemetry plane has
    no natural expiry, so without this a timed-out node lingers as a ghost.

    Note: there is deliberately no "bridge legacy NodeGatheredInfo telemetry
    into the view" path here. Mixed-version clusters are unsupported (an
    anti-pattern — all nodes must run the same Skulk version, deployed
    whole-fleet); engineering for an un-upgraded worker's legacy event stream
    would be supporting exactly that. See the deployment docs and #293.
    """
    if isinstance(event, NodeTimedOut):
        view.prune(event.node_id)
