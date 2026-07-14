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
``NodeHeartbeat`` names the primary liveness reading; ordinary telemetry
receipt remains a separately tracked fallback.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    Event,
    NodeDownloadProgress,
    NodeTimedOut,
    StagedModelEvicted,
    StateSnapshotHydrated,
)
from skulk.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeIdentity,
    NodeRdmaCtlStatus,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    LiveDownloadProgress,
)
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    LinuxGpuMetrics,
    MacmonMetrics,
    MactopMetrics,
    MiscData,
    NodeCapabilities,
    NodeDiskUsage,
    NodeHeartbeat,
    RdmaCtlStatus,
    StaticNodeInformation,
)
from skulk.utils.pydantic_ext import CamelCaseModel

_DOWNLOAD_EVICTION_TOMBSTONE_CAPACITY = 4_096

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
    NodeCapabilities,
    NodeDiskUsage,
    MiscData,
    StaticNodeInformation,
    RdmaCtlStatus,
    NodeHeartbeat,
    # Extension-advertised capability tags (fabric-citizenship): a plugin
    # advertises what it offers so peers discover it, off the event log.
    NodeCapabilities,
)


class NodeTelemetry(CamelCaseModel):
    """A single node's latest telemetry reading, published on the TELEMETRY
    topic. Mirrors ``NodeGatheredInfo`` but is a plain message, not a log
    event: it is gossiped node -> all and never enters the event log."""

    node_id: NodeId
    info: GatheredInfo | LiveDownloadProgress

    def coalescing_key(self) -> str:
        """Return the bounded latest-value identity used before network egress.

        Ordinary telemetry has one latest value per node and reading class.
        Concurrent downloads additionally key by model so progress for one model
        cannot replace another model's progress on the same node.

        Returns:
            Stable process-local key used only by bounded admission.
        """

        suffix = ""
        if isinstance(self.info, (DownloadPending, DownloadOngoing)):
            suffix = f":{self.info.shard_metadata.model_card.model_id}"
        return f"{self.node_id}:{type(self.info).__name__}{suffix}"


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
        # Extension-advertised capability tags per node (fabric-citizenship).
        # An extension on any node publishes what it offers (via the outbound
        # `local_advertised_capabilities` set below); every node coalesces the
        # readings here so a plugin can discover which peers offer which
        # capability. Opaque free-form strings; Skulk core does not interpret them.
        self.node_capabilities: dict[NodeId, frozenset[str]] = {}
        # Non-terminal download status is observational and last-write-wins.
        # Durable State retains only completed/failed terminal outcomes.
        self.node_downloads: dict[NodeId, dict[str, LiveDownloadProgress]] = {}
        self._node_download_attempts: dict[tuple[NodeId, str], DownloadAttemptId] = {}
        self._node_download_terminal_attempts: dict[
            tuple[NodeId, str], DownloadAttemptId | None
        ] = {}
        # A bounded tombstone prevents delayed telemetry from resurrecting a
        # model after a fleet-wide staged eviction. Unlike terminal-attempt
        # records, these no longer mirror durable State and therefore need an
        # explicit cap.
        self._node_download_eviction_tombstones: OrderedDict[
            tuple[NodeId, str], None
        ] = OrderedDict()
        # This node's OWN outbound capability set: the write half of the plane.
        # It lives on the shared view (not in State) so the API-side extension
        # surface can add to it and the worker-side gatherer can read it, without
        # threading a new channel through Node construction. Mutated by
        # `ExtensionContext.advertise_capability`; polled by the InfoGatherer's
        # `_monitor_capabilities` and gossiped as `NodeCapabilities`.
        self.local_advertised_capabilities: set[str] = set()
        # Receipt time of the explicit, payload-free liveness reading. The
        # master treats this as its primary liveness signal.
        self.node_last_heartbeat: dict[NodeId, datetime] = {}
        # Receipt time of any NON-heartbeat telemetry. This remains a fallback
        # liveness signal so a heartbeat collector defect cannot prune a node
        # that is demonstrably still publishing ordinary telemetry.
        self.node_last_telemetry: dict[NodeId, datetime] = {}

    def last_liveness_receipt(self, node_id: NodeId) -> datetime | None:
        """Return the freshest heartbeat or fallback telemetry receipt.

        Args:
            node_id: Node whose local receipt timestamp should be selected.

        Returns:
            The freshest known receipt, or ``None`` before any telemetry from
            the node has arrived.
        """
        receipts = [
            receipt
            for receipt in (
                self.node_last_heartbeat.get(node_id),
                self.node_last_telemetry.get(node_id),
            )
            if receipt is not None
        ]
        if not receipts:
            return None
        return max(
            receipts,
            key=lambda receipt: (
                receipt
                if receipt.tzinfo is not None
                else receipt.replace(tzinfo=timezone.utc)
            ).timestamp(),
        )

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
        self.node_capabilities.pop(node_id, None)
        self.node_downloads.pop(node_id, None)
        for attempts in (
            self._node_download_attempts,
            self._node_download_terminal_attempts,
            self._node_download_eviction_tombstones,
        ):
            for key in [key for key in attempts if key[0] == node_id]:
                attempts.pop(key, None)
        self.node_last_heartbeat.pop(node_id, None)
        self.node_last_telemetry.pop(node_id, None)
        # local_advertised_capabilities is deliberately NOT pruned: it is this
        # node's own outbound advertisement, not a peer reading, so a peer
        # timing out must never clear what this node offers.

    def apply(
        self,
        message: NodeTelemetry,
        *,
        received_at: datetime | None = None,
    ) -> None:
        """Coalesce one telemetry message into the latest-value view.

        Args:
            message: Reading received from the telemetry topic.
            received_at: Local receipt time, injectable for deterministic tests.
                Defaults to the current UTC time; sender clocks are never used.
        """
        info = message.info
        node_id = message.node_id
        receipt_time = received_at or datetime.now(tz=timezone.utc)
        if isinstance(info, NodeHeartbeat):
            self.node_last_heartbeat[node_id] = receipt_time
        else:
            self.node_last_telemetry[node_id] = receipt_time

        if isinstance(info, NodeHeartbeat):
            return
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
        elif isinstance(info, NodeCapabilities):
            self.node_capabilities[node_id] = info.capabilities
        elif isinstance(info, (DownloadPending, DownloadOngoing)):
            model_id = str(info.shard_metadata.model_card.model_id)
            key = (node_id, model_id)
            terminal_recorded = key in self._node_download_terminal_attempts
            terminal_attempt = self._node_download_terminal_attempts.get(key)
            evicted = key in self._node_download_eviction_tombstones
            if isinstance(info, DownloadOngoing) and (terminal_recorded or evicted):
                # A new attempt always starts with Pending. Until that boundary
                # arrives, every ongoing sample is stale relative to the
                # terminal/eviction decision, including legacy samples without
                # an ID.
                return

            current_attempt = self._node_download_attempts.get(key)
            if isinstance(info, DownloadOngoing) and current_attempt is not None:
                if info.attempt_id != current_attempt:
                    # A cancel/reset establishes a fresh attempt before the old
                    # producer necessarily stops. Never let that producer's tail
                    # overwrite the reset on the independent telemetry protocol.
                    return
            elif isinstance(info, DownloadPending):
                if evicted:
                    # Only the ordered Pending event proves that an operator
                    # started a new attempt after eviction. A delayed telemetry
                    # sample can carry an ID too, but must not resurrect state.
                    return
                if terminal_recorded:
                    if info.attempt_id is None or info.attempt_id == terminal_attempt:
                        return
                    self._node_download_terminal_attempts.pop(key, None)

            if info.attempt_id is not None:
                self._node_download_attempts[key] = info.attempt_id
            self.node_downloads.setdefault(node_id, {})[model_id] = info
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

    def effective_downloads(
        self,
        terminal_downloads: Mapping[NodeId, Sequence[DownloadProgress]],
    ) -> dict[NodeId, list[DownloadProgress]]:
        """Overlay live progress onto durable terminal download outcomes.

        Args:
            terminal_downloads: Event-sourced completed/failed outcomes.

        Returns:
            Per-node statuses for placement, worker planning, and operator views.
            The returned mapping is detached from both source stores.
        """

        effective: dict[NodeId, list[DownloadProgress]] = {
            node_id: list(progresses)
            for node_id, progresses in terminal_downloads.items()
        }
        for node_id, live_by_model in self.node_downloads.items():
            current = effective.setdefault(node_id, [])
            for model_id, live_progress in live_by_model.items():
                current[:] = [
                    progress
                    for progress in current
                    if str(progress.shard_metadata.model_card.model_id) != model_id
                ]
                current.append(live_progress)
        return effective

    def record_download_event(self, event: NodeDownloadProgress) -> None:
        """Apply terminal ordering metadata from one durable download event."""

        progress = event.download_progress
        model_id = str(progress.shard_metadata.model_card.model_id)
        key = (progress.node_id, model_id)
        live_by_model = self.node_downloads.get(progress.node_id)

        if isinstance(progress, DownloadPending):
            # A rare Pending event is a durable reset/start decision, not progress
            # sampling. It clears an older terminal outcome and establishes the
            # attempt that subsequent telemetry belongs to.
            if progress.attempt_id is not None:
                self._node_download_attempts[key] = progress.attempt_id
            else:
                self._node_download_attempts.pop(key, None)
            self._node_download_terminal_attempts.pop(key, None)
            self._node_download_eviction_tombstones.pop(key, None)
            if live_by_model is not None:
                current = live_by_model.get(model_id)
                if (
                    current is not None
                    and (
                        progress.attempt_id is None
                        or current.attempt_id != progress.attempt_id
                    )
                ):
                    live_by_model.pop(model_id, None)
            return

        if not isinstance(progress, (DownloadCompleted, DownloadFailed)):
            # Legacy event logs may contain DownloadOngoing. New producers never
            # emit it on the ordered event plane.
            return
        current_attempt = self._node_download_attempts.get(key)
        if (
            progress.attempt_id is not None
            and current_attempt is not None
            and progress.attempt_id != current_attempt
        ):
            # A terminal event from an older attempt can race with the next
            # attempt's Pending telemetry. The ordered Pending control event
            # will clear its durable outcome; do not disturb the newer view.
            return
        if progress.attempt_id is not None:
            self._node_download_attempts[key] = progress.attempt_id
        self._node_download_terminal_attempts[key] = progress.attempt_id
        self._node_download_eviction_tombstones.pop(key, None)
        if live_by_model is not None:
            current = live_by_model.get(model_id)
            if (
                current is not None
                and (
                    progress.attempt_id is None
                    or current.attempt_id is None
                    or current.attempt_id == progress.attempt_id
                )
            ):
                live_by_model.pop(model_id, None)

    def remove_download_model(self, model_id: str) -> None:
        """Drop live and ordering metadata for one globally evicted model."""

        keys = {
            key
            for key in (
                *self._node_download_attempts,
                *self._node_download_terminal_attempts,
            )
            if key[1] == model_id
        }
        for node_id, downloads in list(self.node_downloads.items()):
            if model_id in downloads:
                keys.add((node_id, model_id))
            downloads.pop(model_id, None)
            if not downloads:
                self.node_downloads.pop(node_id, None)
        for key in keys:
            self._node_download_attempts.pop(key, None)
            self._node_download_terminal_attempts.pop(key, None)
            self._node_download_eviction_tombstones.pop(key, None)
            self._node_download_eviction_tombstones[key] = None
        while (
            len(self._node_download_eviction_tombstones)
            > _DOWNLOAD_EVICTION_TOMBSTONE_CAPACITY
        ):
            self._node_download_eviction_tombstones.popitem(last=False)

    def record_download_snapshot(
        self,
        downloads: Mapping[NodeId, Sequence[DownloadProgress]],
    ) -> None:
        """Seed terminal ordering from an authoritative hydrated state.

        Args:
            downloads: Download outcomes carried by the state snapshot.
        """

        for progresses in downloads.values():
            for progress in progresses:
                self.record_download_event(
                    NodeDownloadProgress(download_progress=progress)
                )


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
    elif isinstance(event, NodeDownloadProgress):
        view.record_download_event(event)
    elif isinstance(event, StagedModelEvicted):
        view.remove_download_model(str(event.model_id))
    elif isinstance(event, StateSnapshotHydrated):
        view.record_download_snapshot(event.state.downloads)
