"""Derived per-node health summary for the dashboard topology (#388).

When the master recovers a wedged instance (a node that could not pull its
shard, or whose download terminally failed), the *reason* is invisible in the
UI: the node looks normal while placements quietly route around it. This module
derives a small, read-only health summary per node from signals already present
in the replicated ``State`` and the per-node ``TelemetryView`` -- no new gossip
type, no new command, no remediation automation. The API merges the result into
``GET /state`` under ``nodeHealth`` and the dashboard renders an amber/red
indicator whose hover names the problem *and* its remediation, so the recovery
is legible instead of mysterious.

The derivation is a pure function with an injected ``now`` so it is exercised in
isolation by unit tests; the API supplies the live wall clock and telemetry.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Literal, final

from pydantic import ConfigDict

from skulk.api.build_identity import (
    git_commit_identifiers_disagree,
    known_build_identifier,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.node_facts import CONFLICT_ERROR_CODES
from skulk.shared.types.profiling import (
    DiskUsage,
    NodeDataTransport,
    NodeIdentity,
    NodeResources,
)
from skulk.shared.types.worker.downloads import DownloadFailed, DownloadProgress
from skulk.utils.pydantic_ext import CamelCaseModel

# A node is pruned once every liveness signal is ~30s stale. Warn while it is
# still present so the operator sees "this node is going" rather than a silent
# disappearance.
UNREACHABLE_WARN_AFTER = timedelta(seconds=15)

# Disk thresholds for the models volume. A download needs headroom for the full
# repo plus staging, so warn well before zero (a large model can be tens of GB)
# and escalate to error when the volume is effectively out of space and the next
# download is certain to fail.
DISK_LOW_THRESHOLD = Memory.from_gb(10.0)
DISK_FULL_THRESHOLD = Memory.from_gb(2.0)

HealthLevel = Literal["ok", "warn", "error"]
HealthCode = Literal[
    "data_transport_mismatch",
    "version_mismatch",
    "download_failed",
    "disk_low",
    "disk_full",
    "unreachable",
    # Capability conflicts from backend derivation (#614): observation vs
    # declaration disagreements advertised on NodeResources. The codes are the
    # CapabilityConflictCode vocabulary, mapped one-to-one.
    "gpu_serving_disabled",
    "gpu_detection_degraded",
    "invalid_engine_binary",
    "backend_override_conflict",
]

# Severity rank so an aggregate level is the worst of a node's reasons.
_LEVEL_RANK: dict[HealthLevel, int] = {"ok": 0, "warn": 1, "error": 2}


@final
class NodeHealthReason(CamelCaseModel):
    """One concrete problem on a node, with the path to fix it.

    The dashboard shows ``message`` (what is wrong) and ``remediation`` (how to
    fix it) together, so a node problem is actionable rather than a raw error.
    """

    model_config = ConfigDict(frozen=True)

    code: HealthCode
    """Stable machine code for the condition (for styling / filtering)."""

    message: str
    """Human-readable description of what is wrong."""

    remediation: str
    """The operator's path to resolve it."""


@final
class NodeHealth(CamelCaseModel):
    """Aggregate health for a single node: a level plus its concrete reasons."""

    model_config = ConfigDict(frozen=True)

    level: HealthLevel
    """Worst severity across ``reasons`` (``ok`` when there are none)."""

    reasons: Sequence[NodeHealthReason] = ()
    """The concrete problems contributing to ``level`` (empty when ``ok``)."""


def _aggregate_level(reasons: Sequence[NodeHealthReason]) -> HealthLevel:
    """Return the worst severity across *reasons* (``ok`` when empty)."""
    worst: HealthLevel = "ok"
    for reason in reasons:
        level: HealthLevel = (
            "error"
            if reason.code in ("data_transport_mismatch", "download_failed", "disk_full")
            or reason.code in CONFLICT_ERROR_CODES
            else "warn"
        )
        if _LEVEL_RANK[level] > _LEVEL_RANK[worst]:
            worst = level
    return worst


def _capability_conflict_reasons(
    resources: NodeResources | None,
) -> list[NodeHealthReason]:
    """Reasons for every capability conflict a node advertises (#614).

    The conflict's message and remediation were composed on the node that owns
    the facts (it knows its own paths and env), so this is a pure one-to-one
    mapping: the conflict code is a ``HealthCode`` by construction and its
    severity comes from the shared ``CONFLICT_ERROR_CODES`` source of truth.
    """
    if resources is None:
        return []
    return [
        NodeHealthReason(
            code=conflict.code,
            message=conflict.message,
            remediation=conflict.remediation,
        )
        for conflict in resources.capability_conflicts
    ]


def live_data_transports(
    *,
    live_nodes: Mapping[NodeId, datetime],
    node_resources: Mapping[NodeId, NodeResources],
) -> frozenset[NodeDataTransport]:
    """Return transports advertised by live nodes with resource telemetry.

    Nodes whose first ``NodeResources`` reading has not arrived are omitted. A
    mismatch is reported only from positive evidence of both transports, never
    from a transient telemetry gap during startup.

    Args:
        live_nodes: Nodes currently present in replicated cluster membership.
        node_resources: Latest per-node resource advertisements.

    Returns:
        The distinct resolved DATA transports advertised by live nodes.
    """
    return frozenset(
        resources.data_transport
        for node_id, resources in node_resources.items()
        if node_id in live_nodes
    )


def live_skulk_build_mismatch(
    *,
    live_nodes: Mapping[NodeId, datetime],
    node_identities: Mapping[NodeId, NodeIdentity],
) -> bool:
    """Return whether live telemetry positively identifies multiple builds.

    Unknown version or commit sentinels are omitted so startup telemetry gaps do
    not create false warnings. Either multiple package versions or multiple Git
    commits is sufficient evidence of a mixed deployment.

    Args:
        live_nodes: Nodes currently present in replicated cluster membership.
        node_identities: Latest per-node package and source identity telemetry.

    Returns:
        True when at least one known build identifier disagrees across live nodes.
    """

    identities = (
        identity
        for node_id, identity in node_identities.items()
        if node_id in live_nodes
    )
    versions: set[str] = set()
    commits: list[str] = []
    for identity in identities:
        version = known_build_identifier(identity.skulk_version)
        commit = known_build_identifier(identity.skulk_commit)
        if version is not None:
            versions.add(version)
        if commit is not None:
            commits.append(commit)
    return len(versions) > 1 or git_commit_identifiers_disagree(commits)


def _data_transport_mismatch_reason(
    *,
    node_id: NodeId,
    node_resources: Mapping[NodeId, NodeResources],
    transports: frozenset[NodeDataTransport],
) -> NodeHealthReason:
    """Build the fail-loud health reason for one node in a split DATA fleet."""
    resources = node_resources.get(node_id)
    local_transport = (
        resources.data_transport if resources is not None else "not yet advertised"
    )
    fleet_transports = ", ".join(sorted(transports))
    return NodeHealthReason(
        code="data_transport_mismatch",
        message=(
            "Cluster DATA transports disagree "
            f"({fleet_transports}); this node uses {local_transport}. "
            "Cross-transport inference output cannot be delivered."
        ),
        remediation=(
            "Configure every node to use the same DATA transport, restart the "
            "fleet, and confirm nodeResources reports one transport before "
            "serving inference."
        ),
    )


def _version_mismatch_reason(
    *, node_id: NodeId, node_identities: Mapping[NodeId, NodeIdentity]
) -> NodeHealthReason:
    """Build the degraded-health reason for one node in a mixed-build fleet."""

    identity = node_identities.get(node_id)
    version = identity.skulk_version if identity is not None else "not yet advertised"
    commit = identity.skulk_commit if identity is not None else "not yet advertised"
    return NodeHealthReason(
        code="version_mismatch",
        message=(
            "Cluster nodes report different Skulk builds; this node reports "
            f"version {version} at commit {commit}. Correctness-bearing wire "
            "compatibility is not guaranteed until deployment converges."
        ),
        remediation=(
            "Complete the deployment so every node runs the same version and "
            "commit, then confirm the version warning clears before starting "
            "new inference work."
        ),
    )


def _download_failure_reasons(
    downloads: Sequence[DownloadProgress],
) -> list[NodeHealthReason]:
    """Reasons for any terminally failed download on a node.

    Pairs with the master's download-failure recovery (#381): this is the
    operator-visible explanation for why an instance was re-placed off this node.
    """
    reasons: list[NodeHealthReason] = []
    for download in downloads:
        if isinstance(download, DownloadFailed):
            model_id = download.shard_metadata.model_card.model_id
            reasons.append(
                NodeHealthReason(
                    code="download_failed",
                    message=f"Download of {model_id} failed: {download.error_message}",
                    remediation=(
                        "Free disk space (or lower staging_keep_recent_gb), check "
                        "the node's network, then retry the model."
                    ),
                )
            )
    return reasons


def _disk_reason(disk: DiskUsage) -> NodeHealthReason | None:
    """A disk-pressure reason for a node, or ``None`` when it has headroom."""
    available = disk.available
    if available.in_bytes <= DISK_FULL_THRESHOLD.in_bytes:
        return NodeHealthReason(
            code="disk_full",
            message=(
                f"Models volume is effectively full: {available.in_gb:.1f} GB free "
                f"of {disk.total.in_gb:.1f} GB."
            ),
            remediation=(
                "Free disk space or lower staging_keep_recent_gb; new downloads "
                "will fail until there is headroom."
            ),
        )
    if available.in_bytes < DISK_LOW_THRESHOLD.in_bytes:
        return NodeHealthReason(
            code="disk_low",
            message=(
                f"Models volume is low on space: {available.in_gb:.1f} GB free "
                f"of {disk.total.in_gb:.1f} GB."
            ),
            remediation=(
                "Free disk space or lower staging_keep_recent_gb before pulling a "
                "large model."
            ),
        )
    return None


def _unreachable_reason(
    last_seen: datetime, now: datetime, warn_after: timedelta
) -> NodeHealthReason | None:
    """A reason when a node's heartbeat is stale enough to be at risk of pruning."""
    # last_seen is stamped tz-aware UTC by the master, but guard a tz-naive value
    # (an odd snapshot) defensively: a naive-vs-aware subtraction would raise and
    # 500 the whole /state endpoint, which is the dashboard's primary data source.
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    if now - last_seen > warn_after:
        return NodeHealthReason(
            code="unreachable",
            message=(
                "Node heartbeats are late; it may be dropping out of the cluster."
            ),
            remediation=(
                "Check the node's network and Tailscale connectivity; restart "
                "skulk on it if it does not recover."
            ),
        )
    return None


def compute_node_health(
    *,
    live_nodes: Mapping[NodeId, datetime],
    downloads: Mapping[NodeId, Sequence[DownloadProgress]],
    node_disk: Mapping[NodeId, DiskUsage],
    now: datetime,
    node_resources: Mapping[NodeId, NodeResources] | None = None,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    heartbeat_last_seen: Mapping[NodeId, datetime] | None = None,
    telemetry_last_seen: Mapping[NodeId, datetime] | None = None,
    unreachable_warn_after: timedelta = UNREACHABLE_WARN_AFTER,
) -> dict[str, NodeHealth]:
    """Derive a per-node health summary for every live node.

    Args:
        live_nodes: ``State.last_seen`` -- the nodes currently in the cluster,
            mapped to their last indexed control-plane event. This timestamp is
            not a heartbeat. Health is computed for exactly these nodes (a node
            already pruned for timeout is absent here and from ``/state``).
        downloads: Effective per-node download status, scanned for terminal
            ``DownloadFailed`` entries.
        node_disk: ``TelemetryView.node_disk`` -- per-node models-volume usage.
        now: The wall clock used for heartbeat-staleness; injected for testing.
        node_resources: ``TelemetryView.node_resources`` -- resolved DATA
            transport and placement capability facts for split-fleet detection.
        node_identities: ``TelemetryView.node_identities`` -- package version and
            source commit facts for mixed-build detection.
        heartbeat_last_seen: ``TelemetryView.node_last_heartbeat`` -- local
            receipt time of each peer's dedicated heartbeat, the primary
            liveness signal.
        telemetry_last_seen: ``TelemetryView.node_last_telemetry`` -- local
            receipt time of each peer's ordinary telemetry, retained as a
            fallback liveness signal if the dedicated heartbeat path degrades.
        unreachable_warn_after: Heartbeat-staleness past which a node is flagged
            as at-risk of pruning.

    Returns:
        A mapping from string node id to its :class:`NodeHealth`, one entry per
        live node (``level`` is ``ok`` with no reasons for a healthy node), so
        the dashboard can render an indicator (or none) per topology node.
    """
    heartbeat_last_seen = heartbeat_last_seen or {}
    telemetry_last_seen = telemetry_last_seen or {}
    node_resources = node_resources or {}
    node_identities = node_identities or {}
    transports = live_data_transports(
        live_nodes=live_nodes,
        node_resources=node_resources,
    )
    build_mismatch = live_skulk_build_mismatch(
        live_nodes=live_nodes,
        node_identities=node_identities,
    )
    health: dict[str, NodeHealth] = {}
    for node_id, last_seen in live_nodes.items():
        reasons: list[NodeHealthReason] = []
        if len(transports) > 1:
            reasons.append(
                _data_transport_mismatch_reason(
                    node_id=node_id,
                    node_resources=node_resources,
                    transports=transports,
                )
            )
        if build_mismatch:
            reasons.append(
                _version_mismatch_reason(
                    node_id=node_id,
                    node_identities=node_identities,
                )
            )
        reasons.extend(_capability_conflict_reasons(node_resources.get(node_id)))
        reasons.extend(_download_failure_reasons(downloads.get(node_id, ())))
        disk = node_disk.get(node_id)
        if disk is not None:
            disk_reason = _disk_reason(disk)
            if disk_reason is not None:
                reasons.append(disk_reason)
        # Use the freshest of the control event, dedicated heartbeat, and
        # ordinary telemetry fallback. Normalize a tz-naive snapshot timestamp
        # defensively so /state cannot fail on an aware-vs-naive comparison.
        liveness_signals = [
            last_seen
            if last_seen.tzinfo is not None
            else last_seen.replace(tzinfo=timezone.utc)
        ]
        for signal in (
            heartbeat_last_seen.get(node_id),
            telemetry_last_seen.get(node_id),
        ):
            if signal is not None:
                liveness_signals.append(
                    signal
                    if signal.tzinfo is not None
                    else signal.replace(tzinfo=timezone.utc)
                )
        liveness_seen = max(liveness_signals)
        unreachable = _unreachable_reason(
            liveness_seen, now, unreachable_warn_after
        )
        if unreachable is not None:
            reasons.append(unreachable)
        health[str(node_id)] = NodeHealth(
            level=_aggregate_level(reasons), reasons=reasons
        )
    return health
