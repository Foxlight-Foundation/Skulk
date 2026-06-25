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

from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import DiskUsage
from skulk.shared.types.worker.downloads import DownloadFailed, DownloadProgress
from skulk.utils.pydantic_ext import CamelCaseModel

# A node is pruned from the cluster once its heartbeat is ~30s stale (the
# master's NodeTimedOut threshold), so a still-present node whose last_seen is
# already this stale is missing heartbeats and about to drop -- warn before it
# vanishes so the operator sees "this node is going" rather than a silent
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
    "download_failed",
    "disk_low",
    "disk_full",
    "unreachable",
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
        level: HealthLevel = "error" if reason.code in ("download_failed", "disk_full") else "warn"
        if _LEVEL_RANK[level] > _LEVEL_RANK[worst]:
            worst = level
    return worst


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
    unreachable_warn_after: timedelta = UNREACHABLE_WARN_AFTER,
) -> dict[str, NodeHealth]:
    """Derive a per-node health summary for every live node.

    Args:
        live_nodes: ``State.last_seen`` -- the nodes currently in the cluster,
            mapped to their last heartbeat. Health is computed for exactly these
            (a node already pruned for timeout is absent here and from ``/state``).
        downloads: ``State.downloads`` -- per-node download progress, scanned for
            terminal ``DownloadFailed`` entries.
        node_disk: ``TelemetryView.node_disk`` -- per-node models-volume usage.
        now: The wall clock used for heartbeat-staleness; injected for testing.
        unreachable_warn_after: Heartbeat-staleness past which a node is flagged
            as at-risk of pruning.

    Returns:
        A mapping from string node id to its :class:`NodeHealth`, one entry per
        live node (``level`` is ``ok`` with no reasons for a healthy node), so
        the dashboard can render an indicator (or none) per topology node.
    """
    health: dict[str, NodeHealth] = {}
    for node_id, last_seen in live_nodes.items():
        reasons: list[NodeHealthReason] = []
        reasons.extend(_download_failure_reasons(downloads.get(node_id, ())))
        disk = node_disk.get(node_id)
        if disk is not None:
            disk_reason = _disk_reason(disk)
            if disk_reason is not None:
                reasons.append(disk_reason)
        unreachable = _unreachable_reason(last_seen, now, unreachable_warn_after)
        if unreachable is not None:
            reasons.append(unreachable)
        health[str(node_id)] = NodeHealth(
            level=_aggregate_level(reasons), reasons=reasons
        )
    return health
