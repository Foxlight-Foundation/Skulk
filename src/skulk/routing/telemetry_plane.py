"""Aggregate observability for bounded, latest-value telemetry egress."""

from __future__ import annotations

import time
from collections.abc import Iterable

from skulk.shared.types.diagnostics import TelemetryPlaneDiagnostics

# A brief no-peer window is normal (startup, mesh churn); a SUSTAINED one on
# a node with live connections means telemetry is reaching nobody and the
# node is invisible to membership. Warn only after the state persists, then
# at most once per interval so a long outage stays one-line-per-minute loud
# rather than log-flooding.
NO_PEER_WARNING_AFTER_SECONDS = 30.0
NO_PEER_WARNING_INTERVAL_SECONDS = 60.0


class TelemetryPlaneObserver:
    """Track telemetry pressure without retaining node or model identifiers."""

    def __init__(self, *, admission_capacity: int, network_queue_capacity: int) -> None:
        """Create a zeroed process-local telemetry observer.

        Args:
            admission_capacity: Maximum distinct latest-value keys pending.
            network_queue_capacity: Maximum serialized packets queued for publish.
        """

        self.admission_capacity = admission_capacity
        self.network_queue_capacity = network_queue_capacity
        self.readings_offered = 0
        self.readings_coalesced = 0
        self.readings_dropped = 0
        self.readings_published = 0
        self.publish_failures = 0
        self.no_peer_publishes = 0
        self.bytes_published = 0
        self.max_queue_depth = 0
        self.last_successful_publish_at: float | None = None
        # Sustained no-peer tracking for the operator warning: when the first
        # no-peer outcome happened (None while healthy) and when the warning
        # last fired (rate limit).
        self.no_peers_since: float | None = None
        self.last_no_peer_warning_at: float | None = None

    def record_offered(self) -> None:
        """Record one reading offered by a local telemetry producer."""

        self.readings_offered += 1

    def record_coalesced(self) -> None:
        """Record replacement of one stale pending reading."""

        self.readings_coalesced += 1

    def record_dropped(self) -> None:
        """Record eviction of one pending key at the admission bound."""

        self.readings_dropped += 1

    def record_depth(self, *, pending: int, network_queue: int) -> None:
        """Update the high-water mark for aggregate bounded queue depth."""

        self.max_queue_depth = max(self.max_queue_depth, pending + network_queue)

    def record_published(self, size_bytes: int, *, now: float | None = None) -> None:
        """Record one successful isolated telemetry publish."""

        self.readings_published += 1
        self.bytes_published += size_bytes
        self.last_successful_publish_at = time.monotonic() if now is None else now
        # A successful publish proves peers exist again: end any sustained
        # no-peer window so the next one is timed from its own start.
        self.no_peers_since = None

    def record_publish_failure(self) -> None:
        """Record one rejected isolated telemetry publish."""

        self.publish_failures += 1

    def record_no_peer_publish(self, *, now: float | None = None) -> bool:
        """Record one publish that found no telemetry-protocol peers.

        Distinct from :meth:`record_publish_failure` (transport pressure):
        no-peers means the serialized packet had nowhere to go, which on a
        node with live connections is the signature of a wire-protocol
        mismatch — the node stays invisible to membership while looking
        healthy locally (#660).

        Returns:
            ``True`` when the caller should emit the rate-limited operator
            warning: the no-peer state has persisted past
            ``NO_PEER_WARNING_AFTER_SECONDS`` and no warning fired in the
            last ``NO_PEER_WARNING_INTERVAL_SECONDS``.
        """

        observed_at = time.monotonic() if now is None else now
        self.no_peer_publishes += 1
        if self.no_peers_since is None:
            self.no_peers_since = observed_at
        if observed_at - self.no_peers_since < NO_PEER_WARNING_AFTER_SECONDS:
            return False
        if (
            self.last_no_peer_warning_at is not None
            and observed_at - self.last_no_peer_warning_at
            < NO_PEER_WARNING_INTERVAL_SECONDS
        ):
            return False
        self.last_no_peer_warning_at = observed_at
        return True

    def snapshot(
        self,
        *,
        pending_enqueued_at: Iterable[float],
        pending_readings: int,
        network_queue_depth: int,
        now: float | None = None,
    ) -> TelemetryPlaneDiagnostics:
        """Return one immutable aggregate diagnostics snapshot.

        Args:
            pending_enqueued_at: Monotonic timestamps for pending latest values.
            pending_readings: Number of distinct pending latest-value keys.
            network_queue_depth: Serialized packets waiting for network publish.
            now: Monotonic clock value, injectable for deterministic tests.

        Returns:
            Current telemetry pressure and isolation counters.
        """

        observed_at = time.monotonic() if now is None else now
        pending_times = tuple(pending_enqueued_at)
        oldest_pending_age = (
            max(0.0, observed_at - min(pending_times)) if pending_times else None
        )
        last_publish_age = (
            max(0.0, observed_at - self.last_successful_publish_at)
            if self.last_successful_publish_at is not None
            else None
        )
        self.record_depth(
            pending=pending_readings,
            network_queue=network_queue_depth,
        )
        return TelemetryPlaneDiagnostics(
            transport="isolated_gossipsub",
            admission_capacity=self.admission_capacity,
            pending_readings=pending_readings,
            network_queue_capacity=self.network_queue_capacity,
            network_queue_depth=network_queue_depth,
            max_queue_depth=self.max_queue_depth,
            readings_offered=self.readings_offered,
            readings_coalesced=self.readings_coalesced,
            readings_dropped=self.readings_dropped,
            readings_published=self.readings_published,
            publish_failures=self.publish_failures,
            no_peer_publishes=self.no_peer_publishes,
            bytes_published=self.bytes_published,
            oldest_pending_age_seconds=oldest_pending_age,
            last_successful_publish_age_seconds=last_publish_age,
        )
