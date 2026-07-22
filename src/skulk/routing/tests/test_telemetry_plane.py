"""Unit tests for the telemetry-plane observer's no-peer accounting (#660)."""

from skulk.routing.telemetry_plane import (
    NO_PEER_WARNING_AFTER_SECONDS,
    NO_PEER_WARNING_INTERVAL_SECONDS,
    TelemetryPlaneObserver,
)


def _observer() -> TelemetryPlaneObserver:
    return TelemetryPlaneObserver(admission_capacity=256, network_queue_capacity=1)


def test_no_peer_publishes_counted_distinctly() -> None:
    """No-peer outcomes land in their own counter, not publish_failures.

    The distinction is the whole point: publish_failures means transport
    pressure; no_peer_publishes on a connected node means a wire mismatch
    ghost. Conflating them hides the one failure shape that leaves a node
    invisible to membership while looking healthy.
    """
    observer = _observer()
    observer.record_no_peer_publish(now=0.0)
    observer.record_no_peer_publish(now=1.0)
    assert observer.no_peer_publishes == 2
    assert observer.publish_failures == 0
    snapshot = observer.snapshot(
        pending_enqueued_at=(), pending_readings=0, network_queue_depth=0, now=2.0
    )
    assert snapshot.no_peer_publishes == 2


def test_no_peer_warning_fires_only_after_sustained_window() -> None:
    """The warning gates on persistence, then rate-limits.

    A brief no-peer blip (startup, mesh churn) must stay silent; a sustained
    state warns once per interval so a long outage is loud without flooding.
    """
    observer = _observer()
    assert not observer.record_no_peer_publish(now=0.0)
    assert not observer.record_no_peer_publish(
        now=NO_PEER_WARNING_AFTER_SECONDS - 1.0
    )
    first_warn_at = NO_PEER_WARNING_AFTER_SECONDS + 1.0
    assert observer.record_no_peer_publish(now=first_warn_at)
    # Inside the rate-limit interval: counted, not warned.
    assert not observer.record_no_peer_publish(now=first_warn_at + 5.0)
    # Past the interval: warns again while the state persists.
    assert observer.record_no_peer_publish(
        now=first_warn_at + NO_PEER_WARNING_INTERVAL_SECONDS + 1.0
    )


def test_successful_publish_resets_the_no_peer_window() -> None:
    """A successful publish ends the sustained window.

    The next no-peer outcome starts a fresh window rather than inheriting
    the old start time, so recovery followed by a new outage is timed
    honestly.
    """
    observer = _observer()
    observer.record_no_peer_publish(now=0.0)
    observer.record_published(10, now=5.0)
    assert observer.no_peers_since is None
    # New outage: not yet sustained even though the first one started at 0.
    assert not observer.record_no_peer_publish(now=6.0)
    assert not observer.record_no_peer_publish(
        now=6.0 + NO_PEER_WARNING_AFTER_SECONDS - 1.0
    )
    assert observer.record_no_peer_publish(
        now=6.0 + NO_PEER_WARNING_AFTER_SECONDS + 1.0
    )
