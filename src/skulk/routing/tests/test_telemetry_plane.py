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
    """A successful publish ends the sustained window AND its rate limit.

    The next outage is a fresh incident: timed from its own start, and its
    first warning is not suppressed by a stale rate-limit stamp from the
    previous incident (PR #663 review).
    """
    observer = _observer()
    # Drive the first incident all the way to a warning.
    observer.record_no_peer_publish(now=0.0)
    assert observer.record_no_peer_publish(now=NO_PEER_WARNING_AFTER_SECONDS + 1.0)
    recovery = NO_PEER_WARNING_AFTER_SECONDS + 2.0
    observer.record_published(10, now=recovery)
    assert observer.no_peers_since is None
    assert observer.last_no_peer_warning_at is None
    # New outage: not yet sustained even though the first one started at 0,
    # and once sustained it warns even though the previous warning was
    # within the old rate-limit interval.
    assert not observer.record_no_peer_publish(now=recovery + 1.0)
    assert observer.record_no_peer_publish(
        now=recovery + 1.0 + NO_PEER_WARNING_AFTER_SECONDS + 1.0
    )


def test_transport_failure_breaks_no_peer_continuity() -> None:
    """A pressure failure ends the sustained window.

    Saturated per-peer queues imply peers exist, so a no-peer window
    claiming continuity across the failure would overstate it
    (PR #663 review).
    """
    observer = _observer()
    observer.record_no_peer_publish(now=0.0)
    observer.record_publish_failure()
    assert observer.no_peers_since is None
    assert not observer.record_no_peer_publish(
        now=NO_PEER_WARNING_AFTER_SECONDS + 1.0
    )


def test_no_warning_without_connected_peers() -> None:
    """A lone node never gets the mismatch warning.

    Single-node deployments hit no-peer outcomes as their normal state; the
    count stays honest but the warning gates on live swarm connections
    (PR #663 review). When the state persists after peers connect, the
    caller-side window restart (restart_no_peer_window on first connect)
    plus persistence gives the fresh connection its own grace period.
    """
    observer = _observer()
    observer.record_no_peer_publish(peers_connected=False, now=0.0)
    assert not observer.record_no_peer_publish(
        peers_connected=False, now=NO_PEER_WARNING_AFTER_SECONDS * 4
    )
    assert observer.no_peer_publishes == 2
    # First peer arrives: router restarts the window; a persisting no-peer
    # state then warns only after its own full grace period.
    observer.restart_no_peer_window()
    connect_at = NO_PEER_WARNING_AFTER_SECONDS * 4 + 1.0
    assert not observer.record_no_peer_publish(now=connect_at)
    assert observer.record_no_peer_publish(
        now=connect_at + NO_PEER_WARNING_AFTER_SECONDS + 1.0
    )
