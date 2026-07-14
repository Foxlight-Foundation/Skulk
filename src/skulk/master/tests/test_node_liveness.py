"""Tests for explicit three-signal node liveness and prune evidence."""

from datetime import datetime, timedelta, timezone

from loguru import logger

from skulk.master.main import (
    NODE_HEARTBEAT_GAP_WARNING,
    NODE_LIVENESS_TIMEOUT,
    Master,
    compute_heartbeat_gap_nodes,
    compute_node_timeout_evidence,
    compute_timed_out_nodes,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

_NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
_STALE = _NOW - NODE_LIVENESS_TIMEOUT - timedelta(seconds=5)
_FRESH = _NOW - timedelta(seconds=1)
_NODE = NodeId("node-a")


def _timed_out(
    last_event_at: datetime,
    heartbeat_at: datetime | None,
    fallback_at: datetime | None,
) -> set[NodeId]:
    """Evaluate one node while keeping absent signal maps explicit."""
    return compute_timed_out_nodes(
        {_NODE: last_event_at},
        {_NODE: heartbeat_at} if heartbeat_at is not None else {},
        {_NODE: fallback_at} if fallback_at is not None else {},
        now=_NOW,
    )


def test_stale_last_event_with_fresh_heartbeat_is_alive() -> None:
    """The dedicated heartbeat is the normal steady-state liveness signal."""
    assert _timed_out(_STALE, _FRESH, None) == set()


def test_fresh_ordinary_telemetry_is_a_liveness_fallback() -> None:
    """A heartbeat defect cannot prune a node still publishing telemetry."""
    assert _timed_out(_STALE, _STALE, _FRESH) == set()


def test_all_available_signals_stale_times_out() -> None:
    """The node times out only after every signal exceeds the window."""
    assert _timed_out(_STALE, _STALE, _STALE) == {_NODE}


def test_no_telemetry_falls_back_to_last_logged_event() -> None:
    """A new node receives the normal timeout grace before telemetry arrives."""
    assert _timed_out(_FRESH, None, None) == set()
    assert _timed_out(_STALE, None, None) == {_NODE}


def test_fresh_last_logged_event_with_stale_telemetry_is_alive() -> None:
    """Control-plane activity remains valid fallback evidence of life."""
    assert _timed_out(_FRESH, _STALE, _STALE) == set()


def test_tz_naive_last_seen_does_not_crash() -> None:
    """Odd legacy snapshots are normalized before liveness comparison."""
    naive_stale = _STALE.replace(tzinfo=None)
    assert _timed_out(naive_stale, _FRESH, None) == set()
    assert _timed_out(naive_stale, None, None) == {_NODE}


def test_only_stale_nodes_selected_from_mixed_fleet() -> None:
    """Fresh heartbeat evidence protects only the node that sent it."""
    dead = NodeId("node-dead")
    timed_out = compute_timed_out_nodes(
        {_NODE: _STALE, dead: _STALE},
        {_NODE: _FRESH},
        {},
        now=_NOW,
    )
    assert timed_out == {dead}


def test_timeout_evidence_records_every_deciding_age() -> None:
    """A persisted NodeTimedOut can explain the master's decision later."""
    evidence = compute_node_timeout_evidence(
        {_NODE: _NOW - timedelta(seconds=40)},
        {_NODE: _NOW - timedelta(seconds=35)},
        {_NODE: _NOW - timedelta(seconds=50)},
        now=_NOW,
    )[_NODE]

    assert evidence.last_logged_event_age_seconds == 40
    assert evidence.heartbeat_age_seconds == 35
    assert evidence.fallback_telemetry_age_seconds == 50
    assert evidence.effective_age_seconds == 35
    assert evidence.timeout_seconds == NODE_LIVENESS_TIMEOUT.total_seconds()


def test_heartbeat_gap_warns_at_threshold() -> None:
    """The primary liveness channel is visible before the prune threshold."""
    just_fresh = _NOW - NODE_HEARTBEAT_GAP_WARNING + timedelta(milliseconds=1)
    at_threshold = _NOW - NODE_HEARTBEAT_GAP_WARNING

    assert (
        compute_heartbeat_gap_nodes(
            {_NODE: _FRESH}, {_NODE: just_fresh}, now=_NOW
        )
        == set()
    )
    assert compute_heartbeat_gap_nodes(
        {_NODE: _FRESH}, {_NODE: at_threshold}, now=_NOW
    ) == {_NODE}


def test_missing_heartbeat_warning_window_starts_at_last_event() -> None:
    """A node with no heartbeat receives grace but cannot remain silent forever."""
    assert compute_heartbeat_gap_nodes({_NODE: _FRESH}, {}, now=_NOW) == set()
    assert compute_heartbeat_gap_nodes(
        {_NODE: _NOW - NODE_HEARTBEAT_GAP_WARNING}, {}, now=_NOW
    ) == {_NODE}


def test_master_logs_heartbeat_gap_once_then_recovery() -> None:
    """Planning emits one visible warning and one recovery per transition."""
    master = Master.__new__(Master)
    master.state = State(
        last_seen={_NODE: _NOW - NODE_HEARTBEAT_GAP_WARNING}
    )
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._heartbeat_gap_warned_nodes = set()  # pyright: ignore[reportPrivateUsage]
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        master._report_heartbeat_gap_changes(now=_NOW)  # pyright: ignore[reportPrivateUsage]
        master._report_heartbeat_gap_changes(now=_NOW)  # pyright: ignore[reportPrivateUsage]
        assert sum("late or absent" in message for message in messages) == 1

        master._telemetry_view.node_last_heartbeat[_NODE] = _NOW  # pyright: ignore[reportPrivateUsage]
        master._report_heartbeat_gap_changes(now=_NOW)  # pyright: ignore[reportPrivateUsage]
        assert sum("recovered" in message for message in messages) == 1
    finally:
        logger.remove(sink)
