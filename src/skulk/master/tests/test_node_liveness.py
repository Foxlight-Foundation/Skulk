"""Tests for the master's two-plane liveness predicate.

``compute_timed_out_nodes`` drives ``NodeTimedOut`` (membership prune plus
instance/task cleanup), so its semantics are critical-path: a stable node's
``last_seen`` goes cold by design (connectivity is change-gated), and the
telemetry receipt is the live signal. A node times out only when BOTH are
stale.
"""

from datetime import datetime, timedelta, timezone

from skulk.master.main import NODE_LIVENESS_TIMEOUT, compute_timed_out_nodes
from skulk.shared.types.common import NodeId

_NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
_STALE = _NOW - NODE_LIVENESS_TIMEOUT - timedelta(seconds=5)
_FRESH = _NOW - timedelta(seconds=1)
_NODE = NodeId("node-a")


def test_stale_last_seen_with_fresh_telemetry_is_alive() -> None:
    # The designed steady state after the connectivity change-gate: last_seen
    # cold, telemetry flowing. The node must NOT be timed out.
    timed_out = compute_timed_out_nodes(
        {_NODE: _STALE},
        {_NODE: _FRESH},
        now=_NOW,
    )
    assert timed_out == set()


def test_both_signals_stale_times_out() -> None:
    timed_out = compute_timed_out_nodes(
        {_NODE: _STALE},
        {_NODE: _STALE},
        now=_NOW,
    )
    assert timed_out == {_NODE}


def test_no_telemetry_falls_back_to_last_seen() -> None:
    # A just-appeared node with no telemetry yet: fresh last_seen keeps it
    # alive; stale last_seen (and still no telemetry) times it out.
    assert (
        compute_timed_out_nodes({_NODE: _FRESH}, {}, now=_NOW) == set()
    )
    assert (
        compute_timed_out_nodes({_NODE: _STALE}, {}, now=_NOW) == {_NODE}
    )


def test_fresh_last_seen_with_stale_telemetry_is_alive() -> None:
    # Events still flowing (e.g. an actual topology change just logged) keeps
    # the node alive even if telemetry lagged.
    timed_out = compute_timed_out_nodes(
        {_NODE: _FRESH},
        {_NODE: _STALE},
        now=_NOW,
    )
    assert timed_out == set()


def test_tz_naive_last_seen_does_not_crash() -> None:
    # A tz-naive last_seen (odd snapshot) compared against the tz-aware
    # telemetry stamp would raise inside max() and crash the plan loop; it
    # must be normalized. Covers both the alive and timed-out outcomes.
    naive_stale = _STALE.replace(tzinfo=None)
    assert (
        compute_timed_out_nodes({_NODE: naive_stale}, {_NODE: _FRESH}, now=_NOW)
        == set()
    )
    assert (
        compute_timed_out_nodes({_NODE: naive_stale}, {}, now=_NOW) == {_NODE}
    )


def test_only_stale_nodes_selected_from_mixed_fleet() -> None:
    dead = NodeId("node-dead")
    timed_out = compute_timed_out_nodes(
        {_NODE: _STALE, dead: _STALE},
        {_NODE: _FRESH},
        now=_NOW,
    )
    assert timed_out == {dead}
