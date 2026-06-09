import pytest

from skulk.utils.crash_window import CrashWindow


class _FakeClock:
    """Manually-advanced monotonic clock for deterministic window tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_trips_exactly_at_threshold_within_window():
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(3, 60.0, clock=clock)
    assert breaker.record("a") is False  # 1st
    assert breaker.record("a") is False  # 2nd
    assert breaker.record("a") is True  # 3rd -> trips


def test_trip_is_edge_triggered_not_level_triggered():
    # Once tripped, further failures inside the window must NOT re-trip, so a
    # caller's trip handler (e.g. sending DeleteInstance) runs once per loop.
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(3, 60.0, clock=clock)
    assert breaker.record("a") is False  # 1st
    assert breaker.record("a") is False  # 2nd
    assert breaker.record("a") is True  # 3rd -> crosses, trips once
    assert breaker.record("a") is False  # 4th -> still tripped, no re-fire
    assert breaker.record("a") is False  # 5th -> still tripped, no re-fire


def test_latch_releases_after_window_drains_then_retrips():
    # When the window drains below the threshold the latch resets, so a
    # genuinely fresh crash loop trips again.
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(2, 60.0, clock=clock)
    assert breaker.record("a") is False  # t=0, count 1
    assert breaker.record("a") is True  # t=0, count 2 -> trips
    assert breaker.record("a") is False  # t=0, count 3 -> latched, no re-fire
    clock.now = 100.0  # all earlier failures now outside the 60s window
    assert breaker.record("a") is False  # fresh count 1, latch released
    assert breaker.record("a") is True  # fresh count 2 -> trips again


def test_threshold_one_trips_each_time_after_window_drains():
    # threshold == 1 is the edge case for latch release: every isolated failure
    # (outside the prior window) must re-trip, but a burst still trips only once.
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(1, 60.0, clock=clock)
    assert breaker.record("a") is True  # 1st failure -> trips
    assert breaker.record("a") is False  # same instant, still latched
    clock.now = 100.0  # prior failure aged out
    assert breaker.record("a") is True  # fresh isolated failure -> trips again


def test_fresh_crossing_after_partial_drain_retrips():
    # threshold=3, failures at t=0,1,2 trip at t=2. At t=60.5 the t=0 stamp has
    # aged out, so the pruned count is 2 (< threshold) and the latch must release
    # BEFORE the new failure brings the count back to 3 — a genuine fresh edge.
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(3, 60.0, clock=clock)
    clock.now = 0.0
    assert breaker.record("a") is False
    clock.now = 1.0
    assert breaker.record("a") is False
    clock.now = 2.0
    assert breaker.record("a") is True  # trips
    clock.now = 60.5  # t=0 now outside the 60s window; t=1,t=2 remain (count 2)
    assert breaker.record("a") is True  # 2 -> 3 is a fresh crossing


def test_failures_outside_window_are_forgotten():
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(3, 60.0, clock=clock)
    breaker.record("a")  # t=0
    clock.now = 30.0
    breaker.record("a")  # t=30
    clock.now = 70.0  # t=0 now older than the 60s window
    # only t=30 and t=70 remain in-window -> 2 < 3 -> no trip
    assert breaker.record("a") is False


def test_keys_are_independent():
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(2, 60.0, clock=clock)
    assert breaker.record("a") is False
    assert breaker.record("b") is False
    assert breaker.record("a") is True
    assert breaker.record("b") is True


def test_clear_resets_a_key():
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(2, 60.0, clock=clock)
    breaker.record("a")
    breaker.clear("a")
    assert breaker.record("a") is False  # count restarted at 1


def test_retain_drops_dead_keys_and_keeps_live_ones():
    clock = _FakeClock()
    breaker: CrashWindow[str] = CrashWindow(2, 60.0, clock=clock)
    breaker.record("live")
    breaker.record("live")  # trips -> latched
    breaker.record("dead")  # one failure recorded for a key about to disappear

    breaker.retain({"live"})  # "dead" is no longer a live key

    # "dead" was fully forgotten: it starts counting from scratch.
    assert breaker.record("dead") is False
    # "live" kept both its history and its latch (still tripped, no re-fire).
    assert breaker.record("live") is False


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        CrashWindow(0, 60.0)
