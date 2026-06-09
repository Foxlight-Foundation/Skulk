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


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        CrashWindow(0, 60.0)
