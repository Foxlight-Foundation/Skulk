"""Tests for the warmup deadline watchdog.

A wedged Metal eval parks the runner at 0% CPU forever and previously left
it in ``RunnerWarmingUp`` indefinitely — silently blocking all dispatch on
the node (launch smoke, 2026-06-05). The watchdog bounds that failure by
hard-exiting the runner; these tests exercise the timing contract through
the ``on_timeout`` test seam (the production action is log + ``os._exit``,
which cannot run inside pytest).
"""

import threading
import time

import pytest

from exo.worker.runner.bootstrap import (
    WARMUP_DEADLINE_SECONDS_DEFAULT,
    deadline_watchdog,
    resolve_warmup_deadline_seconds,
)


def test_watchdog_fires_when_block_overruns() -> None:
    fired = threading.Event()

    with deadline_watchdog(0.05, "test block", on_timeout=fired.set):
        assert fired.wait(timeout=2.0), "watchdog did not fire on overrun"


def test_watchdog_does_not_fire_when_block_completes() -> None:
    fired = threading.Event()

    with deadline_watchdog(0.2, "test block", on_timeout=fired.set):
        pass  # completes immediately

    # Give the watchdog thread ample time to (incorrectly) fire.
    time.sleep(0.4)
    assert not fired.is_set(), "watchdog fired after the block completed"


def test_watchdog_does_not_fire_on_exception_exit() -> None:
    """Leaving the block via an exception still disarms the watchdog."""
    fired = threading.Event()

    with (
        pytest.raises(RuntimeError),
        deadline_watchdog(0.2, "test block", on_timeout=fired.set),
    ):
        raise RuntimeError("boom")

    time.sleep(0.4)
    assert not fired.is_set(), "watchdog fired after exceptional exit"


def test_resolver_returns_default_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_WARMUP_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("EXO_WARMUP_DEADLINE_SECONDS", raising=False)
    assert resolve_warmup_deadline_seconds() == WARMUP_DEADLINE_SECONDS_DEFAULT


def test_resolver_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKULK_WARMUP_DEADLINE_SECONDS", "42.5")
    assert resolve_warmup_deadline_seconds() == 42.5


@pytest.mark.parametrize("value", ["abc", "-5", "0"])
def test_resolver_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Garbage or non-positive overrides fall back to the default rather
    than disabling the watchdog or arming it with a nonsense deadline."""
    monkeypatch.setenv("SKULK_WARMUP_DEADLINE_SECONDS", value)
    assert resolve_warmup_deadline_seconds() == WARMUP_DEADLINE_SECONDS_DEFAULT
