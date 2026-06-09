"""Rolling-window failure counter for per-key circuit breakers.

Used by the worker to stop relaunching an instance whose runner keeps failing:
unbounded relaunch of a doomed (e.g. OOM-on-load) runner compounds damage,
because each abnormal Metal termination can leak wired GPU memory reclaimable
only by reboot. The window forgets old failures, so a long-lived instance that
fails once an hour never trips — only a tight crash loop does.
"""

import time
from collections.abc import Callable
from typing import Generic, TypeVar

K = TypeVar("K")


class CrashWindow(Generic[K]):
    """Tracks recent failures per key and trips after ``threshold`` within
    ``window_seconds``.

    The monotonic clock is injectable so the breaker is deterministically
    testable without sleeping.
    """

    def __init__(
        self,
        threshold: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._clock = clock
        self._failures: dict[K, list[float]] = {}
        # Keys currently latched as tripped. The trip is edge-triggered: once a
        # key crosses the threshold it returns True exactly once and stays
        # latched (returning False) for every further failure while it remains
        # at/above the threshold, so callers don't re-run trip side effects
        # (e.g. sending DeleteInstance repeatedly). The latch releases only when
        # the window drains back below the threshold, letting a genuinely fresh
        # crash loop trip again.
        self._tripped: set[K] = set()

    def record(self, key: K) -> bool:
        """Record a failure for ``key``; return ``True`` only on the edge where
        the in-window count *crosses* the threshold.

        Prunes failures older than the window before counting, so the count
        reflects only failures inside ``window_seconds`` of now. Returns ``True``
        the first time the count reaches ``threshold`` and ``False`` on
        subsequent failures while it stays at/above the threshold — so the
        caller's trip handler runs once per crash loop, not once per failure.
        Once the window drains below the threshold the latch resets and a new
        loop can trip again.
        """
        now = self._clock()
        recent = [
            stamp
            for stamp in self._failures.get(key, [])
            if now - stamp < self._window_seconds
        ]
        recent.append(now)
        self._failures[key] = recent
        if len(recent) >= self._threshold:
            if key in self._tripped:
                return False  # already tripped this loop; edge already fired
            self._tripped.add(key)
            return True
        # Window drained below threshold — release the latch so a fresh loop
        # (threshold failures within a new window) can trip again.
        self._tripped.discard(key)
        return False

    def clear(self, key: K) -> None:
        """Forget all recorded failures and the trip latch for ``key``."""
        self._failures.pop(key, None)
        self._tripped.discard(key)
