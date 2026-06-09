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

    def record(self, key: K) -> bool:
        """Record a failure for ``key``; return ``True`` if the breaker tripped.

        Prunes failures older than the window before counting, so the return
        reflects only failures inside ``window_seconds`` of now.
        """
        now = self._clock()
        recent = [
            stamp
            for stamp in self._failures.get(key, [])
            if now - stamp < self._window_seconds
        ]
        recent.append(now)
        self._failures[key] = recent
        return len(recent) >= self._threshold

    def clear(self, key: K) -> None:
        """Forget all recorded failures for ``key`` (e.g. after giving up)."""
        self._failures.pop(key, None)
