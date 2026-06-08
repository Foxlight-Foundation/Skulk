# pyright: reportPrivateUsage=false
"""Tests for the leaked-wired-memory diagnostics warning (#239).

Server-side counterpart of tests/preflight_mem.sh: a node with high wired
memory and no live runners has leaked memory from an abnormal Metal
termination, which otherwise surfaces only as unexplained placement 400s
and decode GPU-timeouts (#236).
"""

from skulk.api.main import _LEAKED_WIRED_THRESHOLD_BYTES, _leaked_wired_warning
from skulk.shared.types.diagnostics import RunnerSupervisorDiagnostics
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage


def _mem(wired_gb: float | None) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_bytes(24 * 2**30),
        ram_available=Memory.from_bytes(8 * 2**30),
        swap_total=Memory(),
        swap_available=Memory(),
        wired=Memory.from_bytes(int(wired_gb * 2**30)) if wired_gb is not None else None,
    )


def _runner(alive: bool) -> RunnerSupervisorDiagnostics:
    # The helper only reads .process_alive; model_construct skips the many
    # unrelated required fields of the full diagnostics model.
    return RunnerSupervisorDiagnostics.model_construct(process_alive=alive)


def test_high_wired_no_live_runners_flags() -> None:
    w = _leaked_wired_warning(_mem(13.2), [])
    assert w is not None and "leaked wired" in w.lower() and "reboot" in w.lower()


def test_high_wired_with_dead_supervisor_still_flags() -> None:
    # A retained-but-dead supervisor (the exact poisoned state — runners
    # killed) must not suppress the warning.
    assert _leaked_wired_warning(_mem(13.2), [_runner(alive=False)]) is not None


def test_high_wired_with_live_runner_does_not_flag() -> None:
    # Legitimate load: a live runner explains the high wired.
    assert _leaked_wired_warning(_mem(13.2), [_runner(alive=True)]) is None


def test_low_wired_does_not_flag() -> None:
    assert _leaked_wired_warning(_mem(2.0), []) is None


def test_wired_unavailable_does_not_flag() -> None:
    # Non-macOS (wired is None) — no signal, no false positive.
    assert _leaked_wired_warning(_mem(None), []) is None


def test_no_memory_reading_does_not_flag() -> None:
    assert _leaked_wired_warning(None, []) is None


def test_threshold_is_a_sane_floor() -> None:
    # ~5GB: comfortably above the ~2GB idle baseline, below the 13GB leak.
    assert 3 * 2**30 < _LEAKED_WIRED_THRESHOLD_BYTES < 10 * 2**30
