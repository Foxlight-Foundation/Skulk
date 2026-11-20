"""Actual outer host sensing and child identity, plus portable Mach parity."""

import os
import pickle
import subprocess
import sys
from typing import cast

import pytest

from skulk.shared.types.native_music import (
    NativeMusicMemoryBudget,
    native_music_host_available_bytes,
)
from skulk.shared.types.profiling import gpu_wireable_memory_bytes, parse_vm_stat_output
from skulk.worker.runner.ace_step import resources
from skulk.worker.runner.ace_step.resources import (
    NativeMusicHostGuard,
    sample_native_music_host_capacity,
)

MACH = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 200.
Pages wired down: 300.
Anonymous pages: 400.
Pages occupied by compressor: 50.
File-backed pages: 9000.
"""


def test_portable_native_mach_accounting_matches_ordinary_core() -> None:
    """Downloaded file cache must not veto a model Core just admitted."""
    categories = parse_vm_stat_output(MACH)
    assert categories is not None
    total = 16 * 1024**3
    expected = gpu_wireable_memory_bytes(total, categories)
    assert native_music_host_available_bytes(total, 1024, MACH) == expected
    assert expected > 1024
    for broken in (None, "changed format", MACH.replace("Anonymous pages", "unknown")):
        assert native_music_host_available_bytes(total, 1024, broken) == 1024
    assert native_music_host_available_bytes(1024, 1024, MACH) == 0


def test_spawn_safe_guard_observes_real_child_and_current_host() -> None:
    """The local process argument contains no captured Worker or native runtime."""
    capacity = sample_native_music_host_capacity()
    guard = NativeMusicHostGuard(
        budget=NativeMusicMemoryBudget(
            host_peak_bytes=2 * 1024**3,
            device_peak_bytes=0,
            shared_peak_bytes=2 * 1024**3,
        ),
        host_total_bytes=capacity.host_total_bytes,
        unified=False,
    )
    restored = cast(object, pickle.loads(pickle.dumps(guard)))
    assert isinstance(restored, NativeMusicHostGuard) and restored == guard
    with subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as child:
        try:
            assert guard(child.pid) is None
            assert "foreign" in (guard(os.getpid()) or "")
        finally:
            child.terminate()
            child.wait(timeout=5)
    assert "unavailable" in (guard(child.pid) or "")


@pytest.mark.parametrize("fault", ["pressure", "capacity", "resident"])
def test_outer_guard_refuses_live_os_or_residency_violation(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Loaded status never excuses physical pressure or a changed host total."""
    observed = sample_native_music_host_capacity()
    guard = NativeMusicHostGuard(
        budget=NativeMusicMemoryBudget(
            host_peak_bytes=1 if fault == "resident" else 2 * 1024**3,
            device_peak_bytes=0,
            shared_peak_bytes=1 if fault == "resident" else 2 * 1024**3,
        ),
        host_total_bytes=observed.host_total_bytes,
        unified=False,
    )
    if fault == "pressure":
        observed = observed.model_copy(update={"host_available_bytes": 0})
    elif fault == "capacity":
        observed = observed.model_copy(
            update={"host_total_bytes": observed.host_total_bytes + 4096}
        )
    monkeypatch.setattr(
        resources, "sample_native_music_host_capacity", lambda: observed
    )
    with subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as child:
        try:
            # Wait for the child interpreter's own allocation before the RSS case.
            if fault == "resident":
                import time

                import psutil

                deadline = time.monotonic() + 5
                while psutil.Process(child.pid).memory_info().rss <= 1:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            assert guard(child.pid) is not None
        finally:
            child.terminate()
            child.wait(timeout=5)
