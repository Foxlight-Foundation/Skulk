"""Spawn-safe outer native guard using live OS facts, never captured Worker State.

The isolated native observer owns actual device allocations and its heartbeat.
This independent Core callback checks the child identity, host pressure and RSS
even if native library sensing stalls. It imports no native engine or Torch.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import final

import psutil
from pydantic import Field

from skulk.shared.types.native_music import (
    NativeMusicMemoryBudget,
    NativeMusicMemoryCapacity,
    native_music_admission_capacity,
    native_music_host_available_bytes,
)
from skulk.utils.pydantic_ext import FrozenModel


def sample_native_music_host_capacity() -> NativeMusicMemoryCapacity:
    """Read current host capacity using Core's Mach semantics with a bounded effect.

    File-backed cache is reclaimable on macOS. Use one consistent wired,
    anonymous and compressor snapshot, as ordinary Core admission does; fall
    back to psutil on unavailable/changed Mach output. Device facts are absent
    here because only the private native process can observe its actual GPU.
    """
    memory = psutil.virtual_memory()
    total, available = int(memory.total), int(memory.available)
    text: str | None = None
    if platform.system() == "Darwin":
        try:
            snapshot = subprocess.run(
                ["/usr/bin/vm_stat"],
                capture_output=True,
                check=False,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            snapshot = None
        if snapshot is not None and snapshot.returncode == 0:
            text = snapshot.stdout.decode("utf-8", errors="replace")
    return NativeMusicMemoryCapacity(
        sampled_at_unix_ms=time.time_ns() // 1_000_000,
        host_total_bytes=total,
        host_available_bytes=native_music_host_available_bytes(total, available, text),
    )


@final
class NativeMusicHostGuard(FrozenModel):
    """Portable local spawn argument containing policy, without locks or services.

    A fresh sample is taken in the runner process on every bounded guard poll.
    No availability or allocation is inferred from a previous loaded status.
    Unified RSS may alias GPU pages, so it uses the same simultaneous phase
    ceiling as the private native guard rather than adding RSS and GPU bytes.
    """

    budget: NativeMusicMemoryBudget = Field(
        description="Complete admitted phase peaks."
    )
    host_total_bytes: int = Field(
        gt=0, description="Physical host total at actual admission."
    )
    unified: bool = Field(
        description="Actual native device's observed shared-memory topology."
    )

    def __call__(self, process_id: int) -> str | None:
        """Refuse a foreign child, changed capacity, resident overrun or host pressure.

        The caller remains responsible for retiring its private native group.
        Device allocation and stale-heartbeat refusal remain independent in
        ResidentMusic and the native process. Unknown OS sensing fails closed.
        """
        try:
            process = psutil.Process(process_id)
            if process.ppid() != os.getpid():
                return "Native music live guard received a foreign child process"
            capacity = sample_native_music_host_capacity()
            if capacity.host_total_bytes != self.host_total_bytes:
                return "Native music physical host capacity changed; re-admit the model"
            normalized = native_music_admission_capacity(capacity)
            floor = capacity.host_total_bytes - normalized.host_total_bytes
            if capacity.host_available_bytes < floor:
                return "Native music stopped for current host memory pressure"
            ceiling = (
                self.budget.shared_peak_bytes
                if self.unified
                else self.budget.host_peak_bytes
            )
            if int(process.memory_info().rss) > ceiling:
                return "Native music process exceeded its admitted host or shared memory ceiling"
        except (OSError, psutil.Error, ValueError):
            return "Native music live host/process sensing is unavailable; inspect runner diagnostics"
        return None
