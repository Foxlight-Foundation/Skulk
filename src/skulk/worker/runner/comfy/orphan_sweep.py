"""Startup sweep for ComfyUI servers a dead runner left behind.

The runner starts ComfyUI with ``PR_SET_PDEATHSIG`` and tears its process
group down on shutdown, so an orphan needs a shape neither covers (a runner
killed while the kernel had not yet delivered the death signal, a platform
without ``prctl``). An orphaned server keeps the H3 weights resident on the
GPU, and the next placement is refused for memory nobody can see. The sweep
runs once at worker startup and kills exactly one shape: an init-parented
process whose command line launches ``main.py`` with a ``--user-directory``
under the node's own ComfyUI launch directory. Only Skulk passes that
directory, so an operator's own ComfyUI (any path, ``nohup`` or not) never
matches. Orphans get SIGKILL: nothing is waiting on them and only exit
releases the GPU memory.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

from loguru import logger

from skulk.shared.constants import SKULK_CACHE_HOME
from skulk.worker.runner.comfy.server import SKULK_LAUNCH_MARKER_DIR


def launch_marker_dir() -> Path:
    """The directory whose children mark a ComfyUI server as Skulk-launched."""
    return SKULK_CACHE_HOME / SKULK_LAUNCH_MARKER_DIR


def _is_orphaned_comfy(entry: Path, marker: str) -> bool:
    try:
        stat_text = (entry / "stat").read_bytes().decode(errors="replace")
        argv = (entry / "cmdline").read_bytes().split(b"\x00")
    except OSError:
        return False
    fields_after_comm = stat_text.rsplit(")", 1)[-1].split()
    if len(fields_after_comm) < 2:
        return False
    try:
        parent_pid = int(fields_after_comm[1])
    except ValueError:
        return False
    if parent_pid != 1:
        return False
    args = [part.decode(errors="replace") for part in argv]
    if not any(arg.endswith("main.py") for arg in args[:2]):
        return False
    for index, arg in enumerate(args):
        if arg == "--user-directory" and index + 1 < len(args):
            return args[index + 1].startswith(marker)
    return False


def find_orphaned_comfy_pids(proc_root: Path = Path("/proc"), marker_dir: Path | None = None) -> list[int]:
    """Pids of Skulk-launched ComfyUI servers reparented to init; pure scan."""
    marker = str((marker_dir or launch_marker_dir()).resolve()) + os.sep
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    return sorted(
        int(entry.name) for entry in entries if entry.name.isdigit() and _is_orphaned_comfy(entry, marker)
    )


def sweep_orphaned_comfy_servers() -> int:
    """Kill orphaned Skulk-launched ComfyUI servers; return how many died."""
    killed = 0
    proc_root = Path("/proc")
    marker = str(launch_marker_dir().resolve()) + os.sep
    for pid in find_orphaned_comfy_pids(proc_root):
        # Re-verify at kill time: the pid may have been recycled since the scan.
        if not _is_orphaned_comfy(proc_root / str(pid), marker):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as error:
            logger.warning(f"failed to reap orphaned ComfyUI server pid {pid}: {error}")
            continue
        logger.warning(f"reaped orphaned ComfyUI server pid {pid} (its runner died without tearing it down)")
        killed += 1
    return killed
