"""Startup sweep for orphaned vLLM engine-core processes (#653).

vLLM spawns its ``EngineCore`` as a grandchild of the runner (runner ->
``vllm serve`` -> ``VLLM::EngineCore``). ``PR_SET_PDEATHSIG`` covers only the
direct ``vllm serve`` child, so when the node dies abruptly mid-teardown the
engine core can survive reparented to init while holding its full GPU
allocation; the next placement is then refused for VRAM the operator cannot
see. Observed live on 2026-07-22: a stale ``VLLM::EngineCore`` held ~73 GB
across a node restart until killed by hand.

The sweep runs once at worker startup and kills exactly that shape: a
process whose command line or name carries the vLLM engine-core marker AND
whose parent is init (pid 1). A healthy engine core always has a live
``vllm serve`` parent, and a manually launched ``vllm serve`` (which may
itself legitimately reparent to init under ``nohup``) is never matched, so
the rule cannot hit anything an operator still wants. Orphans get SIGKILL,
not SIGTERM: an engine core without its parent's ZMQ sockets is
unrecoverable garbage, and only process exit reliably releases the GPU
memory the placement math needs back.
"""

import os
import signal
from pathlib import Path

from loguru import logger

_ENGINE_CORE_MARKER = "VLLM::EngineCore"
# /proc/<pid>/comm is kernel-truncated to 15 characters, so the comm-side
# match uses the truncated marker rather than a broad "VLLM" prefix: an
# init-parented vLLM-adjacent helper with a title like "VLLMRouter" must
# never satisfy the sweep (PR #656 review).
_ENGINE_CORE_COMM_PREFIX = _ENGINE_CORE_MARKER[:15]


def find_orphaned_vllm_engine_pids(proc_root: Path = Path("/proc")) -> list[int]:
    """Return pids of vLLM engine-core processes reparented to init.

    Pure scan, no signalling; separated from :func:`sweep_orphaned_vllm_engines`
    so the matching rule is unit-testable against a fake ``/proc`` tree. On
    platforms without ``/proc`` (macOS) the scan returns empty and the sweep
    is a no-op; the vllm engine is Linux-oriented so nothing is lost.
    """
    orphans: list[int] = []
    if not proc_root.exists():
        return orphans
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # Both files decode with errors="replace": comm is arbitrary
            # bytes (any process can set a non-UTF-8 title), and a strict
            # decode raising UnicodeDecodeError past the OSError catch
            # would crash worker startup over an unrelated process.
            stat_text = (entry / "stat").read_bytes().decode(errors="replace")
            cmdline = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode(errors="replace")
            )
        except OSError:
            # Process vanished mid-scan or is unreadable; either way it is
            # not ours to reap.
            continue
        # /proc/<pid>/stat: "pid (comm) state ppid ..."; comm may itself
        # contain spaces or parens, so split at the LAST closing paren.
        fields_after_comm = stat_text.rsplit(")", 1)[-1].split()
        if len(fields_after_comm) < 2:
            continue
        try:
            parent_pid = int(fields_after_comm[1])
        except ValueError:
            continue
        if parent_pid != 1:
            continue
        # setproctitle rewrites both cmdline and comm; comm is truncated to
        # 15 chars ("VLLM::EngineCor"), so match the truncated marker there
        # and the full marker in cmdline.
        comm = stat_text.split("(", 1)[-1].rsplit(")", 1)[0]
        if _ENGINE_CORE_MARKER in cmdline or comm.startswith(
            _ENGINE_CORE_COMM_PREFIX
        ):
            orphans.append(int(entry.name))
    return orphans


def sweep_orphaned_vllm_engines() -> int:
    """Kill orphaned vLLM engine-core processes; return how many were killed.

    Called once at worker startup, before placement admission can observe
    their GPU usage as unexplained pressure. Each kill is logged loudly with
    the pid so a node that WAS wedged says why it recovered.
    """
    killed = 0
    for pid in find_orphaned_vllm_engine_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as error:
            logger.warning(
                f"failed to reap orphaned vLLM engine core pid {pid}: {error}"
            )
            continue
        logger.warning(
            f"reaped orphaned vLLM engine core pid {pid} (parent died without "
            "tearing it down; it was holding GPU memory invisibly, #653)"
        )
        killed += 1
    return killed
