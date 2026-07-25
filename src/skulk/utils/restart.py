"""Utility for restarting the current Skulk process in-place.

Uses os.execv to replace the current process image with a fresh one.
Because exec replaces the image within the same PID, Metal/GPU allocations are
released by the OS. Open file descriptors that libraries mark inheritable are
reset to close-on-exec immediately before replacement so the new process can
bind fresh sockets.
"""

import os
import sys
import threading
from collections.abc import Iterable

import psutil
from loguru import logger

_restart_scheduled = False
_restart_lock = threading.Lock()
_CHILD_TERMINATION_TIMEOUT_SECONDS = 5.0
_CHILD_KILL_TIMEOUT_SECONDS = 1.0


def _iter_open_file_descriptors() -> Iterable[int]:
    """Yield currently open file descriptors when the platform exposes them."""
    for path in ("/dev/fd", "/proc/self/fd"):
        try:
            names = os.listdir(path)
        except OSError:
            continue
        for name in names:
            try:
                fd = int(name)
            except ValueError:
                continue
            yield fd
        return


def _mark_open_file_descriptors_close_on_exec() -> None:
    """Prevent inherited listener sockets from surviving the process exec."""
    for fd in _iter_open_file_descriptors():
        if fd <= 2:
            continue
        try:
            os.set_inheritable(fd, False)
        except OSError:
            # The fd may have closed after the directory snapshot.
            continue


def _terminate_child_processes() -> None:
    """Terminate and reap descendants before replacing the Skulk process.

    ``exec`` replaces only the current process. Runner, resource-tracker, and
    telemetry children otherwise survive under the replacement process and can
    retain model mappings or GPU allocations that no longer exist in Skulk
    state.
    """

    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error as exc:
        logger.warning(f"Could not enumerate child processes before restart: {exc}")
        return

    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            continue

    _, survivors = psutil.wait_procs(
        children,
        timeout=_CHILD_TERMINATION_TIMEOUT_SECONDS,
    )
    for child in survivors:
        try:
            child.kill()
        except psutil.Error:
            continue
    if survivors:
        psutil.wait_procs(survivors, timeout=_CHILD_KILL_TIMEOUT_SECONDS)


def schedule_restart(delay: float = 1.0) -> bool:
    """Schedule an in-place process restart after *delay* seconds.

    Returns True if a restart was scheduled, False if one is already pending.
    After the delay, replaces the current process via os.execv. If execv
    fails, the current process is left running and the guard is reset.
    """
    global _restart_scheduled
    with _restart_lock:
        if _restart_scheduled:
            return False
        _restart_scheduled = True

    def _do_restart() -> None:
        import time

        time.sleep(delay)
        try:
            # Use `python -m skulk` so restart works even when invoked via the
            # `skulk` console script (where sys.argv[0] is just "skulk", not a
            # valid Python file path). Preserve all original arguments.
            _terminate_child_processes()
            _mark_open_file_descriptors_close_on_exec()
            os.execv(sys.executable, [sys.executable, "-m", "skulk", *sys.argv[1:]])
        except Exception as exc:
            # If we can't exec the replacement, keep the current process alive
            global _restart_scheduled
            logger.exception(f"Failed to exec replacement process: {exc}")
            with _restart_lock:
                _restart_scheduled = False

    threading.Thread(target=_do_restart, daemon=True).start()
    return True
