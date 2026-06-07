"""Utility for restarting the current exo process in-place.

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

from loguru import logger

_restart_scheduled = False
_restart_lock = threading.Lock()


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
            # Use `python -m exo` so restart works even when invoked via the
            # `exo` console script (where sys.argv[0] is just "exo", not a
            # valid Python file path). Preserve all original arguments.
            _mark_open_file_descriptors_close_on_exec()
            os.execv(sys.executable, [sys.executable, "-m", "exo", *sys.argv[1:]])
        except Exception as exc:
            # If we can't exec the replacement, keep the current process alive
            global _restart_scheduled
            logger.exception(f"Failed to exec replacement process: {exc}")
            with _restart_lock:
                _restart_scheduled = False

    threading.Thread(target=_do_restart, daemon=True).start()
    return True
