"""Keep a macOS audio.cpp sidecar bound to the lifetime of its runner."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from types import FrameType


def main(arguments: list[str]) -> int:
    """Run one sidecar and end its process group when the runner disappears.

    Args:
        arguments: Runner PID followed by the pinned server command.

    Returns:
        The server exit code, or zero after an orderly parent-loss cleanup.
    """
    if len(arguments) < 2:
        return 2
    runner_pid = int(arguments[0])
    # The runner may die while Python starts; do not launch an orphan then.
    if os.getppid() != runner_pid:
        return 1

    stopping = False

    def request_stop(_number: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    sidecar = subprocess.Popen(arguments[1:], start_new_session=True)  # noqa: S603
    try:
        while not stopping and os.getppid() == runner_pid:
            exit_code = sidecar.poll()
            if exit_code is not None:
                return exit_code
            time.sleep(0.1)
        return 0
    finally:
        if sidecar.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(sidecar.pid, signal.SIGTERM)
            try:
                sidecar.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(sidecar.pid, signal.SIGKILL)
                sidecar.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
