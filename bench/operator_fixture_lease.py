"""Local relay watchdog: parent-pipe EOF and a monotonic expiry both stop it.

Kept independent of Skulk imports so runner crashes cannot leave the relay
waiting for a Python application cleanup callback. No provider APIs are used.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from types import FrameType


def supervise_local_process(
    command: Sequence[str], *, parent_descriptor: int, lifetime_seconds: float
) -> int:
    """Run a local child until pipe EOF, expiry, signal, or unexpected child exit.

    The caller owns the exact generated command. Output is discarded; teardown
    targets only the child created here. An unexpected child exit returns 1.
    """
    if not 0 < lifetime_seconds <= 7200:
        raise ValueError("fixture lease must be positive and at most two hours")
    stop = False

    def requested_stop(_signal: int, _frame: FrameType | None) -> None:
        nonlocal stop
        stop = True

    previous = signal.signal(signal.SIGTERM, requested_stop)
    process: subprocess.Popen[bytes] | None = None
    try:
        deadline = time.monotonic() + lifetime_seconds
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        while not stop:
            if process.poll() is not None:
                return 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            readable, _, _ = select.select(
                [parent_descriptor], [], [], min(remaining, 0.2)
            )
            if readable:
                # The parent writes nothing: either EOF or an explicit stop
                # byte ends the lease. Input cannot extend the deadline.
                os.read(parent_descriptor, 1)
                break
        return 0
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        signal.signal(signal.SIGTERM, previous)


def main(arguments: Sequence[str]) -> int:
    """Internal watchdog entry point; the parent supplies only generated paths."""
    if len(arguments) != 3:
        return 2
    binary, configuration, lifetime = arguments
    return supervise_local_process(
        (binary, "serve", configuration),
        parent_descriptor=sys.stdin.fileno(),
        lifetime_seconds=float(lifetime),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
