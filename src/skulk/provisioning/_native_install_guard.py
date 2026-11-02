"""Standard-library-only guardian for a single owned native installer command."""

from __future__ import annotations

import os
import signal
import struct
import subprocess
import sys
import threading
import time


def main(arguments: list[str]) -> int:
    """Run one command in this private group until its controller drains it.

    Arguments contain the expected parent PID, completion pipe, optional build
    lock descriptor (-1 when absent), and command. Report its signed exit status
    through the dedicated pipe, then keep the leader and inherited lock alive.
    Parent loss kills the complete group, including installer descendants.
    """
    if len(arguments) < 4:
        return 2
    parent, completion, lock = (int(value) for value in arguments[:3])
    if parent <= 1 or os.getppid() != parent or os.getpgrp() != os.getpid():
        return 2

    def watch_parent() -> None:
        while os.getppid() == parent:
            time.sleep(0.05)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_parent, daemon=True).start()
    # A separate completion pipe keeps command output opaque. Keeping this
    # unreaped group leader alive also prevents lock release or PGID reuse in
    # the gap between command completion and the controller's descendant drain.
    try:
        command = subprocess.Popen(
            arguments[3:], pass_fds=(() if lock < 0 else (lock,))
        )
        status = command.wait()
        os.write(completion, struct.pack("!i", status))
        os.close(completion)
        while True:
            time.sleep(0.05)
    finally:
        # Parent loss can close the completion pipe just as the command exits.
        # A BrokenPipeError must not exit this leader before the parent watcher
        # has killed surviving descendants and released their inherited lock.
        os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
