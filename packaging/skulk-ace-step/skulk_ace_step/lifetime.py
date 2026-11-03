"""Owned process-group lifetime for native work that may outlive Python timeouts."""

from __future__ import annotations

import os
import signal
import threading
import time


def terminate_owned_group() -> None:
    """Kill only this dedicated session, including lingering native threads/children."""
    if os.getpid() != os.getpgrp():
        raise RuntimeError("native music cannot terminate a foreign process group")
    os.killpg(os.getpgrp(), signal.SIGKILL)


def start_parent_guardian(parent_pid: int) -> None:
    """Require an actual parent and private session; kill this group if orphaned."""
    if parent_pid <= 1 or os.getppid() != parent_pid or os.getpid() != os.getpgrp():
        raise ValueError(
            "native music requires its actual supervisor and a private process session"
        )

    def reap() -> None:
        while os.getppid() == parent_pid:
            time.sleep(0.25)
        terminate_owned_group()

    threading.Thread(target=reap, name="native-music-owner", daemon=True).start()
