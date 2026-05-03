"""Startup integrity checks run before Skulk's components come up.

The cluster is event-sourced and most "stale state" recovers naturally:

* The event log uses per-record framing, so a partial-write at process death
  is detected and dropped at replay (`exo.shared.event_log`).
* Runner subprocesses die with the parent, so MLX/Metal heap state is
  reclaimed by the kernel.
* libp2p reconnects from scratch on every startup; there is no on-disk peer
  state that needs cleanup.

What does *not* recover automatically is socket reuse: when Skulk is killed
hard (kernel OOM, `kill -9`, sudden power loss with networking still up via
WoL, or a fast crash-and-relaunch via systemd/launchd), the API port can sit
in `TIME_WAIT` for tens of seconds and the new process's bind fails with a
confusing "address already in use" error.

This module preflights the API port and produces a clear actionable log line
plus a non-zero exit, which is the contract systemd/launchd want for
restart-with-backoff to do its job.
"""

from __future__ import annotations

import errno
import socket
import sys
from typing import Final

from loguru import logger

# Brief grace before failing — covers the common case of a fast restart where
# the previous process's TIME_WAIT entry is about to clear. Longer than this
# and we want the supervisor (systemd/launchd) to take over with its own
# backoff so the operator gets honest feedback instead of a wedged "starting"
# state.
_API_PORT_GRACE_SECONDS: Final[float] = 5.0


def preflight_api_port(api_port: int) -> None:
    """Verify the API port is bindable; on failure log and exit non-zero.

    Calls `socket(...).bind(("0.0.0.0", api_port))` with `SO_REUSEADDR`
    matching what the actual API server uses, so the check has the same
    bind semantics as the eventual production listener. The socket is
    closed immediately — this is a probe, not a reservation.
    """

    deadline_attempts = max(1, int(_API_PORT_GRACE_SECONDS))
    last_error: OSError | None = None

    for attempt in range(deadline_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", api_port))
            return
        except OSError as exception:
            last_error = exception
            if exception.errno == errno.EADDRINUSE and attempt + 1 < deadline_attempts:
                # Common in supervisor-driven restarts. Sleep a beat and retry
                # before escalating to the supervisor's own backoff.
                import time

                time.sleep(1.0)
                continue
            break

    assert last_error is not None
    logger.critical(
        f"API port {api_port} is not bindable: {last_error}. "
        f"This usually means a previous Skulk instance is still releasing the port. "
        f"Exiting; the service supervisor (systemd/launchd) will retry with backoff."
    )
    sys.exit(75)  # EX_TEMPFAIL — supervisor should treat this as transient.
