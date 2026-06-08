"""macOS Local Network Privacy detection.

macOS 15 (Sequoia) and 26 gate access to the *local network* — RFC-1918
subnets (``192.168.x``, ``10.x``), link-local, multicast, and the Thunderbolt
bridge — behind a per-application privacy permission. A process that has not
been granted Local Network access receives ``EHOSTUNREACH`` (errno 65) the
moment it tries to connect to a local-subnet peer, while internet-routed and
VPN traffic (e.g. Tailscale's ``utun`` interface) are unaffected.

The failure is silent and misleading: cluster discovery over Ethernet or
Thunderbolt simply never connects, with no error a user would recognise. This
module probes for the condition at startup so Skulk can tell the operator to
grant the permission instead of failing mysteriously.

The permission can only be granted from the GUI (System Settings → Privacy &
Security → Local Network, or the one-time prompt macOS shows on first access).
There is no supported command-line grant, so detection + a clear message is the
best we can do programmatically.
"""

from __future__ import annotations

import errno
import platform
import socket
import subprocess
from typing import Literal

# Discard protocol port (RFC 863): almost always closed, so a *reachable* host
# answers with a TCP reset (ECONNREFUSED) rather than accepting a connection.
_PROBE_PORT = 9
_PROBE_TIMEOUT_SECONDS = 2.0

LocalNetworkStatus = Literal["ok", "blocked", "unknown"]


def _default_gateway_ipv4() -> str | None:
    """Return the IPv4 default-gateway address, or ``None`` if undiscoverable.

    The gateway is a guaranteed local-subnet host, which makes it a reliable
    target for distinguishing a Local Network Privacy denial (EHOSTUNREACH)
    from a normal closed port (ECONNREFUSED).
    """
    try:
        completed = subprocess.run(
            ["route", "-n", "get", "default"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("gateway:"):
            gateway = line.split(":", 1)[1].strip()
            # Guard against an IPv6 / link-local gateway sneaking in.
            return gateway if gateway and ":" not in gateway else None
    return None


def check_local_network_access() -> LocalNetworkStatus:
    """Best-effort probe for a macOS Local Network Privacy denial.

    Returns:
        ``"blocked"`` if a local-subnet connect fails with ``EHOSTUNREACH`` (the
        Local Network Privacy signature); ``"ok"`` if the local network is
        reachable; ``"unknown"`` when the check does not apply or is
        inconclusive (non-macOS, no IPv4 gateway, probe error).

    The probe is a single short-timeout TCP connect to the default gateway's
    discard port. A denied process is rejected at the ``connect`` syscall with
    ``EHOSTUNREACH``; a permitted process reaches the host and gets
    ``ECONNREFUSED`` (or a timeout), both of which mean the local network is
    usable.
    """
    if platform.system() != "Darwin":
        return "unknown"

    gateway = _default_gateway_ipv4()
    if gateway is None:
        return "unknown"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_PROBE_TIMEOUT_SECONDS)
    try:
        sock.connect((gateway, _PROBE_PORT))
        return "ok"  # an open connection means the local network is reachable
    except OSError as exc:
        if exc.errno == errno.EHOSTUNREACH:
            return "blocked"
        # ECONNREFUSED / ETIMEDOUT / etc.: the host was reachable, so the
        # local network is not blocked — only this port is.
        return "ok"
    finally:
        sock.close()


LOCAL_NETWORK_DENIED_MESSAGE = (
    "macOS Local Network access appears to be DENIED for this process. Skulk "
    "cannot reach peers on your local network or Thunderbolt bridge, so the "
    "cluster will not form over Ethernet/Thunderbolt (a Tailscale overlay, if "
    "present, is exempt and still works). Grant access: System Settings → "
    "Privacy & Security → Local Network → enable the app you launched Skulk "
    "from (e.g. Terminal), then restart Skulk. See "
    "website/docs/thunderbolt-clustering.md for details."
)
