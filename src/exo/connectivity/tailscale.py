# pyright: reportAny=false
"""Tailscale connectivity helpers.

Queries the local ``tailscale status --json`` output to discover whether
tailscaled is running, the node's Tailscale IP, and tailnet membership.  The
module is intentionally thin — it does not manage Tailscale, only reads its
state.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, final

from loguru import logger
from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel


@final
class TailscaleStatus(FrozenModel):
    """Snapshot of the local node's Tailscale connectivity state.

    All fields except ``running`` are ``None`` when tailscaled is not running
    or is not installed.

    Attributes:
        running: ``True`` only when tailscaled reports ``BackendState ==
            "Running"``.
        self_ip: The node's Tailscale IPv4 address (``100.x.x.x`` range).
            ``None`` when not running or not assigned.
        hostname: The node's hostname as registered in the tailnet.
        dns_name: The node's fully-qualified Tailscale MagicDNS name, e.g.
            ``my-node.tailnet-abc.ts.net``.  Trailing dot stripped.
        tailnet: The tailnet name derived from ``dns_name``, e.g.
            ``tailnet-abc.ts.net``.  ``None`` when dns_name is absent or
            cannot be parsed.
        version: The tailscale client version string.
    """

    running: bool = Field(description="True when tailscaled reports BackendState == 'Running'.")
    self_ip: str | None = Field(default=None, description="Node's Tailscale IPv4 address (100.x.x.x).")
    hostname: str | None = Field(default=None, description="Node hostname in the tailnet.")
    dns_name: str | None = Field(default=None, description="Fully-qualified Tailscale MagicDNS name.")
    tailnet: str | None = Field(default=None, description="Tailnet name derived from dns_name.")
    version: str | None = Field(default=None, description="Tailscale client version string.")


def parse_status_json(raw: dict[str, Any]) -> TailscaleStatus:
    """Parse a ``tailscale status --json`` dict into a ``TailscaleStatus``.

    Handles missing keys gracefully — any absent field produces ``None``.
    """

    running = raw.get("BackendState") == "Running"

    self_node: dict[str, Any] = raw.get("Self") or {}
    tailscale_ips: list[str] = self_node.get("TailscaleIPs") or []
    self_ip = next(
        (ip for ip in tailscale_ips if ip.startswith("100.")),
        None,
    )

    hostname: str | None = self_node.get("HostName") or None

    raw_dns = self_node.get("DNSName") or None
    dns_name = raw_dns.rstrip(".") if raw_dns else None

    tailnet: str | None = None
    if dns_name:
        # "my-node.tailnet-abc.ts.net" → strip "my-node." prefix
        parts = dns_name.split(".", 1)
        tailnet = parts[1] if len(parts) == 2 else None

    version: str | None = raw.get("Version") or None

    return TailscaleStatus(
        running=running,
        self_ip=self_ip,
        hostname=hostname,
        dns_name=dns_name,
        tailnet=tailnet,
        version=version,
    )


async def query_tailscale_status() -> TailscaleStatus:
    """Query tailscaled and return the node's current Tailscale state.

    Runs ``tailscale status --json`` as a subprocess.  Returns a
    ``TailscaleStatus`` with ``running=False`` on any error: tailscale not
    installed, tailscaled not running, or the command timing out.
    """

    _not_running = TailscaleStatus(running=False)

    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "tailscale",
                "status",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            ),
            timeout=3.0,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3.0)
    except FileNotFoundError:
        logger.debug("tailscale binary not found; Tailscale not installed")
        return _not_running
    except TimeoutError:
        logger.debug("tailscale status timed out")
        return _not_running
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"tailscale status failed: {exc}")
        return _not_running

    if process.returncode != 0:
        logger.debug(f"tailscale status exited {process.returncode}")
        return _not_running

    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.debug(f"tailscale status JSON parse error: {exc}")
        return _not_running

    return parse_status_json(raw)
