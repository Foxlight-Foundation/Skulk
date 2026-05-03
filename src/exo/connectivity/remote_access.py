"""Remote access info helpers.

Aggregates local LAN and Tailscale connectivity into a single
``RemoteAccessInfo`` snapshot used by ``GET /v1/connectivity/remote-access``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

from pydantic import Field

from exo.connectivity.tailscale import query_tailscale_status
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.utils.pydantic_ext import FrozenModel


@final
class LocalAccess(FrozenModel):
    """Direct LAN access information for the local node.

    Attributes:
        ip: Preferred LAN IPv4 address, or ``None`` if unknown.
        port: API/dashboard port.
        url: ``http://{ip}:{port}`` when ``ip`` is known, otherwise ``None``.
    """

    ip: str | None = Field(description="Preferred LAN IPv4 address, or None if unknown.")
    port: int = Field(description="API/dashboard port.")
    url: str | None = Field(description="http://{ip}:{port} if ip is known, else None.")


@final
class TailscaleAccess(FrozenModel):
    """Tailscale overlay access information for the local node.

    Attributes:
        running: ``True`` when tailscaled is connected.
        ip: Tailscale IPv4 address (``100.x.x.x``), or ``None`` when not running.
        dns_name: Fully-qualified Tailscale MagicDNS name, or ``None``.
        port: API/dashboard port.
        url: ``http://{ip}:{port}`` when running, otherwise ``None``.
    """

    running: bool = Field(description="True when tailscaled is running and connected.")
    ip: str | None = Field(default=None, description="Tailscale IPv4 address (100.x.x.x).")
    dns_name: str | None = Field(default=None, description="Fully-qualified Tailscale MagicDNS name.")
    port: int = Field(description="API/dashboard port.")
    url: str | None = Field(default=None, description="http://{ip}:{port} if running, else None.")


@final
class RemoteAccessInfo(FrozenModel):
    """Aggregated remote access information for the local node.

    Combines LAN and Tailscale access details into a single snapshot so
    operator apps can determine the best URL to reach this node and
    generate a bookmark or QR code.

    Attributes:
        local: Direct LAN access details.
        tailscale: Tailscale overlay access details.
        preferred_url: Best URL for this node: Tailscale if running, otherwise LAN.
        operator_url: ``preferred_url`` with ``/operator`` appended — suitable for
            QR code generation so mobile users land directly on the operator panel.
    """

    local: LocalAccess = Field(description="Direct LAN access details.")
    tailscale: TailscaleAccess = Field(description="Tailscale overlay access details.")
    preferred_url: str | None = Field(
        description="Best URL: Tailscale URL if running, otherwise local LAN URL."
    )
    operator_url: str | None = Field(
        description="preferred_url + /operator. Suitable for QR code generation."
    )


async def build_remote_access_info(
    node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    port: int,
) -> RemoteAccessInfo:
    """Build a :class:`RemoteAccessInfo` snapshot for the local node.

    Args:
        node_id: The local node's ID (used to look up network interfaces from state).
        node_network: Current ``state.node_network`` mapping.
        port: The API/dashboard port this node is listening on.

    Returns:
        A :class:`RemoteAccessInfo` with LAN and Tailscale details populated.
    """

    local_ip: str | None = None
    network = node_network.get(node_id)
    if network:
        for iface in network.interfaces:
            addr = iface.ip_address
            if (
                addr
                and not addr.startswith("127.")
                and ":" not in addr
                and not addr.startswith("169.254.")
            ):
                local_ip = addr
                break

    local_url = f"http://{local_ip}:{port}" if local_ip else None
    local = LocalAccess(ip=local_ip, port=port, url=local_url)

    ts = await query_tailscale_status()
    ts_host = ts.dns_name or ts.self_ip
    ts_url = f"http://{ts_host}:{port}" if ts.running and ts_host else None
    tailscale = TailscaleAccess(
        running=ts.running,
        ip=ts.self_ip,
        dns_name=ts.dns_name,
        port=port,
        url=ts_url,
    )

    preferred_url = ts_url or local_url
    operator_url = f"{preferred_url}/operator" if preferred_url else None

    return RemoteAccessInfo(
        local=local,
        tailscale=tailscale,
        preferred_url=preferred_url,
        operator_url=operator_url,
    )
