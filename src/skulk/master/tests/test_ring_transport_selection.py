# pyright: reportPrivateUsage=false
"""Ring hostfile transport-selection tests (#265).

First coverage for ``get_mlx_ring_hosts_by_node`` and the per-pair transport
ranking. Pins the operator-directed policy: Thunderbolt first whenever the
pair has a TB path, LAN next, and VPN/overlay (Tailscale) strictly last —
the overlay exists for reaching nodes from OUTSIDE the local network and may
be DERP-relayed, so it must never win against any local candidate (the
2026-06-10 incident rode the Dallas relay between two machines on the same
switch).
"""

import pytest

from skulk.master.placement_utils import (
    _find_ip_prioritised,
    _is_vpn_address,
    get_mlx_ring_hosts_by_node,
)
from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from skulk.shared.types.topology import Connection, Cycle, SocketConnection

_TB_IP = "169.254.10.2"
_LAN_IP = "192.168.0.20"
_VPN_IP = "100.79.166.20"


def _socket(ip: str) -> SocketConnection:
    return SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52416"))


def _pair_topology(
    node_a: NodeId, node_b: NodeId, ips_to_b: list[str], ips_to_a: list[str]
) -> Topology:
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    for ip in ips_to_b:
        topology.add_connection(
            Connection(source=node_a, sink=node_b, edge=_socket(ip))
        )
    for ip in ips_to_a:
        topology.add_connection(
            Connection(source=node_b, sink=node_a, edge=_socket(ip))
        )
    return topology


def _network(*entries: tuple[str, str]) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="enX", ip_address=ip, interface_type=kind)  # type: ignore[arg-type]
            for ip, kind in entries
        ]
    )


def test_vpn_address_detection():
    assert _is_vpn_address("100.64.0.1")
    assert _is_vpn_address("100.79.166.20")
    assert _is_vpn_address("100.127.255.254")
    assert not _is_vpn_address("100.63.255.255")  # below CGNAT range
    assert not _is_vpn_address("100.128.0.1")  # above CGNAT range
    assert not _is_vpn_address("192.168.0.20")
    assert not _is_vpn_address("169.254.10.2")
    assert _is_vpn_address("fd7a:115c:a1e0::1")  # Tailscale IPv6
    assert not _is_vpn_address("not-an-ip")


def test_thunderbolt_beats_lan():
    node_a, node_b = NodeId(), NodeId()
    topology = _pair_topology(node_a, node_b, [_LAN_IP, _TB_IP], [])
    node_network = {
        node_b: _network((_TB_IP, "thunderbolt"), (_LAN_IP, "ethernet"))
    }
    chosen = _find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)
    assert chosen == _TB_IP


def test_vpn_never_beats_lan():
    # The tailscale candidate's gossiped label is "unknown" (utun never
    # appears in networksetup output) — but the policy must hold even if a
    # label lies, so the address check is what demotes it.
    node_a, node_b = NodeId(), NodeId()
    topology = _pair_topology(node_a, node_b, [_VPN_IP, _LAN_IP], [])
    node_network = {
        node_b: _network((_VPN_IP, "unknown"), (_LAN_IP, "ethernet"))
    }
    chosen = _find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)
    assert chosen == _LAN_IP


def test_vpn_demoted_even_with_lying_label():
    # Address-based detection overrides the gossiped interface label: a CGNAT
    # address claiming to be "thunderbolt" still ranks last.
    node_a, node_b = NodeId(), NodeId()
    topology = _pair_topology(node_a, node_b, [_VPN_IP, _LAN_IP], [])
    node_network = {
        node_b: _network((_VPN_IP, "thunderbolt"), (_LAN_IP, "ethernet"))
    }
    chosen = _find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)
    assert chosen == _LAN_IP


def test_vpn_only_pair_still_connects():
    # Genuinely cross-network placement: the overlay is the only path and
    # must remain usable.
    node_a, node_b = NodeId(), NodeId()
    topology = _pair_topology(node_a, node_b, [_VPN_IP], [])
    node_network = {node_b: _network((_VPN_IP, "unknown"))}
    chosen = _find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)
    assert chosen == _VPN_IP


def test_ring_hostfile_shape_and_neighbor_selection():
    # 4-node ring: each rank gets self at 0.0.0.0, real IPs for both
    # neighbors (TB preferred), and inert TEST-NET-2 placeholders elsewhere.
    nodes = [NodeId() for _ in range(4)]
    topology = Topology()
    for node in nodes:
        topology.add_node(node)
    ips = {}
    for i in range(4):
        j = (i + 1) % 4
        ip_forward = f"169.254.{i}.{j}"
        ip_back = f"169.254.{j}.{i}"
        ips[(i, j)] = ip_forward
        ips[(j, i)] = ip_back
        topology.add_connection(
            Connection(source=nodes[i], sink=nodes[j], edge=_socket(ip_forward))
        )
        topology.add_connection(
            Connection(source=nodes[j], sink=nodes[i], edge=_socket(ip_back))
        )
    node_network: dict[NodeId, NodeNetworkInfo] = {}
    for i, node in enumerate(nodes):
        entries: list[tuple[str, str]] = [
            (ips[(j, i)], "thunderbolt") for j in range(4) if (j, i) in ips
        ]
        node_network[node] = _network(*entries)

    hosts_by_node = get_mlx_ring_hosts_by_node(
        selected_cycle=Cycle(node_ids=nodes),
        cycle_digraph=topology,
        ephemeral_port=50000,
        node_network=node_network,
    )

    for rank, node in enumerate(nodes):
        hosts = hosts_by_node[node]
        assert len(hosts) == 4
        assert hosts[rank].ip == "0.0.0.0"
        assert hosts[rank].port == 50000
        left, right = (rank - 1) % 4, (rank + 1) % 4
        for idx, host in enumerate(hosts):
            if idx == rank:
                continue
            if idx in (left, right):
                assert host.ip == ips[(rank, idx)]
                assert host.port == 50000
            else:
                assert host.ip == "198.51.100.1"
                assert host.port == 0


def test_ring_requires_neighbor_connectivity():
    nodes = [NodeId() for _ in range(3)]
    topology = Topology()
    for node in nodes:
        topology.add_node(node)
    # Only one edge of the 3-cycle exists — neighbor connectivity is missing.
    topology.add_connection(
        Connection(source=nodes[0], sink=nodes[1], edge=_socket(_LAN_IP))
    )
    with pytest.raises(ValueError, match="requires connectivity"):
        get_mlx_ring_hosts_by_node(
            selected_cycle=Cycle(node_ids=nodes),
            cycle_digraph=topology,
            ephemeral_port=50000,
            node_network={},
        )
