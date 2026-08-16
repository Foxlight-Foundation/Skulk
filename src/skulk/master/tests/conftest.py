import pytest

import skulk.master.placement as placement_module
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from skulk.shared.types.topology import RDMAConnection, SocketConnection


@pytest.fixture(autouse=True)
def approve_synthetic_planner_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep placement fixtures focused on topology unless a test opts into trust."""
    def approval_not_required(_card: ModelCard) -> bool:
        return False

    monkeypatch.setattr(
        placement_module,
        "remote_code_approval_required",
        approval_not_required,
    )


def create_node_memory(memory: int, ram_total: int | None = None) -> MemoryUsage:
    # ram_total defaults to 2x available so the placement GPU working-set
    # ceiling (GPU_WORKING_SET_FRACTION * ram_total) stays above ram_available
    # and does not bind. Pass ram_total explicitly to exercise that ceiling.
    return MemoryUsage.from_bytes(
        ram_total=ram_total if ram_total is not None else memory * 2,
        ram_available=memory,
        swap_total=1000,
        swap_available=1000,
    )


def create_node_network() -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address=f"169.254.0.{i}")
            for i in range(10)
        ]
    )


def create_socket_connection(ip: int, sink_port: int = 1234) -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/169.254.0.{ip}/tcp/{sink_port}"),
    )


def create_rdma_connection(iface: int) -> RDMAConnection:
    return RDMAConnection(
        source_rdma_iface=f"rdma_en{iface}", sink_rdma_iface=f"rdma_en{iface}"
    )
