from collections.abc import AsyncIterator

import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.utils.info_gatherer import net_profile


async def _collect_reachable_targets(
    topology: Topology,
    self_node_id: NodeId,
    node_network: dict[NodeId, NodeNetworkInfo],
) -> list[tuple[str, NodeId]]:
    reachable_targets: list[tuple[str, NodeId]] = []
    reachable_iter: AsyncIterator[tuple[str, NodeId]] = net_profile.check_reachable(
        topology=topology,
        self_node_id=self_node_id,
        node_network=node_network,
    )
    async for reachable in reachable_iter:
        reachable_targets.append(reachable)
    return reachable_targets


@pytest.mark.anyio
async def test_check_reachable_skips_loopback_and_unspecified_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_node_id = NodeId("self")
    remote_node_id = NodeId("remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)
    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="lo0", ip_address="127.0.0.1"),
                NetworkInterfaceInfo(name="lo0", ip_address="::1"),
                NetworkInterfaceInfo(name="lo0", ip_address="0.0.0.0"),
                NetworkInterfaceInfo(name="lo0", ip_address="::"),
                NetworkInterfaceInfo(name="lo0", ip_address="localhost"),
                NetworkInterfaceInfo(name="en0", ip_address="192.168.0.117"),
                NetworkInterfaceInfo(
                    name="en7", ip_address="fe80::20:315a:c2e5:286b%en0"
                ),
            ]
        )
    }
    probed_targets: list[str] = []

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        _client: object,
    ) -> None:
        probed_targets.append(target_ip)
        out[expected_node_id].add(target_ip)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)

    reachable_targets = await _collect_reachable_targets(
        topology=topology,
        self_node_id=self_node_id,
        node_network=node_network,
    )

    assert probed_targets == ["192.168.0.117", "fe80::20:315a:c2e5:286b%en0"]
    assert reachable_targets == [
        ("192.168.0.117", remote_node_id),
        ("fe80::20:315a:c2e5:286b%en0", remote_node_id),
    ]
