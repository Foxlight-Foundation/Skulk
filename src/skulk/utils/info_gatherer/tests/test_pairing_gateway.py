"""Coverage for the phone-pairing gateway reading on the telemetry plane.

The node holding the cluster's relay route advertises it so other nodes'
dashboards can say where phone pairing is managed. Almost no node is the
gateway, so nothing is published while the role stays off.
"""

from collections.abc import Callable
from datetime import datetime, timezone

import anyio

from skulk.shared.types.common import NodeId
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.channels import Sender, channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
    NodePairingGateway,
)


def _pairing_gateway_only_gatherer(
    info_send: Sender[GatheredInfo],
    provider: Callable[[], bool] | None,
) -> InfoGatherer:
    """A gatherer whose only live monitor is the pairing-gateway advertiser."""

    return InfoGatherer(
        info_sender=info_send,
        heartbeat_poll_interval=None,
        interface_watcher_interval=None,
        misc_poll_interval=None,
        system_profiler_interval=None,
        memory_poll_rate=None,
        mactop_interval=None,
        thunderbolt_bridge_poll_interval=None,
        static_info_poll_interval=None,
        node_resources_poll_interval=None,
        rdma_ctl_poll_interval=None,
        disk_poll_interval=None,
        gpu_linux_poll_interval=None,
        capabilities_poll_interval=None,
        capability_nodes_poll_interval=None,
        pairing_gateway_provider=provider,
        pairing_gateway_poll_interval=0.02,
    )


async def test_gateway_role_is_published_while_active_and_once_when_it_ends() -> None:
    """Peers learn the role, then clear it after one inactive reading."""

    active = [True]
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _pairing_gateway_only_gatherer(info_send, lambda: active[0])
    received: list[GatheredInfo] = []

    async def collect() -> None:
        with info_recv as stream:
            async for info in stream:
                received.append(info)
                if len(received) == 2:
                    active[0] = False

    with anyio.move_on_after(0.4):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(gatherer._monitor_pairing_gateway)  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(collect)

    readings = [info for info in received if isinstance(info, NodePairingGateway)]
    assert readings[:2] == [NodePairingGateway(active=True)] * 2
    assert readings[2:] == [NodePairingGateway(active=False)]


async def test_a_node_that_is_not_the_gateway_publishes_nothing() -> None:
    """The common case adds no telemetry traffic."""

    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _pairing_gateway_only_gatherer(info_send, lambda: False)
    received: list[GatheredInfo] = []

    async def collect() -> None:
        with info_recv as stream:
            async for info in stream:
                received.append(info)

    with anyio.move_on_after(0.2):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(gatherer._monitor_pairing_gateway)  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(collect)

    assert received == []


async def test_monitor_is_inert_without_a_provider() -> None:
    """A node without an API never advertises the role."""

    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _pairing_gateway_only_gatherer(info_send, None)
    with anyio.fail_after(30):
        await gatherer._monitor_pairing_gateway()  # pyright: ignore[reportPrivateUsage]
    info_send.close()
    assert [info async for info in info_recv] == []


def test_view_tracks_and_prunes_pairing_gateways() -> None:
    """The view keeps advertising nodes, clears withdrawals, and forgets lost nodes."""

    view = TelemetryView()
    now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    gateway = NodeId("gateway-node")
    other = NodeId("other-node")

    view.apply(
        NodeTelemetry(node_id=gateway, info=NodePairingGateway(active=True)),
        received_at=now,
    )
    view.apply(
        NodeTelemetry(node_id=other, info=NodePairingGateway(active=True)),
        received_at=now,
    )
    assert view.node_pairing_gateways == {gateway, other}

    view.apply(
        NodeTelemetry(node_id=other, info=NodePairingGateway(active=False)),
        received_at=now,
    )
    assert view.node_pairing_gateways == {gateway}

    view.prune(gateway)
    assert view.node_pairing_gateways == set()


def test_pairing_gateway_reading_round_trips_on_the_wire() -> None:
    """The reading survives the telemetry envelope's JSON encoding."""

    message = NodeTelemetry(
        node_id=NodeId("gateway-node"), info=NodePairingGateway(active=True)
    )
    decoded = NodeTelemetry.model_validate_json(message.model_dump_json())
    assert decoded.info == NodePairingGateway(active=True)
