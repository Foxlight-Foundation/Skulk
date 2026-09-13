"""Coverage for the capability-node summary monitor (topology satellites).

Publication discipline matches the tag monitor: quiet while the host runs no
capability node, a reading on every change, one empty reading after the last
withdrawal, plus a periodic republish of a non-empty snapshot so late joiners
learn it on a plane without replay.
"""

from collections.abc import Callable

import anyio

from skulk.shared.types.capability_nodes import CapabilityNodeSummary
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
    NodeCapabilityNodes,
)


def _summary(status: str = "ready") -> CapabilityNodeSummary:
    return CapabilityNodeSummary.model_validate(
        {
            "plugin_id": "foxlight.video-studio",
            "node_id": "studio",
            "bundle_id": "foxlight.video-studio",
            "version": "1.0.0",
            "status": status,
            "owner_available": True,
        }
    )


def _nodes_only_gatherer(
    info_send: Sender[GatheredInfo],
    provider: Callable[[], tuple[CapabilityNodeSummary, ...]] | None,
    republish_interval: float = 30,
) -> InfoGatherer:
    """A gatherer whose only live monitor is the capability-node publisher."""
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
        capability_nodes_provider=provider,
        capability_nodes_poll_interval=0.02,
        capability_nodes_republish_interval=republish_interval,
    )


async def _collect(
    gatherer: InfoGatherer, info_recv: Receiver[GatheredInfo], budget: float
) -> list[tuple[CapabilityNodeSummary, ...]]:
    received: list[tuple[CapabilityNodeSummary, ...]] = []

    async def collect() -> None:
        with info_recv as stream:
            async for info in stream:
                assert isinstance(info, NodeCapabilityNodes)
                received.append(info.nodes)

    with anyio.move_on_after(budget):
        async with anyio.create_task_group() as tg:
            tg.start_soon(gatherer._monitor_capability_nodes)  # pyright: ignore[reportPrivateUsage]
            tg.start_soon(collect)
    return received


async def test_monitor_publishes_on_change_only() -> None:
    ready = (_summary(),)
    degraded = (_summary("degraded"),)
    snapshots = iter([ready, ready, ready, degraded, degraded, degraded])
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _nodes_only_gatherer(info_send, lambda: next(snapshots, degraded))
    received = await _collect(gatherer, info_recv, 0.3)
    assert received == [ready, degraded]


async def test_monitor_stays_quiet_without_nodes() -> None:
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _nodes_only_gatherer(info_send, lambda: ())
    assert await _collect(gatherer, info_recv, 0.2) == []


async def test_monitor_disabled_without_provider() -> None:
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _nodes_only_gatherer(info_send, None)
    with anyio.fail_after(30):
        await gatherer._monitor_capability_nodes()  # pyright: ignore[reportPrivateUsage]
    info_send.close()
    assert [info async for info in info_recv] == []


async def test_monitor_publishes_empty_once_after_withdrawal() -> None:
    ready = (_summary(),)
    empty: tuple[CapabilityNodeSummary, ...] = ()
    snapshots = iter([ready, empty, empty, empty])
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _nodes_only_gatherer(info_send, lambda: next(snapshots, empty))
    received = await _collect(gatherer, info_recv, 0.3)
    assert received == [ready, empty]


async def test_monitor_republishes_unchanged_snapshot_for_late_joiners() -> None:
    ready = (_summary(),)
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _nodes_only_gatherer(info_send, lambda: ready, republish_interval=0.1)
    received = await _collect(gatherer, info_recv, 0.35)
    assert len(received) >= 3
    assert all(nodes == ready for nodes in received)
