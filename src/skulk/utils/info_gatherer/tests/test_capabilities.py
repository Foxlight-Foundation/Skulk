"""Coverage for the capability-advertise monitor (fabric-citizenship Phase 1).

The gatherer publishes extension-advertised capability tags onto the telemetry
plane, but only while the advertised set is non-empty: the overwhelming majority
of nodes advertise nothing, and an empty reading every poll would be pure gossip
volume on a plane engineered to stay quiet (#279).
"""

import anyio

from skulk.utils.channels import Sender, channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
    NodeCapabilities,
)


def _capabilities_only_gatherer(
    info_send: Sender[GatheredInfo], provider: object
) -> InfoGatherer:
    """A gatherer whose only live monitor is the capability advertiser."""
    return InfoGatherer(
        info_sender=info_send,
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
        capabilities_provider=provider,  # pyright: ignore[reportArgumentType]
        capabilities_poll_interval=0.02,
    )


async def test_monitor_publishes_advertised_capabilities() -> None:
    # A non-empty provider yields a NodeCapabilities reading carrying the tags.
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _capabilities_only_gatherer(
        info_send, lambda: frozenset({"memory", "search"})
    )
    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(gatherer._monitor_capabilities)  # pyright: ignore[reportPrivateUsage]
            with info_recv as stream:
                async for info in stream:
                    assert isinstance(info, NodeCapabilities)
                    assert info.capabilities == frozenset({"memory", "search"})
                    break
            tg.cancel_scope.cancel()


async def test_monitor_stays_quiet_when_nothing_advertised() -> None:
    # An empty provider publishes nothing: the receiver times out with no item.
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _capabilities_only_gatherer(info_send, frozenset)
    received: list[GatheredInfo] = []

    async def collect() -> None:
        with info_recv as stream:
            async for info in stream:
                received.append(info)

    with anyio.move_on_after(0.2):
        async with anyio.create_task_group() as tg:
            tg.start_soon(gatherer._monitor_capabilities)  # pyright: ignore[reportPrivateUsage]
            tg.start_soon(collect)

    assert received == []


async def test_monitor_disabled_without_provider() -> None:
    # No provider (the common case: no capability-advertising extension) => the
    # monitor returns immediately and never sends.
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _capabilities_only_gatherer(info_send, None)
    with anyio.fail_after(30):
        await gatherer._monitor_capabilities()  # pyright: ignore[reportPrivateUsage]
    info_send.close()
    received = [info async for info in info_recv]
    assert received == []


async def test_monitor_publishes_empty_once_after_withdrawal() -> None:
    # Provider liveness (fabric-citizenship Phase 2a): when the advertised set
    # transitions non-empty -> empty, ONE empty reading is published so peers
    # clear their last-write-wins entry; after that the monitor goes quiet
    # again (no steady-state empty gossip).
    empty: frozenset[str] = frozenset()
    sets = iter(
        [
            frozenset({"memory"}),  # advertised
            empty,  # withdrawn -> publish empty once
            empty,  # stays empty -> quiet
            empty,
        ]
    )
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _capabilities_only_gatherer(
        info_send, lambda: next(sets, empty)
    )
    received: list[frozenset[str]] = []

    async def collect() -> None:
        with info_recv as stream:
            async for info in stream:
                assert isinstance(info, NodeCapabilities)
                received.append(info.capabilities)

    with anyio.move_on_after(0.3):
        async with anyio.create_task_group() as tg:
            tg.start_soon(gatherer._monitor_capabilities)  # pyright: ignore[reportPrivateUsage]
            tg.start_soon(collect)

    assert received[:2] == [frozenset({"memory"}), frozenset()]
    # Nothing after the single empty reading: the monitor went quiet.
    assert all(not caps for caps in received[1:])
    assert len(received) == 2
