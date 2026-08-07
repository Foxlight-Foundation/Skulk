"""Coverage for the explicit telemetry-plane heartbeat monitor."""

import anyio

from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
    NodeHeartbeat,
)


async def test_monitor_publishes_heartbeats_at_configured_cadence() -> None:
    """The monitor emits immediately and continues until its owner cancels it."""
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = InfoGatherer(
        info_sender=info_send,
        heartbeat_poll_interval=0.01,
    )

    with anyio.fail_after(1):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(gatherer._monitor_heartbeat)  # pyright: ignore[reportPrivateUsage]
            first = await info_recv.receive()
            second = await info_recv.receive()
            assert isinstance(first, NodeHeartbeat)
            assert isinstance(second, NodeHeartbeat)
            task_group.cancel_scope.cancel()


async def test_monitor_is_inert_when_disabled() -> None:
    """A ``None`` cadence disables heartbeat publication for focused tests."""
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = InfoGatherer(
        info_sender=info_send,
        heartbeat_poll_interval=None,
    )

    await gatherer._monitor_heartbeat()  # pyright: ignore[reportPrivateUsage]
    info_send.close()
    assert [info async for info in info_recv] == []
