"""Regression tests for #266: a closed telemetry consumer must stop the
InfoGatherer cleanly, never crash the node.

The live failure: during peer churn the worker's info forwarder exits first
(its downstream event stream closes on a master transition), closing the
info channel's receive side; the gatherer's next send raised
``BrokenResourceError`` through the task group and took the entire process
down — twice in one night, once on the cluster hub.
"""

import anyio
import pytest

from skulk.utils.channels import Sender, channel
from skulk.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer


def _quiet_gatherer(info_send: Sender[GatheredInfo]) -> InfoGatherer:
    """A gatherer whose periodic monitors are all disabled, so the only
    send is the startup NodeConfig — the exact line that crashed in #266."""
    return InfoGatherer(
        info_sender=info_send,
        interface_watcher_interval=None,
        misc_poll_interval=None,
        system_profiler_interval=None,
        memory_poll_rate=None,
        mactop_interval=None,
        thunderbolt_bridge_poll_interval=None,
        static_info_poll_interval=None,
        rdma_ctl_poll_interval=None,
        disk_poll_interval=None,
    )


async def test_closed_consumer_stops_run_cleanly():
    # Receiver closed before the gatherer ever sends — run() must return,
    # not raise, within a bounded time.
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _quiet_gatherer(info_send)
    info_recv.close()
    with anyio.fail_after(30):
        await gatherer.run()


async def test_consumer_closing_mid_run_stops_cleanly():
    # Consumer disappears while the gatherer is live (the churn ordering):
    # a fast monitor keeps sending; the receiver reads one item then closes.
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = _quiet_gatherer(info_send)
    gatherer.static_info_poll_interval = 0.05

    async def consume_one_then_leave():
        with info_recv as stream:
            async for _ in stream:
                return

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(consume_one_then_leave)
            await gatherer.run()


async def test_real_errors_still_propagate(monkeypatch: pytest.MonkeyPatch):
    # The consumer-gone handling must not swallow genuine faults.
    from skulk.utils.info_gatherer import info_gatherer as module

    async def explode():
        raise ValueError("genuine gatherer fault")

    monkeypatch.setattr(module.NodeConfig, "gather", explode)
    info_send, _info_recv = channel[GatheredInfo]()
    gatherer = _quiet_gatherer(info_send)
    with pytest.raises(BaseExceptionGroup) as exc_info:
        with anyio.fail_after(30):
            await gatherer.run()
    assert exc_info.group_contains(ValueError)
