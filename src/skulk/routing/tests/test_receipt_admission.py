"""Receipt subscriptions stay bounded when a remote response cannot publish."""

# pyright: reportPrivateUsage=false

from typing import Literal, cast

import anyio
from skulk_pyo3_bindings import NetworkingHandle

from skulk.master.main import Master
from skulk.routing.router import Router, TopicRouter
from skulk.routing.topics import LOCAL_EVENT_RECEIPTS
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import Event, GlobalForwarderEvent, LocalForwarderEvent
from skulk.shared.types.state_sync import LocalEventReceipt, StateSyncMessage
from skulk.utils.channels import Receiver, channel
from skulk.utils.pydantic_ext import CamelCaseModel


async def _subscription() -> tuple[
    Router,
    TopicRouter[CamelCaseModel],
    Receiver[tuple[str | None, LocalEventReceipt]],
]:
    """Use the production factory without starting or substituting transport."""
    router = Router(cast(NetworkingHandle, object()), node_id="master")
    await router.register_topic(LOCAL_EVENT_RECEIPTS)
    receiver = router.receiver_with_origin(LOCAL_EVENT_RECEIPTS)
    return router, router.topic_routers[LOCAL_EVENT_RECEIPTS.topic], receiver


def _receipt(
    source: SystemId,
    *,
    kind: Literal["request", "response"] = "response",
    requester: str = "peer",
    clock: int = 1,
    head: int = 0,
) -> LocalEventReceipt:
    """Build one actual receipt lineage with an observable replacement head."""
    return LocalEventReceipt(
        kind=kind,
        requester=NodeId(requester),
        source=source,
        session_id=SessionId(master_node_id=NodeId("master"), election_clock=clock),
        through_idx=0 if kind == "response" else -1,
        next_global_idx=head,
    )


async def test_receipt_origin_subscription_coalesces_without_a_consumer() -> None:
    """Repeated retries retain one latest lineage without backpressuring ingress."""
    _, topic, receiver = await _subscription()
    source = SystemId()
    with receiver, anyio.fail_after(4):
        for head in range(2048):
            await topic.publish(_receipt(source, head=head), origin="master")
        assert receiver.statistics().current_buffer_used == 1
        assert receiver.receive_nowait()[1].next_global_idx == 2047


async def test_receipt_origin_subscription_bounds_unique_sources_and_frame_bytes() -> (
    None
):
    """Unbounded producer identities and oversized metadata cannot grow this lane."""
    _, topic, receiver = await _subscription()
    with receiver, anyio.fail_after(4):
        for _ in range(2048):
            await topic.publish(_receipt(SystemId()), origin="master")
        assert receiver.statistics().max_buffer_size == 256
        assert receiver.statistics().current_buffer_used == 256
        assert all(
            len(value.model_dump_json().encode("utf-8")) + len(origin or "") <= 4096
            for origin, value in receiver.collect()
        )
        await topic.publish(_receipt(SystemId(), requester="x" * 4096), origin="master")
        assert receiver.statistics().current_buffer_used == 0
        latest = _receipt(SystemId(), head=99)
        await topic.publish(latest, origin="master")
        assert receiver.receive_nowait() == ("master", latest)


async def test_receipt_coalescing_keeps_peer_session_source_and_kind_separate() -> None:
    """A retry cannot replace another authenticated lineage or election epoch."""
    _, topic, receiver = await _subscription()
    source = SystemId()
    values = [
        ("master", _receipt(source)),
        ("other-master", _receipt(source)),
        ("master", _receipt(source, requester="other-peer")),
        ("master", _receipt(source, clock=2)),
        ("master", _receipt(SystemId())),
        ("peer", _receipt(source, kind="request")),
    ]
    with receiver:
        for origin, value in values:
            await topic.publish(value, origin=origin)
        assert receiver.collect() == values


async def test_closed_receipt_subscriptions_are_retired() -> None:
    """Queue ownership ends with its consumer instead of retaining late probes."""
    _, topic, receiver = await _subscription()
    receiver.close()
    await topic.publish(_receipt(SystemId()), origin="master")
    assert not topic.origin_senders


async def test_receipt_flood_keeps_other_event_loop_work_schedulable() -> None:
    """Disposable retry admission must not starve independent liveness work."""
    _, topic, receiver = await _subscription()
    source = SystemId()
    heartbeat = anyio.Event()

    async def signal_liveness() -> None:
        """Require an independent task to run while the ingress flood continues."""
        await anyio.sleep(0)
        heartbeat.set()

    with receiver, anyio.fail_after(4):
        async with anyio.create_task_group() as group:
            group.start_soon(signal_liveness)
            for head in range(2048):
                await topic.publish(_receipt(source, head=head), origin="master")
            assert heartbeat.is_set()
            assert receiver.statistics().current_buffer_used == 1
            group.cancel_scope.cancel()


async def test_blocked_master_reply_does_not_accumulate_retry_probes() -> None:
    """Actual reply egress can stall while repeated inbound probes stay bounded."""
    router, topic, receiver = await _subscription()
    session = SessionId(master_node_id=NodeId("master"), election_clock=1)
    source = SystemId()
    master = Master(
        NodeId("master"),
        session,
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )
    master._consumed_local_indices[source] = 0
    probe = _receipt(source, kind="request")
    delivered: list[LocalEventReceipt] = []

    async def drain() -> None:
        """Release the same blocked egress without restarting either actor."""
        while True:
            packet = await router._local_events_out_recv.receive()
            delivered.append(LOCAL_EVENT_RECEIPTS.deserialize(packet.data))

    try:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(topic.run)
                group.start_soon(
                    master._local_receipt_processor,
                    router.sender(LOCAL_EVENT_RECEIPTS),
                    receiver,
                )
                await topic.publish(probe, origin="peer")
                while (
                    router._local_events_out_recv.statistics().current_buffer_used != 1
                ):
                    await anyio.sleep(0)
                await topic.publish(probe, origin="peer")
                while topic.networking_sender.statistics().tasks_waiting_send != 1:
                    await anyio.sleep(0)
                await topic.publish(probe, origin="peer")
                while topic.receiver.statistics().tasks_waiting_send != 1:
                    await anyio.sleep(0)
                before = receiver.statistics().current_buffer_used
                for _ in range(2048):
                    await topic.publish(probe, origin="peer")
                assert receiver.statistics().current_buffer_used <= before + 1
                assert receiver.statistics().current_buffer_used <= 256
                group.start_soon(drain)
                while (
                    receiver.statistics().current_buffer_used
                    or topic.receiver.statistics().tasks_waiting_send
                    or len(delivered) < 3
                ):
                    await anyio.sleep(0)
                assert all(value.kind == "response" for value in delivered)
                assert all(value.through_idx == 0 for value in delivered)
                group.cancel_scope.cancel()
    finally:
        master._event_log.close()
        await topic.shutdown()
