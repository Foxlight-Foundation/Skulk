"""Receipt subscriptions stay bounded when a remote response cannot publish."""

# pyright: reportPrivateUsage=false

from typing import Literal, cast

import anyio
import pytest
from skulk_pyo3_bindings import NetworkingHandle

from skulk.master.main import Master
from skulk.routing.router import Router, TopicRouter
from skulk.routing.tests.test_local_control_delivery import _components
from skulk.routing.topics import LOCAL_EVENT_RECEIPTS
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestEventLog,
)
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskDeleted,
    TracingStateChanged,
)
from skulk.shared.types.state_sync import LocalEventReceipt, StateSyncMessage
from skulk.shared.types.tasks import TaskId
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


@pytest.mark.parametrize(
    ("previous_through", "previous_head", "incoming_through", "incoming_head"),
    [(0, 2, 0, 1), (3, 10, 2, 10), (5, 8, 4, 9)],
)
async def test_response_coalescing_preserves_each_progress_high_water_mark(
    previous_through: int,
    previous_head: int,
    incoming_through: int,
    incoming_head: int,
) -> None:
    """Out-of-order arrivals cannot erase acknowledged prefixes or replay targets."""
    _, topic, receiver = await _subscription()
    previous = _receipt(SystemId(), head=previous_head).model_copy(
        update={"through_idx": previous_through}
    )
    incoming = previous.model_copy(
        update={"through_idx": incoming_through, "next_global_idx": incoming_head}
    )
    with receiver:
        await topic.publish(previous, origin="master")
        await topic.publish(incoming, origin="master")
        assert receiver.statistics().current_buffer_used == 1
        result = receiver.receive_nowait()[1]
        assert result.through_idx == max(previous_through, incoming_through)
        assert result.next_global_idx == max(previous_head, incoming_head)


async def test_reversed_queued_response_keeps_replay_after_outbox_retirement() -> None:
    """The real subscriber and consumer recover a lost final frame after coalescing."""
    _, topic, receipts = await _subscription()
    router, outbound, incoming = _components()
    router.nack_base_seconds = router.nack_cap_seconds = 0.01
    commands = router.command_sender.clone_receiver()
    indexed = router.receiver()
    with anyio.fail_after(5):
        async with router._tg as group:
            group.start_soon(router._run_ext_in)
            await router.sender().send(TaskDeleted(task_id=TaskId()))
            local = await outbound.receive()
            latest = LocalEventReceipt(
                kind="response",
                requester=router.node_id,
                source=local.origin,
                session_id=router.session_id,
                through_idx=0,
                next_global_idx=2,
            )
            await topic.publish(latest, origin="master")
            await topic.publish(
                latest.model_copy(update={"next_global_idx": 1}), origin="master"
            )
            group.start_soon(router._receive_delivery_receipts, receipts)
            group.start_soon(router._retry_receipt_replay)
            first_request = await commands.receive()
            assert isinstance(first_request.command, RequestEventLog)
            assert first_request.command.since_idx == 0
            assert not router.out_for_delivery
            assert router._receipt_replay_target == 2
            for index in range(2):
                if index:
                    retry = await commands.receive()
                    assert isinstance(retry.command, RequestEventLog)
                    assert retry.command.since_idx == index
                event = TracingStateChanged(enabled=index % 2 == 0)
                await incoming.send(
                    GlobalForwarderEvent(
                        origin=router.session_id.master_node_id,
                        origin_idx=index,
                        session=router.session_id,
                        event=event,
                    )
                )
                assert (await indexed.receive()).event == event
            await anyio.sleep(0.6)
            assert commands.collect() == []
            group.cancel_scope.cancel()


async def test_coalescing_cannot_expand_metadata_past_the_frame_limit() -> None:
    """Combining individually admitted counters cannot bypass the byte bound."""
    _, topic, receiver = await _subscription()
    large = 10**2000
    previous = _receipt(SystemId()).model_copy(update={"through_idx": large})
    incoming = previous.model_copy(update={"through_idx": 0, "next_global_idx": large})
    with receiver:
        await topic.publish(previous, origin="master")
        await topic.publish(incoming, origin="master")
        assert receiver.statistics().current_buffer_used == 1
        origin, result = receiver.receive_nowait()
        assert result == previous
        assert len(result.model_dump_json().encode("utf-8")) + len(origin or "") <= 4096


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
