"""Local control ordering survives partitions without unbounded admission."""

from typing import cast

import anyio
import pytest
from skulk_pyo3_bindings import NetworkingHandle

from skulk.routing.event_router import EventRouter
from skulk.routing.router import OutboundPacket, Router, TopicRouter
from skulk.routing.topics import (
    ELECTION_MESSAGES,
    GLOBAL_EVENTS,
    LOCAL_EVENTS,
    TELEMETRY,
)
from skulk.shared.types.commands import ForwarderCommand
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    NodeGatheredInfo,
    TestEvent,
)
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.event_buffer import MultiSourceBuffer
from skulk.utils.info_gatherer.info_gatherer import NodeHeartbeat


def _components() -> tuple[
    EventRouter, Receiver[LocalForwarderEvent], Sender[GlobalForwarderEvent]
]:
    commands, _ = channel[ForwarderCommand]()
    requests, _ = channel[StateSyncMessage]()
    _, responses = channel[tuple[str | None, StateSyncMessage]]()
    incoming, incoming_receiver = channel[GlobalForwarderEvent]()
    outgoing, outgoing_receiver = channel[LocalForwarderEvent]()
    router = EventRouter(
        NodeId("worker"),
        SessionId(master_node_id=NodeId("master"), election_clock=1),
        commands,
        requests,
        responses,
        incoming_receiver,
        outgoing,
    )
    router._bootstrap_complete.set()  # pyright: ignore[reportPrivateUsage]
    return router, outgoing_receiver, incoming


async def _echo(
    router: EventRouter,
    sender: Sender[GlobalForwarderEvent],
    event: Event,
    index: int,
) -> None:
    await sender.send(
        GlobalForwarderEvent(
            origin=router.session_id.master_node_id,
            origin_idx=index,
            session=router.session_id,
            event=event,
        )
    )
    with anyio.fail_after(2):
        while event.event_id in router.out_for_delivery:
            await anyio.sleep(0)


async def test_missing_local_sequence_survives_old_retry_cap_and_repairs_tail() -> None:
    """Later publications drain only after the retained missing sequence returns."""
    router, outbound, incoming = _components()
    producer = router.sender()
    missing, following = TestEvent(), TestEvent()
    master = MultiSourceBuffer[SystemId, Event]()
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
        await producer.send(missing)
        lost = await outbound.receive()
        await producer.send(following)
        later = await outbound.receive()
        assert (lost.origin_idx, later.origin_idx) == (0, 1)
        master.ingest(later.origin_idx, later.event, later.origin)
        assert master.drain() == []
        router.out_for_delivery[missing.event_id] = (-1.0, 6, lost)
        group.start_soon(router._simple_retry)  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(4):
            repaired = await outbound.receive()
        assert repaired == lost
        master.ingest(repaired.origin_idx, repaired.event, repaired.origin)
        assert master.drain() == [missing, following]
        await _echo(router, incoming, missing, 0)
        await _echo(router, incoming, following, 1)
        assert not router.out_for_delivery
        assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
        group.cancel_scope.cancel()


@pytest.mark.parametrize("limit", ["events", "bytes"])
async def test_admission_backpressures_producer_and_releases_on_echo(
    limit: str,
) -> None:
    """Full count/byte budgets bound upstream queuing and resume in source order."""
    router, outbound, incoming = _components()
    if limit == "events":
        router.pending_event_limit = 1
    else:
        # A legal oversized event is admitted exclusively, never truncated.
        router.pending_byte_limit = 1
    producer = router.sender()
    events = [TestEvent() for _ in range(4)]
    fourth_enqueued = anyio.Event()

    async def fourth() -> None:
        await producer.send(events[3])
        fourth_enqueued.set()

    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
        await producer.send(events[0])
        first = await outbound.receive()
        await producer.send(events[1])
        await producer.send(events[2])
        group.start_soon(fourth)
        await anyio.sleep(0)
        assert not fourth_enqueued.is_set()
        assert outbound.collect() == []
        assert len(router.out_for_delivery) == 1
        for index, event in enumerate(events):
            frame = first if index == 0 else await outbound.receive()
            assert frame.origin_idx == index and frame.event == event
            assert len(router.out_for_delivery) == 1
            await _echo(router, incoming, event, index)
        assert fourth_enqueued.is_set()
        assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
        group.cancel_scope.cancel()


async def test_duplicate_pending_event_does_not_create_an_ordering_hole() -> None:
    """One acknowledgement cannot cover two different indexes for one event."""
    router, outbound, incoming = _components()
    router.pending_event_limit = 1
    producer = router.sender()
    first, second = TestEvent(), TestEvent()
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
        await producer.send(first)
        initial = await outbound.receive()
        await producer.send(first)
        await producer.send(second)
        await anyio.sleep(0)
        assert outbound.collect() == []
        await _echo(router, incoming, first, 0)
        following = await outbound.receive()
        assert following.origin_idx == initial.origin_idx + 1
        assert following.event == second
        await _echo(router, incoming, second, 1)
        group.cancel_scope.cancel()


async def test_immediate_echo_during_publish_does_not_leave_phantom_outbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local master can echo before the original publish await returns."""
    router, _, _ = _components()
    published = anyio.Event()
    event = TestEvent()

    async def publish(frame: LocalForwarderEvent) -> None:
        assert frame.event.event_id in router.out_for_delivery
        await router._acknowledge_local_event(frame.event.event_id)  # pyright: ignore[reportPrivateUsage]
        published.set()

    monkeypatch.setattr(router.external_outbound, "send", publish)
    producer = router.sender()
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        await producer.send(event)
        with anyio.fail_after(2):
            await published.wait()
        assert not router.out_for_delivery
        assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
        group.cancel_scope.cancel()


async def test_non_control_reading_never_takes_a_reliable_sequence() -> None:
    """A reading the master would reject cannot fill retained admission."""
    router, outbound, incoming = _components()
    producer = router.sender()
    valid = TestEvent()
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
        for _ in range(3):
            await producer.send(
                NodeGatheredInfo(
                    node_id=router.node_id,
                    when="2026-11-07T00:00:00+00:00",
                    info=NodeHeartbeat(),
                )
            )
        await producer.send(valid)
        frame = await outbound.receive()
        assert frame.origin_idx == 0 and frame.event == valid
        assert set(router.out_for_delivery) == {valid.event_id}
        await _echo(router, incoming, valid, 0)
        group.cancel_scope.cancel()


async def test_retry_work_per_tick_is_bounded() -> None:
    """A retained backlog cannot turn one retry wakeup into a resend storm."""
    router, outbound, _ = _components()
    router.retry_batch_size = 2
    router.retry_cap_seconds = 60
    producer = router.sender()
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        for _ in range(5):
            await producer.send(TestEvent())
            frame = await outbound.receive()
            router.out_for_delivery[frame.event.event_id] = (-1.0, 6, frame)
        group.start_soon(router._simple_retry)  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(4):
            first = await outbound.receive()
            second = await outbound.receive()
        await anyio.sleep(0)
        assert first.origin_idx < second.origin_idx
        assert outbound.collect() == []
        assert len(router.out_for_delivery) == 5
        group.cancel_scope.cancel()


async def test_local_master_delivery_does_not_wait_for_full_peer_egress() -> None:
    """The master can index its own control even while peer egress is stalled."""
    network_sender, network_receiver = channel[OutboundPacket](1)
    await network_sender.send(
        OutboundPacket(
            topic=LOCAL_EVENTS.topic,
            routing_key=None,
            stream_key=None,
            is_terminal=False,
            data=b"held",
        )
    )
    topic = TopicRouter(
        LOCAL_EVENTS, network_sender, max_buffer_size=0, local_routing_key="master"
    )
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    topic.senders.add(local_sender)
    producer = topic.new_sender()
    event = LocalForwarderEvent(
        origin=SystemId("self"),
        origin_idx=0,
        session=SessionId(master_node_id=NodeId("master"), election_clock=1),
        event=TestEvent(),
    )
    async with anyio.create_task_group() as group:
        group.start_soon(topic.run)
        with anyio.fail_after(2):
            await producer.send(event)
            assert await local_receiver.receive() == event
        assert network_receiver.statistics().current_buffer_used == 1
        group.cancel_scope.cancel()


class _BlockedLocalControlNetwork:
    """Expose a deliberately blocked local-event transport effect."""

    def __init__(self) -> None:
        self.local_started = anyio.Event()
        self.release_local = anyio.Event()
        self.other_published = anyio.Event()
        self.other_topics: set[str] = set()

    async def gossipsub_publish(self, topic: str, data: bytes) -> None:
        """Hold local events while allowing independent correctness lanes."""
        if topic == LOCAL_EVENTS.topic:
            self.local_started.set()
            await self.release_local.wait()
        else:
            self.other_topics.add(topic)
            if self.other_topics == {
                ELECTION_MESSAGES.topic,
                TELEMETRY.topic,
                GLOBAL_EVENTS.topic,
            }:
                self.other_published.set()


async def test_local_retry_egress_is_bounded_and_cannot_stall_other_lanes() -> None:
    """A blocked local-event publish cannot hide an unbounded retry queue."""
    network = _BlockedLocalControlNetwork()
    router = Router(cast(NetworkingHandle, cast(object, network)), node_id="worker")
    await router.register_topic(LOCAL_EVENTS)
    later_enqueued = anyio.Event()

    def packet(topic: str) -> OutboundPacket:
        return OutboundPacket(
            topic=topic,
            routing_key=None,
            stream_key=None,
            is_terminal=False,
            data=b"test",
        )

    async def third() -> None:
        await router._local_events_out_send.send(packet(LOCAL_EVENTS.topic))  # pyright: ignore[reportPrivateUsage]
        later_enqueued.set()

    async with anyio.create_task_group() as group:
        group.start_soon(router._local_events_networking_publish)  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._networking_publish)  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._election_networking_publish)  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._telemetry_networking_publish)  # pyright: ignore[reportPrivateUsage]
        await router._local_events_out_send.send(packet(LOCAL_EVENTS.topic))  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(2):
            await network.local_started.wait()
        await router._local_events_out_send.send(packet(LOCAL_EVENTS.topic))  # pyright: ignore[reportPrivateUsage]
        group.start_soon(third)
        await anyio.sleep(0)
        assert not later_enqueued.is_set()
        await router._election_out_send.send(packet(ELECTION_MESSAGES.topic))  # pyright: ignore[reportPrivateUsage]
        await router._telemetry_out_send.send(packet(TELEMETRY.topic))  # pyright: ignore[reportPrivateUsage]
        await router.networking_receiver.clone_sender().send(
            packet(GLOBAL_EVENTS.topic)
        )
        with anyio.fail_after(2):
            await network.other_published.wait()
        assert not later_enqueued.is_set()
        network.release_local.set()
        with anyio.fail_after(2):
            await later_enqueued.wait()
        group.cancel_scope.cancel()
