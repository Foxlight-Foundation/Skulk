"""Promotion and local consumption survive blocked peer control publication."""

from typing import cast

import anyio
import pytest

from skulk.master.main import Master
from skulk.routing import event_router as event_router_module
from skulk.routing.event_router import EventRouter
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import LOCAL_EVENT_RECEIPTS, LOCAL_EVENTS, TypedTopic
from skulk.shared.models.model_cards import LicenseCardConfig, ModelCard, ModelTask
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    CustomModelCardAdded,
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskDeleted,
    TestEvent,
    TracingStateChanged,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state_sync import LocalEventReceipt, StateSyncMessage
from skulk.shared.types.tasks import TaskId
from skulk.utils.channels import channel
from skulk.utils.pydantic_ext import CamelCaseModel


@pytest.mark.parametrize("receipt", [False, True])
async def test_promoted_master_admission_bypasses_already_blocked_topic(
    receipt: bool,
) -> None:
    """An old follower publish cannot hold new self-master control behind it."""
    network, network_receiver = channel[OutboundPacket](0)
    session = SessionId(master_node_id=NodeId("self"), election_clock=2)
    old = SessionId(master_node_id=NodeId("old"), election_clock=1)
    topic = cast(
        TypedTopic[CamelCaseModel], LOCAL_EVENT_RECEIPTS if receipt else LOCAL_EVENTS
    )
    router = TopicRouter(topic, network, max_buffer_size=0, local_routing_key="self")
    observed, local = channel[CamelCaseModel]()
    router.senders.add(observed)
    producer = router.new_sender()
    if receipt:
        blocked: CamelCaseModel = LocalEventReceipt(
            kind="response",
            requester=NodeId("peer"),
            source=SystemId(),
            session_id=session,
            through_idx=0,
        )
        promoted: CamelCaseModel = LocalEventReceipt(
            kind="request",
            requester=NodeId("self"),
            source=SystemId(),
            session_id=session,
        )
    else:
        blocked = LocalForwarderEvent(
            origin=SystemId(),
            origin_idx=0,
            session=old,
            event=TestEvent(),
        )
        promoted = LocalForwarderEvent(
            origin=SystemId(),
            origin_idx=0,
            session=session,
            event=TestEvent(),
        )
    with anyio.fail_after(3):
        async with anyio.create_task_group() as group:
            group.start_soon(router.run)
            await producer.send(blocked)
            while network_receiver.statistics().tasks_waiting_send != 1:
                await anyio.sleep(0)
            # A new EventRouter after promotion receives a new handle to the
            # same existing topic. Cloning must preserve the admission policy.
            replacement = router.new_sender().clone()
            await replacement.send(promoted)
            assert await local.receive() == promoted
            assert network_receiver.statistics().tasks_waiting_send == 1
            replacement.close()
            with pytest.raises(anyio.ClosedResourceError):
                await replacement.send(promoted)
            group.cancel_scope.cancel()


async def test_master_noops_retire_while_remote_receipt_actor_and_topic_are_blocked() -> (
    None
):
    """Local consumed-prefix retirement cannot wait behind two remote replies."""
    node = NodeId("master")
    session = SessionId(master_node_id=node, election_clock=1)
    commands, _ = channel[ForwarderCommand]()
    requests, _ = channel[StateSyncMessage]()
    _, snapshots = channel[tuple[str | None, StateSyncMessage]]()
    incoming, globals_in = channel[GlobalForwarderEvent]()
    outgoing, local_events = channel[LocalForwarderEvent]()
    router = EventRouter(
        node, session, commands, requests, snapshots, globals_in, outgoing
    )
    router.pending_event_limit = 1
    router._bootstrap_complete.set()  # pyright: ignore[reportPrivateUsage]
    network, network_receiver = channel[OutboundPacket](0)
    topic = TopicRouter(
        LOCAL_EVENT_RECEIPTS, network, max_buffer_size=0, local_routing_key=str(node)
    )
    probes, probe_input = channel[tuple[str | None, LocalEventReceipt]]()
    master = Master(
        node,
        session,
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=local_events,
        global_event_sender=incoming,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        local_event_consumed=router.acknowledge_master_consumption,
    )
    producer = router.sender()
    remote_source = SystemId()
    try:
        with anyio.fail_after(4):
            async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
                group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(master._event_processor)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(topic.run)
                process_receipts = master._local_receipt_processor  # pyright: ignore[reportPrivateUsage]
                group.start_soon(process_receipts, topic.new_sender(), probe_input)
                await outgoing.send(
                    LocalForwarderEvent(
                        origin=remote_source,
                        origin_idx=0,
                        session=session,
                        event=TaskDeleted(task_id=TaskId()),
                    )
                )
                while remote_source not in master._consumed_local_indices:  # pyright: ignore[reportPrivateUsage]
                    await anyio.sleep(0)
                probe = LocalEventReceipt(
                    kind="request",
                    requester=NodeId("peer"),
                    source=remote_source,
                    session_id=session,
                )
                for _ in range(2):
                    await probes.send(("peer", probe))
                while network_receiver.statistics().tasks_waiting_send != 1:
                    await anyio.sleep(0)
                while topic.receiver.statistics().tasks_waiting_send != 1:
                    await anyio.sleep(0)
                for _ in range(32):
                    await producer.send(TaskDeleted(task_id=TaskId()))
                await producer.send(TracingStateChanged(enabled=True))
                while router.out_for_delivery or not master.state.tracing_enabled:
                    await anyio.sleep(0)
                assert len(master._event_log) == 1  # pyright: ignore[reportPrivateUsage]
                assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
                assert network_receiver.statistics().tasks_waiting_send == 1
                group.cancel_scope.cancel()
    finally:
        master._event_log.close()  # pyright: ignore[reportPrivateUsage]


async def test_oversized_producer_failure_consumes_no_sequence_or_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both send methods and clones reject before poisoning other producers."""
    from skulk.routing.tests.test_local_control_delivery import (
        _components,  # pyright: ignore[reportPrivateUsage]
    )

    router, outbound, _ = _components()
    producer = router.sender()
    clone = producer.clone()
    monkeypatch.setattr(event_router_module, "_MAX_LOCAL_CONTROL_FRAME_BYTES", 1)
    with pytest.raises(ValueError, match="too large for transport"):
        await producer.send(TestEvent())
    with pytest.raises(ValueError, match="too large for transport"):
        clone.send_nowait(TestEvent())
    assert not router.out_for_delivery
    assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        event_router_module, "_MAX_LOCAL_CONTROL_FRAME_BYTES", 8 * 1024 * 1024 - 4096
    )
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        event = TestEvent()
        await producer.send(event)
        frame = await outbound.receive()
        assert frame.origin_idx == 0 and frame.event == event
        group.cancel_scope.cancel()


async def test_real_card_payload_exceeding_transport_budget_fails_before_admission() -> (
    None
):
    """Valid card metadata still needs a byte bound before reliable sequencing."""
    from skulk.routing.tests.test_local_control_delivery import (
        _components,  # pyright: ignore[reportPrivateUsage]
    )

    router, _, _ = _components()
    card = ModelCard(
        model_id=ModelId("example/model"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        license=LicenseCardConfig(name="Example", notice="🎵" * (2 * 1024 * 1024)),
    )
    with pytest.raises(ValueError, match="too large for transport"):
        await router.sender().send(CustomModelCardAdded(model_card=card))
    assert router.out_for_delivery == {}
    assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
