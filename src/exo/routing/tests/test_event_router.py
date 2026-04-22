from dataclasses import dataclass, field
from typing import cast

import anyio
import pytest

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    StateSnapshotHydrated,
    TestEvent,
)
from exo.shared.types.state import State
from exo.shared.types.state_sync import StateSnapshot, StateSyncMessage
from exo.utils.channels import Receiver, Sender, channel


def _make_router(
    session_id: SessionId,
    *,
    snapshot_request_attempts: int = 3,
    router_cls: type[EventRouter] = EventRouter,
) -> tuple[
    EventRouter,
    Receiver[ForwarderCommand],
    Receiver[StateSyncMessage],
    Sender[tuple[str | None, StateSyncMessage]],
    Sender[GlobalForwarderEvent],
]:
    command_sender, command_receiver = channel[ForwarderCommand]()
    state_sync_sender, state_sync_requests = channel[StateSyncMessage]()
    state_sync_responses, state_sync_receiver = channel[
        tuple[str | None, StateSyncMessage]
    ]()
    external_sender, external_receiver = channel[GlobalForwarderEvent]()
    external_outbound, _unused_external_outbound = channel[LocalForwarderEvent]()

    router = router_cls(
        session_id,
        command_sender=command_sender,
        state_sync_sender=state_sync_sender,
        state_sync_receiver=state_sync_receiver,
        external_inbound=external_receiver,
        external_outbound=external_outbound,
        snapshot_request_timeout_seconds=0.05,
        snapshot_request_attempts=snapshot_request_attempts,
        nack_base_seconds=0.01,
        nack_cap_seconds=0.02,
    )
    return (
        router,
        command_receiver,
        state_sync_requests,
        state_sync_responses,
        external_sender,
    )


@dataclass
class _ControlledSendEventRouter(EventRouter):
    delivered_indices: list[int] = field(init=False, default_factory=list)
    first_send_started: anyio.Event = field(init=False, default_factory=anyio.Event)
    allow_first_send_finish: anyio.Event = field(
        init=False, default_factory=anyio.Event
    )

    async def _send_internal_event(self, indexed_event: IndexedEvent) -> None:
        self.delivered_indices.append(indexed_event.idx)
        if indexed_event.idx == 0:
            self.first_send_started.set()
            await self.allow_first_send_finish.wait()

    async def release_ready_events_for_test(self) -> None:
        await self._release_ready_events()


@pytest.mark.asyncio
async def test_snapshot_supported_bootstrap_replays_only_tail() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=3)
    router, command_receiver, state_sync_requests, state_sync_responses, event_sender = (
        _make_router(session_id)
    )
    internal_receiver = router.receiver()

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        request = await state_sync_requests.receive()
        assert request.kind == "request"
        assert request.session_id == session_id

        snapshot_state = State(last_event_applied_idx=4)
        await state_sync_responses.send(
            (
                str(session_id.master_node_id),
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=session_id,
                    snapshot=StateSnapshot(
                        session_id=session_id,
                        last_event_applied_idx=4,
                        state=snapshot_state,
                    ),
                ),
            )
        )

        replay_request = await command_receiver.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == 5

        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=5,
                session=session_id,
                event=TestEvent(),
            )
        )

        hydrated = await internal_receiver.receive()
        assert hydrated.idx == 4
        assert isinstance(hydrated.event, StateSnapshotHydrated)
        assert hydrated.event.state == snapshot_state

        tail_event = await internal_receiver.receive()
        assert tail_event.idx == 5
        assert isinstance(tail_event.event, TestEvent)

        router.shutdown()
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_old_master_fallback_requests_full_replay() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=5)
    router, command_receiver, state_sync_requests, _state_sync_responses, event_sender = (
        _make_router(session_id)
    )
    internal_receiver = router.receiver()

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        request = await state_sync_requests.receive()
        assert request.kind == "request"

        replay_request = await command_receiver.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == 0

        event = TestEvent()
        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=0,
                session=session_id,
                event=event,
            )
        )

        received = await internal_receiver.receive()
        assert received.idx == 0
        assert received.event == event

        router.shutdown()
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_snapshot_session_mismatch_falls_back_to_full_replay() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=7)
    router, command_receiver, state_sync_requests, state_sync_responses, _event_sender = (
        _make_router(session_id)
    )
    internal_receiver = router.receiver()

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        request = await state_sync_requests.receive()
        await state_sync_responses.send(
            (
                str(session_id.master_node_id),
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=SessionId(
                        master_node_id=session_id.master_node_id,
                        election_clock=session_id.election_clock + 1,
                    ),
                    snapshot=StateSnapshot(
                        session_id=SessionId(
                            master_node_id=session_id.master_node_id,
                            election_clock=session_id.election_clock + 1,
                        ),
                        last_event_applied_idx=2,
                        state=State(last_event_applied_idx=2),
                    ),
                ),
            )
        )

        replay_request = await command_receiver.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == 0
        assert internal_receiver.collect() == []

        router.shutdown()
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_snapshot_gap_triggers_nack_recovery() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=9)
    router, command_receiver, state_sync_requests, state_sync_responses, event_sender = (
        _make_router(session_id)
    )
    internal_receiver = router.receiver()

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        request = await state_sync_requests.receive()
        await state_sync_responses.send(
            (
                str(session_id.master_node_id),
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=session_id,
                    snapshot=StateSnapshot(
                        session_id=session_id,
                        last_event_applied_idx=4,
                        state=State(last_event_applied_idx=4),
                    ),
                ),
            )
        )

        initial_replay = await command_receiver.receive()
        assert isinstance(initial_replay.command, RequestEventLog)
        assert initial_replay.command.since_idx == 5

        late_event = TestEvent()
        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=6,
                session=session_id,
                event=late_event,
            )
        )
        await anyio.sleep(0.05)
        hydrated_events = internal_receiver.collect()
        assert len(hydrated_events) == 1
        assert hydrated_events[0].idx == 4
        assert isinstance(hydrated_events[0].event, StateSnapshotHydrated)
        assert hydrated_events[0].event.state.last_event_applied_idx == 4

        nack = await command_receiver.receive()
        assert isinstance(nack.command, RequestEventLog)
        assert nack.command.since_idx == 5

        missing_event = TestEvent()
        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=5,
                session=session_id,
                event=missing_event,
            )
        )

        recovered = await internal_receiver.receive()
        assert recovered.idx == 5
        assert recovered.event == missing_event

        drained_late_event = await internal_receiver.receive()
        assert drained_late_event.idx == 6
        assert drained_late_event.event == late_event

        router.shutdown()
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_snapshot_bootstrap_retries_before_full_replay_fallback() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=11)
    router, command_receiver, state_sync_requests, state_sync_responses, _event_sender = (
        _make_router(session_id)
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        first_request = await state_sync_requests.receive()
        assert first_request.kind == "request"
        assert first_request.session_id == session_id

        second_request = await state_sync_requests.receive()
        assert second_request.kind == "request"
        assert second_request.requester == first_request.requester

        snapshot_state = State(last_event_applied_idx=7)
        await state_sync_responses.send(
            (
                str(session_id.master_node_id),
                StateSyncMessage(
                    kind="response",
                    requester=second_request.requester,
                    session_id=session_id,
                    snapshot=StateSnapshot(
                        session_id=session_id,
                        last_event_applied_idx=7,
                        state=snapshot_state,
                    ),
                ),
            )
        )

        replay_request = await command_receiver.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == 8

        router.shutdown()
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_snapshot_bootstrap_ignores_non_master_origin() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=12)
    router, command_receiver, state_sync_requests, state_sync_responses, _event_sender = (
        _make_router(session_id)
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        request = await state_sync_requests.receive()
        await state_sync_responses.send(
            (
                "not-the-master",
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=session_id,
                    snapshot=StateSnapshot(
                        session_id=session_id,
                        last_event_applied_idx=2,
                        state=State(last_event_applied_idx=2),
                    ),
                ),
            )
        )

        replay_request = await command_receiver.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == 0

        router.shutdown()
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_release_ready_events_preserves_order_during_concurrent_drains() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=13)
    router, _command_receiver, _state_sync_requests, _state_sync_responses, _sender = cast(
        tuple[
            _ControlledSendEventRouter,
            Receiver[ForwarderCommand],
            Receiver[StateSyncMessage],
            Sender[tuple[str | None, StateSyncMessage]],
            Sender[GlobalForwarderEvent],
        ],
        _make_router(session_id, router_cls=_ControlledSendEventRouter),
    )

    router.event_buffer.ingest(0, TestEvent())

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.release_ready_events_for_test)
        await router.first_send_started.wait()

        router.event_buffer.ingest(1, TestEvent())
        tg.start_soon(router.release_ready_events_for_test)
        await anyio.sleep(0)

        assert router.delivered_indices == [0]

        router.allow_first_send_finish.set()

    assert router.delivered_indices == [0, 1]


@pytest.mark.asyncio
async def test_conflicting_duplicate_index_requests_replay_instead_of_crashing() -> None:
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=14)
    router, command_receiver, state_sync_requests, state_sync_responses, event_sender = (
        _make_router(session_id)
    )
    internal_receiver = router.receiver()

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)

        request = await state_sync_requests.receive()

        first_event = TestEvent()
        conflicting_event = TestEvent()
        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=0,
                session=session_id,
                event=first_event,
            )
        )
        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=0,
                session=session_id,
                event=conflicting_event,
            )
        )

        await state_sync_responses.send(
            (
                str(session_id.master_node_id),
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=session_id,
                    snapshot=StateSnapshot(
                        session_id=session_id,
                        last_event_applied_idx=-1,
                        state=State(last_event_applied_idx=-1),
                    ),
                ),
            )
        )

        initial_replay = await command_receiver.receive()
        assert isinstance(initial_replay.command, RequestEventLog)
        assert initial_replay.command.since_idx == 0

        recovery_replay = await command_receiver.receive()
        assert isinstance(recovery_replay.command, RequestEventLog)
        assert recovery_replay.command.since_idx == 0

        authoritative_event = TestEvent()
        await event_sender.send(
            GlobalForwarderEvent(
                origin=session_id.master_node_id,
                origin_idx=0,
                session=session_id,
                event=authoritative_event,
            )
        )

        delivered = await internal_receiver.receive()
        assert delivered.idx == 0
        assert delivered.event == authoritative_event

        router.shutdown()
        tg.cancel_scope.cancel()
