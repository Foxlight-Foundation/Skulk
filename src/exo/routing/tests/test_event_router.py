import anyio
import pytest

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
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
) -> tuple[
    EventRouter,
    Receiver[ForwarderCommand],
    Receiver[StateSyncMessage],
    Sender[StateSyncMessage],
    Sender[GlobalForwarderEvent],
]:
    command_sender, command_receiver = channel[ForwarderCommand]()
    state_sync_sender, state_sync_requests = channel[StateSyncMessage]()
    state_sync_responses, state_sync_receiver = channel[StateSyncMessage]()
    external_sender, external_receiver = channel[GlobalForwarderEvent]()
    external_outbound, _unused_external_outbound = channel[LocalForwarderEvent]()

    router = EventRouter(
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
            StateSyncMessage(
                kind="response",
                requester=request.requester,
                session_id=session_id,
                snapshot=StateSnapshot(
                    session_id=session_id,
                    last_event_applied_idx=4,
                    state=snapshot_state,
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
            StateSyncMessage(
                kind="response",
                requester=request.requester,
                session_id=session_id,
                snapshot=StateSnapshot(
                    session_id=session_id,
                    last_event_applied_idx=4,
                    state=State(last_event_applied_idx=4),
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
            StateSyncMessage(
                kind="response",
                requester=second_request.requester,
                session_id=session_id,
                snapshot=StateSnapshot(
                    session_id=session_id,
                    last_event_applied_idx=7,
                    state=snapshot_state,
                ),
            )
        )

        replay_request = await command_receiver.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == 8

        router.shutdown()
        tg.cancel_scope.cancel()
