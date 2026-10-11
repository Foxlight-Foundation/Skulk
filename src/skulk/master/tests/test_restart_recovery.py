"""A cold-started master carries steward history back from the node's record."""

from datetime import datetime, timedelta, timezone

import anyio

from skulk.master.main import Master
from skulk.recovery.record import (
    RESTARTED_BEFORE_DECISION,
    ClusterIntentRecord,
    RecoveryStamp,
)
from skulk.recovery.tests.fixtures import sample_proposal
from skulk.shared.apply import apply
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    RestartRecoveryCompleted,
    StateSnapshotHydrated,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import Receiver, channel


def _master(
    initial_state: State | None,
    *,
    record: ClusterIntentRecord | None,
    enabled: bool = True,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    global_send, global_recv = channel[GlobalForwarderEvent]()
    master = Master(
        NodeId(),
        SessionId(master_node_id=NodeId(), election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_send,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=initial_state,
        restart_recovery_record=record,
        restart_recovery_enabled=enabled,
    )
    return master, global_recv


def _record() -> ClusterIntentRecord:
    return ClusterIntentRecord(
        stamp=RecoveryStamp(
            session_master_node_id=NodeId("before-restart"),
            session_election_clock=4,
            last_event_applied_idx=120,
            written_at=datetime.now(tz=timezone.utc) - timedelta(minutes=1),
        ),
        steward_proposals=(sample_proposal("rejected", 30), sample_proposal("pending", 2)),
    )


async def _index_seed(master: Master, global_recv: Receiver[GlobalForwarderEvent]) -> list[GlobalForwarderEvent]:
    received: list[GlobalForwarderEvent] = []

    async def consume() -> None:
        with global_recv as events:
            async for event in events:
                received.append(event)
                return

    with anyio.fail_after(10):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume)
            await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]
    return received


async def test_cold_start_restores_terminal_proposal_history_as_event_zero() -> None:
    record = _record()
    master, global_recv = _master(None, record=record)

    received = await _index_seed(master, global_recv)

    assert len(received) == 1 and received[0].origin_idx == 0
    seed = received[0].event
    assert isinstance(seed, StateSnapshotHydrated)
    assert seed.state.restart_recovery_pending is True
    restored = seed.state.steward_action_proposals
    assert {proposal.status for proposal in restored.values()} == {"rejected", "expired"}
    pending_before = record.steward_proposals[1]
    assert restored[pending_before.proposal_id].outcome == RESTARTED_BEFORE_DECISION
    # The master's ordered view matches what it replicated.
    assert dict(master._ordered_steward_proposals) == dict(restored)  # pyright: ignore[reportPrivateUsage]


async def test_cold_start_without_a_record_still_holds_records_until_complete() -> None:
    """Other nodes' records stay protected even when this node has none."""
    master, global_recv = _master(None, record=None)

    received = await _index_seed(master, global_recv)

    seed = received[0].event
    assert isinstance(seed, StateSnapshotHydrated)
    assert seed.state.restart_recovery_pending is True
    assert seed.state.steward_action_proposals == {}


async def test_disabled_recovery_keeps_the_historical_empty_cold_start() -> None:
    master, _ = _master(None, record=_record(), enabled=False)

    await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]

    assert master.state.last_event_applied_idx == -1
    assert master.state.restart_recovery_pending is False
    assert master.state.steward_action_proposals == {}


def test_failover_keeps_live_history_and_ignores_the_disk_record() -> None:
    """A surviving view is fresher than any record; failover is unchanged."""
    live = sample_proposal("dispatched", 1)
    prior = State(
        steward_action_proposals={live.proposal_id: live}, last_event_applied_idx=9
    )
    master, _ = _master(seed_state_for_new_session(prior), record=_record())

    assert dict(master._ordered_steward_proposals) == {live.proposal_id: live}  # pyright: ignore[reportPrivateUsage]
    assert master._restart_recovery_record is None  # pyright: ignore[reportPrivateUsage]
    assert master._finish_restart_recovery() == []  # pyright: ignore[reportPrivateUsage]


async def test_recovery_completes_once_and_clears_the_flag() -> None:
    record = _record()
    master, global_recv = _master(None, record=record)
    await _index_seed(master, global_recv)

    completion = master._finish_restart_recovery()  # pyright: ignore[reportPrivateUsage]

    assert completion == [
        RestartRecoveryCompleted(
            event_id=completion[0].event_id,
            restored_placements=0,
            restored_proposals=2,
        )
    ]
    assert master._finish_restart_recovery() == []  # pyright: ignore[reportPrivateUsage]
    cleared = apply(master.state, IndexedEvent(idx=1, event=completion[0]))
    assert cleared.restart_recovery_pending is False


def test_master_promoted_mid_recovery_finishes_it() -> None:
    """Failover carries an unfinished restore to the new master."""
    prior = State(restart_recovery_pending=True, last_event_applied_idx=3)
    master, _ = _master(seed_state_for_new_session(prior), record=_record())

    assert master._restart_recovery_record is not None  # pyright: ignore[reportPrivateUsage]
    # The flag arrives with the carried seed; once applied, completion follows.
    master.state = master.state.model_copy(update={"restart_recovery_pending": True})
    assert len(master._finish_restart_recovery()) == 1  # pyright: ignore[reportPrivateUsage]
