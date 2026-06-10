"""Master construction honors a failover seed state (#273)."""

from skulk.master.main import Master
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.events import Event, GlobalForwarderEvent, LocalForwarderEvent
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import channel


def _master(initial_state: State | None) -> Master:
    return Master(
        NodeId(),
        SessionId(master_node_id=NodeId(), election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=initial_state,
    )


def test_master_starts_from_seed():
    seed = seed_state_for_new_session(State(tracing_enabled=True))
    master = _master(seed)
    assert master.state is seed
    assert master.state.tracing_enabled is True
    # The seed's event index starts the new session's log at the beginning.
    assert master.state.last_event_applied_idx == -1


def test_master_without_seed_starts_empty():
    master = _master(None)
    assert master.state.instances == {}
    assert master.state.last_event_applied_idx == -1
