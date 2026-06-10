"""Master failover seeding (#273): the seed is event 0 of the new session.

The first deployment assigned the seed directly to ``Master.state`` — and
lost it: a seeded snapshot at idx ``-1`` is indistinguishable from "fresh
empty state", which the event router deliberately skips hydrating, so the
promoted node's own worker (whose bootstrap races the promotion) never saw
it. Indexing the seed as an ordinary ``StateSnapshotHydrated`` event gives
every consumer exactly one delivery path: in the snapshot for late
bootstrappers, as live event 0 for early ones.
"""

import anyio
import pytest

from skulk.master.main import Master
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshotHydrated,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import Receiver, channel


def _master(
    initial_state: State | None,
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
    )
    return master, global_recv


async def test_seed_indexed_as_first_event():
    seed = seed_state_for_new_session(State(tracing_enabled=True))
    master, global_recv = _master(seed)
    # Construction must NOT pre-assign the seed — it flows through the log.
    assert master.state.instances == {}

    received: list[GlobalForwarderEvent] = []

    async def consume() -> None:
        with global_recv as events:
            async for event in events:
                received.append(event)
                return

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(consume)
            await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]

    # State applied from the indexed seed event at idx 0.
    assert master.state.tracing_enabled is True
    assert master.state.last_event_applied_idx == 0
    # The seed was broadcast as an ordinary indexed event — the path early
    # bootstrappers (including the promoted node's own worker) consume.
    assert len(received) == 1
    assert received[0].origin_idx == 0
    assert isinstance(received[0].event, StateSnapshotHydrated)
    assert received[0].event.state.last_event_applied_idx == 0


async def test_no_seed_indexes_nothing():
    master, _global_recv = _master(None)
    with anyio.fail_after(10):
        await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]
    assert master.state.instances == {}
    assert master.state.last_event_applied_idx == -1


@pytest.mark.parametrize("tracing", [True, False])
def test_construction_never_pre_seeds(tracing: bool):
    master, _ = _master(seed_state_for_new_session(State(tracing_enabled=tracing)))
    assert master.state.last_event_applied_idx == -1
