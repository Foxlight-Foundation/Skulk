from collections.abc import Sequence

import anyio
import pytest

from skulk.master.main import (
    EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS,
    EVENT_LOG_REPLAY_CHUNK_SIZE,
    EventLogGrowthMonitor,
    Master,
)
from skulk.routing.router import get_node_id_keypair
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestEventLog,
    SetTracingEnabled,
)
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TestEvent,
    TracingStateChanged,
)
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import Receiver, Sender, channel


class _ReplayEventLog:
    def __init__(self, events: Sequence[Event], *, start_idx: int = 0) -> None:
        self._events = list(events)
        self.start_idx = start_idx

    def __len__(self) -> int:
        return self.start_idx + len(self._events)

    def read_range(self, start: int, end: int) -> list[Event]:
        offset_start = max(start - self.start_idx, 0)
        offset_end = max(end - self.start_idx, 0)
        return self._events[offset_start:offset_end]


def _make_master() -> tuple[
    Master,
    Receiver[GlobalForwarderEvent],
    Sender[ForwarderCommand],
    Receiver[Event],
]:
    node_id = NodeId(get_node_id_keypair().to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    _local_sender, local_receiver = channel[LocalForwarderEvent]()
    _state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, _state_sync_response = channel[StateSyncMessage]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()
    return (
        Master(
            node_id,
            session_id,
            event_sender=event_sender,
            global_event_sender=global_sender,
            local_event_receiver=local_receiver,
            command_receiver=command_receiver,
            state_sync_receiver=state_sync_receiver,
            state_sync_sender=state_sync_sender,
            download_command_sender=download_sender,
        ),
        global_receiver,
        command_sender,
        event_receiver,
    )


@pytest.mark.asyncio
async def test_event_log_replay_is_paced_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, global_receiver, _command_sender, _event_receiver = _make_master()
    event_count = (EVENT_LOG_REPLAY_CHUNK_SIZE * 2) + 3
    master._event_log = _ReplayEventLog(  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
        [TestEvent() for _ in range(event_count)]
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("skulk.master.main.anyio.sleep", record_sleep)

    await master._serve_event_log_replay(0)  # pyright: ignore[reportPrivateUsage]

    replay = global_receiver.collect()
    assert [event.origin_idx for event in replay] == list(range(event_count))
    assert sleeps == [
        EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS,
        EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS,
    ]


@pytest.mark.asyncio
async def test_replay_does_not_block_later_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, _global_receiver, command_sender, event_receiver = _make_master()
    master._event_log = _ReplayEventLog(  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
        [TestEvent() for _ in range(EVENT_LOG_REPLAY_CHUNK_SIZE + 1)]
    )
    replay_pacing_started = anyio.Event()
    release_replay = anyio.Event()

    async def block_replay_between_chunks(_delay: float) -> None:
        replay_pacing_started.set()
        await release_replay.wait()

    monkeypatch.setattr("skulk.master.main.anyio.sleep", block_replay_between_chunks)

    async with master._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("replay-requester"),
                command=RequestEventLog(since_idx=0),
            )
        )
        await replay_pacing_started.wait()

        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("operator"),
                command=SetTracingEnabled(enabled=True),
            )
        )
        with anyio.fail_after(1.0):
            generated = await event_receiver.receive()

        assert isinstance(generated, TracingStateChanged)
        assert generated.enabled is True
        release_replay.set()
        task_group.cancel_scope.cancel()


def test_event_log_replay_requests_coalesce_around_active_range() -> None:
    master, _global_receiver, _command_sender, _event_receiver = _make_master()
    master._replay_worker_running = True  # pyright: ignore[reportPrivateUsage]
    master._active_replay_next_idx = 10  # pyright: ignore[reportPrivateUsage]
    master._active_replay_end_idx = 20  # pyright: ignore[reportPrivateUsage]

    master._schedule_event_log_replay(15)  # pyright: ignore[reportPrivateUsage]
    assert master._pending_replay_start_idx is None  # pyright: ignore[reportPrivateUsage]

    master._schedule_event_log_replay(25)  # pyright: ignore[reportPrivateUsage]
    master._schedule_event_log_replay(5)  # pyright: ignore[reportPrivateUsage]
    assert master._pending_replay_start_idx == 5  # pyright: ignore[reportPrivateUsage]


def test_event_log_growth_monitor_warns_only_after_sustained_idle_growth() -> None:
    monitor = EventLogGrowthMonitor(
        window_seconds=60.0,
        warning_rate_per_minute=4.0,
        warning_cooldown_seconds=300.0,
    )

    assert monitor.observe(now=0.0, idle=True) is None
    assert monitor.observe(now=20.0, idle=True) is None
    assert monitor.observe(now=40.0, idle=True) is None
    assert monitor.observe(now=60.0, idle=True) == 4.0
    assert monitor.observe(now=61.0, idle=True) is None


def test_event_log_growth_monitor_resets_for_active_work() -> None:
    monitor = EventLogGrowthMonitor(
        window_seconds=60.0,
        warning_rate_per_minute=2.0,
        warning_cooldown_seconds=300.0,
    )

    assert monitor.observe(now=0.0, idle=True) is None
    assert monitor.observe(now=60.0, idle=False) is None
    assert monitor.observe(now=61.0, idle=True) is None
    assert monitor.observe(now=121.0, idle=True) == 2.0
