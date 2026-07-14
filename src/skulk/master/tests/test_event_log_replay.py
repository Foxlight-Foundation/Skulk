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
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
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
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.worker.downloads import (
    DownloadOngoing,
    DownloadPending,
    DownloadProgressData,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
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

    def append(self, event: Event) -> None:
        self._events.append(event)


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


@pytest.mark.asyncio
async def test_event_log_replay_clamps_request_beyond_current_tail() -> None:
    master, global_receiver, _command_sender, _event_receiver = _make_master()
    master._event_log = _ReplayEventLog(  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
        [TestEvent() for _ in range(3)]
    )

    await master._serve_event_log_replay(8)  # pyright: ignore[reportPrivateUsage]

    assert global_receiver.collect() == []
    assert master._active_replay_next_idx == 3  # pyright: ignore[reportPrivateUsage]
    assert master._active_replay_end_idx == 3  # pyright: ignore[reportPrivateUsage]


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


def _download_state(*, ongoing: bool) -> State:
    node_id = NodeId("download-node")
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test/model"),
            storage_size=Memory.from_mb(100),
            n_layers=2,
            hidden_size=8,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=2,
        n_layers=2,
    )
    if ongoing:
        progress = DownloadOngoing(
            node_id=node_id,
            shard_metadata=shard,
            download_progress=DownloadProgressData(
                total=Memory.from_mb(100),
                downloaded=Memory.from_mb(50),
                downloaded_this_session=Memory.from_mb(50),
                completed_files=1,
                total_files=2,
                speed=1.0,
                eta_ms=1_000,
                files={},
            ),
        )
    else:
        progress = DownloadPending(
            node_id=node_id,
            shard_metadata=shard,
            total=Memory.from_mb(100),
        )
    return State(downloads={node_id: [progress]})


def test_pending_download_does_not_suppress_idle_growth_alarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, _global_receiver, _command_sender, _event_receiver = _make_master()
    master.state = _download_state(ongoing=False)
    master._event_log = _ReplayEventLog([])  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
    observed_idle: list[bool] = []

    def observe(
        _monitor: EventLogGrowthMonitor, *, now: float, idle: bool
    ) -> None:
        del now
        observed_idle.append(idle)

    monkeypatch.setattr(EventLogGrowthMonitor, "observe", observe)
    master._append_event_log(TestEvent())  # pyright: ignore[reportPrivateUsage]

    assert observed_idle == [True]


def test_ongoing_download_resets_idle_growth_alarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, _global_receiver, _command_sender, _event_receiver = _make_master()
    master.state = _download_state(ongoing=True)
    master._event_log = _ReplayEventLog([])  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
    observed_idle: list[bool] = []

    def observe(
        _monitor: EventLogGrowthMonitor, *, now: float, idle: bool
    ) -> None:
        del now
        observed_idle.append(idle)

    monkeypatch.setattr(EventLogGrowthMonitor, "observe", observe)
    master._append_event_log(TestEvent())  # pyright: ignore[reportPrivateUsage]

    assert observed_idle == [False]
