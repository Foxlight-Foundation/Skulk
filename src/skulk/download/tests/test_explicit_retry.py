"""Explicit operator download retries consume failures without automatic retry loops."""

# pyright: reportPrivateUsage=false

from dataclasses import dataclass
from pathlib import Path
from typing import final

import anyio
import httpx
import pytest

from skulk.api.main import API
from skulk.download import coordinator as coordinator_module
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.tests.test_re_download import FakeShardDownloader, _make_shard
from skulk.routing.topics import DOWNLOAD_COMMANDS
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RetryDownload,
    StartDownload,
)
from skulk.shared.types.common import ModelId, NodeId, SystemId
from skulk.shared.types.events import Event, IndexedEvent, NodeDownloadProgress
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadProgress,
)
from skulk.shared.types.worker.shards import ShardMetadata
from skulk.store.installed_cards import (
    build_installed_card_record,
    require_registry_installed_artifact,
    write_installed_card,
)
from skulk.utils.channels import Receiver, Sender, channel


@final
class _RetryDownloader(FakeShardDownloader):
    """Inject one transient refusal, then finalize actual test-owned bytes."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.calls = 0
        self.failures_remaining = 1
        self.started = anyio.Event()
        self.finish = anyio.Event()
        self.finish.set()

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        """Retain partial files and publish installed identity before completion."""
        self.calls += 1
        self.started.set()
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("Injected transient capacity refusal")
        await self.finish.wait()
        self.directory.mkdir(exist_ok=True)
        (self.directory / "config.json").write_text("{}")
        (self.directory / "model.gguf").write_bytes(b"verified fixture weights")
        (self.directory / ".skulk-source-revision").write_text("a" * 40 + "\n")
        write_installed_card(
            self.directory,
            build_installed_card_record(self.directory, shard.model_card),
        )
        await super().ensure_shard(shard, config_only)
        return self.directory


@dataclass(frozen=True)
class _Harness:
    """Actual API, command processor and durable-progress channels under test."""

    api: API
    coordinator: DownloadCoordinator
    downloader: _RetryDownloader
    shard: ShardMetadata
    commands: Sender[ForwarderDownloadCommand]
    events: Receiver[Event]


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """Construct isolated actors with an authoritative immutable catalog fixture."""
    node = NodeId("retry-node")
    shard = _make_shard(
        source_revision="a" * 40,
        gguf_file="model.gguf",
        registry_card_id="card_" + "a" * 52,
    )
    commands, received_commands = channel[ForwarderDownloadCommand]()
    events, received_events = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    forwarded, _ = channel[ForwarderCommand]()
    _, indexed = channel[IndexedEvent]()
    _, election = channel[ElectionMessage]()
    api = API(
        node,
        port=52415,
        event_receiver=indexed,
        command_sender=forwarded,
        download_command_sender=commands,
        election_receiver=election,
        enable_event_log=False,
        mount_dashboard=False,
    )
    downloader = _RetryDownloader(tmp_path / "installed")
    coordinator = DownloadCoordinator(
        node,
        downloader,
        received_commands,
        events,
        telemetry,
    )

    async def load(_model_id: ModelId) -> ModelCard:
        """Authorize only this fixture's exact immutable card."""
        return shard.model_card

    monkeypatch.setattr(ModelCard, "load", staticmethod(load))
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", tmp_path / "models")
    return _Harness(api, coordinator, downloader, shard, commands, received_events)


async def _post(harness: _Harness) -> None:
    """Submit through the real authorized ASGI route, not a private reset."""
    transport = httpx.ASGITransport(app=harness.api.app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost"
    ) as client:
        response = await client.post(
            "/download/start",
            json={
                "targetNodeId": str(harness.coordinator.node_id),
                "shardMetadata": harness.shard.model_dump(mode="json"),
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["commandId"]


async def _terminal(harness: _Harness) -> DownloadCompleted | DownloadFailed:
    """Wait for an actual ordered terminal event from the same coordinator."""
    while True:
        event = await harness.events.receive()
        if isinstance(event, NodeDownloadProgress) and isinstance(
            event.download_progress, (DownloadCompleted, DownloadFailed)
        ):
            return event.download_progress


async def _idle(harness: _Harness) -> None:
    """Wait until the same failed task has released its ownership."""
    while harness.coordinator.active_downloads:
        await anyio.sleep(0)


async def test_explicit_api_retry_starts_new_attempt_and_preserves_partial_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Correcting a transient refusal can succeed without deletion or restart."""
    harness = _harness(tmp_path, monkeypatch)
    directory = harness.downloader.directory
    directory.mkdir()
    partial = directory / "retained.partial"
    partial.write_bytes(b"previous partial bytes")
    with anyio.fail_after(6):
        async with harness.coordinator._tg:
            harness.coordinator._tg.start_soon(harness.coordinator._command_processor)
            await _post(harness)
            failed = await _terminal(harness)
            assert isinstance(failed, DownloadFailed)
            await _idle(harness)
            await _post(harness)
            completed = await _terminal(harness)
            assert isinstance(completed, DownloadCompleted)
            assert completed.attempt_id != failed.attempt_id
            assert harness.downloader.calls == 2
            assert partial.read_bytes() == b"previous partial bytes"
            assert (
                directory / "model.gguf"
            ).read_bytes() == b"verified fixture weights"
            require_registry_installed_artifact(directory, harness.shard.model_card)
            harness.coordinator._tg.cancel_tasks()


async def test_ordinary_planner_start_retains_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated ordinary starts never turn a persistent failure into a loop."""
    harness = _harness(tmp_path, monkeypatch)
    with anyio.fail_after(6):
        async with harness.coordinator._tg:
            harness.coordinator._tg.start_soon(harness.coordinator._command_processor)
            await _post(harness)
            failed = await _terminal(harness)
            await _idle(harness)
            for _ in range(4):
                await harness.commands.send(
                    ForwarderDownloadCommand(
                        origin=SystemId(),
                        command=StartDownload(
                            target_node_id=harness.coordinator.node_id,
                            shard_metadata=harness.shard,
                        ),
                    )
                )
                assert await _terminal(harness) == failed
            assert harness.downloader.calls == 1
            harness.coordinator._tg.cancel_tasks()


async def test_explicit_starts_keep_active_and_completed_attempts_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry authorization cannot create a second concurrent transfer."""
    harness = _harness(tmp_path, monkeypatch)
    harness.downloader.failures_remaining = 0
    harness.downloader.finish = anyio.Event()
    with anyio.fail_after(6):
        async with harness.coordinator._tg:
            harness.coordinator._tg.start_soon(harness.coordinator._command_processor)
            await _post(harness)
            await harness.downloader.started.wait()
            await _post(harness)
            while harness.commands.statistics().tasks_waiting_receive != 1:
                await anyio.sleep(0)
            assert harness.downloader.calls == 1
            harness.downloader.finish.set()
            completed = await _terminal(harness)
            assert isinstance(completed, DownloadCompleted)
            await _idle(harness)
            await _post(harness)
            assert await _terminal(harness) == completed
            assert harness.downloader.calls == 1
            harness.coordinator._tg.cancel_tasks()


async def test_retry_waits_for_failed_task_cleanup_and_suppresses_old_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry arriving during failure publication hands off the original owner."""
    harness = _harness(tmp_path, monkeypatch)
    release = anyio.Event()
    emit = harness.coordinator._emit_status

    async def hold_failure(status: DownloadProgress) -> DownloadProgress:
        """Keep the failing task alive after its actual terminal publication."""
        prepared = await emit(status)
        if isinstance(status, DownloadFailed):
            await release.wait()
        return prepared

    monkeypatch.setattr(harness.coordinator, "_emit_status", hold_failure)
    with anyio.fail_after(6):
        async with harness.coordinator._tg:
            harness.coordinator._tg.start_soon(harness.coordinator._command_processor)
            await _post(harness)
            failed = await _terminal(harness)
            assert isinstance(failed, DownloadFailed)
            assert harness.coordinator.active_downloads
            await _post(harness)
            while not harness.coordinator._pending_download_starts:
                await anyio.sleep(0)
            assert harness.downloader.calls == 1
            progress = await harness.downloader.get_shard_download_status_for_shard(
                harness.shard
            )
            await harness.coordinator._download_progress_callback(
                harness.shard, progress.model_copy(update={"status": "in_progress"})
            )
            assert (
                harness.shard.model_card.model_id
                not in harness.coordinator.download_status
            )
            release.set()
            completed = await _terminal(harness)
            assert isinstance(completed, DownloadCompleted)
            assert completed.attempt_id != failed.attempt_id
            assert harness.downloader.calls == 2
            harness.coordinator._tg.cancel_tasks()


async def test_explicit_retry_preserves_offline_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry does not bypass the coordinator's offline download boundary."""
    harness = _harness(tmp_path, monkeypatch)
    harness.coordinator.offline = True
    with anyio.fail_after(6):
        async with harness.coordinator._tg:
            harness.coordinator._tg.start_soon(harness.coordinator._command_processor)
            await _post(harness)
            failed = await _terminal(harness)
            assert isinstance(failed, DownloadFailed)
            await _post(harness)
            retried = await _terminal(harness)
            assert isinstance(retried, DownloadFailed)
            assert retried.attempt_id != failed.attempt_id
            assert harness.downloader.calls == 0
            harness.coordinator._tg.cancel_tasks()


def test_retry_wire_variant_preserves_ordinary_start_shape() -> None:
    """Explicit retry round-trips without adding fields to legacy planner starts."""
    shard = _make_shard()
    origin = SystemId()
    ordinary = ForwarderDownloadCommand(
        origin=origin,
        command=StartDownload(target_node_id=NodeId("node"), shard_metadata=shard),
    )
    explicit = ForwarderDownloadCommand(
        origin=origin,
        command=RetryDownload(target_node_id=NodeId("node"), shard_metadata=shard),
    )
    ordinary_bytes = DOWNLOAD_COMMANDS.serialize(ordinary)
    explicit_bytes = DOWNLOAD_COMMANDS.serialize(explicit)
    assert b"RetryDownload" not in ordinary_bytes
    assert b"retryFailed" not in ordinary_bytes
    assert isinstance(
        DOWNLOAD_COMMANDS.deserialize(explicit_bytes).command, RetryDownload
    )
