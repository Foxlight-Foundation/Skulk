"""Tests that re-downloading a previously deleted model completes successfully."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock, patch

import anyio

from skulk.download.coordinator import DownloadCoordinator
from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.impl_shard_downloader import SingletonShardDownloader
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from skulk.shared.types.common import NodeId, SystemId
from skulk.shared.types.events import Event, NodeDownloadProgress
from skulk.shared.types.memory import Memory
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import DownloadCompleted
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MODEL_ID = ModelId("test-org/test-model")


def _make_shard(model_id: ModelId = MODEL_ID) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


class FakeShardDownloader(ShardDownloader):
    """Fake downloader that simulates a successful download by firing the
    progress callback with status='complete' when ensure_shard is called."""

    def __init__(self) -> None:
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        # Simulate a completed download by firing the progress callback
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=1,
            total_files=1,
            downloaded=Memory.from_mb(100),
            downloaded_this_session=Memory.from_mb(100),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="complete",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        return Path("/fake/models") / shard.model_card.model_id.normalize()

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:  # noqa: SIM108  # empty async generator
            yield (
                Path(),
                RepoDownloadProgress(  # pyright: ignore[reportUnreachable]
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(),
                    completed_files=0,
                    total_files=0,
                    downloaded=Memory.from_bytes(0),
                    downloaded_this_session=Memory.from_bytes(0),
                    total=Memory.from_bytes(0),
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="not_started",
                ),
            )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="not_started",
        )


class SlowCancellationShardDownloader(FakeShardDownloader):
    """Downloader whose first cancelled attempt holds cleanup until released."""

    def __init__(self) -> None:
        super().__init__()
        self.ensure_calls = 0
        self.first_started = anyio.Event()
        self.release_cleanup = anyio.Event()
        self.restarted = anyio.Event()

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        self.ensure_calls += 1
        if self.ensure_calls == 1:
            self.first_started.set()
            try:
                await anyio.sleep_forever()
            finally:
                with anyio.CancelScope(shield=True):
                    await self.release_cleanup.wait()
        self.restarted.set()
        return await super().ensure_shard(shard)


async def test_re_download_after_delete_completes() -> None:
    """A model that was downloaded, deleted, and then re-downloaded should
    reach DownloadCompleted status. This is an end-to-end test through
    the DownloadCoordinator."""
    cmd_send: Sender[ForwarderDownloadCommand]
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    telemetry_send, _ = channel[NodeTelemetry]()

    fake_downloader = FakeShardDownloader()
    wrapped_downloader = SingletonShardDownloader(fake_downloader)
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=wrapped_downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )

    shard = _make_shard()
    origin = SystemId("test")

    with patch("skulk.download.coordinator.delete_model", new_callable=AsyncMock):
        # Run the coordinator in the background
        coordinator_task = asyncio.create_task(coordinator.run())

        try:
            # 1. Start first download
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
                )
            )

            # Wait for DownloadCompleted
            first_completed = await _wait_for_download_completed(event_recv, MODEL_ID)
            assert first_completed is not None, "First download should complete"

            # 2. Delete the model
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=DeleteDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
                )
            )
            # Give the coordinator time to process the delete
            await asyncio.sleep(0.05)

            # 3. Re-download the same model
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
                )
            )

            # Wait for second DownloadCompleted — this is the bug: it never arrives
            second_completed = await _wait_for_download_completed(event_recv, MODEL_ID)
            assert second_completed is not None, (
                "Re-download after deletion should complete"
            )
        finally:
            await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task


async def test_start_during_cancellation_runs_after_cleanup() -> None:
    """A replacement start must survive the prior task's cancellation window."""

    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    telemetry_send, _ = channel[NodeTelemetry]()
    downloader = SlowCancellationShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )
    shard = _make_shard()
    origin = SystemId("test")
    coordinator_task = asyncio.create_task(coordinator.run())

    try:
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )
        with anyio.fail_after(2):
            await downloader.first_started.wait()

        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            )
        )
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )
        await anyio.sleep(0)
        assert downloader.ensure_calls == 1

        downloader.release_cleanup.set()
        with anyio.fail_after(2):
            await downloader.restarted.wait()
        completed = await _wait_for_download_completed(event_recv, MODEL_ID)
        assert completed is not None
        assert downloader.ensure_calls == 2
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def _wait_for_download_completed(
    event_recv: Receiver[Event], model_id: ModelId, timeout: float = 2.0
) -> DownloadCompleted | None:
    """Drain events until we see a DownloadCompleted for the given model, or timeout."""
    try:
        async with asyncio.timeout(timeout):
            while True:
                event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and isinstance(event.download_progress, DownloadCompleted)
                    and event.download_progress.shard_metadata.model_card.model_id
                    == model_id
                ):
                    return event.download_progress
    except TimeoutError:
        return None
