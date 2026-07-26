# pyright: reportPrivateUsage=false
"""Disk-capacity admission for direct Hugging Face model downloads."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

import pytest

from skulk.download import impl_shard_downloader
from skulk.download.impl_shard_downloader import (
    DirectDownloadCapacityError,
    ResumableShardDownloader,
    _remaining_direct_download_bytes,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import (
    FileListEntry,
    RepoDownloadProgress,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


def _shard(model_id: str) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(model_id),
            storage_size=Memory.from_bytes(60),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def test_remaining_bytes_credit_partial_and_replaced_files(tmp_path: Path) -> None:
    (tmp_path / "complete.gguf").write_bytes(b"x" * 10)
    (tmp_path / "replace.gguf").write_bytes(b"x" * 3)
    (tmp_path / "resume.gguf.partial").write_bytes(b"x" * 4)
    file_list = [
        FileListEntry(type="file", path="complete.gguf", size=10),
        FileListEntry(type="file", path="replace.gguf", size=8),
        FileListEntry(type="file", path="resume.gguf", size=9),
        FileListEntry(type="file", path="new.gguf", size=7),
    ]

    assert _remaining_direct_download_bytes(tmp_path, file_list) == 5 + 5 + 7


def test_remaining_bytes_reject_unknown_manifest_sizes(tmp_path: Path) -> None:
    with pytest.raises(DirectDownloadCapacityError, match="unknown size"):
        _remaining_direct_download_bytes(
            tmp_path,
            [FileListEntry(type="file", path="missing.safetensors", size=None)],
        )


@pytest.mark.anyio
async def test_direct_download_fails_before_transfer_without_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = ResumableShardDownloader()

    class _DiskUsage:
        free = 50

    def disk_usage(_path: Path) -> _DiskUsage:
        return _DiskUsage()

    monkeypatch.setattr(
        impl_shard_downloader,
        "MINIMUM_STAGING_FREE_DISK_BYTES",
        100,
    )
    monkeypatch.setattr(
        impl_shard_downloader.shutil,
        "disk_usage",
        disk_usage,
    )

    with pytest.raises(DirectDownloadCapacityError):
        await downloader._ensure_direct_download_capacity(
            tmp_path,
            [FileListEntry(type="file", path="model.gguf", size=60)],
        )


@pytest.mark.anyio
async def test_direct_download_reuses_complete_artifact_without_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = ResumableShardDownloader()
    (tmp_path / "model.gguf").write_bytes(b"complete")

    def unexpected_disk_usage(_path: Path) -> object:
        raise AssertionError("zero-byte reuse must not inspect free space")

    monkeypatch.setattr(
        impl_shard_downloader.shutil,
        "disk_usage",
        unexpected_disk_usage,
    )

    await downloader._ensure_direct_download_capacity(
        tmp_path,
        [FileListEntry(type="file", path="model.gguf", size=len(b"complete"))],
    )


@pytest.mark.anyio
async def test_direct_download_admission_and_transfer_are_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = ResumableShardDownloader()
    active_transfers = 0
    maximum_active_transfers = 0

    async def fake_download_shard(
        shard: ShardMetadata,
        on_progress: Callable[
            [ShardMetadata, RepoDownloadProgress],
            Awaitable[None],
        ],
        *,
        max_parallel_downloads: int = 8,
        skip_download: bool = False,
        skip_internet: bool = False,
        allow_patterns: list[str] | None = None,
        on_connection_lost: Callable[[], None] = lambda: None,
        capacity_preflight: Callable[
            [Path, list[FileListEntry]],
            Awaitable[None],
        ]
        | None = None,
    ) -> tuple[Path, RepoDownloadProgress]:
        del (
            on_progress,
            max_parallel_downloads,
            skip_download,
            skip_internet,
            allow_patterns,
            on_connection_lost,
        )
        nonlocal active_transfers, maximum_active_transfers
        assert capacity_preflight is not None
        active_transfers += 1
        maximum_active_transfers = max(maximum_active_transfers, active_transfers)
        await asyncio.sleep(0.01)
        active_transfers -= 1
        return (
            tmp_path / str(shard.model_card.model_id).replace("/", "--"),
            RepoDownloadProgress(
                repo_id=shard.model_card.model_id,
                repo_revision="main",
                shard=shard,
                completed_files=1,
                total_files=1,
                downloaded=Memory.from_bytes(60),
                downloaded_this_session=Memory.from_bytes(60),
                total=Memory.from_bytes(60),
                overall_speed=1.0,
                overall_eta=timedelta(0),
                status="complete",
                file_progress={},
            ),
        )

    monkeypatch.setattr(impl_shard_downloader, "download_shard", fake_download_shard)

    await asyncio.gather(
        downloader.ensure_shard(_shard("org/first")),
        downloader.ensure_shard(_shard("org/second")),
    )

    assert maximum_active_transfers == 1
