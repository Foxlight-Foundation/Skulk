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
    _replacement_identity_for_installed_card,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import (
    FileListEntry,
    RepoDownloadProgress,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
    write_installed_card,
)


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


def test_changed_custom_card_requires_new_direct_generation(tmp_path: Path) -> None:
    """A sidecar mismatch must force a separate direct-download generation."""

    model_directory = tmp_path / "org--model"
    model_directory.mkdir()
    (model_directory / "weights.bin").write_bytes(b"old")
    old_card = _shard("org/model").model_card.model_copy(
        update={"is_custom": True}
    )
    new_card = old_card.model_copy(update={"hidden_size": 2})
    write_installed_card(
        model_directory,
        build_installed_card_record(model_directory, old_card),
    )

    assert (
        _replacement_identity_for_installed_card(
            model_directory,
            old_card,
            artifact_model_id="org/model",
            artifact_role="base",
        )
        is None
    )
    replacement_identity = _replacement_identity_for_installed_card(
        model_directory,
        new_card,
        artifact_model_id="org/model",
        artifact_role="base",
    )
    assert replacement_identity is not None
    assert replacement_identity == _replacement_identity_for_installed_card(
        model_directory,
        new_card,
        artifact_model_id="org/model",
        artifact_role="base",
    )


@pytest.mark.anyio
async def test_config_only_card_change_never_starts_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config probe cannot atomically replace complete installed weights."""

    model_directory = tmp_path / "org--model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text("{}")
    (model_directory / "weights.bin").write_bytes(b"installed-weights")
    old_card = _shard("org/model").model_card.model_copy(
        update={"is_custom": True}
    )
    new_card = old_card.model_copy(update={"hidden_size": 2})
    write_installed_card(
        model_directory,
        build_installed_card_record(model_directory, old_card),
    )
    monkeypatch.setattr(impl_shard_downloader, "SKULK_MODELS_DIR", tmp_path)
    observed: dict[str, object] = {}

    class ProbeCompleteError(RuntimeError):
        """Stop after observing direct-download call arguments."""

    async def observe_download(
        _self: ResumableShardDownloader,
        _shard_metadata: ShardMetadata,
        *,
        allow_patterns: list[str] | None = None,
        replacement_identity: str | None = None,
    ) -> tuple[Path, RepoDownloadProgress]:
        observed["allow_patterns"] = allow_patterns
        observed["replacement_identity"] = replacement_identity
        raise ProbeCompleteError

    monkeypatch.setattr(
        ResumableShardDownloader,
        "_download_with_capacity",
        observe_download,
    )
    downloader = ResumableShardDownloader()
    shard = _shard("org/model").model_copy(update={"model_card": new_card})

    with pytest.raises(ProbeCompleteError):
        await downloader.ensure_shard(shard, config_only=True)

    assert observed == {
        "allow_patterns": ["config.json"],
        "replacement_identity": None,
    }
    assert (model_directory / "weights.bin").read_bytes() == b"installed-weights"


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
    registered_model_ids: list[ModelId] = []

    def register_installed_card(record: InstalledCardRecord) -> None:
        registered_model_ids.append(record.model_card.model_id)

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
        model_path = tmp_path / str(shard.model_card.model_id).replace("/", "--")
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "weights.bin").write_bytes(b"complete")
        return (
            model_path,
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
    monkeypatch.setattr(
        impl_shard_downloader,
        "register_installed_card_record",
        register_installed_card,
    )

    await asyncio.gather(
        downloader.ensure_shard(_shard("org/first")),
        downloader.ensure_shard(_shard("org/second")),
    )

    assert maximum_active_transfers == 1
    assert registered_model_ids == [ModelId("org/first"), ModelId("org/second")]
