"""Tests for startup discovery of resumable direct downloads."""

from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

import pytest

from skulk.download import impl_shard_downloader
from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


def _model_card(model_id: str) -> ModelCard:
    """Build the minimum text model card needed by the status scanner."""

    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(7),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model.gguf",
    )


def _shard(model_card: ModelCard) -> PipelineShardMetadata:
    """Build a complete single-rank shard for one test model."""

    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _progress(shard: ShardMetadata) -> RepoDownloadProgress:
    """Build a terminal progress snapshot returned by the fake downloader."""

    return RepoDownloadProgress(
        repo_id=str(shard.model_card.model_id),
        repo_revision="main",
        shard=shard,
        completed_files=1,
        total_files=1,
        downloaded=Memory.from_bytes(7),
        downloaded_this_session=Memory.from_bytes(0),
        total=Memory.from_bytes(7),
        overall_speed=0,
        overall_eta=timedelta(0),
        status="complete",
        file_progress={},
    )


async def _collect_status(
    downloader: ResumableShardDownloader,
) -> list[tuple[Path, RepoDownloadProgress]]:
    """Collect the async status iterator into a list."""

    return [item async for item in downloader.get_shard_download_status()]


@pytest.mark.asyncio
async def test_startup_scan_does_not_query_untouched_catalog_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fresh cache must cause zero remote-backed download status probes."""

    model_cards = [_model_card("org/one"), _model_card("org/two")]
    probed_models: list[ModelId] = []

    async def get_model_cards() -> list[ModelCard]:
        return model_cards

    async def build_full_shard(model_id: ModelId) -> PipelineShardMetadata:
        probed_models.append(model_id)
        return _shard(next(card for card in model_cards if card.model_id == model_id))

    async def download_shard(
        shard: ShardMetadata,
        on_progress: Callable[
            [ShardMetadata, RepoDownloadProgress], Awaitable[None]
        ],
        **_kwargs: object,
    ) -> tuple[Path, RepoDownloadProgress]:
        del on_progress
        return tmp_path / shard.model_card.model_id.normalize(), _progress(shard)

    monkeypatch.setattr(impl_shard_downloader, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(impl_shard_downloader, "get_model_cards", get_model_cards)
    monkeypatch.setattr(
        impl_shard_downloader, "build_full_shard", build_full_shard
    )
    monkeypatch.setattr(impl_shard_downloader, "download_shard", download_shard)

    results = await _collect_status(ResumableShardDownloader())

    assert results == []
    assert probed_models == []


@pytest.mark.asyncio
async def test_startup_scan_queries_only_models_with_local_download_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An interrupted local model remains discoverable without probing peers."""

    interrupted_card = _model_card("org/interrupted")
    untouched_card = _model_card("org/untouched")
    model_cards = [interrupted_card, untouched_card]
    interrupted_directory = tmp_path / interrupted_card.model_id.normalize()
    interrupted_directory.mkdir()
    (interrupted_directory / "model.gguf.partial").write_bytes(b"partial")
    (tmp_path / untouched_card.model_id.normalize()).mkdir()
    probed_models: list[ModelId] = []

    async def get_model_cards() -> list[ModelCard]:
        return model_cards

    async def build_full_shard(model_id: ModelId) -> PipelineShardMetadata:
        probed_models.append(model_id)
        return _shard(next(card for card in model_cards if card.model_id == model_id))

    async def download_shard(
        shard: ShardMetadata,
        on_progress: Callable[
            [ShardMetadata, RepoDownloadProgress], Awaitable[None]
        ],
        **_kwargs: object,
    ) -> tuple[Path, RepoDownloadProgress]:
        del on_progress
        return interrupted_directory, _progress(shard)

    monkeypatch.setattr(impl_shard_downloader, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(impl_shard_downloader, "get_model_cards", get_model_cards)
    monkeypatch.setattr(
        impl_shard_downloader, "build_full_shard", build_full_shard
    )
    monkeypatch.setattr(impl_shard_downloader, "download_shard", download_shard)

    results = await _collect_status(ResumableShardDownloader())

    assert len(results) == 1
    assert results[0][0] == interrupted_directory
    assert probed_models == [interrupted_card.model_id]
