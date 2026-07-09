# pyright: reportPrivateUsage=false
"""Pinned-GGUF completeness checks for node-local staging reuse."""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import (
    ModelStoreClient,
    ModelStoreDownloader,
    _staged_pinned_gguf_missing,
)

_MODEL_ID = "org/multi-quant-GGUF"


class _UnusedInnerDownloader(ShardDownloader):
    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        raise AssertionError("store-backed test must not use the inner downloader")

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        pass

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:
            yield (Path("/unused"), cast(RepoDownloadProgress, object()))

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        raise AssertionError("status queries are not used in this test")


class _RecordingStoreClient:
    def __init__(self) -> None:
        self.availability_checks = 0
        self.stage_calls = 0

    async def is_model_available(self, model_id: str) -> bool:
        assert model_id == _MODEL_ID
        self.availability_checks += 1
        return True

    async def stage_shard(
        self,
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        assert model_id == _MODEL_ID
        self.stage_calls += 1
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "model-IQ3_XXS.gguf").write_bytes(b"selected")
        return dest_path


def _shard(gguf_file: str) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(_MODEL_ID),
            storage_size=Memory.from_bytes(8),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            gguf_file=gguf_file,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def test_pinned_shard_group_requires_every_sibling(tmp_path: Path) -> None:
    shard = _shard("weights/model-IQ3-00001-of-00002.gguf")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model-IQ3-00001-of-00002.gguf").write_bytes(b"one")

    assert _staged_pinned_gguf_missing(shard, tmp_path) is True

    (weights / "model-IQ3-00002-of-00002.gguf").write_bytes(b"two")
    assert _staged_pinned_gguf_missing(shard, tmp_path) is False


@pytest.mark.anyio
async def test_different_staged_quant_is_replaced_from_store(tmp_path: Path) -> None:
    staged = tmp_path / "org--multi-quant-GGUF"
    staged.mkdir()
    (staged / "model-Q4_K_M.gguf").write_bytes(b"old")
    store = _RecordingStoreClient()
    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(ModelStoreClient, cast(object, store)),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(tmp_path),
        ),
    )

    path = await downloader.ensure_shard(_shard("model-IQ3_XXS.gguf"))

    assert path == staged
    assert store.availability_checks == 1
    assert store.stage_calls == 1
    assert (staged / "model-IQ3_XXS.gguf").is_file()
