from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from exo.download.download_utils import RepoDownloadProgress
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.store.config import StagingNodeConfig
from exo.store.model_store_client import ModelStoreClient, ModelStoreDownloader


class _FakeInnerDownloader(ShardDownloader):
    def __init__(self) -> None:
        self.callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        raise AssertionError("store-backed path should not fall back to inner downloader")

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.callbacks.append(callback)

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:
            yield (Path("/tmp/unused"), cast(RepoDownloadProgress, object()))

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        raise AssertionError("status queries are not used in this test")


class _FakeStoreClient:
    async def is_model_available(self, model_id: str) -> bool:
        assert model_id == "mlx-community/gemma-4-26b-a4b-it-4bit"
        return True

    async def stage_shard(
        self,
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        assert model_id == "mlx-community/gemma-4-26b-a4b-it-4bit"
        assert on_progress is not None
        await on_progress(512, 2048)
        await on_progress(2048, 2048)
        return dest_path


def _build_shard() -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
            storage_size=Memory.from_bytes(2048),
            n_layers=30,
            hidden_size=2816,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=30,
        n_layers=30,
    )


@pytest.mark.anyio
async def test_model_store_downloader_emits_real_stage_progress() -> None:
    shard = _build_shard()
    observed: list[RepoDownloadProgress] = []
    downloader = ModelStoreDownloader(
        inner=_FakeInnerDownloader(),
        store_client=cast(ModelStoreClient, cast(object, _FakeStoreClient())),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path="~/.exo/staging",
        ),
    )
    downloader.on_progress(lambda _shard, progress: _record_progress(observed, progress))

    path = await downloader.ensure_shard(shard)

    assert path == Path("~/.exo/staging").expanduser() / "mlx-community--gemma-4-26b-a4b-it-4bit"
    assert [progress.status for progress in observed] == [
        "in_progress",
        "in_progress",
        "complete",
    ]
    assert observed[0].downloaded.in_bytes == 512
    assert observed[0].downloaded_this_session.in_bytes == 512
    assert observed[0].total.in_bytes == 2048
    assert observed[1].downloaded.in_bytes == 2048
    assert observed[1].downloaded_this_session.in_bytes == 2048
    assert observed[1].total.in_bytes == 2048
    assert observed[2].downloaded.in_bytes == 2048
    assert observed[2].completed_files == 1


async def _record_progress(
    observed: list[RepoDownloadProgress], progress: RepoDownloadProgress
) -> None:
    observed.append(progress)
