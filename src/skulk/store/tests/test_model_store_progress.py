from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store import model_store_client
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient, ModelStoreDownloader


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
    async def is_model_available(
        self, model_id: str, source_revision: str | None = None
    ) -> bool:
        assert source_revision is None
        assert model_id == "mlx-community/gemma-4-26b-a4b-it-4bit"
        return True

    async def stage_shard(
        self,
        model_id: str,
        staging_root: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        source_revision: str | None = None,
    ) -> Path:
        assert source_revision is None
        assert model_id == "mlx-community/gemma-4-26b-a4b-it-4bit"
        assert on_progress is not None
        await on_progress(512, 2048)
        await on_progress(2048, 2048)
        return staging_root.expanduser() / model_id.replace("/", "--")


class _FakeRegistryResponse:
    status = 200

    async def __aenter__(self) -> "_FakeRegistryResponse":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def json(self) -> object:
        return [{"model_id": "org/model", "total_bytes": 1_000}]


class _FakeRegistrySession:
    async def __aenter__(self) -> "_FakeRegistrySession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def get(self, _url: str) -> _FakeRegistryResponse:
        return _FakeRegistryResponse()


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


@pytest.mark.anyio
async def test_http_stage_uses_registered_total_for_resumed_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ModelStoreClient(store_host="store.local", store_port=58080)
    model_id = "org/model"
    dest_path = tmp_path / "staging"
    dest_path.mkdir()
    (dest_path / "config.json").write_bytes(b"x" * 250)
    observed: list[tuple[int, int]] = []
    download_sizes = {
        "weights-1.safetensors": 300,
        "weights-2.safetensors": 450,
    }

    async def fetch_file_list(
        _model_id: str, source_revision: str | None
    ) -> list[str]:
        assert source_revision is None
        return ["config.json", *download_sizes]

    async def fetch_model_total_bytes(
        _model_id: str, source_revision: str | None
    ) -> int:
        assert source_revision is None
        return 1_000

    async def download_store_file(
        _model_id: str,
        file_path: str,
        _dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None,
        total_bytes_offset: int,
        grand_total: int,
        source_revision: str | None,
    ) -> int:
        assert source_revision is None
        size = download_sizes[file_path]
        assert on_progress is not None
        await on_progress(total_bytes_offset + size, grand_total)
        return size

    async def record_progress(downloaded: int, total: int) -> None:
        observed.append((downloaded, total))

    monkeypatch.setattr(client, "_fetch_file_list", fetch_file_list)
    monkeypatch.setattr(client, "_fetch_model_total_bytes", fetch_model_total_bytes)
    monkeypatch.setattr(client, "_download_store_file", download_store_file)

    path = await client._stage_http(  # pyright: ignore[reportPrivateUsage]
        model_id,
        dest_path,
        record_progress,
        None,
    )

    assert path == dest_path
    assert observed == [(550, 1_000), (1_000, 1_000)]


@pytest.mark.anyio
async def test_registered_total_lookup_reads_canonical_store_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_create_http_session(**_kwargs: object) -> _FakeRegistrySession:
        return _FakeRegistrySession()

    monkeypatch.setattr(
        model_store_client,
        "create_http_session",
        fake_create_http_session,
    )
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    total_bytes = await client._fetch_model_total_bytes(  # pyright: ignore[reportPrivateUsage]
        "org/model"
    )

    assert total_bytes == 1_000


async def _record_progress(
    observed: list[RepoDownloadProgress], progress: RepoDownloadProgress
) -> None:
    observed.append(progress)
