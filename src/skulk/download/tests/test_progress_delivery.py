"""Regression coverage for ordered, bounded repository progress delivery."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.download import download_utils
from skulk.download.download_utils import (
    download_shard,
    map_repo_download_progress_to_download_progress_data,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import (
    FileListEntry,
    RepoDownloadProgress,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


def _shard() -> ShardMetadata:
    card = ModelCard(
        model_id=ModelId("test-org/progress-order"),
        storage_size=Memory.from_bytes(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


@pytest.mark.asyncio
async def test_chunk_callbacks_are_coalesced_and_terminal_progress_is_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A slow callback cannot leave stale progress behind completion."""

    async def file_list(*args: object, **kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="weights.bin", size=100)]

    async def downloaded_size(path: Path) -> int:
        return 0

    async def path_exists(path: str | Path) -> bool:
        return False

    async def fake_download(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None],
        skip_internet: bool,
    ) -> Path:
        # Let the first callback enter a deliberately slow consumer, then queue
        # a chunk storm followed by completion without yielding.
        on_progress(1, 100, False)
        await asyncio.sleep(0)
        for downloaded in range(2, 100):
            on_progress(downloaded, 100, False)
        on_progress(100, 100, True)
        await asyncio.sleep(0)
        return target_dir / path

    seen: list[RepoDownloadProgress] = []

    async def collect(
        shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        if progress.status == "in_progress" and progress.downloaded.in_bytes == 1:
            await asyncio.sleep(0.05)
        seen.append(progress)

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "get_downloaded_size", downloaded_size)
    monkeypatch.setattr(download_utils.aios.path, "exists", path_exists)
    monkeypatch.setattr(download_utils, "download_file_with_retry", fake_download)

    await download_shard(_shard(), collect, allow_patterns=["*"])
    await asyncio.sleep(0.1)

    assert seen[-1].status == "complete"
    assert len(seen) <= 4, f"chunk callbacks were not coalesced: {len(seen)}"

    aggregate = map_repo_download_progress_to_download_progress_data(
        seen[-1], include_files=False
    )
    detailed = map_repo_download_progress_to_download_progress_data(seen[-1])
    assert aggregate.files == {}
    assert set(detailed.files) == {"weights.bin"}
    assert len(aggregate.model_dump_json()) < len(detailed.model_dump_json())


@pytest.mark.asyncio
async def test_cancellation_stops_the_owned_progress_emitter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancelling a shard download cannot leave a progress callback alive."""

    async def file_list(*args: object, **kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="weights.bin", size=100)]

    async def downloaded_size(path: Path) -> int:
        return 0

    async def path_exists(path: str | Path) -> bool:
        return False

    download_blocked = asyncio.Event()

    async def fake_download(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None],
        skip_internet: bool,
    ) -> Path:
        on_progress(1, 100, False)
        await download_blocked.wait()
        return target_dir / path

    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def collect(
        shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            callback_cancelled.set()
            raise

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "get_downloaded_size", downloaded_size)
    monkeypatch.setattr(download_utils.aios.path, "exists", path_exists)
    monkeypatch.setattr(download_utils, "download_file_with_retry", fake_download)

    task = asyncio.create_task(
        download_shard(_shard(), collect, allow_patterns=["*"])
    )
    await asyncio.wait_for(callback_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert callback_cancelled.is_set()
