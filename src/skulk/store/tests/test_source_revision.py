# pyright: reportPrivateUsage=false
"""Immutable source-revision behavior for store downloads and staging."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from skulk.download import download_utils
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.model_store import ModelStore, StoreDownloadStatus
from skulk.store.model_store_client import ModelStoreClient

_OLD_REVISION = "0" * 40
_NEW_REVISION = "1" * 40


async def test_pinned_store_download_writes_revision_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A qualified store entry must remain discoverable by generic runners."""

    store = ModelStore(tmp_path)
    model_id = "org/model"
    store._active_downloads[model_id] = StoreDownloadStatus(
        model_id=model_id,
        source_revision=_NEW_REVISION,
    )

    async def file_list(
        _model_id: ModelId,
        revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert revision == _NEW_REVISION
        assert recursive is True
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    async def download_file(
        _model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None] = lambda: None,
        skip_internet: bool = False,
    ) -> Path:
        del on_connection_lost, skip_internet
        assert revision == _NEW_REVISION
        target = target_dir / path
        target.write_bytes(b"weights")
        on_progress(7, 7, True)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download_file)

    await store._do_download(
        model_id,
        pinned_gguf="model.gguf",
        source_revision=_NEW_REVISION,
    )

    entry = store.get_entry(model_id)
    assert entry is not None
    model_dir = tmp_path / entry.store_path
    assert entry.source_revision == _NEW_REVISION
    assert (model_dir / ".skulk-source-revision").read_text().strip() == _NEW_REVISION


async def test_store_redownloads_when_registered_revision_differs(
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    model_id = "org/model"
    model_dir = tmp_path / "org--model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"old")
    store.register_model(
        model_id,
        model_dir,
        ["model.gguf"],
        3,
        source_revision=_OLD_REVISION,
    )

    status = await store.request_download(
        model_id,
        pinned_gguf="model.gguf",
        source_revision=_NEW_REVISION,
    )

    assert status.status in {"pending", "downloading"}
    assert status.source_revision == _NEW_REVISION
    for task in tuple(store._download_tasks):
        task.cancel()
    await asyncio.gather(*store._download_tasks, return_exceptions=True)


async def test_staging_replaces_files_from_another_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "org--model"
    destination.mkdir()
    (destination / ".skulk-source-revision").write_text(f"{_OLD_REVISION}\n")
    (destination / "model.gguf").write_bytes(b"old")
    unrelated = tmp_path / "unrelated-cache-root"
    unrelated.mkdir()
    (unrelated / "sentinel").write_text("preserve")

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        source_revision: str | None,
    ) -> Path:
        assert source_revision == _NEW_REVISION
        assert not (dest_path / "model.gguf").exists()
        (dest_path / "model.gguf").write_bytes(b"new")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    staged = await client.stage_shard(
        "org/model",
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert (staged / "model.gguf").read_bytes() == b"new"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
    assert (unrelated / "sentinel").read_text() == "preserve"


async def test_staging_recovers_from_corrupted_revision_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A malformed staging marker is a mismatch, not a terminal load error."""

    destination = tmp_path / "org--model"
    destination.mkdir()
    (destination / ".skulk-source-revision").write_bytes(b"\xff")
    (destination / "model.gguf").write_bytes(b"old")

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        _source_revision: str | None,
    ) -> Path:
        (dest_path / "model.gguf").write_bytes(b"new")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    staged = await client.stage_shard(
        "org/model",
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert (staged / "model.gguf").read_bytes() == b"new"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION


async def test_pinned_staging_retry_preserves_partial_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A retry for the same pinned revision must retain resumable bytes."""

    attempts = 0

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        source_revision: str | None,
    ) -> Path:
        nonlocal attempts
        attempts += 1
        assert source_revision == _NEW_REVISION
        partial = dest_path / "model.gguf.partial"
        if attempts == 1:
            partial.write_bytes(b"partial")
            raise ConnectionError("transfer interrupted")
        assert partial.read_bytes() == b"partial"
        partial.replace(dest_path / "model.gguf")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    with pytest.raises(ConnectionError, match="transfer interrupted"):
        await client.stage_shard(
            "org/model",
            tmp_path,
            source_revision=_NEW_REVISION,
        )

    destination = tmp_path / "org--model"
    assert (destination / "model.gguf.partial").read_bytes() == b"partial"
    assert (
        destination / ".skulk-source-revision-staging"
    ).read_text().strip() == _NEW_REVISION

    staged = await client.stage_shard(
        "org/model",
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert attempts == 2
    assert (staged / "model.gguf").read_bytes() == b"partial"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
    assert not (staged / ".skulk-source-revision-staging").exists()


async def test_request_rechecks_revision_after_waiting_for_download_lock(
    tmp_path: Path,
) -> None:
    """A lock wait must not reuse a stale pre-lock revision comparison."""

    store = ModelStore(tmp_path)
    model_id = "org/model"
    model_dir = tmp_path / "org--model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    store.register_model(
        model_id,
        model_dir,
        ["model.safetensors"],
        len(b"weights"),
        source_revision=_NEW_REVISION,
    )

    await store._download_lock.acquire()
    request = asyncio.create_task(
        store.request_download(model_id, source_revision=_NEW_REVISION)
    )
    await asyncio.sleep(0)
    store.register_model(
        model_id,
        model_dir,
        ["model.safetensors"],
        len(b"weights"),
        source_revision=_OLD_REVISION,
    )
    store._download_lock.release()

    status = await request

    assert status.status in {"pending", "downloading"}
    assert status.source_revision == _NEW_REVISION
    for task in tuple(store._download_tasks):
        task.cancel()
    await asyncio.gather(*store._download_tasks, return_exceptions=True)
