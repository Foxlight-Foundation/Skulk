# pyright: reportPrivateUsage=false
"""Canonical model-store download cancellation behavior."""

import asyncio
from pathlib import Path

import pytest

import skulk.store.model_store as model_store_module
from skulk.download import download_utils
from skulk.download.download_utils import FileListEntry
from skulk.shared.models.model_cards import ModelId
from skulk.store.model_store import ModelStore, StoreDownloadStatus


async def test_cancel_store_download_is_idempotent_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation stops work, preserves status, and permits a later retry."""

    transfer_started = asyncio.Event()

    async def file_list(
        _model_id: ModelId,
        _revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert recursive
        return [FileListEntry(type="file", path="model.safetensors", size=1)]

    async def blocked_download(*_args: object, **_kwargs: object) -> Path:
        transfer_started.set()
        await asyncio.Event().wait()
        raise AssertionError("the cancelled transfer must not resume")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", blocked_download)
    # Cancellation behavior must not depend on the development host having the
    # production store's 10 GiB operating reserve available. Capacity admission
    # has dedicated tests; this fixture needs to reach its mocked transfer.
    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)

    first_status = await store.request_download("org/model")
    await transfer_started.wait()
    cancelled_status = await store.cancel_download("org/model")

    assert cancelled_status is first_status
    assert cancelled_status is not None
    assert cancelled_status.status == "cancelled"
    assert store.list_active_downloads() == []
    assert store._download_tasks == set()
    assert store._download_tasks_by_model == {}
    assert await store.cancel_download("org/model") is cancelled_status

    transfer_started.clear()
    retry_status = await store.request_download("org/model")
    await transfer_started.wait()

    assert retry_status is not first_status
    assert retry_status.status == "downloading"
    await store.cancel_download("org/model")


async def test_cancel_store_download_rejects_unknown_work(
    tmp_path: Path,
) -> None:
    """An unknown transfer is not cancellable."""

    store = ModelStore(tmp_path)

    assert await store.cancel_download("org/missing") is None
    store._active_downloads["org/complete"] = StoreDownloadStatus(
        model_id="org/complete",
        status="complete",
        progress=1.0,
    )
    assert await store.cancel_download("org/complete") is None
