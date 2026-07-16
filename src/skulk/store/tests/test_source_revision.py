# pyright: reportPrivateUsage=false
"""Immutable source-revision behavior for store downloads and staging."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from skulk.store.model_store import ModelStore
from skulk.store.model_store_client import ModelStoreClient

_OLD_REVISION = "0" * 40
_NEW_REVISION = "1" * 40


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
        destination,
        source_revision=_NEW_REVISION,
    )

    assert (staged / "model.gguf").read_bytes() == b"new"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
