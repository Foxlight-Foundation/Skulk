"""Worker staging-capacity preflight tests."""

import os
from pathlib import Path
from typing import NamedTuple

import pytest

from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event, IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.utils.channels import channel
from skulk.worker import main as worker_main
from skulk.worker.main import Worker

_GIB = 1024**3


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _sparse_model(root: Path, model_id: str, size_bytes: int) -> Path:
    directory = root / model_id.replace("/", "--")
    directory.mkdir(parents=True)
    with (directory / "model.safetensors.partial").open("wb") as partial:
        partial.truncate(size_bytes)
    return directory


def _shard(model_id: str, size_bytes: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(model_id),
            storage_size=Memory.from_bytes(size_bytes),
            n_layers=32,
            hidden_size=2048,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


def _worker(staging_root: Path, *, cleanup: bool) -> Worker:
    _, event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    return Worker(
        node_id=NodeId("node-a"),
        event_receiver=event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(staging_root),
            cleanup_on_deactivate=cleanup,
            staging_keep_recent_gb=40,
        ),
    )


@pytest.mark.asyncio
async def test_capacity_preflight_evicts_idle_cache_and_keeps_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = _sparse_model(tmp_path, "org/incoming", 8 * _GIB)
    idle = _sparse_model(tmp_path, "org/idle", 15 * _GIB)
    def _disk_usage(_path: os.PathLike[str] | str) -> _DiskUsage:
        free_bytes = 20 * _GIB if not idle.exists() else 5 * _GIB
        return _DiskUsage(100 * _GIB, 100 * _GIB - free_bytes, free_bytes)

    monkeypatch.setattr(worker_main.shutil, "disk_usage", _disk_usage)
    worker = _worker(tmp_path, cleanup=True)

    error = await worker._prepare_staging_capacity(  # pyright: ignore[reportPrivateUsage]
        _shard("org/incoming", 12 * _GIB)
    )

    assert error is None
    assert incoming.exists()
    assert not idle.exists()


@pytest.mark.asyncio
async def test_capacity_preflight_fails_before_writing_when_cleanup_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = _sparse_model(tmp_path, "org/incoming", 8 * _GIB)
    idle = _sparse_model(tmp_path, "org/idle", 15 * _GIB)
    def _disk_usage(_path: os.PathLike[str] | str) -> _DiskUsage:
        return _DiskUsage(100 * _GIB, 95 * _GIB, 5 * _GIB)

    monkeypatch.setattr(worker_main.shutil, "disk_usage", _disk_usage)
    worker = _worker(tmp_path, cleanup=False)

    error = await worker._prepare_staging_capacity(  # pyright: ignore[reportPrivateUsage]
        _shard("org/incoming", 12 * _GIB)
    )

    assert error is not None
    assert "Insufficient staging capacity" in error
    assert incoming.exists()
    assert idle.exists()
