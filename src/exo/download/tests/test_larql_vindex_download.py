"""Tests for LARQL vindex download path handling."""

from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import VindexPathResolution
from exo.download.impl_shard_downloader import ResumableShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.shards import LarqlShardMetadata
from exo.utils.channels import channel


class _SuccessfulProcess:
    async def wait(self) -> int:
        return 0


class _FailedProcess:
    async def wait(self) -> int:
        return 1


def _larql_shard() -> LarqlShardMetadata:
    return LarqlShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("skulk/test-vindex"),
            storage_size=Memory.from_mb(128),
            n_layers=12,
            hidden_size=2048,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=12,
        n_layers=12,
        vindex_uri="hf://skulk/test-vindex",
        preset="expert-server",
    )


@pytest.mark.asyncio
async def test_larql_vindex_pull_uses_configured_models_dir(tmp_path: Path) -> None:
    """Pulled vindexes are written under the configured writable models dir."""

    models_dir = tmp_path / "configured-models"
    shard = _larql_shard()
    recorded_commands: list[tuple[str, ...]] = []

    async def fake_create_subprocess_exec(*args: str) -> _SuccessfulProcess:
        recorded_commands.append(tuple(args))
        output_dir = Path(args[-1])
        output_dir.mkdir(parents=True)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "weights.bin").write_bytes(b"vindex")
        return _SuccessfulProcess()

    with (
        patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir),
        patch(
            "exo.download.impl_shard_downloader.resolve_vindex_in_path",
            return_value=None,
        ),
        patch("asyncio.create_subprocess_exec", new=fake_create_subprocess_exec),
    ):
        result = await ResumableShardDownloader().ensure_shard(shard)

    target_dir = models_dir / shard.model_card.model_id.normalize()
    assert result == target_dir
    assert target_dir.is_dir()
    assert recorded_commands[0][:4] == ("larql", "pull", shard.vindex_uri, "--output")
    assert recorded_commands[0][4].startswith(str(models_dir / f".{target_dir.name}"))


@pytest.mark.asyncio
async def test_larql_vindex_pull_cleans_partial_directory_on_failure(
    tmp_path: Path,
) -> None:
    """Failed pulls leave no reusable partial vindex directory behind."""

    models_dir = tmp_path / "configured-models"
    shard = _larql_shard()
    partial_dir: Path | None = None

    async def fake_create_subprocess_exec(*args: str) -> _FailedProcess:
        nonlocal partial_dir
        partial_dir = Path(args[-1])
        partial_dir.mkdir(parents=True)
        (partial_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (partial_dir / "weights.bin").write_bytes(b"partial")
        return _FailedProcess()

    with (
        patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir),
        patch(
            "exo.download.impl_shard_downloader.resolve_vindex_in_path",
            return_value=None,
        ),
        patch("asyncio.create_subprocess_exec", new=fake_create_subprocess_exec),
        pytest.raises(RuntimeError, match="larql pull failed"),
    ):
        await ResumableShardDownloader().ensure_shard(shard)

    target_dir = models_dir / shard.model_card.model_id.normalize()
    assert partial_dir is not None
    assert not partial_dir.exists()
    assert not target_dir.exists()


@pytest.mark.asyncio
async def test_larql_vindex_writable_cache_hit_remains_deletable(
    tmp_path: Path,
) -> None:
    """Coordinator preserves writable ownership when reusing local vindexes."""

    _, command_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()
    shard = _larql_shard()
    vindex_dir = tmp_path / "configured-models" / shard.model_card.model_id.normalize()
    coordinator = DownloadCoordinator(
        node_id=NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        shard_downloader=ResumableShardDownloader(),
        download_command_receiver=command_receiver,
        event_sender=event_sender,
    )

    with patch(
        "exo.download.coordinator.resolve_vindex_location",
        return_value=VindexPathResolution(path=vindex_dir, read_only=False),
    ):
        await coordinator._start_download(shard)  # pyright: ignore[reportPrivateUsage]

    event = await event_receiver.receive()

    assert isinstance(event, NodeDownloadProgress)
    assert isinstance(event.download_progress, DownloadCompleted)
    assert event.download_progress.model_directory == str(vindex_dir)
    assert not event.download_progress.read_only
