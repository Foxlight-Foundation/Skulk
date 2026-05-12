"""Tests for LARQL vindex download path handling."""

from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.impl_shard_downloader import ResumableShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import LarqlShardMetadata


class _SuccessfulProcess:
    async def wait(self) -> int:
        return 0


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
    assert recorded_commands == [
        (
            "larql",
            "pull",
            shard.vindex_uri,
            "--output",
            str(target_dir),
        )
    ]
