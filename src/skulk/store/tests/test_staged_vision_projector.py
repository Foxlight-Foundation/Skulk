# pyright: reportPrivateUsage=false
"""Staged-directory projector completeness, scoped to GGUF vision models (#346).

A GGUF vision model carries a separate ``mmproj`` projector that the generic
completeness probe ignores, so a staged dir without it must be re-staged. An
MLX / safetensors vision model bundles its vision weights and has no separate
projector, so it must NOT be flagged (doing so would disable the staged-cache
fast path that keeps inference working when the store is unreachable).
"""

from pathlib import Path

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    VisionCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.model_store_client import _staged_vision_projector_missing


def _shard(
    *, vision: bool, gguf_file: str | None
) -> PipelineShardMetadata:
    card = ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_gb(1.0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file=gguf_file,
        vision=VisionCardConfig() if vision else None,
    )
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _write(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x")
    return directory


def test_gguf_vision_without_projector_is_flagged(tmp_path: Path) -> None:
    staged = _write(tmp_path, "model-Q4_K_M.gguf", "config.json")
    shard = _shard(vision=True, gguf_file="model-Q4_K_M.gguf")
    assert _staged_vision_projector_missing(shard, staged) is True


def test_gguf_vision_with_projector_is_complete(tmp_path: Path) -> None:
    staged = _write(tmp_path, "model-Q4_K_M.gguf", "mmproj-F16.gguf", "config.json")
    shard = _shard(vision=True, gguf_file="model-Q4_K_M.gguf")
    assert _staged_vision_projector_missing(shard, staged) is False


def test_mlx_vision_without_projector_is_not_flagged(tmp_path: Path) -> None:
    # An MLX vision model has no GGUF projector; it must not be flagged, or the
    # staged-cache fast path is wrongly disabled for it.
    staged = _write(tmp_path, "model.safetensors", "config.json")
    shard = _shard(vision=True, gguf_file=None)
    assert _staged_vision_projector_missing(shard, staged) is False


def test_non_vision_gguf_is_not_flagged(tmp_path: Path) -> None:
    staged = _write(tmp_path, "model-Q4_K_M.gguf", "config.json")
    shard = _shard(vision=False, gguf_file="model-Q4_K_M.gguf")
    assert _staged_vision_projector_missing(shard, staged) is False
