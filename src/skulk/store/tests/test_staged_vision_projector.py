# pyright: reportPrivateUsage=false
"""Staged-directory projector completeness, scoped to GGUF vision models (#346).

A GGUF vision model carries a separate ``mmproj`` projector that the generic
completeness probe ignores, so a staged dir without it must be re-staged. An
MLX / safetensors vision model bundles its vision weights and has no separate
projector, so it must NOT be flagged (doing so would disable the staged-cache
fast path that keeps inference working when the store is unreachable).
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    VisionCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store_client import (
    _remove_invalid_staged_projector,
    _staged_vision_projector_missing,
    _staged_vision_projector_missing_async,
)


def _shard(
    *,
    vision: bool,
    gguf_file: str | None,
    projector_file: str | None = None,
    projector_size: int | None = None,
) -> PipelineShardMetadata:
    card = ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_gb(1.0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file=gguf_file,
        source_revision="a" * 40 if projector_file is not None else None,
        vision=(
            VisionCardConfig(
                projector_file=projector_file,
                projector_size=projector_size,
            )
            if vision
            else None
        ),
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


def test_pinned_projector_requires_exact_path_and_size(tmp_path: Path) -> None:
    """Another projector variant cannot satisfy an immutable card selection."""

    staged = _write(
        tmp_path,
        "model-Q4_K_M.gguf",
        "mmproj-Q4_K_M.gguf",
        "config.json",
    )
    shard = _shard(
        vision=True,
        gguf_file="model-Q4_K_M.gguf",
        projector_file="mmproj-F16.gguf",
        projector_size=1,
    )

    assert _staged_vision_projector_missing(shard, staged) is True
    (staged / "mmproj-F16.gguf").write_bytes(b"wrong")
    assert _staged_vision_projector_missing(shard, staged) is True
    (staged / "mmproj-F16.gguf").write_bytes(b"x")
    write_installed_card(
        staged,
        build_installed_card_record(staged, shard.model_card),
    )
    assert _staged_vision_projector_missing(shard, staged) is False


def test_same_size_corrupt_projector_is_removed_for_recovery(tmp_path: Path) -> None:
    """Manifest-invalid staged bytes cannot survive the store recovery path."""

    staged = _write(
        tmp_path,
        "model-Q4_K_M.gguf",
        "mmproj-F16.gguf",
        "config.json",
    )
    shard = _shard(
        vision=True,
        gguf_file="model-Q4_K_M.gguf",
        projector_file="mmproj-F16.gguf",
        projector_size=1,
    )
    write_installed_card(
        staged,
        build_installed_card_record(staged, shard.model_card),
    )
    (staged / "mmproj-F16.gguf").write_bytes(b"y")

    assert _staged_vision_projector_missing(shard, staged) is True
    _remove_invalid_staged_projector(shard, staged)

    assert not (staged / "mmproj-F16.gguf").exists()


def test_corrupt_projector_symlink_is_unlinked_without_following(
    tmp_path: Path,
) -> None:
    """Recovery removes the staged symlink entry and preserves its target."""

    staged = _write(tmp_path / "staged", "model-Q4_K_M.gguf", "config.json")
    external = tmp_path / "external-projector.gguf"
    external.write_bytes(b"x")
    (staged / "mmproj-F16.gguf").symlink_to(external)
    shard = _shard(
        vision=True,
        gguf_file="model-Q4_K_M.gguf",
        projector_file="mmproj-F16.gguf",
        projector_size=1,
    )

    _remove_invalid_staged_projector(shard, staged)

    assert not (staged / "mmproj-F16.gguf").exists()
    assert external.read_bytes() == b"x"


async def test_async_projector_check_hashes_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker's cache fast path delegates projector hashing to a thread."""

    staged = _write(
        tmp_path,
        "model-Q4_K_M.gguf",
        "mmproj-F16.gguf",
        "config.json",
    )
    shard = _shard(
        vision=True,
        gguf_file="model-Q4_K_M.gguf",
        projector_file="mmproj-F16.gguf",
        projector_size=1,
    )
    write_installed_card(
        staged,
        build_installed_card_record(staged, shard.model_card),
    )
    delegated: list[Callable[..., bool]] = []

    async def _to_thread(function: Callable[..., bool], *args: object) -> bool:
        delegated.append(function)
        return function(*args)

    monkeypatch.setattr("skulk.store.model_store_client.asyncio.to_thread", _to_thread)

    assert await _staged_vision_projector_missing_async(shard, staged) is False
    assert delegated == [_staged_vision_projector_missing]


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
