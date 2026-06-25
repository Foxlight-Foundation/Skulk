"""GGUF staged-directory completeness recognition (slice 3a, downloader side)."""

from pathlib import Path

import pytest

from skulk.download import download_utils
from skulk.download.download_utils import (
    build_model_path,
    directory_has_gguf_weights,
    is_model_directory_complete,
)
from skulk.shared.types.common import ModelId


def test_gguf_dir_recognized_complete(tmp_path: Path) -> None:
    # A GGUF repo has no *.safetensors.index.json, so the safetensors probe
    # returns None; the directory must still be recognized as complete.
    (tmp_path / "model-q4.gguf").write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text("{}")
    assert directory_has_gguf_weights(tmp_path)
    assert is_model_directory_complete(tmp_path)


def test_mmproj_only_is_not_complete(tmp_path: Path) -> None:
    # An mmproj projector is not the LM weights; on its own the dir is not done.
    (tmp_path / "mmproj-model.gguf").write_bytes(b"GGUF")
    assert not directory_has_gguf_weights(tmp_path)
    assert not is_model_directory_complete(tmp_path)


def test_in_progress_gguf_partial_is_not_complete(tmp_path: Path) -> None:
    # An in-progress download is *.gguf.partial and must not count as complete.
    (tmp_path / "model-q4.gguf.partial").write_bytes(b"GG")
    assert not directory_has_gguf_weights(tmp_path)
    assert not is_model_directory_complete(tmp_path)


def test_empty_dir_not_complete(tmp_path: Path) -> None:
    assert not is_model_directory_complete(tmp_path)


def test_sharded_gguf_complete_only_with_all_shards(tmp_path: Path) -> None:
    # All shards present -> complete.
    for i in (1, 2, 3):
        (tmp_path / f"model-{i:05d}-of-00003.gguf").write_bytes(b"GGUF")
    assert directory_has_gguf_weights(tmp_path)
    assert is_model_directory_complete(tmp_path)


def test_sharded_gguf_incomplete_missing_shard(tmp_path: Path) -> None:
    # Only the first shard finalized (no .partial) -> NOT complete; otherwise the
    # store would skip restaging and llama.cpp would fail to find the rest.
    (tmp_path / "model-00001-of-00003.gguf").write_bytes(b"GGUF")
    assert not directory_has_gguf_weights(tmp_path)
    assert not is_model_directory_complete(tmp_path)


def _no_path_match(_model_id: ModelId) -> Path | None:
    """A resolve_model_in_path stub: no SKULK_MODELS_PATH hit."""
    return None


def test_build_model_path_accepts_bare_gguf_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare GGUF repo (no config.json) downloaded to the default models dir is
    resolvable at load time, not just when it lands in SKULK_MODELS_PATH (#327).
    """
    model_id = ModelId("org/bare-gguf-repo")
    model_dir = tmp_path / model_id.normalize()
    model_dir.mkdir(parents=True)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"GGUF")  # no config.json

    # No SKULK_MODELS_PATH match, default dir is the tmp models dir.
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "resolve_model_in_path", _no_path_match)

    assert build_model_path(model_id) == model_dir


def test_build_model_path_missing_is_filenotfound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "resolve_model_in_path", _no_path_match)
    with pytest.raises(FileNotFoundError):
        build_model_path(ModelId("org/not-on-disk"))


async def test_resolve_allow_patterns_gguf_vs_safetensors() -> None:
    """A GGUF card restricts the download to its pinned shard group + config.json;
    a non-GGUF card keeps the broad ["*"] fetch (#332)."""
    from skulk.download.download_utils import resolve_allow_patterns
    from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
    from skulk.shared.types.memory import Memory
    from skulk.shared.types.worker.shards import PipelineShardMetadata

    def _shard(card: ModelCard) -> PipelineShardMetadata:
        return PipelineShardMetadata(
            model_card=card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=card.n_layers,
            n_layers=card.n_layers,
        )

    def _card(model_id: str, gguf_file: str | None = None) -> ModelCard:
        return ModelCard(
            model_id=ModelId(model_id),
            storage_size=Memory.from_gb(1),
            n_layers=16,
            hidden_size=2048,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            gguf_file=gguf_file,
        )

    gguf = _card("o/r", gguf_file="m-Q4_K_M.gguf")
    # LM quant + the always-included multimodal projector glob (#346) + config.
    assert await resolve_allow_patterns(_shard(gguf)) == [
        "m-Q4_K_M.gguf",
        "*mmproj*.gguf",
        "config.json",
    ]

    mlx = _card("o/r2")
    assert await resolve_allow_patterns(_shard(mlx)) == ["*"]
