"""GGUF staged-directory completeness recognition (slice 3a, downloader side)."""

from pathlib import Path

from skulk.download.download_utils import (
    directory_has_gguf_weights,
    is_model_directory_complete,
)


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
