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
