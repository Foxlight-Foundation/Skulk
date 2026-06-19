"""Store-host selective GGUF download (#339).

When the store host downloads a multi-quant GGUF repo from HuggingFace, it
should fetch exactly what the direct-HuggingFace path fetches: the preferred
quant's shard group plus ``config.json``, and nothing else (not other quants,
not ``original/*`` full-precision weights, not ``metal/*`` artifacts).
"""

from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.model_store import select_store_gguf_download_files


def _entry(path: str, size: int = 100) -> FileListEntry:
    return FileListEntry(type="file", path=path, size=size)


def test_multi_quant_repo_keeps_only_preferred_quant_and_config() -> None:
    files = [
        _entry("config.json"),
        _entry("tokenizer.json"),
        _entry("model-BF16.gguf", 4000),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("model-Q8_0.gguf", 1300),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    # Q4_K_M is preferred; other quants and tokenizer.json are dropped to match
    # the direct-HF allow-list (gguf group + config.json only).
    assert kept == {"config.json", "model-Q4_K_M.gguf"}


def test_sharded_preferred_quant_keeps_whole_group() -> None:
    files = [
        _entry("config.json"),
        _entry("big-Q4_K_M-00001-of-00002.gguf", 500),
        _entry("big-Q4_K_M-00002-of-00002.gguf", 600),
        _entry("big-BF16-00001-of-00003.gguf", 3000),
        _entry("big-BF16-00002-of-00003.gguf", 3000),
        _entry("big-BF16-00003-of-00003.gguf", 3000),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {
        "config.json",
        "big-Q4_K_M-00001-of-00002.gguf",
        "big-Q4_K_M-00002-of-00002.gguf",
    }


def test_original_and_metal_weight_artifacts_are_dropped() -> None:
    # Some GGUF repos ship the original full-precision weights and metal/*
    # artifacts; these must NOT be downloaded (the direct-HF path ignores them).
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("original/consolidated.safetensors", 16000),
        _entry("original/params.json"),
        _entry("metal/ggml-common.h"),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf"}


def test_mmproj_projector_is_dropped_to_match_direct_path() -> None:
    # The direct-HF GGUF allow-list does not include the projector, so the store
    # matches it (multimodal GGUF projector handling is a separate concern).
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("mmproj-model-f16.gguf", 600),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf"}


def test_non_gguf_repo_unchanged() -> None:
    files = [
        _entry("config.json"),
        _entry("model.safetensors.index.json"),
        _entry("model-00001-of-00002.safetensors", 5000),
        _entry("model-00002-of-00002.safetensors", 5000),
    ]
    kept = select_store_gguf_download_files(files)
    assert kept == files  # no .gguf weights: returned unchanged


def test_matches_repo_relative_paths_like_the_direct_path() -> None:
    # Patterns match repo-relative paths (HuggingFace's allow_patterns basis):
    # a non-root config.json is not the model config and is dropped, just as the
    # direct path's "config.json" allow-pattern would not match a subdir file.
    files = [
        _entry("config.json"),
        _entry("nested/config.json"),
        _entry("model-Q4_K_M.gguf", 800),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf"}


def test_single_quant_repo_keeps_quant_and_config() -> None:
    files = [
        _entry("config.json"),
        _entry("tokenizer.json"),
        _entry("model-Q4_K_M.gguf", 800),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf"}
