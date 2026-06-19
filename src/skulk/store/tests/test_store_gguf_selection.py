"""Store-host selective GGUF download (#339).

When the store host downloads a multi-quant GGUF repo from HuggingFace, it
should fetch only the preferred quant's shard group plus the non-weight files,
not every quantization, mirroring the direct-HF selective allow-patterns.
"""

from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.model_store import select_store_gguf_download_files


def _entry(path: str, size: int = 100) -> FileListEntry:
    return FileListEntry(type="file", path=path, size=size)


def test_multi_quant_repo_keeps_only_preferred_quant() -> None:
    files = [
        _entry("config.json"),
        _entry("tokenizer.json"),
        _entry("model-BF16.gguf", 4000),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("model-Q8_0.gguf", 1300),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    # Q4_K_M is the preferred quant; other quants are dropped; non-weights stay.
    assert kept == {"config.json", "tokenizer.json", "model-Q4_K_M.gguf"}


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


def test_mmproj_projector_is_kept() -> None:
    # An mmproj projector is a companion, not an alternate quant: keep it.
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("model-BF16.gguf", 4000),
        _entry("mmproj-model-f16.gguf", 600),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf", "mmproj-model-f16.gguf"}


def test_non_gguf_repo_unchanged() -> None:
    files = [
        _entry("config.json"),
        _entry("model.safetensors.index.json"),
        _entry("model-00001-of-00002.safetensors", 5000),
        _entry("model-00002-of-00002.safetensors", 5000),
    ]
    kept = select_store_gguf_download_files(files)
    assert kept == files  # no .gguf weights: returned unchanged


def test_single_quant_repo_unchanged() -> None:
    # A repo that ships exactly one quant should keep it (and config).
    files = [_entry("config.json"), _entry("model-Q4_K_M.gguf", 800)]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf"}
