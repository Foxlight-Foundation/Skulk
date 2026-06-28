"""Store-host selective GGUF download (#339).

When the store host downloads a multi-quant GGUF repo from HuggingFace, it
should fetch exactly what the direct-HuggingFace path fetches: the preferred
quant's shard group plus ``config.json``, and nothing else (not other quants,
not ``original/*`` full-precision weights, not ``metal/*`` artifacts).
"""

from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.model_store import (
    has_gguf_projector,
    select_store_gguf_download_files,
)


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


def test_mmproj_projector_is_kept_for_vision_models() -> None:
    # The GGUF allow-list now includes the multimodal projector (#346), so the
    # store keeps it alongside the LM quant and config -- a vision GGUF model
    # needs the projector or llama.cpp cannot do image inference. Still matches
    # the direct-HF path, which includes the same projector glob.
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("mmproj-model-f16.gguf", 600),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf", "mmproj-model-f16.gguf"}


def test_has_gguf_projector_detects_mmproj() -> None:
    # The store uses this to (a) detect a vision GGUF from its full repo listing
    # and (b) verify the projector actually landed before registering (#346).
    assert has_gguf_projector(["model-Q4_K_M.gguf", "mmproj-F16.gguf"]) is True
    assert has_gguf_projector(["nested/mmproj-model-f16.gguf"]) is True
    # Name is case-insensitive; the .gguf extension follows the HF convention
    # (and the resolver/runner): lowercase extension required.
    assert has_gguf_projector(["MMPROJ-F16.gguf"]) is True
    assert has_gguf_projector(["mmproj-F32.GGUF"]) is False  # uppercase ext


def test_has_gguf_projector_matches_basename_not_directory() -> None:
    # Matches the runner's find_mmproj_file (basename), so a non-projector GGUF
    # under a directory whose name contains "mmproj" is NOT a false positive.
    assert has_gguf_projector(["mmproj-tools/model.gguf"]) is False
    assert has_gguf_projector(["sub/mmproj-f16.gguf"]) is True


def test_has_gguf_projector_false_without_projector() -> None:
    # A text-only GGUF set has no projector; an mmproj-named non-gguf file does
    # not count (the projector is always a .gguf).
    assert has_gguf_projector(["model-Q4_K_M.gguf", "config.json"]) is False
    assert has_gguf_projector(["mmproj-notes.txt"]) is False
    assert has_gguf_projector([]) is False


def test_mixed_case_projector_name_is_kept() -> None:
    # The projector NAME is matched case-insensitively (matching the card
    # resolver and the runner's find_mmproj_file), so an uppercase-named
    # projector with the conventional lowercase .gguf extension is still kept and
    # selected -- detection and selection stay aligned.
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("MMPROJ-F16.gguf", 600),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files)}
    assert kept == {"config.json", "model-Q4_K_M.gguf", "MMPROJ-F16.gguf"}


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


def test_pinned_quant_overrides_default_preference() -> None:
    # #344: when the card pins a non-default quant, the store fetches THAT quant's
    # shard group, not its default preference (Q4_K_M).
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),  # would be the default pick
        _entry("model-Q8_0.gguf", 1300),
    ]
    kept = {e.path for e in select_store_gguf_download_files(files, "model-Q8_0.gguf")}
    assert kept == {"config.json", "model-Q8_0.gguf"}


def test_pinned_sharded_quant_keeps_its_whole_group() -> None:
    files = [
        _entry("config.json"),
        _entry("m-Q4_K_M.gguf", 800),
        _entry("m-Q6_K-00001-of-00002.gguf", 900),
        _entry("m-Q6_K-00002-of-00002.gguf", 950),
    ]
    kept = {
        e.path
        for e in select_store_gguf_download_files(files, "m-Q6_K-00001-of-00002.gguf")
    }
    assert kept == {
        "config.json",
        "m-Q6_K-00001-of-00002.gguf",
        "m-Q6_K-00002-of-00002.gguf",
    }


def test_pin_absent_from_repo_falls_back_to_default() -> None:
    # A stale/typo'd pin not present in the repo degrades to the default pick
    # rather than selecting nothing.
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("model-Q8_0.gguf", 1300),
    ]
    kept = {
        e.path for e in select_store_gguf_download_files(files, "model-Q3_K_M.gguf")
    }
    assert kept == {"config.json", "model-Q4_K_M.gguf"}  # default preference


def test_pinned_none_matches_prior_default_behavior() -> None:
    files = [
        _entry("config.json"),
        _entry("model-Q4_K_M.gguf", 800),
        _entry("model-Q8_0.gguf", 1300),
    ]
    assert (
        {e.path for e in select_store_gguf_download_files(files, None)}
        == {e.path for e in select_store_gguf_download_files(files)}
        == {"config.json", "model-Q4_K_M.gguf"}
    )


def test_extra_pinned_same_repo_draft_is_cofetched() -> None:
    # A served-engine draft GGUF bundled in the base repo (served_spec_draft_repo
    # == base repo) must be co-fetched with the base quant in one store download,
    # or the served runner's --model-draft path 404s on the staged dir.
    files = [
        _entry("config.json"),
        _entry("gemma-4-31B-it-IQ4_XS.gguf", 16000),
        _entry("mtp-gemma-4-31B-it.gguf", 500),
        _entry("gemma-4-31B-it-Q8_0.gguf", 30000),  # other quant: dropped
    ]
    kept = {
        e.path
        for e in select_store_gguf_download_files(
            files,
            "gemma-4-31B-it-IQ4_XS.gguf",
            ["mtp-gemma-4-31B-it.gguf"],
        )
    }
    assert kept == {
        "config.json",
        "gemma-4-31B-it-IQ4_XS.gguf",
        "mtp-gemma-4-31B-it.gguf",
    }


def test_extra_pinned_sharded_draft_keeps_whole_group() -> None:
    files = [
        _entry("config.json"),
        _entry("base-Q4_K_M.gguf", 800),
        _entry("draft-00001-of-00002.gguf", 200),
        _entry("draft-00002-of-00002.gguf", 200),
    ]
    kept = {
        e.path
        for e in select_store_gguf_download_files(
            files, "base-Q4_K_M.gguf", ["draft-00001-of-00002.gguf"]
        )
    }
    assert kept == {
        "config.json",
        "base-Q4_K_M.gguf",
        "draft-00001-of-00002.gguf",
        "draft-00002-of-00002.gguf",
    }


def test_extra_pin_absent_from_repo_is_skipped() -> None:
    # An extra pin that names no file in the repo is dropped (warned), not an
    # error: a genuinely absent draft surfaces loudly later at runner launch.
    files = [
        _entry("config.json"),
        _entry("base-Q4_K_M.gguf", 800),
    ]
    kept = {
        e.path
        for e in select_store_gguf_download_files(
            files, "base-Q4_K_M.gguf", ["nonexistent-draft.gguf"]
        )
    }
    assert kept == {"config.json", "base-Q4_K_M.gguf"}


def test_extra_pinned_none_matches_prior_behavior() -> None:
    files = [
        _entry("config.json"),
        _entry("base-Q4_K_M.gguf", 800),
    ]
    assert (
        {e.path for e in select_store_gguf_download_files(files, "base-Q4_K_M.gguf", None)}
        == {e.path for e in select_store_gguf_download_files(files, "base-Q4_K_M.gguf")}
        == {"config.json", "base-Q4_K_M.gguf"}
    )
