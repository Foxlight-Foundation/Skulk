# pyright: reportPrivateUsage=false
"""Tests for the served-runner SSE delta parser and spec-flag mapping.

The runner's HTTP/subprocess plumbing is exercised live on a GPU node; this
covers the pure parsing surface (the subtle part) without a server.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from skulk.worker.runner.llama_server.runner import (
    _SPEC_TYPE_FLAG,
    _draft_model_args,
    _gpu_layers_for_backend,
    _parse_sse_line,
)


def _runtime(repo: str | None = None, file: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(served_spec_draft_repo=repo, served_spec_draft_file=file)


def test_draft_args_none_when_no_draft_and_not_required() -> None:
    # draft_mtp without a draft = baked-in heads (Qwen); ngram needs no model.
    assert _draft_model_args(_runtime(), "draft_mtp") == []
    assert _draft_model_args(_runtime(), "ngram") == []
    assert _draft_model_args(None, "draft_mtp") == []


def test_draft_args_required_modes_raise_without_draft() -> None:
    for mode in ("draft_simple", "draft_eagle3"):
        with pytest.raises(RuntimeError, match="requires a draft model"):
            _draft_model_args(_runtime(), mode)


def test_draft_args_repo_without_file_raises() -> None:
    with pytest.raises(RuntimeError, match="served_spec_draft_file is"):
        _draft_model_args(_runtime(repo="org/draft-GGUF"), "draft_mtp")


def test_draft_args_resolves_model_draft_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With a draft repo+file present on disk, returns --model-draft <path>.
    (tmp_path / "draft.gguf").write_bytes(b"GGUF")
    import skulk.download.download_utils as du

    def _fake_build_model_path(_model_id: object) -> Path:
        return tmp_path

    monkeypatch.setattr(du, "build_model_path", _fake_build_model_path)
    args = _draft_model_args(_runtime(repo="org/draft-GGUF", file="draft.gguf"), "draft_mtp")
    assert args == ["--model-draft", str(tmp_path / "draft.gguf")]


def test_draft_args_missing_file_on_disk_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import skulk.download.download_utils as du

    def _fake_build_model_path(_model_id: object) -> Path:
        return tmp_path

    monkeypatch.setattr(du, "build_model_path", _fake_build_model_path)
    with pytest.raises(RuntimeError, match="not found under"):
        _draft_model_args(_runtime(repo="org/draft-GGUF", file="missing.gguf"), "draft_simple")


def test_parse_content_delta() -> None:
    d = _parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert d is not None
    assert d.content == "hi"
    assert d.reasoning == ""
    assert d.finish is None
    assert d.done is False


def test_parse_reasoning_delta_is_separate() -> None:
    # reasoning_content rides its own field so the runner can flag is_thinking.
    d = _parse_sse_line('data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}')
    assert d is not None
    assert d.reasoning == "hmm"
    assert d.content == ""


def test_parse_finish_length_is_preserved() -> None:
    # A max_tokens truncation must surface as "length", not be masked as "stop".
    d = _parse_sse_line(
        'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}'
    )
    assert d is not None
    assert d.content == "x"
    assert d.finish == "length"


def test_parse_finish_stop_and_eos_map_to_stop() -> None:
    for reason in ("stop", "eos_token"):
        d = _parse_sse_line(
            f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{reason}"}}]}}'
        )
        assert d is not None
        assert d.finish == "stop"


def test_parse_done_sentinel() -> None:
    d = _parse_sse_line("data: [DONE]")
    assert d is not None
    assert d.done is True


def test_parse_skips_non_data_and_blank_lines() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": keep-alive comment") is None
    assert _parse_sse_line("event: message") is None


def test_parse_skips_malformed_json_and_choiceless() -> None:
    # A stray/garbled line must not break the stream (returns None to skip).
    assert _parse_sse_line("data: {not json") is None
    assert _parse_sse_line('data: {"choices":[]}') is None
    assert _parse_sse_line('data: {"id":"x"}') is None


def test_spec_type_flag_maps_to_llama_server_flags() -> None:
    # The card's served_spec_type underscores become the llama-server hyphen flags.
    assert _SPEC_TYPE_FLAG["draft_mtp"] == "draft-mtp"
    assert _SPEC_TYPE_FLAG["draft_eagle3"] == "draft-eagle3"
    assert _SPEC_TYPE_FLAG["draft_simple"] == "draft-simple"
    # ngram is the special case: it maps to ngram-cache, not "ngram".
    assert _SPEC_TYPE_FLAG["ngram"] == "ngram-cache"


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        ("llama_server-vulkan", "99"),  # GPU compute tag -> full offload
        ("llama_server-rocm", "99"),
        ("llama_server-cuda", "99"),
        ("llama_server-cpu", "0"),  # CPU tag was RAM-admitted -> no GPU offload
        ("llama_server", "0"),  # bare tag is NOT GPU-offload (RAM-admitted)
        (None, "99"),  # no resolution (manual/fallback) -> default GPU offload
    ],
)
def test_gpu_layers_match_vram_admission(resolved: str | None, expected: str) -> None:
    # The runner's -ngl decision must mirror placement_utils._has_gpu_offload_backend
    # so a RAM-admitted placement never grabs an unbudgeted GPU.
    assert _gpu_layers_for_backend(resolved) == expected
