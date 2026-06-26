# pyright: reportPrivateUsage=false
"""Tests for the served-runner SSE delta parser and spec-flag mapping.

The runner's HTTP/subprocess plumbing is exercised live on a GPU node; this
covers the pure parsing surface (the subtle part) without a server.
"""

from skulk.worker.runner.llama_server.runner import (
    _SPEC_TYPE_FLAG,
    _parse_sse_line,
)


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
