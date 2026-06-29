"""Tests for the Gemma 4 ``<|channel>`` streaming parser."""

from __future__ import annotations

from skulk.worker.runner.llama_server.channel_text_parser import GemmaChannelTextParser


def _collapse(emissions: list[tuple[str, bool]]) -> tuple[str, str]:
    """Reduce emissions to ``(reasoning, content)`` strings."""
    reasoning = "".join(t for t, thinking in emissions if thinking)
    content = "".join(t for t, thinking in emissions if not thinking)
    return reasoning, content


def _parse_whole(text: str) -> tuple[str, str]:
    p = GemmaChannelTextParser()
    em = p.feed(text)
    em += p.flush()
    return _collapse(em)


def _parse_streamed(text: str, *, step: int = 1) -> tuple[str, str]:
    p = GemmaChannelTextParser()
    em: list[tuple[str, bool]] = []
    for i in range(0, len(text), step):
        em += p.feed(text[i : i + step])
    em += p.flush()
    return _collapse(em)


# The exact shape observed on kite4 (google/gemma-4-31B-it-qat-q4_0-gguf).
_SAMPLE = (
    "<|channel>thought\n"
    'The user wants me to reply with exactly "Hello world".\n'
    "<channel|>Hello world"
)


def test_sample_splits_reasoning_and_content() -> None:
    reasoning, content = _parse_whole(_SAMPLE)
    assert content == "Hello world"
    assert reasoning == 'The user wants me to reply with exactly "Hello world".\n'
    # No markers leak into either channel.
    assert "<|channel>" not in reasoning + content
    assert "<channel|>" not in reasoning + content
    assert "thought" not in content


def test_sample_streamed_char_by_char_matches_whole() -> None:
    # Markers split across single-char deltas must parse identically.
    assert _parse_streamed(_SAMPLE, step=1) == _parse_whole(_SAMPLE)


def test_sample_streamed_various_chunk_sizes() -> None:
    whole = _parse_whole(_SAMPLE)
    for step in (2, 3, 5, 7, 11):
        assert _parse_streamed(_SAMPLE, step=step) == whole


def test_plain_text_without_markers_passes_through_as_content() -> None:
    reasoning, content = _parse_whole("The capital of France is Paris.")
    assert reasoning == ""
    assert content == "The capital of France is Paris."


def test_thought_then_answer_without_explicit_close_falls_back() -> None:
    # If a generation ends mid-thought (no <channel|>), the thought text is still
    # surfaced as reasoning by flush(), never swallowed.
    reasoning, content = _parse_whole("<|channel>thought\nstill thinking")
    assert reasoning == "still thinking"
    assert content == ""


def test_analysis_channel_is_also_reasoning() -> None:
    reasoning, content = _parse_whole("<|channel>analysis\nhmm\n<channel|>answer")
    assert reasoning == "hmm\n"
    assert content == "answer"


def test_non_thinking_channel_is_content() -> None:
    # A channel that isn't a known thinking channel is treated as content.
    reasoning, content = _parse_whole("<|channel>final\nthe answer")
    assert reasoning == ""
    assert content == "the answer"


def test_leading_content_before_any_channel() -> None:
    reasoning, content = _parse_whole("hi <|channel>thought\nx\n<channel|>bye")
    assert reasoning == "x\n"
    assert content == "hi bye"


def test_partial_marker_held_until_complete() -> None:
    p = GemmaChannelTextParser()
    em: list[tuple[str, bool]] = []
    # Feeding a prefix of <|channel> must not emit it as literal text: only the
    # safe "answer " is emitted; "<|cha" is held back.
    em += p.feed("answer <|cha")
    assert em == [("answer ", False)]
    # Completing the marker + header resolves to a thinking channel.
    em += p.feed("nnel>thought\nreasoning")
    em += p.flush()
    reasoning, content = _collapse(em)
    assert content == "answer "
    assert reasoning == "reasoning"
