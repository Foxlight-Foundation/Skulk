"""Tests for the string-based ``<think>`` reasoning parser used by llama.cpp."""

from skulk.worker.runner.llm_inference.think_text_parser import ThinkTextParser


def _run(parser: ThinkTextParser, deltas: list[str]) -> list[tuple[str, bool]]:
    """Feed deltas then flush, returning all (text, is_thinking) emissions."""
    out: list[tuple[str, bool]] = []
    for delta in deltas:
        out.extend(parser.feed(delta))
    out.extend(parser.flush())
    return out


def _content(emissions: list[tuple[str, bool]]) -> str:
    return "".join(t for t, thinking in emissions if not thinking)


def _reasoning(emissions: list[tuple[str, bool]]) -> str:
    return "".join(t for t, thinking in emissions if thinking)


def test_prefilled_open_stream_starts_in_thinking_and_splits_on_close() -> None:
    # The observed Ornith / Qwen3.5 GGUF shape: the chat template pre-fills the
    # opening <think> in the PROMPT, so the generated stream begins mid-reasoning
    # and only </think> appears. starts_in_thinking=True (the default).
    raw = "Let me reason about this carefully.</think>\n\nThe answer is 42."
    out = _run(ThinkTextParser(), [raw])
    assert _reasoning(out) == "Let me reason about this carefully."
    assert _content(out) == "\n\nThe answer is 42."
    # No markers leak into any emission.
    assert "think" not in "".join(t for t, _ in out)


def test_explicit_open_and_close_when_not_starting_in_thinking() -> None:
    # A template that echoes the opening <think> in the output stream.
    raw = "Here is the answer: <think>internal reasoning</think>final answer."
    out = _run(ThinkTextParser(starts_in_thinking=False), [raw])
    assert _content(out) == "Here is the answer: final answer."
    assert _reasoning(out) == "internal reasoning"
    assert "<think>" not in _content(out) and "</think>" not in _content(out)


def test_markers_split_across_deltas_are_not_emitted() -> None:
    # Chunk boundaries fall inside the closing marker.
    deltas = ["reason", "ing here</thi", "nk>", "\nthe answer"]
    out = _run(ThinkTextParser(), deltas)
    assert _reasoning(out) == "reasoning here"
    assert _content(out) == "\nthe answer"
    assert "</think>" not in _content(out)


def test_truncated_mid_thought_is_all_reasoning() -> None:
    # Generation cut off by a token cap before </think>: everything is reasoning,
    # nothing silently dropped, no answer invented.
    raw = "still thinking, never finished"
    out = _run(ThinkTextParser(), [raw])
    assert _reasoning(out) == "still thinking, never finished"
    assert _content(out) == ""


def test_no_markers_passthrough_when_not_thinking() -> None:
    # A turn with no markers at all (thinking disabled) starting in content mode
    # passes straight through as content.
    raw = "just a plain answer with no reasoning markers"
    out = _run(ThinkTextParser(starts_in_thinking=False), [raw])
    assert _content(out) == raw
    assert _reasoning(out) == ""


def test_close_marker_split_at_every_boundary() -> None:
    # Worst case: each character of </think> arrives in its own delta.
    deltas = ["abc"] + list("</think>") + ["xyz"]
    out = _run(ThinkTextParser(), deltas)
    assert _reasoning(out) == "abc"
    assert _content(out) == "xyz"
