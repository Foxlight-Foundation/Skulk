"""Tests for the string-based gpt-oss harmony parser used by the llama.cpp runner."""

from skulk.worker.runner.llm_inference.harmony_text_parser import HarmonyTextParser


def _run(parser: HarmonyTextParser, deltas: list[str]) -> list[tuple[str, bool]]:
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


def test_splits_analysis_and_final_and_strips_markers() -> None:
    raw = (
        "<|channel|>analysis<|message|>Let me think about it."
        "<|end|><|start|>assistant<|channel|>final<|message|>The answer is 42."
    )
    out = _run(HarmonyTextParser(), [raw])
    assert _reasoning(out) == "Let me think about it."
    assert _content(out) == "The answer is 42."
    # No control markers leak into any emission.
    assert "<|" not in "".join(t for t, _ in out)


def test_markers_split_across_deltas_are_not_emitted() -> None:
    # Chunk boundaries fall in the middle of several markers.
    deltas = [
        "<|chan",
        "nel|>ana",
        "lysis<|mess",
        "age|>think",
        "ing<|end|><|start|>assistant<|cha",
        "nnel|>final<|message|>done",
    ]
    out = _run(HarmonyTextParser(), deltas)
    assert _reasoning(out) == "thinking"
    assert _content(out) == "done"
    assert "<|" not in "".join(t for t, _ in out)


def test_plain_text_without_markers_passes_through_as_content() -> None:
    # A non-harmony stream (or a template that pre-opened the final channel)
    # must not be swallowed: every char is content.
    out = _run(HarmonyTextParser(), ["Hello ", "world", "!"])
    assert _content(out) == "Hello world!"
    assert _reasoning(out) == ""


def test_final_channel_terminated_by_return() -> None:
    raw = (
        "<|channel|>analysis<|message|>reasoning"
        "<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>"
    )
    out = _run(HarmonyTextParser(), [raw])
    assert _reasoning(out) == "reasoning"
    assert _content(out) == "answer"


def test_ordered_integers_coherence_not_garbled() -> None:
    # Regression for the battery failure: the analysis text mentioned the range,
    # but the parser must only surface the final channel as content.
    raw = (
        "<|channel|>analysis<|message|>The user wants 1 to 20. Provide 1 2 3 ... 20."
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    )
    out = _run(HarmonyTextParser(), [raw])
    assert _content(out) == "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"


def test_commentary_header_with_recipient_and_constrain_is_stripped() -> None:
    # A commentary/tool header carries `to=...` and `<|constrain|>json`; none of
    # that header text should leak, and the body is surfaced (as content here).
    raw = (
        "<|channel|>commentary to=functions.get_weather "
        "<|constrain|>json<|message|>{\"city\": \"SF\"}<|call|>"
    )
    out = _run(HarmonyTextParser(), [raw])
    text = "".join(t for t, _ in out)
    assert "to=functions" not in text
    assert "constrain" not in text
    assert "<|" not in text
    assert '{"city": "SF"}' in text


def test_trailing_partial_marker_is_flushed_as_literal() -> None:
    # A stream that ends on a lone '<' (a marker prefix) holds it back mid-stream;
    # flush must still surface it rather than swallow it. Use feed() directly so
    # the held-back '<' is observable before flush.
    parser = HarmonyTextParser()
    fed = parser.feed("answer <")
    # The trailing '<' is held back as a potential marker prefix, not yet emitted.
    assert "".join(t for t, _ in fed) == "answer "
    flushed = parser.flush()
    assert "".join(t for t, _ in flushed) == "<"
    # And a literal '<' that is clearly not at the end passes straight through.
    assert _content(_run(HarmonyTextParser(), ["a < b"])) == "a < b"
