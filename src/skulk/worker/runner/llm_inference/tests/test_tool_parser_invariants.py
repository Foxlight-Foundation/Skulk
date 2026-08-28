"""Invariant sweep over the streaming tool-call parser.

The parser's bugs have all had the same shape: a message arrives split in a way
nobody wrote a case for, and one exit path handles it differently from the
others. Enumerating cases by hand finds them one at a time. This instead states
what must be true of *every* message however it is split, and checks it across
every single split point of each message, so a new exit path that gets one of
these wrong fails here rather than in review.

The invariants:

1. Exactly one chunk terminates the stream, and it is the last one. The
   consumer stops at the first chunk carrying a finish reason, so anything
   after a terminal chunk is invisible to the caller.
2. The calls delivered do not depend on where the message was split.
3. Dialect markup never reaches the caller as content. A block naming a tool
   the caller did not offer is still delivered as content so the caller can
   see what the model did, but with its markers stripped, the same as any
   other block handed back. Only the error path keeps the raw block, as
   evidence of what was malformed.
4. When the request permits no calls (``emit_calls`` of ``False``, the
   ``tool_choice: "none"`` path), no chunk is ever a call, however many
   complete blocks the message contains and wherever it is split.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from skulk.api.types import FinishReason, ToolCallItem
from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.worker.runner.llm_inference.model_output_parsers import (
    ParserChunk,
    parse_tool_calls,
)
from skulk.worker.runner.llm_inference.tool_parsers import (
    ToolParser,
    make_mlx_parser,
    make_text_dialect_parser,
)

WEATHER: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    }
]

CALL = "<tool_call><function=get_weather><parameter=location>Denver</parameter></function></tool_call>"

DROPPED = "<tool_call><function=print></function></tool_call>"

# message, expected call names
MARKED_MESSAGES: list[tuple[str, list[str]]] = [
    ("The weather is fine.", []),
    (CALL, ["get_weather"]),
    (f"I'll check. {CALL}", ["get_weather"]),
    (f"{CALL} Done.", ["get_weather"]),
    (f"{CALL}{CALL}", ["get_weather", "get_weather"]),
    (f"{CALL} and then {CALL}", ["get_weather", "get_weather"]),
    (DROPPED, []),
    (f"{DROPPED}{CALL}", ["get_weather"]),
    # The rejected block arriving AFTER the accepted one lands in the
    # terminal-suffix scan rather than the streaming scan, which once handed
    # it back verbatim, markers and all.
    (f"{CALL}{DROPPED}", ["get_weather"]),
    ("Braces {like this} are fine.", []),
]

UNMARKED_CALL = '{"name": "get_weather", "parameters": {"location": "Denver"}}'
UNMARKED_MESSAGES: list[tuple[str, list[str]]] = [
    ("Just an answer.", []),
    (UNMARKED_CALL, ["get_weather"]),
    (f"{UNMARKED_CALL} Done.", ["get_weather"]),
    ('{"city": "Denver", "population": 715522}', []),
    ("The set is {1, 2, 3}.", []),
    (f"Let me look. <|python_tag|>{UNMARKED_CALL}", ["get_weather"]),
    # A model reaching for its own built-in is dropped and delivered as
    # content, with the dialect's markers stripped.
    ('{"name": "print", "parameters": {}}', []),
]

MARKED_MARKERS = ("<tool_call>", "</tool_call>", "<|python_tag|>")
UNMARKED_MARKERS = ("<|python_tag|>", "<|eom_id|>")


def generic_parser() -> ToolParser:
    def inner(text: str) -> dict[str, object]:
        from skulk.worker.runner.llm_inference.tool_text_parser import (
            parse_tool_calls_from_text,
        )

        items = parse_tool_calls_from_text(f"<tool_call>{text}</tool_call>")
        if not items:
            raise ValueError("no tool calls")
        item: ToolCallItem = items[0]
        return {"name": item.name, "arguments": item.arguments}

    return make_mlx_parser("<tool_call>", "</tool_call>", inner)


def chunk(text: str, finish_reason: FinishReason | None = None) -> GenerationResponse:
    return GenerationResponse(
        text=text, token=1, finish_reason=finish_reason, usage=None
    )


def run(
    pieces: list[str],
    parser: ToolParser,
    *,
    tools: list[dict[str, Any]] | None = WEATHER,
    emit_calls: bool = True,
) -> list[ParserChunk]:
    def source() -> Generator[ParserChunk]:
        for piece in pieces[:-1]:
            yield chunk(piece)
        yield chunk(pieces[-1], finish_reason="stop")

    return list(
        parse_tool_calls(source(), parser, tools=tools, emit_calls=emit_calls)
    )


def splits(message: str) -> list[list[str]]:
    """Every single split point, plus whole and character-by-character."""

    variants: list[list[str]] = [[message]]
    variants.extend(
        [message[:index], message[index:]] for index in range(1, len(message))
    )
    variants.append(list(message))
    return variants


def call_names(chunks: list[ParserChunk]) -> list[str]:
    names: list[str] = []
    for item in chunks:
        if isinstance(item, ToolCallResponse):
            names.extend(call.name for call in item.tool_calls)
    return names


def content(chunks: list[ParserChunk]) -> str:
    return "".join(
        item.text
        for item in chunks
        if isinstance(item, GenerationResponse) and not item.is_thinking
    )


def terminals(chunks: list[ParserChunk]) -> list[ParserChunk]:
    return [
        item
        for item in chunks
        if isinstance(item, ToolCallResponse)
        or (item is not None and item.finish_reason is not None)
    ]


def check_message(
    message: str,
    expected: list[str],
    parser: ToolParser,
    markers: tuple[str, ...],
    *,
    tools: list[dict[str, Any]] | None = WEATHER,
    emit_calls: bool = True,
) -> None:
    for pieces in splits(message):
        chunks = run(pieces, parser, tools=tools, emit_calls=emit_calls)
        where = f"{message!r} split as {pieces!r}"

        found = terminals(chunks)
        assert len(found) == 1, f"{len(found)} terminal chunks for {where}"
        assert found[0] is chunks[-1], f"terminal chunk is not last for {where}"

        assert call_names(chunks) == expected, f"calls differ for {where}"

        answer = content(chunks)
        for marker in markers:
            assert marker not in answer, f"{marker!r} leaked for {where}"


class TestMarkedDialectInvariants:
    def test_every_split_of_every_message(self) -> None:
        parser = generic_parser()
        for message, expected in MARKED_MESSAGES:
            check_message(message, expected, parser, MARKED_MARKERS)


class TestUnmarkedDialectInvariants:
    def test_every_split_of_every_message(self) -> None:
        parser = make_text_dialect_parser("{", "<|eom_id|>")
        for message, expected in UNMARKED_MESSAGES:
            check_message(message, expected, parser, UNMARKED_MARKERS)


class TestNoCallsAllowedInvariants:
    """``tool_choice: "none"`` reaches the parser as no tools and no emission.

    However many complete, well-formed blocks the message carries, none may
    come back as a call, and every one of them is delivered as content with
    its markers stripped. The two-blocks-in-one-terminal-chunk shape matters:
    the first block is met by the streaming scan but the second only by the
    terminal-suffix scan, and the two paths must agree.
    """

    def test_every_split_of_every_message(self) -> None:
        parser = generic_parser()
        for message in (
            CALL,
            f"I'll check. {CALL}",
            f"{CALL}{CALL}",
            f"{CALL} and then {CALL}",
            DROPPED,
        ):
            check_message(
                message, [], parser, MARKED_MARKERS, tools=None, emit_calls=False
            )
