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
3. The markup of an accepted call never reaches the caller as content. A
   block naming a tool the caller did not offer is a deliberate exception: it
   is delivered verbatim so the caller can see what the model did, which is
   why those messages opt out of this one.
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

# message, expected call names, whether content may carry markup
MARKED_MESSAGES: list[tuple[str, list[str], bool]] = [
    ("The weather is fine.", [], False),
    (CALL, ["get_weather"], False),
    (f"I'll check. {CALL}", ["get_weather"], False),
    (f"{CALL} Done.", ["get_weather"], False),
    (f"{CALL}{CALL}", ["get_weather", "get_weather"], False),
    (f"{CALL} and then {CALL}", ["get_weather", "get_weather"], False),
    (DROPPED, [], True),
    (f"{DROPPED}{CALL}", ["get_weather"], True),
    ("Braces {like this} are fine.", [], False),
]

UNMARKED_CALL = '{"name": "get_weather", "parameters": {"location": "Denver"}}'
UNMARKED_MESSAGES: list[tuple[str, list[str], bool]] = [
    ("Just an answer.", [], False),
    (UNMARKED_CALL, ["get_weather"], False),
    (f"{UNMARKED_CALL} Done.", ["get_weather"], False),
    ('{"city": "Denver", "population": 715522}', [], False),
    ("The set is {1, 2, 3}.", [], False),
    (f"Let me look. <|python_tag|>{UNMARKED_CALL}", ["get_weather"], False),
]


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


def run(pieces: list[str], parser: ToolParser) -> list[ParserChunk]:
    def source() -> Generator[ParserChunk]:
        for piece in pieces[:-1]:
            yield chunk(piece)
        yield chunk(pieces[-1], finish_reason="stop")

    return list(parse_tool_calls(source(), parser, tools=WEATHER))


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
    message: str, expected: list[str], parser: ToolParser, markup_ok: bool
) -> None:
    for pieces in splits(message):
        chunks = run(pieces, parser)
        where = f"{message!r} split as {pieces!r}"

        found = terminals(chunks)
        assert len(found) == 1, f"{len(found)} terminal chunks for {where}"
        assert found[0] is chunks[-1], f"terminal chunk is not last for {where}"

        assert call_names(chunks) == expected, f"calls differ for {where}"

        if expected and not markup_ok:
            answer = content(chunks)
            assert "<tool_call>" not in answer, f"markup leaked for {where}"
            assert "<|python_tag|>" not in answer, f"markup leaked for {where}"


class TestMarkedDialectInvariants:
    def test_every_split_of_every_message(self) -> None:
        parser = generic_parser()
        for message, expected, markup_ok in MARKED_MESSAGES:
            check_message(message, expected, parser, markup_ok)


class TestUnmarkedDialectInvariants:
    def test_every_split_of_every_message(self) -> None:
        parser = make_text_dialect_parser("{", "<|eom_id|>")
        for message, expected, markup_ok in UNMARKED_MESSAGES:
            check_message(message, expected, parser, markup_ok)
