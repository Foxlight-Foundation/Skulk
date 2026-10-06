"""Coverage for the unmarked tool-call dialect.

Llama 3.1+ writes a tool call as a bare JSON object with no opening marker and
ends the message with ``<|eom_id|>`` rather than a closing marker, so neither
half of the marker mechanism applies. These tests pin the two behaviors that
makes possible: a call that opens on ``{`` and is closed by the end of
generation is parsed, and a block that turns out not to be a call is delivered
as content instead of being reported as a failure.
"""

from __future__ import annotations

from collections.abc import Generator

from skulk.api.types import FinishReason, ToolCallItem
from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.worker.runner.llm_inference.model_output_parsers import (
    ParserChunk,
    parse_tool_calls,
)
from skulk.worker.runner.llm_inference.tool_parsers import make_text_dialect_parser

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["location"],
        },
    },
}


def chunk(text: str, finish_reason: FinishReason | None = None) -> GenerationResponse:
    return GenerationResponse(
        text=text, token=1, finish_reason=finish_reason, usage=None
    )


def run(texts: list[str], final: str) -> list[ParserChunk]:
    """Feed the chunks through the parser and collect what a caller would see."""

    def source() -> Generator[ParserChunk]:
        for text in texts:
            yield chunk(text)
        yield chunk(final, finish_reason="stop")

    return list(
        parse_tool_calls(
            source(),
            make_text_dialect_parser("{", "<|eom_id|>"),
            tools=[WEATHER_TOOL],
        )
    )


def tool_calls(chunks: list[ParserChunk]) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    for item in chunks:
        if isinstance(item, ToolCallResponse):
            calls.extend(item.tool_calls)
    return calls


def text_of(chunks: list[ParserChunk]) -> str:
    return "".join(
        getattr(item, "text", "") or "" for item in chunks if item is not None
    )


class TestUnmarkedCall:
    def test_bare_call_closed_by_end_of_generation_is_parsed(self) -> None:
        chunks = run(
            ['{"name": "get_weather", ', '"parameters": {"location": '],
            '"Cedar Rapids, Iowa"}}',
        )
        calls = tool_calls(chunks)
        assert [call.name for call in calls] == ["get_weather"]
        assert "Cedar Rapids" in calls[0].arguments

    def test_the_call_does_not_also_leak_out_as_content(self) -> None:
        # The failure this guards is the one seen live: the call arriving as
        # text with finish_reason "stop", so a client sees JSON in the answer
        # and no tool call at all.
        chunks = run(['{"name": "get_weather", '], '"parameters": {}}')
        assert tool_calls(chunks)
        assert "get_weather" not in text_of(chunks)

    def test_a_python_tag_call_in_the_same_block_is_parsed(self) -> None:
        chunks = run(
            ['{"name": "get_weather", "parameters": {}}'],
            '<|python_tag|>{"name": "get_weather", "parameters": {"location": "x"}}',
        )
        assert [call.name for call in tool_calls(chunks)] == ["get_weather"]

    def test_arguments_are_coerced_to_the_tool_schema(self) -> None:
        # Llama writes its arguments under "parameters", and models routinely
        # quote numbers. Both have to be normalized before a caller sees the
        # call, so this pins that the block goes through schema coercion.
        chunks = run(['{"name": "get_weather", '], '"parameters": {"days": "3"}}')
        assert '"days": 3' in tool_calls(chunks)[0].arguments


class TestNotACall:
    def test_a_json_answer_is_delivered_as_content_not_an_error(self) -> None:
        # A caller may supply tools and still ask for a JSON answer. Opening the
        # block on "{" means that answer lands here, and reporting it as a
        # parse failure would turn a correct response into an error.
        chunks = run(['{"city": "Cedar Rapids", '], '"population": 137710}')
        assert tool_calls(chunks) == []
        assert '"population": 137710' in text_of(chunks)
        assert all(
            getattr(item, "finish_reason", None) != "error"
            for item in chunks
            if item is not None
        )

    def test_prose_never_enters_the_block_at_all(self) -> None:
        chunks = run(["The weather in "], "Cedar Rapids is fine.")
        assert tool_calls(chunks) == []
        assert text_of(chunks) == "The weather in Cedar Rapids is fine."
