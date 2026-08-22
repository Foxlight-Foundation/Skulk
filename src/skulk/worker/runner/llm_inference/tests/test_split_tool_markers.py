"""Coverage for tool-call markers that arrive split across chunks.

A generation chunk is whatever the streaming detokenizer could resolve that
step, not a token. A marker that is a single token id still reaches the parser
as several chunks (`<tool`, `_`, `c`, `all>`), which was observed live on a
Qwen model served by the MLX engine: the block never opened and the caller
received the raw markup as content. These tests feed markers split the way the
detokenizer actually splits them.
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
from skulk.worker.runner.llm_inference.tool_parsers import (
    ToolParser,
    make_mlx_parser,
    make_text_dialect_parser,
)

WEATHER = [
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


def chunk(text: str, finish_reason: FinishReason | None = None) -> GenerationResponse:
    return GenerationResponse(text=text, token=1, finish_reason=finish_reason, usage=None)


def feed(pieces: list[str], parser: ToolParser) -> list[ParserChunk]:
    def source() -> Generator[ParserChunk]:
        for piece in pieces[:-1]:
            yield chunk(piece)
        yield chunk(pieces[-1], finish_reason="stop")

    return list(parse_tool_calls(source(), parser, tools=WEATHER))


def calls_of(chunks: list[ParserChunk]) -> list[str]:
    names: list[str] = []
    for item in chunks:
        if isinstance(item, ToolCallResponse):
            names.extend(call.name for call in item.tool_calls)
    return names


def text_of(chunks: list[ParserChunk]) -> str:
    return "".join(
        getattr(item, "text", "") or "" for item in chunks if item is not None
    )


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


class TestSplitMarkers:
    def test_a_marker_split_across_chunks_still_opens_the_block(self) -> None:
        # Exactly the split seen live from the MLX detokenizer.
        chunks = feed(
            [
                "<tool",
                "_",
                "c",
                "all>\n<fu",
                "nction=get_weather>\n<parameter=location>\nDenver\n",
                "</parameter>\n</function>\n</tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "<tool_call>" not in text_of(chunks)

    def test_a_closing_marker_split_across_chunks_still_closes(self) -> None:
        chunks = feed(
            [
                "<tool_call>",
                "<function=get_weather><parameter=location>Denver</parameter></function>",
                "</tool",
                "_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_an_unmarked_call_split_across_chunks_is_parsed(self) -> None:
        chunks = feed(
            ["<|py", "thon_tag|>", '{"name": "get_weather", ', '"parameters": {}}'],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "python_tag" not in text_of(chunks)


class TestOrdinaryAnswersStillStream:
    def test_prose_is_released_as_soon_as_it_cannot_be_a_marker(self) -> None:
        chunks = feed(["The ", "weather ", "is fine."], generic_parser())
        assert calls_of(chunks) == []
        assert text_of(chunks) == "The weather is fine."

    def test_text_that_starts_like_a_marker_and_diverges_is_not_swallowed(self) -> None:
        chunks = feed(["<tool", "box is open."], generic_parser())
        assert calls_of(chunks) == []
        assert text_of(chunks) == "<toolbox is open."

    def test_a_stream_of_whitespace_does_not_buffer_without_bound(self) -> None:
        chunks = feed([" "] * 40, generic_parser())
        assert calls_of(chunks) == []
        assert text_of(chunks).strip() == ""

    def test_a_message_ending_while_still_ambiguous_is_released(self) -> None:
        chunks = feed(["<tool"], generic_parser())
        assert calls_of(chunks) == []
        assert text_of(chunks) == "<tool"
