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


def chunk(
    text: str,
    finish_reason: FinishReason | None = None,
    *,
    thinking: bool = False,
) -> GenerationResponse:
    return GenerationResponse(
        text=text,
        token=1,
        finish_reason=finish_reason,
        usage=None,
        is_thinking=thinking,
    )


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


class TestThinkingBeforeTheCall:
    """Reasoning must not settle the opening decision.

    This parser runs downstream of the thinking parser, so a thinking model
    that reasons before calling a tool sends its reasoning through here first.
    Letting that text decide the opening would mean the later marker is never
    examined and the caller receives raw tool markup as content, which is the
    failure this whole change exists to remove.
    """

    @staticmethod
    def run_with_reasoning(pieces: list[str]) -> list[ParserChunk]:
        def source() -> Generator[ParserChunk]:
            yield chunk("Let me think about which tool fits.", thinking=True)
            yield chunk(" Probably the weather one.", thinking=True)
            for piece in pieces[:-1]:
                yield chunk(piece)
            yield chunk(pieces[-1], finish_reason="stop")

        return list(parse_tool_calls(source(), generic_parser(), tools=WEATHER))

    def test_a_call_after_reasoning_is_still_parsed(self) -> None:
        chunks = self.run_with_reasoning(
            [
                "<tool",
                "_call>",
                "<function=get_weather><parameter=location>Denver</parameter></function>",
                "</tool_call>",
            ]
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_the_reasoning_still_reaches_the_caller(self) -> None:
        chunks = self.run_with_reasoning(
            ["<tool_call>", "<function=get_weather></function>", "</tool_call>"]
        )
        reasoning = "".join(
            item.text
            for item in chunks
            if isinstance(item, GenerationResponse) and item.is_thinking
        )
        assert "Let me think" in reasoning

    def test_a_call_only_contemplated_in_reasoning_is_not_executed(self) -> None:
        def source() -> Generator[ParserChunk]:
            yield chunk("<tool_call><function=get_weather></function></tool_call>", thinking=True)
            yield chunk("I will not call it.", finish_reason="stop")

        chunks = list(parse_tool_calls(source(), generic_parser(), tools=WEATHER))
        assert calls_of(chunks) == []


class TestVisiblePreamble:
    """A sentence before the call must not hide it.

    Models routinely announce what they are about to do ("I'll check that.")
    and then call. The opening scan therefore has to keep looking after
    ordinary text has been released, not decide once and stop.
    """

    def test_a_call_after_a_visible_preamble_is_parsed(self) -> None:
        chunks = feed(
            [
                "I'll check ",
                "that for you. ",
                "<tool_call>",
                "<function=get_weather><parameter=location>Denver</parameter></function>",
                "</tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_the_preamble_still_reaches_the_caller(self) -> None:
        chunks = feed(
            [
                "I'll check that. ",
                "<tool_call><function=get_weather></function></tool_call>",
            ],
            generic_parser(),
        )
        assert "I'll check that." in text_of(chunks)

    def test_a_preamble_and_a_split_marker_together(self) -> None:
        chunks = feed(
            [
                "Sure thing. ",
                "<tool",
                "_call><function=get_weather></function>",
                "</tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "Sure thing." in text_of(chunks)

    def test_two_calls_in_one_message_are_both_parsed(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather></function></tool_call>",
                " and also ",
                "<tool_call><function=get_weather></function></tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather", "get_weather"]


class TestUnmarkedDialectStaysAnchored:
    """The unmarked dialect opens on `{`, so it must only open at the start.

    Letting a brace open a block anywhere would turn any answer that mentions
    one into a tool call.
    """

    def test_a_brace_mid_answer_is_not_a_call(self) -> None:
        chunks = feed(
            ["The set is ", '{"name": "get_weather", "parameters": {}}'],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == []
        assert "The set is" in text_of(chunks)

    def test_a_call_at_the_start_still_opens(self) -> None:
        chunks = feed(
            ['{"name": "get_weather", ', '"parameters": {"location": "Denver"}}'],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_the_distinctive_marker_still_opens_after_a_preamble(self) -> None:
        # <|python_tag|> is unambiguous, so unlike the brace it may appear
        # after a sentence and still open the call.
        chunks = feed(
            [
                "Let me look that up. ",
                '<|python_tag|>{"name": "get_weather", "parameters": {}}',
            ],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]


class TestTextAfterTheCall:
    """A model may keep writing after closing the call.

    Requiring the block to end at the closing marker swallowed everything that
    followed, so trailing text was lost and a second call in the same message
    was folded into the first block.
    """

    def test_trailing_text_after_the_call_is_delivered(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather></function></tool_call>",
                " Done.",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "Done." in text_of(chunks)

    def test_trailing_text_in_the_closing_chunk_is_delivered(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather></function>",
                "</tool_call> Done.",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "Done." in text_of(chunks)

    def test_a_second_call_after_trailing_text_is_also_parsed(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather></function></tool_call>",
                " and then ",
                "<tool_call><function=get_weather></function></tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather", "get_weather"]
        assert "and then" in text_of(chunks)


class TestTailAfterADroppedBlock:
    """A block that named no offered tool must not swallow what follows.

    Emitting the trailing text along with the dropped block would keep a real
    call in that tail from ever being scanned.
    """

    def test_a_real_call_after_a_dropped_block_is_still_found(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=print><parameter=value>hi</parameter></function></tool_call>",
                " then ",
                "<tool_call><function=get_weather></function></tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_the_dropped_block_still_reaches_the_caller_as_content(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=print><parameter=value>hi</parameter></function></tool_call>",
                " done.",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == []
        assert "print" in text_of(chunks)
        assert "done." in text_of(chunks)


class TestParallelCallsReachTheCaller:
    """Several blocks in one message must arrive as one response.

    The consumer of this stream stops at the first chunk carrying a finish
    reason, so a response per block would deliver the first call and drop the
    rest. Families that write each parallel call in its own block would lose
    every call after the first.
    """

    @staticmethod
    def tool_responses(chunks: list[ParserChunk]) -> list[ToolCallResponse]:
        return [item for item in chunks if isinstance(item, ToolCallResponse)]

    def test_two_blocks_arrive_as_one_response_carrying_both_calls(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather><parameter=location>Denver</parameter></function></tool_call>",
                "<tool_call><function=get_weather><parameter=location>Boston</parameter></function></tool_call>",
            ],
            generic_parser(),
        )
        responses = self.tool_responses(chunks)
        assert len(responses) == 1
        assert len(responses[0].tool_calls) == 2

    def test_nothing_terminates_the_stream_before_the_calls(self) -> None:
        # A text chunk carrying a finish reason would end the stream at the
        # consumer, so the trailing text is released without one and the tool
        # response is the terminal chunk.
        chunks = feed(
            [
                "<tool_call><function=get_weather></function></tool_call>",
                " and then ",
                "<tool_call><function=get_weather></function></tool_call>",
                " done.",
            ],
            generic_parser(),
        )
        index = next(
            i for i, item in enumerate(chunks) if isinstance(item, ToolCallResponse)
        )
        assert all(
            getattr(item, "finish_reason", None) is None
            for item in chunks[:index]
            if item is not None
        )
        assert len(self.tool_responses(chunks)[0].tool_calls) == 2

    def test_a_single_call_is_unchanged(self) -> None:
        chunks = feed(
            ["<tool_call><function=get_weather></function></tool_call>"],
            generic_parser(),
        )
        responses = self.tool_responses(chunks)
        assert len(responses) == 1
        assert len(responses[0].tool_calls) == 1


class TestDroppedBlockOnTheTerminalChunk:
    """A dropped block must not end the stream while more is coming.

    The consumer stops at the first chunk carrying a finish reason, so a
    dropped block emitted with one would hide everything after it, including a
    real call in the same message.
    """

    def test_a_real_call_after_a_dropped_block_in_the_last_chunk(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=print></function></tool_call>"
                "<tool_call><function=get_weather></function></tool_call>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_the_dropped_block_does_not_carry_the_finish_reason(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=print></function></tool_call>"
                "<tool_call><function=get_weather></function></tool_call>",
            ],
            generic_parser(),
        )
        index = next(
            i for i, item in enumerate(chunks) if isinstance(item, ToolCallResponse)
        )
        assert all(
            getattr(item, "finish_reason", None) is None
            for item in chunks[:index]
            if item is not None
        )

    def test_trailing_text_after_a_dropped_block_still_arrives(self) -> None:
        chunks = feed(
            ["<tool_call><function=print></function></tool_call> done."],
            generic_parser(),
        )
        assert calls_of(chunks) == []
        assert "done." in text_of(chunks)
