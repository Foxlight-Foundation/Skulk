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
from typing import cast

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


class TestEverythingInOneTerminalChunk:
    """The whole message can arrive as a single terminal chunk.

    There is no next chunk to drive the streaming scan there, so the rest of
    the message has to be parsed in place. Each exit from the close site used
    to decide this for itself, and each got it wrong differently.
    """

    def test_two_calls_in_one_terminal_chunk_are_both_delivered(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather><parameter=location>Denver</parameter></function></tool_call>"
                "<tool_call><function=get_weather><parameter=location>Boston</parameter></function></tool_call>",
            ],
            generic_parser(),
        )
        responses = [c for c in chunks if isinstance(c, ToolCallResponse)]
        assert len(responses) == 1
        assert len(responses[0].tool_calls) == 2

    def test_a_call_then_text_in_one_terminal_chunk(self) -> None:
        chunks = feed(
            ["<tool_call><function=get_weather></function></tool_call> Done."],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "Done." in text_of(chunks)

    def test_exactly_one_chunk_carries_the_finish_reason(self) -> None:
        for pieces in (
            ["<tool_call><function=get_weather></function></tool_call> Done."],
            ["<tool_call><function=print></function></tool_call> Done."],
            ["The weather is fine."],
        ):
            chunks = feed(pieces, generic_parser())
            terminals = [
                c
                for c in chunks
                if isinstance(c, ToolCallResponse)
                or (c is not None and c.finish_reason is not None)
            ]
            assert len(terminals) == 1, pieces
            assert terminals[0] is chunks[-1], pieces


class TestBlockClosedByEndOfGeneration:
    """A block the model never closed must not lose the calls before it.

    Several families end a tool-calling message rather than emitting a closing
    marker, so this path is normal rather than exceptional, and it has to hold
    the same rules as a marker-closed block.
    """

    @staticmethod
    def assert_reaches_the_consumer(chunks: list[ParserChunk]) -> None:
        """The consumer stops at the first chunk carrying a finish reason.

        Checking the whole list is not enough: calls yielded after something
        terminal never reach a caller.
        """

        index = next(
            i for i, item in enumerate(chunks) if isinstance(item, ToolCallResponse)
        )
        assert all(
            getattr(item, "finish_reason", None) is None
            for item in chunks[:index]
            if item is not None
        )

    def test_an_earlier_call_survives_a_final_unoffered_block(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather></function></tool_call>",
                "<tool_call><function=print></function>",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        self.assert_reaches_the_consumer(chunks)

    def test_an_earlier_call_survives_a_final_truncated_block(self) -> None:
        chunks = feed(
            [
                "<tool_call><function=get_weather></function></tool_call>",
                "<tool_call><func",
            ],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        self.assert_reaches_the_consumer(chunks)

    def test_a_truncated_block_alone_is_still_an_error(self) -> None:
        chunks = feed(["<tool_call><func"], generic_parser())
        assert calls_of(chunks) == []
        assert any(
            getattr(item, "finish_reason", None) == "error"
            for item in chunks
            if item is not None
        )


class TestTruncationIsNotACall:
    """A call cut off at max_tokens must not be handed to the caller.

    Several families end a tool-calling message without a closing marker, so an
    unclosed block is parsed rather than discarded. Truncation looks identical
    at the parser, and a marker dialect's inner parser strips the closing
    marker only if it is present, so the finish reason is what separates them.
    """

    @staticmethod
    def truncated(finish_reason: FinishReason) -> list[ParserChunk]:
        def source() -> Generator[ParserChunk]:
            yield chunk("<tool_call><function=get_weather>")
            yield chunk("<parameter=location>Denver</parameter></function>", finish_reason)

        return list(parse_tool_calls(source(), generic_parser(), tools=WEATHER))

    def test_a_block_cut_off_at_max_tokens_is_not_a_call(self) -> None:
        chunks = self.truncated("length")
        assert calls_of(chunks) == []

    def test_the_same_block_ended_normally_is_a_call(self) -> None:
        # The families this exists for end the message rather than closing the
        # block, so a normal stop must still produce the call.
        chunks = self.truncated("stop")
        assert calls_of(chunks) == ["get_weather"]


class TestRejectedBlockDoesNotLeakMarkers:
    """A block handed back as content must not carry its dialect's markers.

    Found by the harness's tool-contract suite: a Llama model called a tool
    that a named tool_choice had narrowed away, the call was correctly
    rejected, and the caller received `<|python_tag|>` in the answer text.
    """

    def test_an_unmarked_rejected_call_loses_its_marker(self) -> None:
        chunks = feed(
            [
                '<|python_tag|>{"name": "print", ',
                '"parameters": {"value": "hi"}}',
            ],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == []
        answer = text_of(chunks)
        assert "<|python_tag|>" not in answer
        # The call itself is still shown, so the caller can see what happened.
        assert "print" in answer

    def test_a_marked_rejected_call_loses_its_markers(self) -> None:
        chunks = feed(
            ["<tool_call><function=print></function></tool_call>"],
            generic_parser(),
        )
        assert calls_of(chunks) == []
        answer = text_of(chunks)
        assert "<tool_call>" not in answer
        assert "</tool_call>" not in answer
        assert "print" in answer

    def test_an_accepted_call_is_unaffected(self) -> None:
        chunks = feed(
            ["<tool_call><function=get_weather></function></tool_call>"],
            generic_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]


class TestAJsonAnswerStreamsIncrementally:
    """A message-opening brace no longer holds the whole answer.

    The unmarked dialect's closing token is a generation stop that never
    arrives as text, so an anchored open used to hold every chunk until the
    terminal one: a plain JSON answer lost all incremental output and
    time-to-first-token grew to the full generation time (deferred #879
    review finding). The open is now provisional, and the buffered prefix is
    released the moment it can no longer be a call.
    """

    def test_a_json_answer_is_released_at_the_first_decisive_key(self) -> None:
        chunks = feed(
            ['{"result": ', '{"answer": 42', ', "unit": "mm"}}'],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == []
        assert text_of(chunks) == '{"result": {"answer": 42, "unit": "mm"}}'
        streamed_early = [
            item
            for item in chunks
            if isinstance(item, GenerationResponse)
            and item.text
            and item.finish_reason is None
        ]
        # Held-to-terminal behavior would produce one terminal blob; the
        # released answer streams across several non-terminal chunks.
        assert len(streamed_early) >= 2

    def test_a_name_first_answer_is_released_at_the_second_key(self) -> None:
        chunks = feed(
            ['{"name": "John"', ', "age": 30}'],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == []
        assert text_of(chunks) == '{"name": "John", "age": 30}'

    def test_a_brace_that_is_not_json_is_released_immediately(self) -> None:
        chunks = feed(
            ["{oops, this is prose", " that keeps going"],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == []
        assert text_of(chunks) == "{oops, this is prose that keeps going"
        # Released as content, never reported as a parse failure.
        assert all(
            item.finish_reason != "error"
            for item in chunks
            if isinstance(item, GenerationResponse)
        )

    def test_a_reversed_key_order_call_still_parses(self) -> None:
        chunks = feed(
            [
                '{"parameters": {"location": "Denver"}',
                ', "name": "get_weather"}',
            ],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_a_llama_type_wrapper_key_still_parses(self) -> None:
        # Some Llama variants wrap the signature with a "type" key; it
        # neither makes the object a call nor rules one out.
        chunks = feed(
            [
                '{"type": "function", "name": "get_weather", ',
                '"parameters": {"location": "Denver"}}',
            ],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]

    def test_a_released_answer_still_yields_a_later_marker_call(self) -> None:
        chunks = feed(
            [
                '{"a": 1} ',
                '<|python_tag|>{"name": "get_weather", "parameters": {}}',
            ],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert '{"a": 1}' in text_of(chunks)

    def test_release_and_marker_call_in_one_terminal_chunk(self) -> None:
        chunks = feed(
            [
                '{"a": 1} <|python_tag|>'
                '{"name": "get_weather", "parameters": {}}'
            ],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert '{"a": 1}' in text_of(chunks)


class TestTextAfterAnUnmarkedCall:
    """Trailing prose after an anchored call reaches the caller as content."""

    def test_a_remark_after_the_call_is_delivered(self) -> None:
        chunks = feed(
            ['{"name": "get_weather", "parameters": {}}', " Done."],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "Done." in text_of(chunks)

    def test_the_tool_response_stays_the_terminal_chunk(self) -> None:
        chunks = feed(
            ['{"name": "get_weather", "parameters": {}} All set.'],
            make_text_dialect_parser("{", "<|eom_id|>"),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "All set." in text_of(chunks)
        last = [item for item in chunks if item is not None][-1]
        assert isinstance(last, ToolCallResponse)


class TestClosingMarkerInsideArguments:
    """A quoted argument may contain the dialect's own closing marker.

    An HTML-writing tool passing "</tool_call>" used to end the block at the
    quoted marker: the truncated block failed parsing and the caller received
    a generation error instead of the call (deferred #879 review finding).
    The scan mode comes from the wiring, which is the only place the block
    interior's quoting rules are known.
    """

    @staticmethod
    def _hermes_json_parser() -> ToolParser:
        import json as json_module

        def inner(text: str) -> dict[str, object]:
            payload = cast("dict[str, object]", json_module.loads(text))
            return {
                "name": payload["name"],
                "arguments": json_module.dumps(payload["arguments"]),
            }

        return make_mlx_parser(
            "<tool_call>", "</tool_call>", inner, close_scan="json_strings"
        )

    def test_a_quoted_closer_does_not_truncate_a_json_block(self) -> None:
        chunks = feed(
            [
                '<tool_call>{"name": "get_weather", ',
                '"arguments": {"location": "</tool_call>"}}',
                "</tool_call> after",
            ],
            self._hermes_json_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "after" in text_of(chunks)

    def test_a_gemma_quoted_closer_does_not_truncate(self) -> None:
        from skulk.worker.runner.llm_inference.tool_text_parser import (
            gemma4_calls,
        )

        def inner(text: str) -> list[dict[str, object]]:
            items = gemma4_calls(text)
            if not items:
                raise ValueError("no tool calls")
            return [
                {"name": item.name, "arguments": item.arguments}
                for item in items
            ]

        parser = make_mlx_parser(
            "<|tool_call>", "<tool_call|>", inner, close_scan="gemma_quotes"
        )
        chunks = feed(
            [
                "<|tool_call>call:get_weather{location:",
                '<|"|>a<tool_call|>b<|"|>}',
                "<tool_call|>",
            ],
            parser,
        )
        assert calls_of(chunks) == ["get_weather"]


class TestTextAroundAMistralArray:
    """Mistral trailing text on the MLX streaming path.

    Mistral closes its call by ending the message (the wired end marker is an
    impossible sentinel), so the block resolves on the end-of-generation path
    and only the split-aware text parser knows where the array ends. The
    split reading is an overlay: the displaced upstream ``NAME[ARGS]`` form
    still parses through the inner parser, with no remainder.
    """

    _SENTINEL = "\x00test:mistral:end\x00"

    def _mistral_parser(self) -> ToolParser:
        def inner(text: str) -> list[dict[str, object]]:
            # Stands in for the displaced upstream parser: reads the
            # NAME[ARGS] form only.
            stripped = text.strip()
            if not stripped.startswith("upstream_form"):
                raise ValueError("not the upstream form")
            return [{"name": "get_weather", "arguments": "{}"}]

        return make_mlx_parser("[TOOL_CALLS]", self._SENTINEL, inner)

    def test_trailing_text_after_the_array_is_delivered(self) -> None:
        chunks = feed(
            [
                "[TOOL_CALLS] ",
                '[{"name": "get_weather", "arguments": {"location": "Paris"}}]',
                " I will check that.",
            ],
            self._mistral_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
        assert "I will check that." in text_of(chunks)
        last = [item for item in chunks if item is not None][-1]
        assert isinstance(last, ToolCallResponse)

    def test_the_displaced_upstream_form_still_parses(self) -> None:
        chunks = feed(
            ["[TOOL_CALLS]", "upstream_form"],
            self._mistral_parser(),
        )
        assert calls_of(chunks) == ["get_weather"]
