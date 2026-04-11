from collections.abc import Generator
from typing import Any

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import Model
from exo.shared.types.worker.runner_response import (
    FinishReason,
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.dsml_encoding import (
    DSML_TOKEN,
    THINKING_END,
    THINKING_START,
    TOOL_CALLS_END,
    TOOL_CALLS_START,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    apply_all_parsers,
    parse_deepseek_v32,
    parse_gemma4_thinking_channels,
    parse_thinking_models,
    parse_tool_calls,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _make_response(
    text: str, token: int, finish_reason: FinishReason | None = None
) -> GenerationResponse:
    return GenerationResponse(
        text=text, token=token, finish_reason=finish_reason, usage=None
    )


def _queue_source(
    tokens: list[GenerationResponse],
) -> Generator[GenerationResponse | None]:
    for token in tokens:
        yield token
        yield None
    while True:
        yield None


def _step_until_finish(
    parser_gen: Generator[GenerationResponse | ToolCallResponse | None],
    max_steps: int = 200,
) -> list[GenerationResponse | ToolCallResponse]:
    results: list[GenerationResponse | ToolCallResponse] = []
    for _ in range(max_steps):
        try:
            result = next(parser_gen)
        except StopIteration:
            break
        if result is None:
            continue
        results.append(result)
        if isinstance(result, GenerationResponse) and result.finish_reason is not None:
            return results
        if isinstance(result, ToolCallResponse):
            return results
    return results


def _got_finish(results: list[GenerationResponse | ToolCallResponse]) -> bool:
    for r in results:
        if isinstance(r, ToolCallResponse):
            return True
        if r.finish_reason is not None:
            return True
    return False


# ── parse_deepseek_v32 ──────────────────────────────────────────


class TestDeepSeekV32FinishReason:
    def test_finish_reason_with_buffered_dsml_prefix(self):
        tokens = [
            _make_response("Hello! The answer is x", 0),
            _make_response("<", 1),
            _make_response("", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)
        full_text = "".join(
            r.text for r in results if isinstance(r, GenerationResponse)
        )
        assert "Hello" in full_text
        assert "<" in full_text

    def test_finish_reason_completes_tool_call_block(self):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 2),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n',
                3,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 4),
            _make_response(TOOL_CALLS_END, 5, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_mid_tool_call_before_close(self):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(
                f'<{DSML_TOKEN}invoke name="get_weather">\n', 2, finish_reason="stop"
            ),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)

    def test_finish_reason_single_token_complete_dsml_block(self):
        dsml_block = (
            f"{TOOL_CALLS_START}\n"
            f'<{DSML_TOKEN}invoke name="get_weather">\n'
            f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n'
            f"</{DSML_TOKEN}invoke>\n"
            f"{TOOL_CALLS_END}"
        )
        tokens = [_make_response(dsml_block, 0, finish_reason="stop")]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_during_thinking(self):
        tokens = [
            _make_response(THINKING_START, 0),
            _make_response("I need to think about this", 1),
            _make_response(" carefully before responding", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)

    def test_finish_reason_after_thinking_then_tool_call(self):
        tokens = [
            _make_response(THINKING_START, 0),
            _make_response("Let me check the weather.", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n\n", 3),
            _make_response(TOOL_CALLS_START, 4),
            _make_response("\n", 5),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 6),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
                7,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 8),
            _make_response(TOOL_CALLS_END, 9, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_normal_text_no_buffering(self):
        tokens = [
            _make_response("Hello", 0),
            _make_response(" world", 1),
            _make_response("!", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)
        full_text = "".join(
            r.text for r in results if isinstance(r, GenerationResponse)
        )
        assert full_text == "Hello world!"

    def test_finish_reason_multiple_buffered_prefix_tokens(self):
        tokens = [
            _make_response("text ", 0),
            _make_response("<", 1),
            _make_response("not a tag", 2),
            _make_response(" more<", 3),
            _make_response("", 4, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)


# ── parse_thinking_models ────────────────────────────────────────


class TestThinkingModelsFinishReason:
    def test_finish_reason_during_thinking(self):
        tokens = [
            _make_response("<think>", 0),
            _make_response("reasoning here", 1),
            _make_response("more reasoning", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _got_finish(results)
        last_gen = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(last_gen) == 1
        assert last_gen[0].is_thinking is False

    def test_finish_reason_after_thinking(self):
        tokens = [
            _make_response("<think>", 0),
            _make_response("hmm", 1),
            _make_response("</think>", 2),
            _make_response("The answer is 42.", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _got_finish(results)

    def test_finish_reason_starts_in_thinking(self):
        tokens = [
            _make_response("still thinking", 0),
            _make_response("</think>", 1),
            _make_response("done", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=True,
            )
        )
        assert _got_finish(results)

    def test_split_and_fused_thinking_markers_are_hidden(self):
        tokens = [
            _make_response("<th", 0),
            _make_response("ink>\nWorking", 1),
            _make_response(" through it</th", 2),
            _make_response("ink>Answer", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )

        all_text = "".join(r.text for r in results if isinstance(r, GenerationResponse))
        thinking_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and r.is_thinking
        )
        visible_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        )

        assert "<think>" not in all_text
        assert "</think>" not in all_text
        assert thinking_text == "\nWorking through it"
        assert visible_text == "Answer"
        assert _got_finish(results)


class _NoThinkingTokenizer:
    has_thinking = False
    think_start = None
    think_end = None


class TestGemma4ThinkingChannels:
    def test_closed_thinking_block_is_flagged_and_markers_are_stripped(self):
        tokens = [
            _make_response("<|channel>thought\n", 100),
            _make_response("Reason about the image.", 101),
            _make_response("<channel|>", 102),
            _make_response("It is a starship.", 103, finish_reason="stop"),
        ]

        results = _step_until_finish(
            parse_gemma4_thinking_channels(_queue_source(tokens))
        )

        thinking_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and r.is_thinking
        )
        visible_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        )

        assert "Reason about the image." in thinking_text
        assert "<|channel>thought" not in thinking_text
        assert "<channel|>" not in thinking_text
        assert visible_text == "It is a starship."

    def test_split_markers_are_buffered_until_complete(self):
        tokens = [
            _make_response("<|chan", 100),
            _make_response("nel>thought\n", 101),
            _make_response("Hidden reasoning.", 102),
            _make_response("<chan", 103),
            _make_response("nel|>", 104),
            _make_response("Visible answer.", 105, finish_reason="stop"),
        ]

        results = _step_until_finish(
            parse_gemma4_thinking_channels(_queue_source(tokens))
        )

        all_text = "".join(r.text for r in results if isinstance(r, GenerationResponse))
        thinking_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and r.is_thinking
        )
        visible_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        )

        assert "<|channel>thought" not in all_text
        assert "<channel|>" not in all_text
        assert thinking_text == "Hidden reasoning."
        assert visible_text == "Visible answer."

    def test_unclosed_thinking_block_at_finish_stays_hidden(self):
        tokens = [
            _make_response("<|channel>thought\n", 100),
            _make_response(
                "Model stayed in the thought channel.", 101, finish_reason="stop"
            ),
        ]

        results = _step_until_finish(
            parse_gemma4_thinking_channels(_queue_source(tokens))
        )

        thinking_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and r.is_thinking
        )
        visible_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        )

        assert thinking_text == "Model stayed in the thought channel."
        assert visible_text == ""
        assert _got_finish(results)

    def test_apply_all_parsers_uses_gemma4_channel_parser_without_tokenizer_metadata(
        self,
    ):
        tokens = [
            _make_response("<|channel>thought\n", 100),
            _make_response("Reason silently.", 101),
            _make_response("<channel|>", 102),
            _make_response("Final answer.", 103, finish_reason="stop"),
        ]

        results = _step_until_finish(
            apply_all_parsers(
                _queue_source(tokens),
                prompt="<bos><|turn>user\nDescribe this.<turn|>\n<|turn>model\n",
                tool_parser=None,
                tokenizer=_NoThinkingTokenizer(),
                model_type=Model,
                model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
                tools=None,
            )
        )

        thinking_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and r.is_thinking
        )
        visible_text = "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        )

        assert thinking_text == "Reason silently."
        assert visible_text == "Final answer."

    def test_apply_all_parsers_uses_deepseek_parser_from_family_without_model_class(
        self,
    ):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 2),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n',
                3,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 4),
            _make_response(TOOL_CALLS_END, 5, finish_reason="stop"),
        ]

        results = _step_until_finish(
            apply_all_parsers(
                _queue_source(tokens),
                prompt="",
                tool_parser=None,
                tokenizer=_NoThinkingTokenizer(),
                model_type=Model,
                model_id=ModelId("custom/deepseek-compatible"),
                tools=None,
                model_card=ModelCard(
                    model_id=ModelId("custom/deepseek-compatible"),
                    storage_size=Memory.from_bytes(1024),
                    n_layers=1,
                    hidden_size=1,
                    supports_tensor=False,
                    tasks=[ModelTask.TextGeneration],
                    family="deepseek-v3.2",
                    capabilities=["text", "thinking"],
                ),
            )
        )

        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"


# ── parse_tool_calls (generic) ──────────────────────────────────


def _dummy_parser_fn(text: str) -> dict[str, Any]:
    return {"name": "test_fn", "arguments": {"arg": text}}


_dummy_parser = make_mlx_parser("<tool_call>", "</tool_call>", _dummy_parser_fn)


class TestGenericToolCallsFinishReason:
    def test_finish_reason_after_complete_tool_call(self):
        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("body", 1),
            _make_response("</tool_call>", 2),
            _make_response("extra text", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1

    def test_finish_reason_mid_tool_call_unclosed(self):
        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("partial content", 1, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        assert _got_finish(results)

    def test_finish_reason_no_tool_calls(self):
        tokens = [
            _make_response("Just", 0),
            _make_response(" a", 1),
            _make_response(" normal", 2),
            _make_response(" response.", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        assert _got_finish(results)


# ── Double parser chain (parse_thinking_models → parse_deepseek_v32) ──


class TestBatchGeneratorSingleNext:
    def test_finish_reason_with_buffered_tokens_drain_loop(self):
        from exo.worker.runner.llm_inference.batch_generator import GeneratorQueue

        queue: GeneratorQueue[GenerationResponse] = GeneratorQueue()
        parser = parse_deepseek_v32(queue.gen())

        tokens = [
            _make_response("Hello ", 0),
            _make_response(" `<", 1),
            _make_response("", 2, finish_reason="stop"),
        ]

        collected: list[GenerationResponse | ToolCallResponse] = []
        for token in tokens:
            queue.push(token)
            while (parsed := next(parser, None)) is not None:
                collected.append(parsed)
            if token.finish_reason is not None:
                break

        assert _got_finish(collected), (
            f"No finish_reason in collected: {[(type(r).__name__, getattr(r, 'finish_reason', None) if isinstance(r, GenerationResponse) else 'tool') for r in collected]}"
        )
