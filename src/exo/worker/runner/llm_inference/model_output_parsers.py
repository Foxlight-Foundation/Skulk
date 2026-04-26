from collections.abc import Generator
from functools import cache
from typing import Any

from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.api.types import ToolCallItem
from exo.shared.constants import preferred_env_value
from exo.shared.models.capabilities import resolve_model_capability_profile
from exo.shared.models.model_cards import (
    ModelCard,
    OutputParserType,
    ReasoningFormat,
)
from exo.shared.tracing import record_trace_marker
from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser

_GEMMA4_THINK_START = "<|channel>thought\n"
_GEMMA4_THINK_END = "<channel|>"
_DEFAULT_TOKEN_THINK_START = "<think>"
_DEFAULT_TOKEN_THINK_END = "</think>"
ParserChunk = GenerationResponse | ToolCallResponse | None


def _thinking_stream_debug_enabled() -> bool:
    """Return whether opt-in thinking stream tracing is enabled."""
    value = preferred_env_value(
        "SKULK_TRACE_THINKING_STREAM",
        "EXO_TRACE_THINKING_STREAM",
    )
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _trace_generation_stream(
    label: str,
    model_id: ModelId,
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    """Log parser-stage generation chunks when thinking stream tracing is enabled."""
    if not _thinking_stream_debug_enabled():
        yield from responses
        return

    for response in responses:
        if response is None:
            logger.info(f"[thinking-stream] stage={label} model={model_id} chunk=None")
            yield None
            continue

        if isinstance(response, ToolCallResponse):
            logger.info(
                f"[thinking-stream] stage={label} model={model_id} "
                f"tool_calls={len(response.tool_calls)}"
            )
            yield response
            continue

        logger.info(
            f"[thinking-stream] stage={label} model={model_id} "
            f"text={response.text!r} token={response.token} "
            f"is_thinking={response.is_thinking} finish_reason={response.finish_reason!r}"
        )
        yield response


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_type: type[Model],
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
    model_card: ModelCard | None = None,
    trace_task_id: str | None = None,
    trace_rank: int = 0,
) -> Generator[ParserChunk]:
    mlx_generator = receiver
    mlx_generator = _trace_generation_stream("raw", model_id, mlx_generator)
    capability_profile = resolve_model_capability_profile(
        model_id,
        model_card=model_card,
        tokenizer=tokenizer,
    )

    if capability_profile.thinking_format == ReasoningFormat.ChannelDelimited:
        mlx_generator = parse_gemma4_thinking_channels(mlx_generator)
    elif capability_profile.thinking_format == ReasoningFormat.TokenDelimited:
        think_start, think_end = _resolve_token_delimited_markers(tokenizer)
        mlx_generator = parse_thinking_models(
            mlx_generator,
            think_start,
            think_end,
            starts_in_thinking=_detect_thinking_prompt_suffix(
                prompt,
                tokenizer,
                fallback_think_start=think_start,
            ),
        )
        mlx_generator = _trace_generation_stream("post-thinking-parser", model_id, mlx_generator)

    if capability_profile.output_parser == OutputParserType.GptOss or issubclass(
        model_type, GptOssModel
    ):
        mlx_generator = parse_gpt_oss(mlx_generator)
    elif capability_profile.output_parser == OutputParserType.DeepseekV32 or issubclass(
        model_type, DeepseekV32Model
    ):
        mlx_generator = parse_deepseek_v32(mlx_generator)
    elif tool_parser:
        mlx_generator = parse_tool_calls(
            mlx_generator,
            tool_parser,
            tools,
            trace_task_id=trace_task_id,
            trace_rank=trace_rank,
        )

    mlx_generator = _trace_generation_stream("post-all-parsers", model_id, mlx_generator)
    return mlx_generator


def _resolve_token_delimited_markers(
    tokenizer: TokenizerWrapper,
) -> tuple[str, str]:
    """Resolve token-delimited thinking markers from tokenizer metadata or fallbacks."""
    think_start = tokenizer.think_start or _DEFAULT_TOKEN_THINK_START
    think_end = tokenizer.think_end or _DEFAULT_TOKEN_THINK_END
    return think_start, think_end


def _detect_thinking_prompt_suffix(
    prompt: str,
    tokenizer: TokenizerWrapper,
    *,
    fallback_think_start: str | None = None,
) -> bool:
    """Detect whether the prompt already ends in an opening thinking marker."""
    if detect_thinking_prompt_suffix(prompt, tokenizer):
        return True
    return (
        fallback_think_start is not None
        and prompt.rstrip().endswith(fallback_think_start)
    )


def parse_gemma4_thinking_channels(
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    """Route Gemma 4 channel-delimited reasoning via ``is_thinking``.

    Gemma 4 does not expose ``TokenizerWrapper.has_thinking`` metadata, but its
    tokenizer config defines assistant reasoning as a ``<|channel>thought``
    block terminated by ``<channel|>``. We strip those channel markers from the
    visible stream and mark the enclosed text as thinking so API adapters can
    route it to reasoning fields instead of assistant content.
    """

    buffer = ""
    is_thinking = False

    def _emit_text(
        template: GenerationResponse,
        text: str,
        *,
        thinking: bool,
    ) -> GenerationResponse | None:
        if not text:
            return None
        return template.model_copy(
            update={"text": text, "is_thinking": thinking, "finish_reason": None}
        )

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        buffer += response.text

        if response.finish_reason is None:
            while True:
                if not is_thinking:
                    start_index = buffer.find(_GEMMA4_THINK_START)
                    if start_index != -1:
                        emitted = _emit_text(
                            response,
                            buffer[:start_index],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[start_index + len(_GEMMA4_THINK_START) :]
                        is_thinking = True
                        continue

                    safe_length = len(buffer) - (len(_GEMMA4_THINK_START) - 1)
                    if safe_length > 0:
                        emitted = _emit_text(
                            response,
                            buffer[:safe_length],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[safe_length:]
                    break

                end_index = buffer.find(_GEMMA4_THINK_END)
                if end_index != -1:
                    emitted = _emit_text(
                        response,
                        buffer[:end_index],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[end_index + len(_GEMMA4_THINK_END) :]
                    is_thinking = False
                    continue

                safe_length = len(buffer) - (len(_GEMMA4_THINK_END) - 1)
                if safe_length > 0:
                    emitted = _emit_text(
                        response,
                        buffer[:safe_length],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[safe_length:]
                break
            continue

        while buffer:
            if not is_thinking:
                start_index = buffer.find(_GEMMA4_THINK_START)
                if start_index == -1:
                    emitted = _emit_text(response, buffer, thinking=False)
                    if emitted is not None:
                        yield emitted
                    buffer = ""
                    break

                emitted = _emit_text(response, buffer[:start_index], thinking=False)
                if emitted is not None:
                    yield emitted
                buffer = buffer[start_index + len(_GEMMA4_THINK_START) :]
                is_thinking = True
                continue

            end_index = buffer.find(_GEMMA4_THINK_END)
            if end_index == -1:
                emitted = _emit_text(response, buffer, thinking=True)
                if emitted is not None:
                    yield emitted
                buffer = ""
                break

            emitted = _emit_text(response, buffer[:end_index], thinking=True)
            if emitted is not None:
                yield emitted
            buffer = buffer[end_index + len(_GEMMA4_THINK_END) :]
            is_thinking = False

        # Always emit a terminal chunk with the finish reason so SSE clients close cleanly.
        yield response.model_copy(
            update={"text": "", "is_thinking": False, "finish_reason": response.finish_reason}
        )


def parse_gpt_oss(
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        # Debug: log every token with state
        logger.debug(
            f"parse_gpt_oss token={response.token} text={response.text!r} "
            f"recipient={recipient!r} ch={ch!r} delta={delta!r} "
            f"state={stream.state} current_tool={current_tool_name!r}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    f"parse_gpt_oss yielding tool call: name={current_tool_name!r}"
                )
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "".join(tool_arg_parts)})
                tool_arg_parts = []
            continue

        if ch == "analysis" and not thinking:
            thinking = True

        if ch != "analysis" and thinking:
            thinking = False

        if delta:
            yield response.model_copy(update={"text": delta, "is_thinking": thinking})

        if response.finish_reason is not None:
            yield response


def parse_deepseek_v32(
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Also handles <think>...</think> blocks for thinking mode.
    """
    from exo.worker.engines.mlx.dsml_encoding import (
        THINKING_END,
        THINKING_START,
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    accumulated = ""
    in_tool_call = False
    thinking = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""

    def _try_parse_tool_call(
        text: str, response: GenerationResponse
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_dsml_output(text)
        if parsed is not None:
            return ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
        logger.warning(f"DSML tool call parsing failed for: {text}")
        return response.model_copy(update={"text": text})

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        if response.finish_reason is not None:
            yield from pending_buffer
            pending_buffer.clear()
            if in_tool_call:
                tool_call_text += response.text
                yield (
                    _try_parse_tool_call(tool_call_text, response)
                    if TOOL_CALLS_END in tool_call_text
                    else response.model_copy(update={"text": tool_call_text})
                )
            elif TOOL_CALLS_START in response.text and TOOL_CALLS_END in response.text:
                dsml_start = response.text.index(TOOL_CALLS_START)
                before = response.text[:dsml_start]
                if before:
                    yield response.model_copy(update={"text": before})
                yield _try_parse_tool_call(response.text[dsml_start:], response)
            else:
                yield response
            break

        # ── Handle thinking tags ──
        if not thinking and THINKING_START in response.text:
            thinking = True
            # Yield any text before the <think> tag
            before = response.text[: response.text.index(THINKING_START)]
            if before:
                yield response.model_copy(update={"text": before})
            continue

        if thinking and THINKING_END in response.text:
            thinking = False
            # Yield any text after the </think> tag
            after = response.text[
                response.text.index(THINKING_END) + len(THINKING_END) :
            ]
            if after:
                yield response.model_copy(update={"text": after, "is_thinking": False})
            continue

        if thinking:
            yield response.model_copy(update={"is_thinking": True})
            continue

        # ── Handle tool call accumulation ──
        if in_tool_call:
            tool_call_text += response.text
            if TOOL_CALLS_END in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                in_tool_call = False
                tool_call_text = ""
            continue

        # ── Detect start of tool call block ──
        accumulated += response.text

        if TOOL_CALLS_START in accumulated:
            # The start marker might be split across pending_buffer + current token
            start_idx = accumulated.index(TOOL_CALLS_START)
            # Yield any pending tokens that are purely before the marker
            pre_text = accumulated[:start_idx]
            if pre_text:
                # Flush pending buffer tokens that contributed text before the marker
                for buf_resp in pending_buffer:
                    if not pre_text:
                        break
                    chunk = buf_resp.text
                    if len(chunk) <= len(pre_text):
                        yield buf_resp
                        pre_text = pre_text[len(chunk) :]
                    else:
                        yield buf_resp.model_copy(update={"text": pre_text})
                        pre_text = ""
            pending_buffer = []
            tool_call_text = accumulated[start_idx:]
            accumulated = ""

            # Check if the end marker is already present (entire tool call in one token)
            if TOOL_CALLS_END in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                tool_call_text = ""
            else:
                in_tool_call = True
            continue

        # Check if accumulated text might be the start of a DSML marker
        # Buffer tokens if we see a partial match at the end
        if _could_be_dsml_prefix(accumulated):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        yield from pending_buffer
        pending_buffer.clear()
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    yield from pending_buffer


def _could_be_dsml_prefix(text: str) -> bool:
    """Check if the end of text could be the start of a DSML function_calls marker.

    We look for suffixes of text that are prefixes of the TOOL_CALLS_START pattern.
    This allows us to buffer tokens until we can determine if a tool call is starting.
    """
    from exo.worker.engines.mlx.dsml_encoding import TOOL_CALLS_START

    # Only check the last portion of text that could overlap with the marker
    max_check = len(TOOL_CALLS_START)
    tail = text[-max_check:] if len(text) > max_check else text

    # Check if any suffix of tail is a prefix of TOOL_CALLS_START
    for i in range(len(tail)):
        suffix = tail[i:]
        if TOOL_CALLS_START.startswith(suffix):
            return True
    return False


def parse_thinking_models(
    responses: Generator[ParserChunk],
    think_start: str | None,
    think_end: str | None,
    starts_in_thinking: bool = True,
) -> Generator[ParserChunk]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets ``is_thinking`` on all others, and buffers
    partial marker fragments so split or fused ``<think>`` tags do not leak into
    visible output.

    Always yields a terminal chunk with ``finish_reason`` so the stream closes
    cleanly even when the model ends inside a thinking block.
    """
    if think_start is None or think_end is None:
        for response in responses:
            yield response
        return

    buffer = ""
    is_thinking = starts_in_thinking

    def _emit_text(
        template: GenerationResponse,
        text: str,
        *,
        thinking: bool,
    ) -> GenerationResponse | None:
        if not text:
            return None
        return template.model_copy(
            update={"text": text, "is_thinking": thinking, "finish_reason": None}
        )

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        buffer += response.text

        if response.finish_reason is None:
            while True:
                if not is_thinking:
                    start_index = buffer.find(think_start)
                    if start_index != -1:
                        emitted = _emit_text(
                            response,
                            buffer[:start_index],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[start_index + len(think_start) :]
                        is_thinking = True
                        continue

                    safe_length = len(buffer) - (len(think_start) - 1)
                    if safe_length > 0:
                        emitted = _emit_text(
                            response,
                            buffer[:safe_length],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[safe_length:]
                    break

                end_index = buffer.find(think_end)
                if end_index != -1:
                    emitted = _emit_text(
                        response,
                        buffer[:end_index],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[end_index + len(think_end) :]
                    is_thinking = False
                    continue

                safe_length = len(buffer) - (len(think_end) - 1)
                if safe_length > 0:
                    emitted = _emit_text(
                        response,
                        buffer[:safe_length],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[safe_length:]
                break
            continue

        while buffer:
            if not is_thinking:
                start_index = buffer.find(think_start)
                if start_index == -1:
                    emitted = _emit_text(response, buffer, thinking=False)
                    if emitted is not None:
                        yield emitted
                    buffer = ""
                    break

                emitted = _emit_text(response, buffer[:start_index], thinking=False)
                if emitted is not None:
                    yield emitted
                buffer = buffer[start_index + len(think_start) :]
                is_thinking = True
                continue

            end_index = buffer.find(think_end)
            if end_index == -1:
                emitted = _emit_text(response, buffer, thinking=True)
                if emitted is not None:
                    yield emitted
                buffer = ""
                break

            emitted = _emit_text(response, buffer[:end_index], thinking=True)
            if emitted is not None:
                yield emitted
            buffer = buffer[end_index + len(think_end) :]
            is_thinking = False

        yield response.model_copy(
            update={"text": "", "is_thinking": False, "finish_reason": response.finish_reason}
        )


def parse_tool_calls(
    responses: Generator[ParserChunk],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
    *,
    trace_task_id: str | None = None,
    trace_rank: int = 0,
) -> Generator[ParserChunk]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if not in_tool_call:
            yield response
            continue

        tool_call_text_parts.append(response.text)
        if response.text.endswith(tool_parser.end_parsing):
            # parse the actual tool calls from the tool call text
            combined = "".join(tool_call_text_parts)
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None:
                logger.warning(f"tool call parsing failed for text {combined}")
                if trace_task_id is not None:
                    record_trace_marker(
                        "tool_call_parse_error",
                        trace_rank,
                        category="tooling",
                        task_id=trace_task_id,
                        tags=["tool_call", "error"],
                        attrs={"raw_length": len(combined)},
                    )
                yield response.model_copy(
                    update={"text": combined, "token": 0, "finish_reason": "error"}
                )
                break

            if trace_task_id is not None:
                record_trace_marker(
                    "tool_call_parsed",
                    trace_rank,
                    category="tooling",
                    task_id=trace_task_id,
                    tags=["tool_call"],
                    attrs={"tool_call_count": len(parsed)},
                )
            yield ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
            continue

        if response.finish_reason is not None:
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={
                    "text": "".join(tool_call_text_parts),
                    "token": 0,
                    "finish_reason": "error",
                }
            )
            yield response
