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

from skulk.api.types import ToolCallItem
from skulk.shared.constants import preferred_env_value
from skulk.shared.models.capabilities import resolve_model_capability_profile
from skulk.shared.models.model_cards import (
    ModelCard,
    OutputParserType,
    ReasoningFormat,
)
from skulk.shared.tracing import record_trace_marker
from skulk.shared.types.common import ModelId
from skulk.shared.types.mlx import Model
from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.llm_inference.tool_parsers import (
    ToolParser,
    declared_tool_calls,
)

_GEMMA4_THINK_START = "<|channel>thought\n"
_GEMMA4_THINK_END = "<channel|>"
_DEFAULT_TOKEN_THINK_START = "<think>"
_DEFAULT_TOKEN_THINK_END = "</think>"
ParserChunk = GenerationResponse | ToolCallResponse | None


def _thinking_stream_debug_enabled() -> bool:
    """Return whether opt-in thinking stream tracing is enabled."""
    value = preferred_env_value(
        "SKULK_TRACE_THINKING_STREAM",
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
    elif tool_parser and tools:
        # The parser is wired from the tokenizer, which does not know whether
        # this request offered tools. Skipping it when none were offered is
        # what keeps a model that spontaneously writes something call-shaped
        # from returning tool calls to a caller who asked for none, and is
        # what makes tool_choice "none" hold, since that removes the tools.
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

        # Keep parser-state diagnostics useful without retaining generated text.
        logger.debug(
            f"parse_gpt_oss token={response.token} "
            f"text_chars={len(response.text)} "
            f"has_recipient={recipient is not None} channel={ch!r} "
            f"delta_chars={len(delta or '')} state={stream.state} "
            f"has_current_tool={current_tool_name is not None}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    "parse_gpt_oss yielding tool call "
                    f"(name_chars={len(current_tool_name)})"
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
    from skulk.worker.engines.mlx.dsml_encoding import (
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
        logger.warning(
            f"DSML tool call parsing failed (generated_chars={len(text)})"
        )
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
    from skulk.worker.engines.mlx.dsml_encoding import TOOL_CALLS_START

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


def _block_start_index(
    text: str, tool_parser: ToolParser, *, at_message_start: bool
) -> int | None:
    """Index in ``text`` where a tool-call block begins, or ``None``.

    Distinctive markers open a block wherever they appear, because models
    routinely write a sentence before calling ("I'll check that." then the
    call). The unmarked dialect's opening marker is ``{``, which appears in
    ordinary prose and JSON answers, so it opens a block only at the start of
    the message, which is the only place the families using it write a call.
    """

    earliest: int | None = None
    for marker in tool_parser.extra_start_parsing:
        found = text.find(marker)
        if found != -1 and (earliest is None or found < earliest):
            earliest = found

    if not tool_parser.anchored:
        found = text.find(tool_parser.start_parsing)
        if found != -1 and (earliest is None or found < earliest):
            earliest = found
    elif at_message_start:
        stripped = text.lstrip()
        if stripped.startswith(tool_parser.start_parsing):
            found = len(text) - len(stripped)
            if earliest is None or found < earliest:
                earliest = found
    return earliest


def _partial_marker_suffix_length(text: str, markers: tuple[str, ...]) -> int:
    """Length of the trailing run of ``text`` that could still become a marker.

    Held back rather than emitted, so a marker split across chunks is still
    recognized. Bounded by the longest marker, so this is a few characters of
    latency at most and never an unbounded buffer.
    """

    longest = max(len(marker) for marker in markers) - 1
    for length in range(min(longest, len(text)), 0, -1):
        tail = text[-length:]
        if any(marker.startswith(tail) for marker in markers):
            return length
    return 0


def _scan_remaining_blocks(
    text: str, tool_parser: ToolParser, tools: list[dict[str, Any]] | None
) -> tuple[list[ToolCallItem], str]:
    """Parse every remaining block in a complete text.

    Used once generation has ended, where there is no further chunk to drive
    the streaming scan and the rest of the message is already in hand. Returns
    the calls found and the text that was not part of any block, so a message
    that puts a call the caller cannot run before one they can still delivers
    the second call and the surrounding prose.
    """

    calls: list[ToolCallItem] = []
    leftover: list[str] = []
    remaining = text
    while remaining:
        start = _block_start_index(remaining, tool_parser, at_message_start=False)
        if start is None:
            leftover.append(remaining)
            break
        end = remaining.find(tool_parser.end_parsing, start)
        if end == -1:
            leftover.append(remaining)
            break
        end_of_block = end + len(tool_parser.end_parsing)
        block = remaining[start:end_of_block]
        parsed = tool_parser.parse(block.strip(), tools=tools)
        kept = declared_tool_calls(parsed, tools) if parsed is not None else []
        if kept:
            leftover.append(remaining[:start])
            calls.extend(kept)
        else:
            # Not a call the caller can run, so it is text like any other.
            leftover.append(remaining[:end_of_block])
        remaining = remaining[end_of_block:]
    return calls, "".join(leftover)


def parse_tool_calls(
    responses: Generator[ParserChunk],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
    *,
    trace_task_id: str | None = None,
    trace_rank: int = 0,
) -> Generator[ParserChunk]:
    """Recover tool calls from the generated stream, one response per message.

    The calls of every block in a message are coalesced into a single
    ``ToolCallResponse``. That is the OpenAI shape, where one assistant message
    carries a ``tool_calls`` array, and it is what makes a model's parallel
    calls survive: several families write each call in its own block, and the
    consumer of this stream stops at the first chunk carrying a finish reason,
    so a response per block would deliver the first call and drop the rest.
    """

    in_tool_call = False
    # Held until the message ends rather than emitted per block, so several
    # blocks arrive as one response. The stream does not end when generation
    # does (the source keeps idling), so the terminal chunk is the signal.
    accumulated_calls: list[ToolCallItem] = []
    last_response: GenerationResponse | None = None
    tool_call_text_parts: list[str] = []
    # A chunk is whatever the streaming detokenizer could resolve this step, not
    # a token: an opening marker that is one token id still arrives split across
    # chunks ("<tool", "_", "c", "all>"). Testing each chunk on its own misses
    # the marker for most models, so text is scanned across chunk boundaries by
    # carrying forward only the trailing run that could still become a marker.
    # That run is shorter than the longest marker, so ordinary answers stream
    # with at most a few characters of latency and nothing is ever held for a
    # message that turns out not to contain a call.
    held_text = ""
    at_message_start = True
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        # Reasoning is never part of a tool-call block: this parser runs
        # downstream of the thinking parser, and a call a model only
        # contemplated inside its reasoning must not be executed. Passing those
        # chunks straight through also keeps them out of the opening decision,
        # so a thinking model that reasons before calling still has its marker
        # examined when the visible answer begins.
        if response.is_thinking:
            yield response
            continue

        last_response = response
        just_opened = False
        if not in_tool_call:
            scanned = held_text + response.text
            start = _block_start_index(
                scanned, tool_parser, at_message_start=at_message_start
            )
            if start is not None:
                preamble = scanned[:start]
                if preamble:
                    yield response.model_copy(
                        update={
                            "text": preamble,
                            "token": 0,
                            "finish_reason": None,
                        }
                    )
                in_tool_call = True
                just_opened = True
                held_text = ""
                at_message_start = False
                tool_call_text_parts.append(scanned[start:])
            else:
                keep = _partial_marker_suffix_length(
                    scanned, tool_parser.start_markers
                )
                if response.finish_reason is not None:
                    # Nothing more is coming, so a partial marker is just text.
                    keep = 0
                emitted = scanned[: len(scanned) - keep]
                held_text = scanned[len(scanned) - keep :]
                if emitted.strip():
                    at_message_start = False
                if response.finish_reason is not None and accumulated_calls:
                    # A call was found earlier in this message and the tool
                    # response has to be the terminal chunk, so this trailing
                    # text is released without the finish reason.
                    if emitted:
                        yield response.model_copy(
                            update={
                                "text": emitted,
                                "token": 0,
                                "finish_reason": None,
                            }
                        )
                    yield ToolCallResponse(
                        tool_calls=accumulated_calls,
                        usage=response.usage,
                        stats=response.stats,
                    )
                    accumulated_calls = []
                    continue
                if emitted == response.text and not held_text:
                    yield response
                elif emitted or response.finish_reason is not None:
                    yield response.model_copy(
                        update={"text": emitted, "token": 0}
                    )
                continue

        if not just_opened:
            tool_call_text_parts.append(response.text)
        # The closing marker splits across chunks for the same reason the
        # opening one does, so it is located in the accumulated block rather
        # than tested against one chunk. Locating rather than matching the end
        # also matters because a model may keep writing after the call ("...
        # </tool_call> Done."): everything past the marker is ordinary text and
        # goes back to the opening scan, where a second call in the same
        # message is still found.
        block_so_far = "".join(tool_call_text_parts)
        end_index = block_so_far.find(tool_parser.end_parsing)
        if end_index != -1:
            end_of_block = end_index + len(tool_parser.end_parsing)
            combined = block_so_far[:end_of_block]
            held_text = block_so_far[end_of_block:]
            tool_call_text_parts = [combined]
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            if parsed is not None:
                kept = declared_tool_calls(parsed, tools)
                if not kept:
                    logger.info(
                        "Block named no offered tool, emitting it as content "
                        f"(parsed_calls={len(parsed)})"
                    )
                    in_tool_call = False
                    tool_call_text_parts = []
                    # The remainder stays with the scan rather than being
                    # emitted here, so a further call in the trailing text is
                    # still found. The finish reason is withheld while anything
                    # remains, since the consumer stops at the first chunk
                    # carrying one and would never see the rest.
                    remainder_follows = bool(held_text) or bool(accumulated_calls)
                    yield response.model_copy(
                        update={
                            "text": combined,
                            "token": 0,
                            "finish_reason": None
                            if remainder_follows
                            else response.finish_reason,
                        }
                    )
                    if response.finish_reason is None:
                        continue
                    if held_text:
                        # No further chunk will drive the scan, so the rest of
                        # the message is parsed here rather than emitted whole.
                        more_calls, leftover = _scan_remaining_blocks(
                            held_text, tool_parser, tools
                        )
                        accumulated_calls.extend(more_calls)
                        held_text = ""
                        if leftover:
                            yield response.model_copy(
                                update={
                                    "text": leftover,
                                    "token": 0,
                                    "finish_reason": None
                                    if accumulated_calls
                                    else response.finish_reason,
                                }
                            )
                    if accumulated_calls:
                        yield ToolCallResponse(
                            tool_calls=accumulated_calls,
                            usage=response.usage,
                            stats=response.stats,
                        )
                        accumulated_calls = []
                    continue
                parsed = kept
            logger.info(
                "Parsed generated tool-call block "
                f"(chunks={len(tool_call_text_parts)}, "
                f"generated_chars={len(combined)}, "
                f"parsed_calls={len(parsed) if parsed is not None else 0})"
            )
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None and tool_parser.unparsed_is_text:
                logger.info(
                    "Unmarked block did not parse as a tool call, "
                    f"emitting it as content (generated_chars={len(combined)})"
                )
                yield response.model_copy(
                    update={"text": combined, "token": 0}
                )
                if held_text and response.finish_reason is not None:
                    yield response.model_copy(
                        update={"text": held_text, "token": 0}
                    )
                    held_text = ""
                continue

            if parsed is None:
                logger.warning(
                    "Tool-call parsing failed "
                    f"(generated_chars={len(combined)})"
                )
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
            accumulated_calls.extend(parsed)
            if response.finish_reason is None:
                # More of the message may follow, including a second call.
                continue
            if held_text:
                # Trailing text with no further chunk coming to carry it.
                yield response.model_copy(
                    update={"text": held_text, "token": 0, "finish_reason": None}
                )
                held_text = ""
            yield ToolCallResponse(
                tool_calls=accumulated_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_calls = []
            continue

        if response.finish_reason is not None:
            # Generation ended while inside a tool-call block. That is not
            # always truncation: several families close a call by ending the
            # message rather than by emitting a closing marker. Llama 3.1+ is
            # the clearest case, where <|eom_id|> means "end of message,
            # handing off to a tool", so the block is complete and the closing
            # marker never arrives. Try to parse before declaring it garbage;
            # only a block that genuinely does not parse falls through to the
            # error path, which is what truncation actually looks like.
            combined = "".join(tool_call_text_parts)
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            if parsed is not None and not declared_tool_calls(parsed, tools):
                logger.info(
                    "Block named no offered tool, emitting it as content "
                    f"(parsed_calls={len(parsed)})"
                )
                yield response.model_copy(update={"text": combined, "token": 0})
                break
            if parsed:
                parsed = declared_tool_calls(parsed, tools)
                logger.info(
                    "Parsed tool-call block closed by end of generation "
                    f"(generated_chars={len(combined)}, parsed_calls={len(parsed)})"
                )
                if trace_task_id is not None:
                    record_trace_marker(
                        "tool_call_parsed",
                        trace_rank,
                        category="tooling",
                        task_id=trace_task_id,
                        tags=["tool_call"],
                        attrs={"tool_call_count": len(parsed)},
                    )
                accumulated_calls.extend(parsed)
                yield ToolCallResponse(
                    tool_calls=accumulated_calls,
                    usage=response.usage,
                    stats=response.stats,
                )
                accumulated_calls = []
                break
            if tool_parser.unparsed_is_text:
                logger.info(
                    "Unmarked block ended without parsing as a tool call, "
                    f"emitting it as content (generated_chars={len(combined)})"
                )
                yield response.model_copy(update={"text": combined, "token": 0})
                break
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={
                    "text": combined,
                    "token": 0,
                    "finish_reason": "error",
                }
            )
            yield response

    if accumulated_calls and last_response is not None:
        # A finite source can end without ever carrying a finish reason, so the
        # calls held for coalescing are released here rather than dropped.
        yield ToolCallResponse(
            tool_calls=accumulated_calls,
            usage=last_response.usage,
            stats=last_response.stats,
        )
