"""OpenAI Chat Completions API adapter for converting requests/responses."""

import base64
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger

from skulk.api.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
    StreamingChoiceResponse,
    ToolCall,
    Usage,
)
from skulk.download.download_utils import create_http_session
from skulk.shared.constants import (
    CONTEXT_LENGTH_EXCEEDED_PREFIX,
    preferred_env_value,
)
from skulk.shared.models.capabilities import resolve_model_capability_profile
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from skulk.shared.types.common import CommandId
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
    resolve_reasoning_params,
)


def _thinking_stream_debug_enabled() -> bool:
    """Return whether opt-in thinking stream tracing is enabled."""
    value = preferred_env_value(
        "SKULK_TRACE_THINKING_STREAM",
    )
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def error_chunk_response(error_message: str | None) -> ErrorResponse:
    """Map a runner ``ErrorChunk`` message to an OpenAI-style error envelope.

    Context-admission rejections (#145) arrive as plain strings carrying the
    ``CONTEXT_LENGTH_EXCEEDED_PREFIX`` sentinel (the chunk types are shared
    across mixed-version clusters, so a structured field is not wire-safe);
    they map to an ``invalid_request_error`` with code 400. Everything else
    keeps the historical 500 internal-error shape.
    """
    message = error_message or "Internal server error"
    if message.startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX):
        return ErrorResponse(
            error=ErrorInfo(
                message=message,
                type="invalid_request_error",
                code=400,
            )
        )
    return ErrorResponse(
        error=ErrorInfo(
            message=message,
            type="InternalServerError",
            code=500,
        )
    )


def extract_base64_from_data_url(data_url: str) -> str:
    match = re.match(r"data:[^;]+;base64,(.+)", data_url)
    if match:
        return match.group(1)
    return data_url


async def fetch_image_url(url: str) -> str:
    headers = {"User-Agent": "skulk/1.0"}
    async with (
        create_http_session(timeout_profile="short") as session,
        session.get(url, headers=headers) as resp,
    ):
        resp.raise_for_status()
        data = await resp.read()
        return base64.b64encode(data).decode("ascii")


async def chat_request_to_text_generation(
    request: ChatCompletionRequest,
    *,
    model_card: ModelCard | None = None,
) -> TextGenerationTaskParams:
    instructions: str | None = None
    input_messages: list[InputMessage] = []
    chat_template_messages: list[dict[str, Any]] = []
    images: list[str] = []

    for msg in request.messages:
        # Normalize content to string
        content: str
        has_images = False
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, ChatCompletionMessageText):
            content = msg.content.text
        elif isinstance(msg.content, ChatCompletionMessageImageUrl):
            url = msg.content.image_url.get("url", "")
            if url:
                if url.startswith(("http://", "https://")):
                    images.append(await fetch_image_url(url))
                else:
                    images.append(extract_base64_from_data_url(url))
                has_images = True
            content = ""
        else:
            text_parts: list[str] = []
            for part in msg.content:
                if isinstance(part, ChatCompletionMessageText):
                    text_parts.append(part.text)
                else:
                    url = part.image_url.get("url", "")
                    if url:
                        if url.startswith(("http://", "https://")):
                            images.append(await fetch_image_url(url))
                        else:
                            images.append(extract_base64_from_data_url(url))
                        has_images = True
            content = "\n".join(text_parts)

        # Extract system message as instructions
        if msg.role == "system":
            if instructions is None:
                instructions = content
            else:
                # Append additional system messages
                instructions = f"{instructions}\n{content}"
            chat_template_messages.append({"role": "system", "content": content})
        else:
            # Skip messages with no meaningful content
            if (
                msg.content is None
                and msg.reasoning_content is None
                and msg.tool_calls is None
            ):
                continue

            if msg.role in ("user", "assistant", "developer"):
                input_messages.append(InputMessage(role=msg.role, content=content))

            # Build full message dict for chat template (preserves tool_calls etc.)
            # Normalize content for model_dump
            if has_images:
                multimodal_content: list[dict[str, Any]] = []
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, ChatCompletionMessageText):
                            multimodal_content.append(
                                {"type": "text", "text": part.text}
                            )
                        else:
                            multimodal_content.append({"type": "image"})
                else:
                    if content:
                        multimodal_content.append({"type": "text", "text": content})
                    multimodal_content.append({"type": "image"})
                chat_template_messages.append(
                    {"role": msg.role, "content": multimodal_content}
                )
                continue
            msg_copy = msg.model_copy(update={"content": content})

            dumped: dict[str, Any] = msg_copy.model_dump(exclude_none=True)
            chat_template_messages.append(dumped)

    capability_profile = resolve_model_capability_profile(
        request.model,
        model_card=model_card,
    )
    resolved_effort, resolved_thinking = resolve_reasoning_params(
        request.reasoning_effort,
        request.enable_thinking,
        capability_profile,
    )

    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages
        if input_messages
        else [InputMessage(role="user", content="")],
        instructions=instructions,
        max_output_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        stream=request.stream,
        tools=request.tools,
        reasoning_effort=resolved_effort,
        enable_thinking=resolved_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
        # `top_logprobs` set alone implies a logprobs request (OpenAI semantics).
        # Normalize it here, at the boundary, so every engine sees a consistent
        # `logprobs` flag rather than each runner re-deriving it (#385).
        logprobs=bool(request.logprobs) or request.top_logprobs is not None,
        top_logprobs=request.top_logprobs,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        repetition_context_size=request.repetition_context_size,
        images=images,
    )


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a TokenChunk to a streaming ChatCompletionResponse."""
    # Build logprobs if available
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    if chunk.is_thinking:
        delta = ChatCompletionMessage(role="assistant", reasoning_content=chunk.text)
    else:
        delta = ChatCompletionMessage(role="assistant", content=chunk.text)

    if _thinking_stream_debug_enabled():
        logger.info(
            "[thinking-stream] stage=chat-completions "
            f"model={chunk.model} text={chunk.text!r} is_thinking={chunk.is_thinking} "
            f"mapped_field={'reasoning_content' if chunk.is_thinking else 'content'} "
            f"finish_reason={chunk.finish_reason!r}"
        )

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=delta,
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from chunks."""
    last_usage: Usage | None = None

    # Emit the command id immediately so first-party clients can cancel during
    # long prefill/model-load phases before the first visible token arrives.
    yield f": command_id {command_id}\n\n"

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                # Use SSE comment so third-party clients ignore it
                yield f": prefill_progress {chunk.model_dump_json()}\n\n"

            case ErrorChunk():
                error_response = error_chunk_response(chunk.error_message)
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case ToolCallChunk():
                last_usage = chunk.usage or last_usage

                tool_call_deltas = [
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                ]
                tool_response = ChatCompletionResponse(
                    id=command_id,
                    created=int(time.time()),
                    model=chunk.model,
                    choices=[
                        StreamingChoiceResponse(
                            index=0,
                            delta=ChatCompletionMessage(
                                role="assistant",
                                tool_calls=tool_call_deltas,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=last_usage,
                )
                yield f"data: {tool_response.model_dump_json()}\n\n"
                if chunk.stats is not None:
                    yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case TokenChunk():
                last_usage = chunk.usage or last_usage

                chunk_response = chunk_to_response(chunk, command_id)
                if chunk.finish_reason is not None:
                    chunk_response = chunk_response.model_copy(
                        update={"usage": last_usage}
                    )
                yield f"data: {chunk_response.model_dump_json()}\n\n"

                if chunk.finish_reason is not None:
                    if chunk.stats is not None:
                        yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    return


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    logprobs_content: list[LogprobsContentItem] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                error_message = chunk.error_message or "Internal server error"
                break

            case TokenChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                else:
                    text_parts.append(chunk.text)
                if chunk.logprob is not None:
                    logprobs_content.append(
                        LogprobsContentItem(
                            token=chunk.text,
                            logprob=chunk.logprob,
                            top_logprobs=chunk.top_logprobs or [],
                        )
                    )
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            case ToolCallChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                tool_calls.extend(
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                )
                finish_reason = chunk.finish_reason

    if error_message is not None:
        # The HTTP status is already committed (the body itself streams), so a
        # runner-side error surfaces as a structured OpenAI error envelope in the
        # body rather than a broken stream or a bogus empty success.
        # error_chunk_response maps the context sentinel to a 400 and everything
        # else to the historical 500 internal-error shape.
        yield error_chunk_response(error_message).model_dump_json()
        return

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts) if thinking_parts else None
    assert model is not None

    yield ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=combined_text,
                    reasoning_content=combined_thinking,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                logprobs=Logprobs(content=logprobs_content)
                if logprobs_content
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=last_usage,
    ).model_dump_json()
    return
