"""Tests for streaming chat-completions adapter behavior."""

from typing import cast

import pytest

from skulk.api.adapters.chat_completions import generate_chat_stream
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.common import CommandId, ModelId


async def _single_token_stream():
    yield TokenChunk(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        text="hello",
        token_id=1,
        usage=None,
        finish_reason="stop",
    )


@pytest.mark.anyio
async def test_generate_chat_stream_emits_command_id_before_tokens() -> None:
    """Streaming responses should expose the command id before prefill completes."""
    chunks = [chunk async for chunk in generate_chat_stream(CommandId("cmd-123"), _single_token_stream())]

    assert chunks[0] == ": command_id cmd-123\n\n"
    assert 'data: {"id":"cmd-123"' in chunks[1]
    assert chunks[-1] == "data: [DONE]\n\n"


def _stats(prompt_tokens: int, generation_tokens: int):
    from skulk.api.types import GenerationStats
    from skulk.shared.types.memory import Memory

    return GenerationStats(
        prompt_tps=10.0,
        generation_tps=20.0,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        peak_memory_usage=Memory.from_bytes(0),
    )


async def _final_token_stream_with_stats():
    yield TokenChunk(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        text="hello",
        token_id=1,
        usage=None,
        finish_reason="stop",
        stats=_stats(43, 30),
    )


@pytest.mark.anyio
async def test_streaming_final_chunk_synthesizes_usage_from_stats() -> None:
    # Runners report token accounting via GenerationStats, not per-chunk usage
    # envelopes; the OpenAI-standard usage object must still be populated on
    # the terminal chunk (#644).
    chunks = [
        chunk
        async for chunk in generate_chat_stream(
            CommandId("cmd-644"), _final_token_stream_with_stats()
        )
    ]
    final_data = [c for c in chunks if c.startswith("data: {")][-1]
    assert '"usage":' in final_data
    assert '"prompt_tokens":43' in final_data
    assert '"completion_tokens":30' in final_data
    assert '"total_tokens":73' in final_data


@pytest.mark.anyio
async def test_collect_response_synthesizes_usage_from_stats() -> None:
    from skulk.api.adapters.chat_completions import collect_chat_response

    parts = [
        part
        async for part in collect_chat_response(
            CommandId("cmd-644"), _final_token_stream_with_stats()
        )
    ]
    body = "".join(parts)
    assert '"usage":' in body
    assert '"prompt_tokens":43' in body
    assert '"completion_tokens":30' in body
    assert '"total_tokens":73' in body


def test_usage_from_stats_unmeasured_and_none() -> None:
    from skulk.api.adapters.chat_completions import usage_from_stats

    assert usage_from_stats(None) is None
    # A zero PROMPT means unmeasured (runner fallbacks produce prompt 0 with
    # real decode counts); a fabricated zero would mislead cost accounting,
    # so those stay null. A zero completion with a measured prompt is a real
    # immediately-stopped outcome and keeps usage (#644, PR #645 review).
    assert usage_from_stats(_stats(0, 0)) is None
    assert usage_from_stats(_stats(0, 30)) is None
    empty_completion = usage_from_stats(_stats(30, 0))
    assert empty_completion is not None
    assert empty_completion.completion_tokens == 0
    assert empty_completion.total_tokens == 30
    usage = usage_from_stats(_stats(5, 7))
    assert usage is not None and usage.total_tokens == 12


async def _tool_call_stream_with_stats():
    from skulk.api.types import ToolCallItem
    from skulk.shared.types.chunks import ToolCallChunk

    yield ToolCallChunk(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        tool_calls=[ToolCallItem(name="get_weather", arguments='{"city":"Oslo"}')],
        usage=None,
        stats=_stats(43, 30),
    )


@pytest.mark.anyio
async def test_streaming_tool_call_terminal_synthesizes_usage_from_stats() -> None:
    # The tool-calls finish path is a distinct terminal from the token path
    # and must carry the synthesized usage too (#644, PR #645 review).
    chunks = [
        chunk
        async for chunk in generate_chat_stream(
            CommandId("cmd-644-tools"), _tool_call_stream_with_stats()
        )
    ]
    final_data = [c for c in chunks if c.startswith("data: {")][-1]
    assert '"finish_reason":"tool_calls"' in final_data
    assert '"usage":' in final_data
    assert '"prompt_tokens":43' in final_data
    assert '"completion_tokens":30' in final_data
    assert '"total_tokens":73' in final_data


async def _tool_call_stream():
    from skulk.api.types import ToolCallItem
    from skulk.shared.types.chunks import ToolCallChunk

    yield ToolCallChunk(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        tool_calls=[ToolCallItem(id="call_1", name="get_weather", arguments="{}")],
        usage=None,
        stats=None,
    )


def _sse_payloads(chunks: list[str]) -> list[dict[str, object]]:
    """Parse the JSON payload out of every non-sentinel SSE data line."""
    import json

    payloads: list[dict[str, object]] = []
    for chunk in chunks:
        if not chunk.startswith("data: ") or "[DONE]" in chunk:
            continue
        parsed = cast("object", json.loads(chunk.removeprefix("data: ").strip()))
        assert isinstance(parsed, dict)
        payloads.append(cast("dict[str, object]", parsed))
    return payloads


def _first_choice(payload: dict[str, object]) -> dict[str, object]:
    """Return the first choice of a completion payload."""
    choices = payload["choices"]
    assert isinstance(choices, list)
    first = cast("object", choices[0])
    assert isinstance(first, dict)
    return cast("dict[str, object]", first)


@pytest.mark.anyio
async def test_streaming_chunks_use_the_chunk_object_discriminator() -> None:
    """Streaming frames must not carry the non-streaming ``object`` value.

    OpenAI's streaming format requires ``chat.completion.chunk``. Lenient
    clients read ``choices[0].delta`` and never notice a wrong discriminator,
    but strict ones reject the stream outright, so this is pinned rather than
    left to a response model shared with the non-streaming path.
    """
    chunks = [
        chunk
        async for chunk in generate_chat_stream(
            CommandId("cmd-123"), _single_token_stream()
        )
    ]
    payloads = _sse_payloads(chunks)

    assert payloads, "expected at least one streamed frame"
    for payload in payloads:
        assert payload["object"] == "chat.completion.chunk"
        choice = _first_choice(payload)
        assert "delta" in choice
        assert "message" not in choice


@pytest.mark.anyio
async def test_streaming_tool_call_frames_use_the_chunk_discriminator() -> None:
    """The tool-call frame takes a separate construction path from token frames."""
    chunks = [
        chunk
        async for chunk in generate_chat_stream(
            CommandId("cmd-tool"), _tool_call_stream()
        )
    ]
    payloads = _sse_payloads(chunks)

    assert payloads, "expected a tool-call frame"
    for payload in payloads:
        assert payload["object"] == "chat.completion.chunk"


@pytest.mark.anyio
async def test_non_streaming_response_keeps_the_completion_discriminator() -> None:
    """The collected response is not a chunk and must keep ``chat.completion``."""
    import json

    from skulk.api.adapters.chat_completions import collect_chat_response

    parts = [
        part
        async for part in collect_chat_response(
            CommandId("cmd-456"), _single_token_stream()
        )
    ]
    parsed = cast("object", json.loads("".join(parts)))
    assert isinstance(parsed, dict)
    payload = cast("dict[str, object]", parsed)

    assert payload["object"] == "chat.completion"
    assert "message" in _first_choice(payload)


async def _empty_stream():
    """A producer that ends without a token, a tool call or an error.

    This is what a cancelled or timed-out task looks like from the adapter's
    side: the chunk stream simply finishes having produced nothing.
    """
    return
    yield  # pragma: no cover - unreachable, makes this an async generator


@pytest.mark.anyio
async def test_collected_response_never_returns_an_empty_body() -> None:
    """A task that produced nothing must still yield a readable body.

    This previously asserted, and because the status is committed before the
    body streams, the assertion reached callers as HTTP 200 with zero bytes.
    Every OpenAI client treats 2xx as success and then fails parsing, so the
    real failure surfaced far from its cause.
    """
    import json

    from skulk.api.adapters.chat_completions import collect_chat_response

    body = "".join(
        [part async for part in collect_chat_response(CommandId("cmd-empty"), _empty_stream())]
    )

    assert body.strip(), "a committed response must never have an empty body"
    parsed = cast("object", json.loads(body))
    assert isinstance(parsed, dict)
    payload = cast("dict[str, object]", parsed)
    error = cast("dict[str, object]", payload["error"])
    message = error["message"]
    assert isinstance(message, str)
    assert message


@pytest.mark.anyio
async def test_stream_terminates_even_when_the_producer_stops_early() -> None:
    """A stream that ends without a finish reason must still send [DONE].

    Without a terminator a client cannot tell a finished turn from a dropped
    connection, so it either hangs or reports a transport error instead of the
    server's actual problem.
    """
    chunks = [
        chunk
        async for chunk in generate_chat_stream(CommandId("cmd-empty"), _empty_stream())
    ]

    assert chunks[-1] == "data: [DONE]\n\n"
    payloads = _sse_payloads(chunks)
    assert payloads, "expected an explanatory frame before the terminator"
    error = cast("dict[str, object]", payloads[-1]["error"])
    message = error["message"]
    assert isinstance(message, str)
    assert message


async def _truncated_stream():
    """A producer that emits text and then stops without a finish reason.

    This is what cancellation mid-generation looks like: real output arrived,
    but nothing ever marked the turn as ended.
    """
    yield TokenChunk(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        text="partial answer",
        token_id=1,
        usage=None,
        finish_reason=None,
    )


@pytest.mark.anyio
async def test_truncated_generation_is_reported_rather_than_returned() -> None:
    """Partial output with no finish reason must not look like a completion.

    Returning it as a normal `chat.completion` hands the caller a silently
    truncated answer, which is worse than an empty body because nothing marks
    it incomplete. A finish reason is the producer's only end-of-turn signal;
    the streaming path already refuses to send `[DONE]` without one.
    """
    import json

    from skulk.api.adapters.chat_completions import collect_chat_response

    body = "".join(
        [
            part
            async for part in collect_chat_response(
                CommandId("cmd-trunc"), _truncated_stream()
            )
        ]
    )
    parsed = cast("object", json.loads(body))
    assert isinstance(parsed, dict)
    payload = cast("dict[str, object]", parsed)

    assert "choices" not in payload, "a truncated turn must not look successful"
    error = cast("dict[str, object]", payload["error"])
    message = error["message"]
    assert isinstance(message, str)
    assert "finish reason" in message


@pytest.mark.anyio
async def test_non_streaming_failure_carries_a_real_status_code() -> None:
    """A non-streaming failure answers with a status, not a 200 plus an error.

    Nothing forces a non-streaming request to commit its status before the
    outcome is known: the body is one complete object produced once. Every
    OpenAI client branches on status first, so reporting a failure only in the
    body means the client treats it as success and discovers the problem later
    through missing ``choices``.
    """
    from skulk.api.adapters.chat_completions import collect_chat_response_body

    status, body = await collect_chat_response_body(
        CommandId("cmd-empty"), _empty_stream()
    )

    assert status >= 400, "a failed generation must not answer 2xx"
    assert body.strip()


@pytest.mark.anyio
async def test_non_streaming_success_still_answers_200() -> None:
    """The status derivation must not turn ordinary completions into errors."""
    from skulk.api.adapters.chat_completions import collect_chat_response_body

    status, body = await collect_chat_response_body(
        CommandId("cmd-ok"), _single_token_stream()
    )

    assert status == 200
    assert '"chat.completion"' in body
