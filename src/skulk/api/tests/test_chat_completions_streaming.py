"""Tests for streaming chat-completions adapter behavior."""

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
    # A zero on EITHER side means that side was unmeasured (runner fallbacks
    # produce prompt 0 with real decode counts); a fabricated zero would
    # mislead cost accounting, so partial measurements stay null (#644,
    # PR #645 review).
    assert usage_from_stats(_stats(0, 0)) is None
    assert usage_from_stats(_stats(0, 30)) is None
    assert usage_from_stats(_stats(30, 0)) is None
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
