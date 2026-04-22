"""Tests for streaming chat-completions adapter behavior."""

import pytest

from exo.api.adapters.chat_completions import generate_chat_stream
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId


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
