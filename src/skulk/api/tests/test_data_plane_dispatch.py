"""API data-plane demux coverage (#279 Phase 2).

The API routes each output chunk to the per-command stream queue by type. This
exercises `_dispatch_generation_chunk` directly (the shared routing used by the
DATA-plane consumer) without standing up the full gossip path.
"""

import pytest

from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import (
    EmbeddingChunk,
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel


def _build_api() -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


@pytest.mark.asyncio
async def test_dispatch_routes_token_chunk_to_text_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-text")
    send, recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="hi",
        token_id=1,
        usage=None,
        finish_reason=None,
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        assert stream.receive_nowait() is chunk


@pytest.mark.asyncio
async def test_dispatch_routes_embedding_chunk_to_embedding_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-embed")
    send, recv = channel[EmbeddingChunk | ErrorChunk]()
    api._embedding_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = EmbeddingChunk(
        model=ModelId("mlx-community/embed"),
        embeddings=[[0.1, 0.2]],
        token_count=2,
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        assert stream.receive_nowait() is chunk


@pytest.mark.asyncio
async def test_dispatch_for_unknown_command_is_a_noop() -> None:
    # A chunk for a command with no registered queue (client already gone) must
    # not raise — the data loop has to survive late/orphan chunks.
    api = _build_api()
    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="x",
        token_id=2,
        usage=None,
        finish_reason="stop",
    )
    await api._dispatch_generation_chunk(  # pyright: ignore[reportPrivateUsage]
        CommandId("nobody-home"), chunk
    )
