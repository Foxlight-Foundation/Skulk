"""API data-plane demux coverage (#279 Phase 2).

The API routes each output chunk to the per-command stream queue by type. This
exercises `_dispatch_generation_chunk` directly (the shared routing used by the
DATA-plane consumer) without standing up the full gossip path.
"""

from unittest.mock import AsyncMock

import anyio
import pytest

import skulk.api.main as api_main
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
async def test_dispatch_routes_error_chunk_to_image_queue() -> None:
    # Regression (#297 review): a runner failure on an ImageGeneration command
    # emits an ErrorChunk; it must reach the image client, not crash the data
    # loop on a too-narrow `assert isinstance(chunk, ImageChunk)`.
    from skulk.shared.types.chunks import ImageChunk

    api = _build_api()
    cmd = CommandId("cmd-img")
    send, recv = channel[ImageChunk | ErrorChunk]()
    api._image_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    err = ErrorChunk(
        model=ModelId("mlx-community/image"),
        error_message="runner shutdown before completing command",
    )
    await api._dispatch_generation_chunk(cmd, err)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        got = stream.receive_nowait()
        assert isinstance(got, ErrorChunk)
        assert got is err


@pytest.mark.asyncio
async def test_dispatch_survives_closed_sender() -> None:
    # Regression (#297 review): cancel_command() closes the queue's sender; a
    # late chunk then raises ClosedResourceError on send. The dispatcher must
    # swallow it (and drop the queue), not let it crash the whole data loop.
    api = _build_api()
    cmd = CommandId("cmd-closed")
    send, _recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]
    send.close()  # simulate cancel_command() closing the sender

    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="late",
        token_id=9,
        usage=None,
        finish_reason=None,
    )
    # must not raise
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    assert cmd not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


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


@pytest.mark.asyncio
async def test_token_stream_stall_after_first_chunk_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #279 Phase 2b backstop: once streaming has started, a dropped chunk would
    # leave _token_chunk_stream blocked on receive() forever. Feed one chunk
    # (started=True), then go silent: the idle timeout must fire, send a
    # TaskCancelled (not TaskFinished — don't orphan the runner), yield a terminal
    # ErrorChunk, and end the stream. fail_after is the anti-hang guard.
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.15)
    api = _build_api()
    finish_send = AsyncMock()
    cancel_send = AsyncMock()
    api._send = finish_send  # pyright: ignore[reportPrivateUsage]  # finalize: suppress real channel send
    api.command_sender = cancel_send  # stall path sends TaskCancelled here
    cmd = CommandId("cmd-stall")

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            # wait for the generator to register its per-command queue, then feed
            # exactly one chunk so the stream is "started", then go silent.
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            await api._text_generation_queues[cmd].send(  # pyright: ignore[reportPrivateUsage]
                TokenChunk(
                    model=ModelId("mlx-community/test"),
                    text="hi",
                    token_id=1,
                    usage=None,
                    finish_reason=None,
                )
            )

    # first the delivered token, then the synthetic terminal error
    assert len(chunks) == 2
    assert isinstance(chunks[0], TokenChunk)
    assert isinstance(chunks[1], ErrorChunk)
    assert "stall" in chunks[1].error_message.lower()
    # stall is treated as a cancellation (TaskCancelled sent on the command
    # channel), NOT a clean finish (finalize's TaskFinished suppressed because the
    # command was marked cancelled) — so the runner isn't orphaned.
    cancel_send.send.assert_awaited()  # pyright: ignore[reportAny]
    finish_send.assert_not_awaited()
    assert cmd not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_token_stream_stall_with_terminal_task_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #298 review: a mid-stream stall whose task has ALREADY reached a terminal
    # status is a dropped FINAL data-plane chunk, not a stuck runner. Sending
    # TaskCancelled there is a no-op on a completed runner and leaks the master's
    # task/command mapping forever — so the stall must clean up via the normal
    # TaskFinished path (no TaskCancelled, command not marked cancelled).
    from skulk.shared.types.state import State
    from skulk.shared.types.tasks import (
        TaskId,
        TaskStatus,
        TextGeneration,
    )
    from skulk.shared.types.text_generation import (
        InputMessage,
        TextGenerationTaskParams,
    )
    from skulk.shared.types.worker.instances import InstanceId

    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.15)
    api = _build_api()
    finish_send = AsyncMock()
    cancel_send = AsyncMock()
    api._send = finish_send  # pyright: ignore[reportPrivateUsage]
    api.command_sender = cancel_send
    cmd = CommandId("cmd-stall-done")

    # The master already considers this command's task Complete.
    task = TextGeneration(
        task_id=TaskId(),
        task_status=TaskStatus.Complete,
        instance_id=InstanceId(),
        command_id=cmd,
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test"),
            input=[InputMessage(role="user", content="hi")],
        ),
    )
    api.state = State(tasks={task.task_id: task})

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            await api._text_generation_queues[cmd].send(  # pyright: ignore[reportPrivateUsage]
                TokenChunk(
                    model=ModelId("mlx-community/test"),
                    text="hi",
                    token_id=1,
                    usage=None,
                    finish_reason=None,
                )
            )

    assert len(chunks) == 2
    assert isinstance(chunks[0], TokenChunk)
    assert isinstance(chunks[1], ErrorChunk)
    # terminal task -> no TaskCancelled, command NOT marked cancelled, so
    # _finalize_command_stream sends the normal TaskFinished (master cleans up).
    cancel_send.send.assert_not_awaited()  # pyright: ignore[reportAny]
    finish_send.assert_awaited()
    assert cmd not in api._cancelled_command_ids  # pyright: ignore[reportPrivateUsage]
    assert cmd not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_token_stream_does_not_timeout_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A request queued behind a long decode (no chunk yet) must NOT be timed out
    # (Codex P1): with a tiny idle timeout and no chunk delivered, the stream
    # stays open until we close the producer (EndOfStream) — no spurious error.
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.05)
    api = _build_api()
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    cmd = CommandId("cmd-queued")

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            # stay silent well past the idle timeout; must NOT yield an error
            await anyio.sleep(0.3)
            assert chunks == []  # no spurious stall before the first chunk
            # close the producer to end the stream cleanly
            api._text_generation_queues[cmd].close()  # pyright: ignore[reportPrivateUsage]

    assert chunks == []  # EndOfStream -> clean return, no error chunk
    assert cmd not in api._cancelled_command_ids  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_prefill_progress_does_not_arm_idle_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #298 review: a PrefillProgressChunk is not real output. Receiving one must
    # NOT arm the idle timer (prefill of a huge prompt can outlast the inter-token
    # bound between progress updates). Send one progress chunk, then stay silent
    # well past a tiny idle timeout: no stall error must be emitted.
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.05)
    api = _build_api()
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    cmd = CommandId("cmd-prefill")

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            await api._text_generation_queues[cmd].send(  # pyright: ignore[reportPrivateUsage]
                PrefillProgressChunk(
                    model=ModelId("mlx-community/test"),
                    processed_tokens=10,
                    total_tokens=1000,
                )
            )
            await anyio.sleep(0.3)  # >> idle timeout, but timer must stay disarmed
            assert all(isinstance(c, PrefillProgressChunk) for c in chunks)
            api._text_generation_queues[cmd].close()  # pyright: ignore[reportPrivateUsage]

    # only the progress chunk(s); no synthetic stall ErrorChunk
    assert all(isinstance(c, PrefillProgressChunk) for c in chunks)
    assert cmd not in api._cancelled_command_ids  # pyright: ignore[reportPrivateUsage]
