import gc
import tempfile
import weakref
from collections.abc import AsyncGenerator
from pathlib import Path

# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from skulk.api.data_plane import DataPlaneObserver
from skulk.api.main import API
from skulk.api.music_jobs import MusicJobRegistry
from skulk.api.video_jobs import VideoJobRegistry
from skulk.api.video_store import VideoStore
from skulk.shared.types.common import CommandId
from skulk.utils.channels import Sender


def _ample_free_bytes(_store: VideoStore) -> int:
    return 1 << 40


@pytest.fixture(autouse=True)
def ample_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the store's free-space check off the host's real disk.

    The store leaves a reserve on its filesystem; a test must not depend on
    how full the machine running it happens to be.
    """

    monkeypatch.setattr(VideoStore, "_free_disk_bytes", _ample_free_bytes)


def _make_api() -> Any:
    """Create a minimal API instance with cancel route and error handler."""

    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._video_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._video_jobs = VideoJobRegistry(None)  # pyright: ignore[reportPrivateUsage]
    api._music_jobs = MusicJobRegistry(  # pyright: ignore[reportPrivateUsage]
        Path(tempfile.mkdtemp()) / "jobs.json"
    )
    api._video_store = VideoStore(Path(tempfile.mkdtemp()))  # pyright: ignore[reportPrivateUsage]
    api._video_job_media_deadlines = {}  # pyright: ignore[reportPrivateUsage]
    api._video_output_sources = {}  # pyright: ignore[reportPrivateUsage]
    api._early_output_packets = {}  # pyright: ignore[reportPrivateUsage]
    api._early_output_packet_bytes = 0  # pyright: ignore[reportPrivateUsage]
    api._pending_output_completions = {}  # pyright: ignore[reportPrivateUsage]
    api._video_upload_inflight_bytes = 0  # pyright: ignore[reportPrivateUsage]
    api._embedding_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._audio_speech_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._audio_transcription_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._realtime_audio_transcription_commands = set()  # pyright: ignore[reportPrivateUsage]
    api._speech_media_commands = set()  # pyright: ignore[reportPrivateUsage]
    api._speech_media_targets = {}  # pyright: ignore[reportPrivateUsage]
    api._transcription_media_targets = {}  # pyright: ignore[reportPrivateUsage]
    api._pending_speech_media = {}  # pyright: ignore[reportPrivateUsage]
    api._pending_speech_media_bytes = 0  # pyright: ignore[reportPrivateUsage]
    api._speech_media_packet_sender = None  # pyright: ignore[reportPrivateUsage]
    api._pending_vision_media = {}  # pyright: ignore[reportPrivateUsage]
    api._pending_vision_media_bytes = 0  # pyright: ignore[reportPrivateUsage]
    api._active_vision_media_bytes = {}  # pyright: ignore[reportPrivateUsage]
    api._active_vision_media_total_bytes = 0  # pyright: ignore[reportPrivateUsage]
    api._vision_media_commands = set()  # pyright: ignore[reportPrivateUsage]
    api._vision_media_targets = {}  # pyright: ignore[reportPrivateUsage]
    api._vision_media_pending_acks = {}  # pyright: ignore[reportPrivateUsage]
    api._vision_media_ack_deadlines = {}  # pyright: ignore[reportPrivateUsage]
    api._vision_media_models = {}  # pyright: ignore[reportPrivateUsage]
    api._vision_media_failures = {}  # pyright: ignore[reportPrivateUsage]
    api._vision_media_packet_sender = None  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api._steward_turns = weakref.WeakValueDictionary()  # pyright: ignore[reportPrivateUsage]
    api._chunk_reorder = {}  # pyright: ignore[reportPrivateUsage]
    api._data_dedup_cursor = {}  # pyright: ignore[reportPrivateUsage]
    api._data_plane_observer = DataPlaneObserver(  # pyright: ignore[reportPrivateUsage]
        transport="disabled",
        reorder_buffer_enabled=True,
    )
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/cancel/{command_id}")(api.cancel_command)
    return api


def test_cancel_nonexistent_command_returns_404() -> None:
    """Cancel for an unknown command_id returns 404 in OpenAI error format."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.post("/v1/cancel/nonexistent-id")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Command not found or already completed"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_cancel_active_text_generation() -> None:
    """Cancel an active text generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-123")
    sender = MagicMock()
    api._text_generation_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    assert cid in api._cancelled_command_ids
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_active_image_generation() -> None:
    """Cancel an active image generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("img-cmd-456")
    sender = MagicMock()
    api._image_generation_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    assert cid in api._cancelled_command_ids
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_active_audio_speech() -> None:
    """Cancel an active speech synthesis command: returns 200 and closes sender."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("speech-cmd-789")
    sender = MagicMock()
    api._audio_speech_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    assert cid in api._cancelled_command_ids
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_active_audio_transcription() -> None:
    """Cancel an active speech transcription command."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("transcription-cmd-789")
    sender = MagicMock()
    api._audio_transcription_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    assert cid in api._cancelled_command_ids
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


@pytest.mark.asyncio
async def test_finalize_command_stream_suppresses_task_finished_for_cancelled_command() -> None:
    """Local cancellation should skip TaskFinished so workers can observe Cancelled."""

    api = _make_api()
    cid = CommandId("cancelled-cmd")
    sender = MagicMock()
    queue: dict[CommandId, Sender[object]] = {
        cid: cast(Sender[object], cast(object, sender))
    }
    api._cancelled_command_ids.add(cid)

    await api._finalize_command_stream(cid, queue)

    api._send.assert_not_called()
    assert cid not in queue
    assert cid not in api._cancelled_command_ids


@pytest.mark.asyncio
async def test_finalize_command_stream_reports_natural_completion() -> None:
    """Natural completion should still emit TaskFinished."""

    api = _make_api()
    cid = CommandId("finished-cmd")
    sender = MagicMock()
    queue: dict[CommandId, Sender[object]] = {
        cid: cast(Sender[object], cast(object, sender))
    }

    await api._finalize_command_stream(cid, queue)

    api._send.assert_called_once()
    task_finished = api._send.call_args[0][0]
    assert task_finished.finished_command_id == cid
    assert cid not in queue


class _RecordingTurn:
    """Stands in for a live steward turn's harness in the turn registry."""

    def __init__(self) -> None:
        self.cancelled = 0

    async def cancel_turn(self) -> None:
        self.cancelled += 1


def test_cancel_routes_a_registered_steward_turn() -> None:
    """The id a steward turn advertises cancels the turn instead of a 404."""
    api = _make_api()
    client = TestClient(api.app)
    outer = CommandId("steward-turn-1")
    turn = _RecordingTurn()
    api._steward_turns[outer] = turn

    response = client.post(f"/v1/cancel/{outer}")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Steward turn cancelled.",
        "command_id": str(outer),
    }
    assert turn.cancelled == 1
    # The turn cancels its own inner generations; the outer id itself is
    # never a worker task.
    api._send.assert_not_called()


@pytest.mark.asyncio
async def test_release_wrapper_removes_the_turn_when_the_response_ends() -> None:
    """A finished response leaves nothing for a later cancel to find."""
    api = _make_api()
    outer = CommandId("steward-turn-2")
    turn = _RecordingTurn()
    api._steward_turns[outer] = turn

    async def _body() -> AsyncGenerator[str, None]:
        yield "data: one\n\n"
        yield "data: [DONE]\n\n"

    items = [item async for item in api._release_steward_turn_after(outer, _body())]

    assert items == ["data: one\n\n", "data: [DONE]\n\n"]
    assert outer not in api._steward_turns
    with pytest.raises(HTTPException) as raised:
        await api.cancel_command(outer)
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_release_wrapper_removes_the_turn_on_early_disconnect() -> None:
    """The wrapper cleans up even when every inner generator never started.

    The keepalive layer emits a byte before it pulls its source, so a client
    can leave while the inner generators are still unstarted, and an
    unstarted generator never runs its ``finally``. The outermost wrapper is
    the iterator Starlette drives, so its cleanup always runs.
    """
    api = _make_api()
    outer = CommandId("steward-turn-3")
    turn = _RecordingTurn()
    api._steward_turns[outer] = turn
    source_started: list[bool] = []

    async def _source() -> AsyncGenerator[str, None]:
        source_started.append(True)
        yield "never reached"

    async def _keepalive_like(
        source: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        yield ": keep-alive\n\n"
        async for item in source:
            yield item

    stream = api._release_steward_turn_after(outer, _keepalive_like(_source()))
    assert await stream.__anext__() == ": keep-alive\n\n"
    await stream.aclose()

    assert source_started == []
    assert outer not in api._steward_turns


def test_turn_whose_response_never_started_leaves_no_entry() -> None:
    """A response discarded before streaming cannot pin its turn forever.

    No generator cleanup runs for a response that never started, so the
    registry holds turns weakly and drops one when nothing else references it.
    """
    api = _make_api()
    outer = CommandId("steward-turn-4")
    turn = _RecordingTurn()
    api._steward_turns[outer] = turn
    assert outer in api._steward_turns

    del turn
    gc.collect()

    assert outer not in api._steward_turns


@pytest.mark.asyncio
async def test_send_task_cancellation_without_a_stream_retains_no_marker() -> None:
    """A cancellation no stream will finalize must not leave its marker.

    Only stream finalization discards the local-finish marker, so retaining
    it for a command whose stream never opens would grow the set forever.
    """
    api = _make_api()
    cid = CommandId("no-stream-cmd")

    await api.send_task_cancellation(cid, suppress_local_finish=False)
    assert cid not in api._cancelled_command_ids
    api._send.assert_called_once()

    await api.send_task_cancellation(cid)
    assert cid in api._cancelled_command_ids
