# pyright: reportPrivateUsage=false
"""API coverage for OpenAI-compatible text-to-speech serving."""

import base64
from collections.abc import AsyncIterator
from typing import Never, cast

import anyio
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.api.types import AudioSpeechRequest
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.shared.types.audio import SpeechSynthesisTaskParams
from skulk.shared.types.chunks import AudioChunk, ErrorChunk
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SpeechSynthesis,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    SpeechSynthesis as SpeechSynthesisTask,
)
from skulk.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import channel


def _build_api() -> API:
    """Create an API instance with in-memory channels for direct route tests."""

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


def test_audio_speech_request_accepts_audio_format_strings() -> None:
    """The public request model should coerce OpenAI-style format strings."""

    request = AudioSpeechRequest.model_validate(
        {
            "model": "mlx-community/kokoro-test",
            "input": "hello",
            "response_format": "wav",
        }
    )

    assert request.response_format == AudioResponseFormat.Wav


def test_audio_speech_request_allows_model_default_response_format() -> None:
    """Omitted response_format is resolved from the mounted model card later."""

    request = AudioSpeechRequest.model_validate(
        {
            "model": "mlx-community/kokoro-test",
            "input": "hello",
        }
    )

    assert request.response_format is None


def test_audio_speech_request_rejects_unknown_audio_format() -> None:
    """Unsupported audio container names should fail request validation."""

    with pytest.raises(ValidationError):
        AudioSpeechRequest.model_validate(
            {
                "model": "mlx-community/kokoro-test",
                "input": "hello",
                "response_format": "aac",
            }
        )


def test_audio_speech_route_is_documented_in_openapi() -> None:
    """The speech endpoint must appear in FastAPI's generated OpenAPI schema."""

    api = _build_api()
    schema = cast(dict[str, object], api.app.openapi())
    paths = cast(dict[str, object], schema["paths"])
    speech_path = cast(dict[str, object], paths["/v1/audio/speech"])
    operation = cast(dict[str, object], speech_path["post"])

    assert operation["tags"] == ["Audio"]
    assert operation["summary"] == "Generate speech audio"


@pytest.mark.anyio
async def test_audio_speech_streams_audio_chunks_and_sends_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming TTS request should stream decoded audio chunks as they arrive."""

    api = _build_api()
    model_id = ModelId("mlx-community/kokoro-test")
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API, requested_model: ModelId, response_format: AudioResponseFormat | None
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Mp3
        return model_id, AudioResponseFormat.Mp3

    async def _send(command: object) -> None:
        if isinstance(command, SpeechSynthesis):
            sent_commands.append(command)
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data=base64.b64encode(b"mp3-a").decode("ascii"),
                    chunk_index=0,
                    total_chunks=None,
                    format=AudioResponseFormat.Mp3,
                    sample_rate=24000,
                    is_partial=True,
                    finish_reason=None,
                )
            )
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data=base64.b64encode(b"mp3-b").decode("ascii"),
                    chunk_index=1,
                    total_chunks=None,
                    format=AudioResponseFormat.Mp3,
                    sample_rate=24000,
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)

    response = await api.audio_speech(
        AudioSpeechRequest(
            model=str(model_id),
            input="hello",
            stream=True,
            streaming_interval=0.25,
        )
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "audio/mpeg"
    body_iterator = cast(AsyncIterator[bytes], response.body_iterator)
    body = b"".join([chunk async for chunk in body_iterator])
    assert body == b"mp3-amp3-b"
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert command.task_params.stream is True
    assert command.task_params.streaming_interval == 0.25
    assert command.command_id not in api._audio_speech_queues


@pytest.mark.anyio
async def test_audio_speech_streaming_defaults_to_mp3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming TTS should request MP3 instead of the non-streaming card default."""

    api = _build_api()
    model_id = ModelId("mlx-community/fish-audio-s2-pro-8bit")

    async def _validate_model(
        self: API, requested_model: ModelId, response_format: AudioResponseFormat | None
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Mp3
        return model_id, AudioResponseFormat.Mp3

    async def _send(command: object) -> None:
        if isinstance(command, SpeechSynthesis):
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data=base64.b64encode(b"mp3").decode("ascii"),
                    chunk_index=0,
                    total_chunks=None,
                    format=AudioResponseFormat.Mp3,
                    sample_rate=44100,
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)

    response = await api.audio_speech(
        AudioSpeechRequest(model=str(model_id), input="hello", stream=True)
    )

    assert isinstance(response, StreamingResponse)
    body_iterator = cast(AsyncIterator[bytes], response.body_iterator)
    assert b"".join([chunk async for chunk in body_iterator]) == b"mp3"


@pytest.mark.anyio
async def test_audio_speech_stream_surfaces_initial_runner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner error before first audio should still become a normal HTTP error."""

    api = _build_api()
    model_id = ModelId("mlx-community/fish-audio-s2-pro-8bit")
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API, requested_model: ModelId, response_format: AudioResponseFormat | None
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Mp3
        return model_id, AudioResponseFormat.Mp3

    async def _send(command: object) -> None:
        if isinstance(command, SpeechSynthesis):
            sent_commands.append(command)
            await api._audio_speech_queues[command.command_id].send(
                ErrorChunk(model=model_id, error_message="voice not found")
            )

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_speech(
            AudioSpeechRequest(model=str(model_id), input="hello", stream=True)
        )

    assert exc_info.value.status_code == 500
    assert "voice not found" in str(exc_info.value.detail)
    assert len(sent_commands) == 1
    assert sent_commands[0].command_id not in api._audio_speech_queues


@pytest.mark.anyio
async def test_audio_speech_stream_errors_when_terminal_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal task with no initial chunk should fail before headers commit."""

    api = _build_api()
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.01)
    command_id = CommandId("speech-terminal-before-first-chunk")
    sender, receiver = channel[AudioChunk | ErrorChunk]()
    api._audio_speech_queues[command_id] = sender
    task = SpeechSynthesisTask(
        task_id=TaskId("terminal-before-first-chunk-task"),
        instance_id=InstanceId("terminal-before-first-chunk-instance"),
        task_status=TaskStatus.Complete,
        command_id=command_id,
        task_params=SpeechSynthesisTaskParams(
            model=ModelId("mlx-community/fish-audio-s2-pro-8bit"),
            input_text="hello",
            response_format=AudioResponseFormat.Mp3,
        ),
    )
    api.state = State(tasks={task.task_id: task})

    with pytest.raises(HTTPException) as exc_info:
        await api._receive_initial_audio_speech_chunk(command_id, receiver)

    assert exc_info.value.status_code == 500
    assert "no audio response" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_speech_stream_finishes_cleanly_for_terminal_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal task with a dropped final audio chunk should finalize cleanly."""

    api = _build_api()
    finished_commands: list[object] = []
    cancel_sender, cancel_receiver = channel[ForwarderCommand]()

    async def _record_finished(command: object) -> None:
        finished_commands.append(command)

    api._send = _record_finished
    api.command_sender = cancel_sender
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.01)
    command_id = CommandId("speech-terminal-idle")
    sender, receiver = channel[AudioChunk | ErrorChunk]()
    api._audio_speech_queues[command_id] = sender
    task = SpeechSynthesisTask(
        task_id=TaskId("terminal-speech-task"),
        instance_id=InstanceId("terminal-speech-instance"),
        task_status=TaskStatus.Complete,
        command_id=command_id,
        task_params=SpeechSynthesisTaskParams(
            model=ModelId("mlx-community/fish-audio-s2-pro-8bit"),
            input_text="hello",
            response_format=AudioResponseFormat.Mp3,
        ),
    )
    api.state = State(tasks={task.task_id: task})
    chunks: list[bytes] = []

    async with anyio.create_task_group() as task_group:

        async def _consume() -> None:
            async for chunk in api._stream_audio_speech_chunks(command_id, receiver):
                chunks.append(chunk)

        task_group.start_soon(_consume)
        await sender.send(
            AudioChunk(
                model=ModelId("mlx-community/fish-audio-s2-pro-8bit"),
                data=base64.b64encode(b"partial").decode("ascii"),
                chunk_index=0,
                total_chunks=None,
                format=AudioResponseFormat.Mp3,
                sample_rate=44100,
                is_partial=True,
                finish_reason=None,
            )
        )

    assert chunks == [b"partial"]
    assert cancel_receiver.collect() == []
    assert finished_commands
    assert command_id not in api._cancelled_command_ids
    assert command_id not in api._audio_speech_queues


@pytest.mark.anyio
async def test_audio_speech_rejects_streaming_interval_without_stream_before_model_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming interval without stream=true is a malformed request."""

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("speech model validation should not run")

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _fail_if_called)
    api = _build_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_speech(
            AudioSpeechRequest(
                model="mlx-community/kokoro-test",
                input="hello",
                streaming_interval=0.25,
            )
        )

    assert exc_info.value.status_code == 400
    assert "stream=true" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_speech_rejects_non_mp3_streaming_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming TTS stays limited to response formats safe to concatenate."""

    api = _build_api()
    model_id = ModelId("mlx-community/kokoro-test")

    async def _validate_model(
        self: API, requested_model: ModelId, response_format: AudioResponseFormat | None
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Wav
        return model_id, AudioResponseFormat.Wav

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("streaming request should fail before command send")

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_speech(
            AudioSpeechRequest(
                model=str(model_id),
                input="hello",
                response_format=AudioResponseFormat.Wav,
                stream=True,
            )
        )

    assert exc_info.value.status_code == 400
    assert "mp3" in str(exc_info.value.detail)
    assert "wav" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_speech_rejects_reference_fields_before_model_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference audio/text stays disabled until Skulk has managed upload flows."""

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("speech model validation should not run")

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _fail_if_called)
    api = _build_api()

    with pytest.raises(HTTPException) as audio_exc:
        await api.audio_speech(
            AudioSpeechRequest(
                model="mlx-community/kokoro-test",
                input="hello",
                reference_audio="local.wav",
            )
        )
    with pytest.raises(HTTPException) as text_exc:
        await api.audio_speech(
            AudioSpeechRequest(
                model="mlx-community/kokoro-test",
                input="hello",
                reference_text="reference transcript",
            )
        )

    assert audio_exc.value.status_code == 400
    assert "reference_audio" in str(audio_exc.value.detail)
    assert text_exc.value.status_code == 400
    assert "reference_text" in str(text_exc.value.detail)


@pytest.mark.anyio
async def test_audio_speech_collects_audio_chunks_and_sends_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-streaming TTS request should return decoded audio bytes."""

    api = _build_api()
    model_id = ModelId("mlx-community/kokoro-test")
    audio_bytes = b"RIFFtestWAVE"
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API, requested_model: ModelId, response_format: AudioResponseFormat | None
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Wav
        return model_id, AudioResponseFormat.Wav

    async def _send(command: object) -> None:
        if isinstance(command, SpeechSynthesis):
            sent_commands.append(command)
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data=base64.b64encode(audio_bytes[:4]).decode("ascii"),
                    chunk_index=0,
                    total_chunks=2,
                    format=AudioResponseFormat.Wav,
                    sample_rate=24000,
                    finish_reason=None,
                )
            )
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data=base64.b64encode(audio_bytes[4:]).decode("ascii"),
                    chunk_index=1,
                    total_chunks=2,
                    format=AudioResponseFormat.Wav,
                    sample_rate=24000,
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)

    response = await api.audio_speech(
        AudioSpeechRequest(
            model=str(model_id),
            input="hello there",
            response_format=AudioResponseFormat.Wav,
            voice="af_heart",
            speed=1.1,
        )
    )

    assert response.media_type == "audio/wav"
    assert response.body == audio_bytes
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert command.owner_node == api.node_id
    assert command.task_params.model == model_id
    assert command.task_params.input_text == "hello there"
    assert command.task_params.voice == "af_heart"
    assert command.task_params.speed == 1.1
    assert command.command_id not in api._audio_speech_queues


@pytest.mark.anyio
async def test_audio_speech_collect_terminal_before_any_chunk_reports_no_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-streaming terminal task with no chunks should report no audio."""

    api = _build_api()
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.01)
    command_id = CommandId("speech-collect-terminal-before-first-chunk")
    sender, receiver = channel[AudioChunk | ErrorChunk]()
    api._audio_speech_queues[command_id] = sender
    task = SpeechSynthesisTask(
        task_id=TaskId("collect-terminal-before-first-chunk-task"),
        instance_id=InstanceId("collect-terminal-before-first-chunk-instance"),
        task_status=TaskStatus.Complete,
        command_id=command_id,
        task_params=SpeechSynthesisTaskParams(
            model=ModelId("mlx-community/fish-audio-s2-pro-8bit"),
            input_text="hello",
            response_format=AudioResponseFormat.Mp3,
        ),
    )
    api.state = State(tasks={task.task_id: task})

    with pytest.raises(HTTPException) as exc_info:
        await api._collect_audio_speech_chunks(command_id, receiver)

    assert exc_info.value.status_code == 500
    assert "no audio response" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_speech_uses_resolved_model_default_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests without response_format should use the mounted model default."""

    api = _build_api()
    model_id = ModelId("mlx-community/kokoro-test")
    audio_bytes = b"RIFFtestWAVE"
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API, requested_model: ModelId, response_format: AudioResponseFormat | None
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format is None
        return model_id, AudioResponseFormat.Wav

    async def _send(command: object) -> None:
        if isinstance(command, SpeechSynthesis):
            sent_commands.append(command)
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data=base64.b64encode(audio_bytes).decode("ascii"),
                    chunk_index=0,
                    total_chunks=1,
                    format=AudioResponseFormat.Wav,
                    sample_rate=24000,
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)

    response = await api.audio_speech(
        AudioSpeechRequest(
            model=str(model_id),
            input="hello there",
        )
    )

    assert response.media_type == "audio/wav"
    assert response.body == audio_bytes
    assert len(sent_commands) == 1
    assert sent_commands[0].task_params.response_format == AudioResponseFormat.Wav
