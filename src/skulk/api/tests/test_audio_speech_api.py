# pyright: reportPrivateUsage=false
"""API coverage for OpenAI-compatible text-to-speech serving."""

import base64
from collections.abc import AsyncIterator
from typing import Never, cast

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from skulk.api.main import API
from skulk.api.types import AudioSpeechRequest
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.shared.types.chunks import AudioChunk
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SpeechSynthesis,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
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
            response_format=AudioResponseFormat.Mp3,
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
