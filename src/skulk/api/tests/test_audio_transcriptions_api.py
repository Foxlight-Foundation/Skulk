# pyright: reportPrivateUsage=false
"""API coverage for OpenAI-compatible speech-to-text serving."""

import base64
import hashlib
import io
import json
from typing import Never, cast

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.experimental import EXPERIMENTAL_MODE_ENV_VAR
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import AudioInputChunk, TranscriptionChunk
from skulk.shared.types.commands import (
    AudioTranscription,
    ForwarderCommand,
    ForwarderDownloadCommand,
    SendInputChunk,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioTranscription as AudioTranscriptionTask
from skulk.shared.types.tasks import TaskId, TaskStatus
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


def _upload(
    audio_bytes: bytes,
    *,
    filename: str = "sample.wav",
    content_type: str = "audio/wav",
) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=io.BytesIO(audio_bytes),
        headers=Headers({"content-type": content_type}),
    )


def test_audio_transcriptions_route_is_documented_in_openapi() -> None:
    """The transcription endpoint must appear in FastAPI's OpenAPI schema."""

    api = _build_api()
    schema = cast(dict[str, object], api.app.openapi())
    paths = cast(dict[str, object], schema["paths"])
    transcription_path = cast(dict[str, object], paths["/v1/audio/transcriptions"])
    operation = cast(dict[str, object], transcription_path["post"])

    assert operation["tags"] == ["Audio"]
    assert operation["summary"] == "Transcribe speech audio"


def test_audio_translations_route_is_documented_in_openapi() -> None:
    """The experimental translation endpoint must appear in OpenAPI."""

    api = _build_api()
    schema = cast(dict[str, object], api.app.openapi())
    paths = cast(dict[str, object], schema["paths"])
    translation_path = cast(dict[str, object], paths["/v1/audio/translations"])
    operation = cast(dict[str, object], translation_path["post"])

    assert operation["tags"] == ["Audio"]
    assert operation["summary"] == "Translate speech audio to English"


@pytest.mark.anyio
async def test_audio_translations_marks_shared_stt_task_for_english(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translation should reuse batch STT while setting explicit task intent."""

    api = _build_api()
    model_id = ModelId("mlx-community/canary-test")
    captured_params: list[object] = []

    async def _validate_model(
        self: API,
        requested_model: ModelId,
        source_language: str | None,
    ) -> ModelId:
        assert self is api
        assert requested_model == model_id
        assert source_language == "fr"
        return model_id

    async def _execute(
        self: API,
        params: object,
        audio_bytes: bytes,
    ) -> list[TranscriptionChunk]:
        assert self is api
        assert audio_bytes == b"RIFFtestWAVE"
        captured_params.append(params)
        return [
            TranscriptionChunk(
                model=model_id,
                text="hello world",
                language="en",
                segments=[],
                finish_reason="stop",
            )
        ]

    monkeypatch.setattr(API, "_validate_audio_translation_model", _validate_model)
    monkeypatch.setattr(API, "_execute_audio_transcription", _execute)

    response = await api.audio_translations(
        file=_upload(b"RIFFtestWAVE"),
        model=str(model_id),
        language="fr",
    )

    assert response.media_type == "application/json"
    assert len(captured_params) == 1
    params = cast(api_main.AudioTranscriptionTaskParams, captured_params[0])
    assert params.translate_to_english is True
    assert params.language == "fr"


@pytest.mark.anyio
async def test_audio_translations_rejects_before_reading_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled or ineligible translation must not consume the request body."""

    api = _build_api()
    upload = _upload(b"RIFFtestWAVE")

    async def _reject(
        self: API,
        requested_model: ModelId,
        source_language: str | None,
    ) -> Never:
        del requested_model, source_language
        raise HTTPException(status_code=404, detail="not mounted")

    monkeypatch.setattr(API, "_validate_audio_translation_model", _reject)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_translations(file=upload, model="org/not-mounted")

    assert exc_info.value.status_code == 404
    assert upload.file.tell() == 0


@pytest.mark.anyio
async def test_audio_translations_requires_experiment_before_reading_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feature gate must reject translation before consuming media."""

    api = _build_api()
    upload = _upload(b"RIFFtestWAVE")
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    monkeypatch.setattr(api, "_speech_translation_experiment_enabled", lambda: False)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_translations(file=upload, model="org/canary")

    assert exc_info.value.status_code == 400
    assert upload.file.tell() == 0


@pytest.mark.anyio
async def test_audio_translation_validation_rejects_unsupported_mounted_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mounted transcription capacity must not imply translation support."""

    api = _build_api()
    model_id = ModelId("org/transcription-only")
    api.state = cast(
        State,
        cast(
            object,
            type(
                "_State",
                (),
                {
                    "instances": {
                        "instance": type(
                            "_Instance",
                            (),
                            {
                                "shard_assignments": type(
                                    "_Assignments", (), {"model_id": model_id}
                                )()
                            },
                        )()
                    }
                },
            )(),
        ),
    )

    async def _running_card(self: API, requested: ModelId) -> object:
        assert self is api
        assert requested == model_id
        return type("_Card", (), {"model_id": model_id})()

    def _unsupported_profile(*_args: object, **_kwargs: object) -> object:
        return type("_Profile", (), {"supports_speech_translation": False})()

    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    monkeypatch.setattr(api, "_speech_translation_experiment_enabled", lambda: True)
    monkeypatch.setattr(API, "_get_running_model_card", _running_card)
    monkeypatch.setattr(
        api_main,
        "resolve_model_capability_profile",
        _unsupported_profile,
    )

    with pytest.raises(HTTPException) as exc_info:
        await api._validate_audio_translation_model(model_id, "fr")

    assert exc_info.value.status_code == 400
    assert "does not support" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_canary_translation_requires_source_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canary must not silently default non-English input to English source."""

    api = _build_api()
    model_id = ModelId("org/canary")
    api.state = cast(
        State,
        cast(
            object,
            type(
                "_State",
                (),
                {
                    "instances": {
                        "instance": type(
                            "_Instance",
                            (),
                            {
                                "shard_assignments": type(
                                    "_Assignments", (), {"model_id": model_id}
                                )()
                            },
                        )()
                    }
                },
            )(),
        ),
    )

    async def _running_card(self: API, requested: ModelId) -> object:
        assert self is api
        assert requested == model_id
        return type("_Card", (), {"model_id": model_id, "family": "canary"})()

    def _translation_profile(*_args: object, **_kwargs: object) -> object:
        return type("_Profile", (), {"supports_speech_translation": True})()

    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    monkeypatch.setattr(api, "_speech_translation_experiment_enabled", lambda: True)
    monkeypatch.setattr(API, "_get_running_model_card", _running_card)
    monkeypatch.setattr(
        api_main,
        "resolve_model_capability_profile",
        _translation_profile,
    )

    with pytest.raises(HTTPException) as exc_info:
        await api._validate_audio_translation_model(model_id, None)

    assert exc_info.value.status_code == 400
    assert "source `language`" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_transcriptions_rejects_streaming_before_model_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 2 is explicitly non-streaming and should fail before card lookup."""

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("transcription model validation should not run")

    monkeypatch.setattr(API, "_validate_audio_transcription_model", _fail_if_called)
    api = _build_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_transcriptions(
            file=_upload(b"RIFFtestWAVE"),
            model="mlx-community/whisper-test",
            stream=True,
        )

    assert exc_info.value.status_code == 400
    assert "stream" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_transcriptions_rejects_non_audio_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearly non-audio multipart uploads should be rejected at the API edge."""

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("transcription model validation should not run")

    monkeypatch.setattr(API, "_validate_audio_transcription_model", _fail_if_called)
    api = _build_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_transcriptions(
            file=_upload(
                b"<html></html>",
                filename="page.html",
                content_type="text/html",
            ),
            model="mlx-community/whisper-test",
        )

    assert exc_info.value.status_code == 415


@pytest.mark.anyio
async def test_audio_transcriptions_sends_input_chunks_and_returns_verbose_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-streaming STT request should return the final transcript."""

    api = _build_api()
    model_id = ModelId("mlx-community/whisper-test")
    audio_bytes = b"RIFFtestWAVE"
    sent_input_chunks: list[AudioInputChunk] = []
    sent_commands: list[AudioTranscription] = []

    async def _validate_model(self: API, requested_model: ModelId) -> ModelId:
        assert self is api
        assert requested_model == model_id
        return model_id

    async def _send(command: object) -> None:
        if isinstance(command, SendInputChunk):
            assert isinstance(command.chunk, AudioInputChunk)
            sent_input_chunks.append(command.chunk)
        if isinstance(command, AudioTranscription):
            sent_commands.append(command)
            await api._audio_transcription_queues[command.command_id].send(
                TranscriptionChunk(
                    model=model_id,
                    text="hello world",
                    language="en",
                    segments=[
                        {
                            "id": 0,
                            "text": "hello world",
                            "start": 0.0,
                            "end": 1.2,
                        }
                    ],
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(API, "_validate_audio_transcription_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)

    response = await api.audio_transcriptions(
        file=_upload(audio_bytes),
        model=str(model_id),
        language="en",
        response_format="verbose_json",
        timestamp_granularities='["segment"]',
    )

    assert response.media_type == "application/json"
    payload = cast(dict[str, object], json.loads(bytes(response.body).decode("utf-8")))
    assert payload["text"] == "hello world"
    assert payload["language"] == "en"
    segments = cast(list[dict[str, object]], payload["segments"])
    assert segments[0]["text"] == "hello world"
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert command.owner_node == api.node_id
    assert command.task_params.model == model_id
    assert command.task_params.language == "en"
    assert command.task_params.total_input_chunks == len(sent_input_chunks)
    assert command.task_params.audio_sha256 == hashlib.sha256(audio_bytes).hexdigest()
    assert command.task_params.audio_data is None
    assert command.task_params.timestamp_granularities == ("segment",)
    assert len(sent_input_chunks) == 1
    input_chunk = sent_input_chunks[0]
    assert input_chunk.command_id == command.command_id
    assert input_chunk.content_type == "audio/wav"
    assert input_chunk.filename == "sample.wav"
    assert (
        base64.b64decode(input_chunk.data.encode("ascii"), validate=True)
        == audio_bytes
    )
    assert command.command_id not in api._audio_transcription_queues


@pytest.mark.anyio
async def test_audio_transcriptions_errors_if_terminal_chunk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed task with no final data-plane chunk should not hang forever."""

    api = _build_api()
    model_id = ModelId("mlx-community/whisper-test")

    async def _validate_model(self: API, requested_model: ModelId) -> ModelId:
        assert self is api
        assert requested_model == model_id
        return model_id

    async def _send(command: object) -> None:
        if isinstance(command, AudioTranscription):
            task = AudioTranscriptionTask(
                task_id=TaskId("terminal-transcription-task"),
                instance_id=InstanceId("terminal-transcription-instance"),
                task_status=TaskStatus.Complete,
                command_id=command.command_id,
                task_params=command.task_params,
            )
            api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    monkeypatch.setattr(API, "_validate_audio_transcription_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_transcriptions(
            file=_upload(b"RIFFtestWAVE"),
            model=str(model_id),
        )

    assert exc_info.value.status_code == 500
    assert "final response chunk" in str(exc_info.value.detail)
