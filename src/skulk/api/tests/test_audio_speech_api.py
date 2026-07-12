# pyright: reportPrivateUsage=false
"""API coverage for OpenAI-compatible text-to-speech serving."""

import base64
import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Never, cast

import anyio
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.testclient import TestClient

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.api.types import AudioSpeechRequest
from skulk.extensions import (
    TTS_CAPABILITY_DESCRIPTOR,
    CapabilityCall,
    CapabilityStreamFrame,
    CapabilityStreamSession,
    InlineMediaAttachment,
    descriptor_revision,
)
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.routing.speech_media import SpeechMediaPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.experimental import EXPERIMENTAL_MODE_ENV_VAR
from skulk.shared.models.model_cards import (
    AudioCardConfig,
    AudioCardKind,
    AudioResponseFormat,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.audio import SpeechSynthesisTaskParams
from skulk.shared.types.chunks import AudioChunk, ErrorChunk
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SpeechSynthesis,
    TaskCancelled,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    SpeechSynthesis as SpeechSynthesisTask,
)
from skulk.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver, Sender, channel


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


def _build_builtin_provider_api() -> tuple[API, Receiver[ForwarderCommand]]:
    """Create an API with the first-party speech provider and local DATA loop."""

    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider_sender, provider_receiver = channel[ProviderStreamPacket](256)
    return (
        API(
            NodeId("api-node"),
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=False,
            telemetry_view=TelemetryView(),
            provider_stream_sender=provider_sender,
            provider_stream_receiver=provider_receiver,
            enable_builtin_providers=True,
        ),
        command_receiver,
    )


def _tts_card(
    *,
    supports_streaming: bool = True,
    supports_reference_audio: bool = False,
) -> ModelCard:
    """Create a minimal TTS model card for speech API validation tests."""

    return ModelCard(
        model_id=ModelId("mlx-community/kokoro-test"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextToSpeech],
        family="kokoro",
        capabilities=["tts"],
        audio=AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            default_response_format=AudioResponseFormat.Wav,
            response_formats=(AudioResponseFormat.Mp3, AudioResponseFormat.Wav),
            supports_streaming=supports_streaming,
            supports_reference_audio=supports_reference_audio,
        ),
    )


def _stt_card() -> ModelCard:
    """Create a minimal mounted STT card for negative voice-catalog tests."""

    return ModelCard(
        model_id=ModelId("mlx-community/stt-test"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.SpeechToText],
        family="parakeet",
        capabilities=["stt"],
        audio=AudioCardConfig(kind=AudioCardKind.SpeechToText),
    )


def _state_with_running_card(card: ModelCard) -> State:
    """Build a minimal state containing one mounted speech model instance."""

    runner_id = RunnerId("speech-runner")
    node_id = NodeId("speech-node")
    return State(
        instances={
            InstanceId("speech-instance"): MlxRingInstance(
                instance_id=InstanceId("speech-instance"),
                shard_assignments=ShardAssignments(
                    model_id=card.model_id,
                    runner_to_shard={
                        runner_id: PipelineShardMetadata(
                            model_card=card,
                            device_rank=0,
                            world_size=1,
                            start_layer=0,
                            end_layer=1,
                            n_layers=1,
                        )
                    },
                    node_to_runner={node_id: runner_id},
                ),
                hosts_by_node={node_id: []},
                ephemeral_port=52415,
            )
        }
    )


def _write_tts_streaming_experiment_config(api: API, tmp_path: Path) -> None:
    """Point the API at a temp config file with TTS streaming opted in."""

    api._config_path = tmp_path / "skulk.yaml"
    api._config_path.write_text("experiments:\n  tts_streaming: true\n")


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
    request_body = cast(dict[str, object], operation["requestBody"])
    content = cast(dict[str, object], request_body["content"])
    assert "application/json" in content
    assert "multipart/form-data" in content


def test_audio_speech_http_parses_multipart_reference_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multipart fields and upload should reach the strict core handler."""

    api = _build_api()
    captured: list[tuple[AudioSpeechRequest, str | None]] = []

    async def _handle(
        request: AudioSpeechRequest,
        *,
        reference_audio_file: UploadFile | None = None,
    ) -> Response:
        captured.append(
            (
                request,
                None
                if reference_audio_file is None
                else reference_audio_file.filename,
            )
        )
        return Response(content=b"audio", media_type="audio/wav")

    monkeypatch.setattr(api, "audio_speech", _handle)
    client = TestClient(api.app)

    response = client.post(
        "/v1/audio/speech",
        data={
            "model": "org/voice-model",
            "input": "hello",
            "response_format": "wav",
            "speed": "1.25",
            "stream": "false",
            "reference_text": "sample transcript",
        },
        files={"reference_audio": ("sample.wav", b"RIFFsample", "audio/wav")},
    )

    assert response.status_code == 200
    assert len(captured) == 1
    request, filename = captured[0]
    assert request.speed == 1.25
    assert request.stream is False
    assert request.reference_text == "sample transcript"
    assert filename == "sample.wav"


@pytest.mark.parametrize(
    ("field", "value"),
    (("speed", "fast"), ("max_tokens", "many"), ("stream", "sometimes")),
)
def test_audio_speech_http_rejects_invalid_multipart_scalars(
    field: str,
    value: str,
) -> None:
    """Malformed multipart scalar fields produce client errors, not server errors."""

    api = _build_api()
    client = TestClient(api.app)
    response = client.post(
        "/v1/audio/speech",
        data={"model": "org/voice-model", "input": "hello", field: value},
        files={"reference_audio": ("sample.wav", b"RIFFsample", "audio/wav")},
    )

    assert response.status_code == 422


def test_audio_speech_http_rejects_non_file_reference_audio() -> None:
    """A named reference field must not silently degrade to ordinary TTS."""

    api = _build_api()
    response = TestClient(api.app).post(
        "/v1/audio/speech",
        data={"model": "org/voice-model", "input": "hello"},
        files={"reference_audio": (None, "not-an-upload")},
    )

    assert response.status_code == 422


def test_audio_speech_http_rejects_invalid_json_payload() -> None:
    """Strict JSON validation failures are exposed as 422 responses."""

    api = _build_api()
    response = TestClient(api.app).post(
        "/v1/audio/speech",
        json={"model": "org/voice-model", "input": "hello", "speed": "fast"},
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_audio_speech_routes_reference_audio_outside_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference bytes travel as media packets while commands carry metadata."""

    api = _build_api()
    media_sender, media_receiver = channel[SpeechMediaPacket](8)
    actions: list[str] = []

    class _RecordingMediaSender:
        async def send(self, packet: SpeechMediaPacket) -> None:
            actions.append("media")
            await media_sender.send(packet)

    api._speech_media_packet_sender = cast(
        "Sender[SpeechMediaPacket]", cast("object", _RecordingMediaSender())
    )
    api._data_plane_zenoh = True
    card = _tts_card(supports_reference_audio=True)
    state = _state_with_running_card(card)
    runner_id = RunnerId("speech-runner")
    api.state = state.model_copy(update={"runners": {runner_id: RunnerReady()}})
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert requested_model == card.model_id
        assert stream is False
        return card.model_id, AudioResponseFormat.Wav

    async def _send(command: object) -> None:
        if isinstance(command, SpeechSynthesis):
            actions.append("command")
            sent_commands.append(command)
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=card.model_id,
                    data=base64.b64encode(b"WAVE-output").decode("ascii"),
                    chunk_index=0,
                    total_chunks=1,
                    format=AudioResponseFormat.Wav,
                    sample_rate=24000,
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(API, "_validate_speech_synthesis_model", _validate_model)
    monkeypatch.setattr(api, "_send", _send)
    upload = UploadFile(
        filename="sample.wav",
        file=io.BytesIO(b"RIFF-reference"),
        headers=Headers({"content-type": "audio/wav"}),
    )

    response = await api.audio_speech(
        AudioSpeechRequest(
            model=str(card.model_id),
            input="hello",
            response_format=AudioResponseFormat.Wav,
            reference_text="reference transcript",
        ),
        reference_audio_file=upload,
    )

    assert bytes(response.body) == b"WAVE-output"
    first = await media_receiver.receive()
    terminal = await media_receiver.receive()
    assert first.kind == "chunk"
    assert first.data == b"RIFF-reference"
    assert terminal.kind == "completed"
    assert len(sent_commands) == 1
    assert actions == ["command", "media", "media"]
    command = sent_commands[0]
    assert command.target_instance_id == InstanceId("speech-instance")
    assert command.task_params.reference_audio_present is True
    assert command.task_params.reference_audio_data is None


def test_audio_voices_route_is_documented_in_openapi() -> None:
    """The Skulk voice-catalog extension must appear in OpenAPI."""

    api = _build_api()
    schema = cast(dict[str, object], api.app.openapi())
    paths = cast(dict[str, object], schema["paths"])
    voices_path = cast(dict[str, object], paths["/v1/audio/voices"])
    operation = cast(dict[str, object], voices_path["get"])

    assert operation["tags"] == ["Audio"]
    assert operation["summary"] == "List voices for a mounted speech model"


@pytest.mark.anyio
async def test_audio_voices_returns_static_mounted_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voice listing should return only identifiers declared by the card."""

    api = _build_api()
    card = _tts_card()
    assert card.audio is not None
    card = card.model_copy(
        update={
            "audio": card.audio.model_copy(
                update={
                    "supports_voice_listing": True,
                    "voices": ("alloy", "coral"),
                }
            )
        }
    )

    async def _running_card(self: API, requested: ModelId) -> ModelCard:
        assert self is api
        assert requested == card.model_id
        return card

    monkeypatch.setattr(API, "_get_running_model_card", _running_card)
    api.state = _state_with_running_card(card)

    response = await api.audio_voices(str(card.model_id))

    assert [voice.id for voice in response.data] == ["alloy", "coral"]
    assert all(voice.model == str(card.model_id) for voice in response.data)


@pytest.mark.anyio
async def test_audio_voices_rejects_unmounted_catalog_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voice discovery must not advertise unavailable model capacity."""

    api = _build_api()
    card = _tts_card()
    assert card.audio is not None
    card = card.model_copy(
        update={
            "audio": card.audio.model_copy(
                update={"supports_voice_listing": True, "voices": ("alloy",)}
            )
        }
    )

    async def _catalog_card(self: API, requested: ModelId) -> ModelCard:
        raise AssertionError("unmounted voice discovery must not load a card")

    monkeypatch.setattr(API, "_get_running_model_card", _catalog_card)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_voices(str(card.model_id))

    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_audio_voices_rejects_mounted_non_tts_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mounted STT capacity should return 400 rather than a voice catalog."""

    api = _build_api()
    card = _stt_card()

    async def _running_card(self: API, requested: ModelId) -> ModelCard:
        assert self is api
        assert requested == card.model_id
        return card

    monkeypatch.setattr(API, "_get_running_model_card", _running_card)
    api.state = _state_with_running_card(card)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_voices(str(card.model_id))

    assert exc_info.value.status_code == 400
    assert "not a text-to-speech model" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_voices_rejects_tts_without_voice_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mounted TTS card must explicitly opt into voice discovery."""

    api = _build_api()
    card = _tts_card()

    async def _running_card(self: API, requested: ModelId) -> ModelCard:
        assert self is api
        assert requested == card.model_id
        return card

    monkeypatch.setattr(API, "_get_running_model_card", _running_card)
    api.state = _state_with_running_card(card)

    with pytest.raises(HTTPException) as exc_info:
        await api.audio_voices(str(card.model_id))

    assert exc_info.value.status_code == 400
    assert "does not support voice listing" in str(exc_info.value.detail)


def test_builtin_tts_capability_tracks_live_experimental_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Discovery should advertise TTS only while core can really stream it."""

    api, _ = _build_builtin_provider_api()
    card = _tts_card(supports_streaming=True)
    assert api._telemetry_view.local_advertised_capabilities == set()
    assert api._extensions is not None
    assert TTS_CAPABILITY_DESCRIPTOR in api._extensions.capability_descriptors

    api.state = _state_with_running_card(card)
    _write_tts_streaming_experiment_config(api, tmp_path)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == {"tts"}

    api.state = State()
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == set()


@pytest.mark.anyio
async def test_builtin_tts_admission_rejects_unmounted_model_without_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mounted-state admission should not perform a remote model-card lookup."""

    api, _ = _build_builtin_provider_api()
    card = _tts_card(supports_streaming=True)
    api.state = _state_with_running_card(card)
    _write_tts_streaming_experiment_config(api, tmp_path)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")

    async def _unexpected_prepare(
        _call: CapabilityCall,
    ) -> SpeechSynthesisTaskParams:
        raise AssertionError("unmounted model reached model-card preparation")

    monkeypatch.setattr(api, "_prepare_builtin_tts_task", _unexpected_prepare)
    error = await api._admit_builtin_tts_stream(
        CapabilityCall(
            call_id="unmounted-model",
            capability_id="tts",
            version="1.0.0",
            descriptor_revision=descriptor_revision(TTS_CAPABILITY_DESCRIPTOR),
            caller_node="api-node",
            target_node="api-node",
            timeout_seconds=2.0,
            payload={"model": "mlx-community/not-mounted", "text": "hello"},
        )
    )

    assert error is not None
    assert error.code == "not_found"


@pytest.mark.anyio
async def test_builtin_tts_provider_streams_core_audio_over_provider_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The generic TTS facade should preserve core command and audio semantics."""

    api, _ = _build_builtin_provider_api()
    card = _tts_card(supports_streaming=True)
    api.state = _state_with_running_card(card)
    _write_tts_streaming_experiment_config(api, tmp_path)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    api._sync_builtin_speech_capability()
    sent_commands: list[SpeechSynthesis] = []

    async def _send(command: object) -> None:
        if not isinstance(command, SpeechSynthesis):
            return
        sent_commands.append(command)
        sender = api._audio_speech_queues[command.command_id]
        await sender.send(
            AudioChunk(
                model=card.model_id,
                data=base64.b64encode(b"mp3-a").decode("ascii"),
                chunk_index=0,
                format=AudioResponseFormat.Mp3,
                sample_rate=24000,
                is_partial=True,
            )
        )
        await sender.send(
            AudioChunk(
                model=card.model_id,
                data=base64.b64encode(b"mp3-b").decode("ascii"),
                chunk_index=1,
                format=AudioResponseFormat.Mp3,
                sample_rate=24000,
                is_partial=True,
            )
        )
        await sender.send(
            AudioChunk(
                model=card.model_id,
                data="",
                chunk_index=2,
                format=AudioResponseFormat.Mp3,
                sample_rate=24000,
                finish_reason="stop",
            )
        )

    monkeypatch.setattr(api, "_send", _send)

    session: CapabilityStreamSession | None = None
    frames: list[CapabilityStreamFrame] = []
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            descriptor_revision(TTS_CAPABILITY_DESCRIPTOR),
            {
                "model": str(card.model_id),
                "text": "hello from the fabric",
                "streaming_interval": 0.25,
            },
            timeout_seconds=2.0,
        )
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert session is not None
    assert session.open_result.ok is True
    assert [frame.kind for frame in frames] == [
        "started",
        "chunk",
        "chunk",
        "completed",
    ]
    media = [frame.media for frame in frames if frame.media is not None]
    assert all(isinstance(item, InlineMediaAttachment) for item in media)
    assert b"".join(cast(InlineMediaAttachment, item).data for item in media) == (
        b"mp3-amp3-b"
    )
    assert all(cast(InlineMediaAttachment, item).channels is None for item in media)
    assert len(sent_commands) == 1
    assert sent_commands[0].task_params.input_text == "hello from the fabric"
    assert sent_commands[0].task_params.stream is True
    assert sent_commands[0].task_params.streaming_interval == 0.25
    assert sent_commands[0].command_id not in api._audio_speech_queues


@pytest.mark.anyio
async def test_builtin_tts_provider_rejects_before_started_when_gate_is_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A disabled experiment should fail opening without creating a lifecycle."""

    api, _ = _build_builtin_provider_api()
    card = _tts_card(supports_streaming=True)
    api.state = _state_with_running_card(card)
    api._config_path = tmp_path / "skulk.yaml"
    api._config_path.write_text("experiments:\n  tts_streaming: false\n")
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")

    session: CapabilityStreamSession | None = None
    frames: list[CapabilityStreamFrame] = []
    async with api._tg:
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            descriptor_revision(TTS_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id), "text": "hello"},
            timeout_seconds=2.0,
        )
        frames = [frame async for frame in session.frames]

    assert session is not None
    assert session.open_result.ok is False
    assert session.open_result.error is not None
    assert session.open_result.error.code == "not_found"
    assert frames == []
    assert api._active_capability_streams == {}


@pytest.mark.anyio
async def test_builtin_tts_provider_cancellation_reaches_core_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Closing a Fabric TTS stream should cancel its core synthesis command."""

    api, command_receiver = _build_builtin_provider_api()
    card = _tts_card(supports_streaming=True)
    api.state = _state_with_running_card(card)
    _write_tts_streaming_experiment_config(api, tmp_path)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    sent_commands: list[SpeechSynthesis] = []

    async def _send(command: object) -> None:
        if not isinstance(command, SpeechSynthesis):
            return
        sent_commands.append(command)
        await api._audio_speech_queues[command.command_id].send(
            AudioChunk(
                model=card.model_id,
                data=base64.b64encode(b"first").decode("ascii"),
                chunk_index=0,
                format=AudioResponseFormat.Mp3,
                sample_rate=24000,
                is_partial=True,
            )
        )

    monkeypatch.setattr(api, "_send", _send)

    forwarded: ForwarderCommand | None = None
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            descriptor_revision(TTS_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id), "text": "cancel me"},
            timeout_seconds=5.0,
        )
        iterator = session.frames.__aiter__()
        assert (await iterator.__anext__()).kind == "started"
        assert (await iterator.__anext__()).kind == "chunk"
        await session.frames.aclose()  # type: ignore[attr-defined]
        with anyio.fail_after(1.0):
            forwarded = await command_receiver.receive()
        task_group.cancel_scope.cancel()

    assert len(sent_commands) == 1
    assert forwarded is not None
    assert isinstance(forwarded.command, TaskCancelled)
    assert forwarded.command.cancelled_command_id == sent_commands[0].command_id
    assert sent_commands[0].command_id not in api._audio_speech_queues


@pytest.mark.anyio
async def test_audio_speech_streams_audio_chunks_and_sends_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming TTS request should stream decoded audio chunks as they arrive."""

    api = _build_api()
    model_id = ModelId("mlx-community/kokoro-test")
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Mp3
        assert stream is True
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
                    is_partial=True,
                    finish_reason=None,
                )
            )
            await api._audio_speech_queues[command.command_id].send(
                AudioChunk(
                    model=model_id,
                    data="",
                    chunk_index=2,
                    total_chunks=None,
                    format=AudioResponseFormat.Mp3,
                    sample_rate=24000,
                    is_partial=False,
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
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Mp3
        assert stream is True
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
async def test_audio_speech_rejects_streaming_without_experimental_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Streaming TTS stays inert until the node opts into experimental mode."""

    api = _build_api()
    card = _tts_card(supports_streaming=True)
    api.state = _state_with_running_card(card)
    _write_tts_streaming_experiment_config(api, tmp_path)
    monkeypatch.delenv(EXPERIMENTAL_MODE_ENV_VAR, raising=False)

    assert await api._validate_speech_synthesis_model(
        card.model_id, AudioResponseFormat.Mp3
    ) == (card.model_id, AudioResponseFormat.Mp3)
    with pytest.raises(HTTPException) as exc_info:
        await api._validate_speech_synthesis_model(
            card.model_id,
            AudioResponseFormat.Mp3,
            stream=True,
        )

    assert exc_info.value.status_code == 400
    assert "experimental" in str(exc_info.value.detail)
    assert EXPERIMENTAL_MODE_ENV_VAR in str(exc_info.value.detail)

    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    assert await api._validate_speech_synthesis_model(
        card.model_id,
        AudioResponseFormat.Mp3,
        stream=True,
    ) == (card.model_id, AudioResponseFormat.Mp3)


@pytest.mark.anyio
async def test_audio_speech_rejects_streaming_without_tts_experiment_toggle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The generic experiment gate still needs the TTS-specific toggle."""

    api = _build_api()
    card = _tts_card(supports_streaming=True)
    api.state = _state_with_running_card(card)
    api._config_path = tmp_path / "skulk.yaml"
    api._config_path.write_text("experiments:\n  tts_streaming: false\n")
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")

    with pytest.raises(HTTPException) as exc_info:
        await api._validate_speech_synthesis_model(
            card.model_id,
            AudioResponseFormat.Mp3,
            stream=True,
        )

    assert exc_info.value.status_code == 400
    assert "experiments.tts_streaming" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_speech_rejects_streaming_when_card_disallows_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A TTS card must explicitly opt in before stream=true is accepted."""

    api = _build_api()
    card = _tts_card(supports_streaming=False)
    api.state = _state_with_running_card(card)
    _write_tts_streaming_experiment_config(api, tmp_path)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")

    assert await api._validate_speech_synthesis_model(
        card.model_id, AudioResponseFormat.Mp3
    ) == (card.model_id, AudioResponseFormat.Mp3)
    with pytest.raises(HTTPException) as exc_info:
        await api._validate_speech_synthesis_model(
            card.model_id,
            AudioResponseFormat.Mp3,
            stream=True,
        )

    assert exc_info.value.status_code == 400
    assert "does not declare streaming speech support" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_audio_speech_stream_surfaces_initial_runner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner error before first audio should still become a normal HTTP error."""

    api = _build_api()
    model_id = ModelId("mlx-community/fish-audio-s2-pro-8bit")
    sent_commands: list[SpeechSynthesis] = []

    async def _validate_model(
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Mp3
        assert stream is True
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
async def test_audio_speech_stream_rejects_empty_initial_terminal_chunk() -> None:
    """A dropped real audio chunk plus terminal marker should not return 200."""

    api = _build_api()
    command_id = CommandId("speech-empty-initial-terminal")
    sender, receiver = channel[AudioChunk | ErrorChunk]()
    api._audio_speech_queues[command_id] = sender
    await sender.send(
        AudioChunk(
            model=ModelId("mlx-community/fish-audio-s2-pro-8bit"),
            data="",
            chunk_index=0,
            total_chunks=None,
            format=AudioResponseFormat.Mp3,
            sample_rate=44100,
            is_partial=False,
            finish_reason="stop",
        )
    )

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
async def test_audio_speech_stream_errors_when_channel_closes_before_terminal() -> None:
    """A closed data-plane stream before a terminal chunk must not look complete."""

    api = _build_api()
    cancel_sender, cancel_receiver = channel[ForwarderCommand]()
    api.command_sender = cancel_sender
    command_id = CommandId("speech-stream-closed-before-terminal")
    sender, receiver = channel[AudioChunk | ErrorChunk]()
    api._audio_speech_queues[command_id] = sender
    await sender.aclose()
    first_chunk = AudioChunk(
        model=ModelId("mlx-community/fish-audio-s2-pro-8bit"),
        data=base64.b64encode(b"partial").decode("ascii"),
        chunk_index=0,
        total_chunks=None,
        format=AudioResponseFormat.Mp3,
        sample_rate=44100,
        is_partial=True,
        finish_reason=None,
    )
    chunks: list[bytes] = []

    with pytest.raises(RuntimeError) as exc_info:
        async for chunk in api._stream_audio_speech_chunks(
            command_id, receiver, first_chunk
        ):
            chunks.append(chunk)

    assert chunks == [b"partial"]
    assert "terminal chunk" in str(exc_info.value)
    forwarded = await cancel_receiver.receive()
    assert isinstance(forwarded.command, TaskCancelled)
    assert forwarded.command.cancelled_command_id == command_id
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
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Wav
        assert stream is True
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
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format == AudioResponseFormat.Wav
        assert stream is False
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
async def test_audio_speech_collect_errors_when_channel_closes_before_terminal() -> None:
    """Collected TTS responses require a terminal audio chunk."""

    api = _build_api()
    cancel_sender, cancel_receiver = channel[ForwarderCommand]()
    api.command_sender = cancel_sender
    command_id = CommandId("speech-collect-closed-before-terminal")
    sender, receiver = channel[AudioChunk | ErrorChunk]()
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
    await sender.aclose()

    with pytest.raises(HTTPException) as exc_info:
        await api._collect_audio_speech_chunks(command_id, receiver)

    assert exc_info.value.status_code == 500
    assert "terminal chunk" in str(exc_info.value.detail)
    forwarded = await cancel_receiver.receive()
    assert isinstance(forwarded.command, TaskCancelled)
    assert forwarded.command.cancelled_command_id == command_id


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
        self: API,
        requested_model: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        assert self is api
        assert requested_model == model_id
        assert response_format is None
        assert stream is False
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
