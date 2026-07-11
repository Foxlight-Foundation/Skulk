# pyright: reportPrivateUsage=false
"""Built-in bounded batch STT provider facade coverage."""

import base64
from collections.abc import AsyncIterator

import pytest

from skulk.api.main import API
from skulk.extensions import (
    STT_CAPABILITY_DESCRIPTOR,
    CapabilityStreamFrame,
    InlineMediaAttachment,
    descriptor_revision,
)
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import (
    AudioCardConfig,
    AudioCardKind,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.chunks import AudioInputChunk, TranscriptionChunk
from skulk.shared.types.commands import (
    AudioTranscription,
    ForwarderCommand,
    ForwarderDownloadCommand,
    SendInputChunk,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerLoading,
    RunnerReady,
    RunnerStatus,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def _stt_card(model_id: str = "mlx-community/parakeet-test") -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.SpeechToText],
        family="parakeet",
        capabilities=["stt"],
        audio=AudioCardConfig(kind=AudioCardKind.SpeechToText),
    )


def _state(
    card: ModelCard,
    *,
    runner_status: RunnerStatus | None = None,
    suffix: str = "",
) -> State:
    runner_id = RunnerId(f"speech-runner{suffix}")
    node_id = NodeId("api-node")
    return State(
        instances={
            InstanceId(f"speech-instance{suffix}"): MlxRingInstance(
                instance_id=InstanceId(f"speech-instance{suffix}"),
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
        },
        runners={runner_id: runner_status or RunnerReady()},
    )


def _build_api() -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider_sender, provider_receiver = channel[ProviderStreamPacket](32)
    return API(
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
    )


def test_batch_stt_discovery_tracks_ready_mounted_capacity() -> None:
    api = _build_api()
    api.state = _state(_stt_card())

    api._sync_builtin_speech_capability()

    assert api._telemetry_view.local_advertised_capabilities == {"stt"}
    assert api._extensions is not None
    assert STT_CAPABILITY_DESCRIPTOR in api._extensions.capability_descriptors

    api.state = State()
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == set()


@pytest.mark.anyio
async def test_batch_stt_provider_transcribes_binary_input_after_half_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api()
    card = _stt_card()
    api.state = _state(card)
    api._sync_builtin_speech_capability()
    input_chunks: list[AudioInputChunk] = []
    commands: list[AudioTranscription] = []

    async def send(command: object) -> None:
        if isinstance(command, SendInputChunk):
            assert isinstance(command.chunk, AudioInputChunk)
            input_chunks.append(command.chunk)
        elif isinstance(command, AudioTranscription):
            commands.append(command)
            await api._audio_transcription_queues[command.command_id].send(
                TranscriptionChunk(
                    model=card.model_id,
                    text="hello world",
                    language="en",
                    segments=[{"id": 0, "text": "hello world"}],
                    finish_reason="stop",
                )
            )

    monkeypatch.setattr(api, "_send", send)
    frames: list[CapabilityStreamFrame] = []
    audio = b"RIFF-test-wave-data"
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            STT_CAPABILITY_DESCRIPTOR.id,
            STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(STT_CAPABILITY_DESCRIPTOR),
            {
                "model": str(card.model_id),
                "filename": "speech.wav",
            },
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        await session.input.send_chunk(
            media=InlineMediaAttachment(data=audio, media_type="audio/wav")
        )
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert len(commands) == 1
    assert commands[0].task_params.content_type == "audio/wav"
    assert b"".join(
        base64.b64decode(chunk.data.encode("ascii")) for chunk in input_chunks
    ) == audio
    assert [frame.kind for frame in frames] == ["started", "completed"]
    assert frames[-1].payload == {
        "model": str(card.model_id),
        "text": "hello world",
        "language": "en",
        "segments": [{"id": 0, "text": "hello world"}],
    }


@pytest.mark.anyio
async def test_batch_stt_admission_requires_requested_model_runner_ready() -> None:
    api = _build_api()
    card = _stt_card()
    ready = _state(card)
    loading = _state(
        card,
        runner_status=RunnerLoading(layers_loaded=0, total_layers=1),
        suffix="-loading",
    )
    loading_instance_id, loading_instance = next(iter(loading.instances.items()))
    placement_runner_id, placement_shard = next(
        iter(loading_instance.shard_assignments.runner_to_shard.items())
    )
    auxiliary_runner_id = RunnerId("speech-runner-loading-auxiliary")
    loading_assignments = loading_instance.shard_assignments.model_copy(
        update={
            "runner_to_shard": {
                placement_runner_id: placement_shard,
                auxiliary_runner_id: placement_shard,
            }
        }
    )
    loading = loading.model_copy(
        update={
            "instances": {
                loading_instance_id: loading_instance.model_copy(
                    update={"shard_assignments": loading_assignments}
                )
            },
            "runners": {
                **loading.runners,
                auxiliary_runner_id: RunnerReady(),
            },
        }
    )
    api.state = State(
        instances={**ready.instances, **loading.instances},
        runners={**ready.runners, **loading.runners},
    )
    api._sync_builtin_speech_capability()

    async with api._tg as task_group:
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            STT_CAPABILITY_DESCRIPTOR.id,
            STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id)},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is False
        assert session.open_result.error is not None
        assert session.open_result.error.code == "overloaded"
        assert session.input is None
        task_group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_batch_stt_input_cancel_emits_cancelled_terminal() -> None:
    api = _build_api()
    card = _stt_card()
    api.state = _state(card)
    api._sync_builtin_speech_capability()
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            STT_CAPABILITY_DESCRIPTOR.id,
            STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id)},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        await session.input.cancel("caller stopped recording")
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "cancelled"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "cancelled"
    assert frames[-1].error.message == "caller stopped recording"


@pytest.mark.anyio
async def test_batch_stt_empty_input_emits_invalid_frame_terminal() -> None:
    api = _build_api()
    card = _stt_card()
    api.state = _state(card)
    api._sync_builtin_speech_capability()
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            STT_CAPABILITY_DESCRIPTOR.id,
            STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id)},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "invalid_frame"
    assert frames[-1].error.message == "batch STT audio input is empty"


@pytest.mark.anyio
async def test_batch_stt_provider_rejects_oversized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api()

    async def frames() -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id="batch",
            direction="caller_to_provider",
            sequence=0,
            kind="started",
        )
        yield CapabilityStreamFrame(
            call_id="batch",
            direction="caller_to_provider",
            sequence=1,
            kind="chunk",
            media=InlineMediaAttachment(
                data=b"xx",
                media_type="audio/wav",
            ),
        )

    import skulk.api.main as api_main

    monkeypatch.setattr(api_main, "_MAX_AUDIO_UPLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        await api._collect_builtin_stt_audio(frames())
