# pyright: reportPrivateUsage=false
"""Built-in realtime STT provider facade coverage."""

from pathlib import Path

import pytest

from skulk.api.main import API
from skulk.extensions import (
    REALTIME_STT_CAPABILITY_DESCRIPTOR,
    CapabilityStreamFrame,
    InlineMediaAttachment,
    descriptor_revision,
)
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.experimental import EXPERIMENTAL_MODE_ENV_VAR
from skulk.shared.models.model_cards import (
    AudioCardConfig,
    AudioCardKind,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.chunks import TranscriptionChunk
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RealtimeAudioTranscription,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver, channel


def _realtime_card(*, supports_realtime: bool = True) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/voxtral-realtime-test"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.SpeechToText],
        family="voxtral_realtime",
        capabilities=["stt"],
        audio=AudioCardConfig(
            kind=AudioCardKind.SpeechToText,
            supports_streaming=supports_realtime,
            supports_realtime=supports_realtime,
            sample_rates=(16000,),
        ),
    )


def _local_state(card: ModelCard) -> State:
    runner_id = RunnerId("speech-runner")
    node_id = NodeId("api-node")
    instance_id = InstanceId("speech-instance")
    return State(
        instances={
            instance_id: MlxRingInstance(
                instance_id=instance_id,
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


def _build_api() -> tuple[API, Receiver[RealtimeAudioInputFrame]]:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider_sender, provider_receiver = channel[ProviderStreamPacket](256)
    realtime_sender, realtime_receiver = channel[RealtimeAudioInputFrame](64)
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
            realtime_audio_sender=realtime_sender,
            enable_builtin_providers=True,
        ),
        realtime_receiver,
    )


def _enable_realtime(api: API, tmp_path: Path) -> None:
    api._config_path = tmp_path / "skulk.yaml"
    api._config_path.write_text("experiments:\n  stt_realtime: true\n")


def test_realtime_stt_discovery_requires_truthful_local_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, _ = _build_api()
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _enable_realtime(api, tmp_path)
    api.state = _local_state(_realtime_card())

    api._sync_builtin_speech_capability()

    assert api._telemetry_view.local_advertised_capabilities == {"stt.realtime"}
    assert api._extensions is not None
    assert (
        REALTIME_STT_CAPABILITY_DESCRIPTOR
        in api._extensions.capability_descriptors
    )

    api.state = _local_state(_realtime_card(supports_realtime=False))
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == set()


@pytest.mark.anyio
async def test_realtime_stt_provider_forwards_pcm_and_streams_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, realtime_receiver = _build_api()
    card = _realtime_card()
    api.state = _local_state(card)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _enable_realtime(api, tmp_path)
    api._sync_builtin_speech_capability()
    commands: list[RealtimeAudioTranscription] = []
    input_frames: list[RealtimeAudioInputFrame] = []

    async def send(command: object) -> None:
        if isinstance(command, RealtimeAudioTranscription):
            commands.append(command)

    async def emulate_worker() -> None:
        while True:
            frame = await realtime_receiver.receive()
            input_frames.append(frame)
            if frame.kind != "completed":
                continue
            command = commands[0]
            output = api._audio_transcription_queues[command.command_id]
            await output.send(
                TranscriptionChunk(
                    model=card.model_id,
                    text="hello ",
                    is_partial=True,
                )
            )
            await output.send(
                TranscriptionChunk(
                    model=card.model_id,
                    text="hello world",
                    finish_reason="stop",
                )
            )
            return

    monkeypatch.setattr(api, "_send", send)
    frames: list[CapabilityStreamFrame] = []
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        task_group.start_soon(emulate_worker)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            REALTIME_STT_CAPABILITY_DESCRIPTOR.id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(REALTIME_STT_CAPABILITY_DESCRIPTOR),
            {
                "model": str(card.model_id),
                "sample_rate": 16000,
                "temperature": 0.0,
                "transcription_delay_ms": 480,
            },
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        await session.input.send_chunk(
            payload={
                "format": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
            },
            media=InlineMediaAttachment(
                data=b"\x00\x00\x01\x00",
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=16000,
                channels=1,
            ),
        )
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert len(commands) == 1
    assert commands[0].target_instance_id == InstanceId("speech-instance")
    assert [frame.kind for frame in input_frames] == ["chunk", "completed"]
    assert [frame.sequence for frame in input_frames] == [1, 2]
    assert input_frames[0].data == b"\x00\x00\x01\x00"
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert frames[1].payload == {
        "model": str(card.model_id),
        "text": "hello ",
        "is_partial": True,
    }
    assert frames[2].payload == {
        "model": str(card.model_id),
        "text": "hello world",
        "is_partial": False,
    }
