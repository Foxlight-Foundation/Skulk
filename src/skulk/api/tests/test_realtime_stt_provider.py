# pyright: reportPrivateUsage=false
"""Built-in realtime STT provider facade coverage."""

from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest

from skulk.api.main import API
from skulk.extensions import (
    REALTIME_STT_CAPABILITY_DESCRIPTOR,
    CapabilityStreamError,
    CapabilityStreamFrame,
    InlineMediaAttachment,
    descriptor_revision,
)
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.routing.realtime_audio import RealtimeAudioPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.experimental import EXPERIMENTAL_MODE_ENV_VAR
from skulk.shared.models.model_cards import (
    AudioCardConfig,
    AudioCardKind,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.audio import (
    RealtimeAudioInputFrame,
    RealtimeAudioTranscriptionTaskParams,
)
from skulk.shared.types.chunks import ErrorChunk, TranscriptionChunk
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RealtimeAudioTranscription,
    TaskCancelled,
    TaskFinished,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import IndexedEvent, NodeTimedOut, RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    RealtimeAudioTranscription as RealtimeAudioTranscriptionTask,
)
from skulk.shared.types.tasks import TaskId, TaskStatus
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
from skulk.utils.channels import Receiver, Sender, channel


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


def _local_state(
    card: ModelCard,
    *,
    runner_status: RunnerStatus | None = None,
    hosting_node: NodeId | None = None,
    runner_name: str = "speech-runner",
    instance_name: str = "speech-instance",
) -> State:
    runner_id = RunnerId(runner_name)
    instance_id = InstanceId(instance_name)
    hosting_node = hosting_node or NodeId("api-node")
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
                    node_to_runner={hosting_node: runner_id},
                ),
                hosts_by_node={hosting_node: []},
                ephemeral_port=52415,
            )
        },
        runners={runner_id: runner_status or RunnerReady()},
    )


def _build_api() -> tuple[
    API, Receiver[RealtimeAudioInputFrame], Sender[IndexedEvent]
]:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[IndexedEvent]()
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
        event_sender,
    )


def _build_remote_api(
    *, data_plane_zenoh: bool = True
) -> tuple[API, Receiver[RealtimeAudioPacket]]:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider_sender, provider_receiver = channel[ProviderStreamPacket](256)
    realtime_packet_sender, realtime_packet_receiver = channel[
        RealtimeAudioPacket
    ](64)
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
            realtime_audio_packet_sender=realtime_packet_sender,
            data_plane_zenoh=data_plane_zenoh,
            enable_builtin_providers=True,
        ),
        realtime_packet_receiver,
    )


def _write_legacy_disabled_realtime_config(api: API, tmp_path: Path) -> None:
    """Write the old disabled toggle to prove it no longer gates capacity."""

    api._config_path = tmp_path / "skulk.yaml"
    api._config_path.write_text("experiments:\n  stt_realtime: false\n")


def test_realtime_stt_discovery_requires_truthful_local_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, _, _ = _build_api()
    monkeypatch.delenv(EXPERIMENTAL_MODE_ENV_VAR, raising=False)
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api.state = _local_state(_realtime_card())

    api._sync_builtin_speech_capability()

    assert api._telemetry_view.local_advertised_capabilities == {
        "stt",
        "stt.realtime",
    }
    assert api._extensions is not None
    assert (
        REALTIME_STT_CAPABILITY_DESCRIPTOR
        in api._extensions.capability_descriptors
    )

    api.state = _local_state(_realtime_card(supports_realtime=False))
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == {"stt"}

    api.state = _local_state(
        _realtime_card(), runner_status=RunnerLoading(layers_loaded=0, total_layers=1)
    )
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == set()


def test_remote_realtime_stt_requires_private_unicast_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote PCM is never advertised over the broadcast fallback transport."""

    api, _ = _build_remote_api(data_plane_zenoh=False)
    api.state = _local_state(
        _realtime_card(), hosting_node=NodeId("worker-node")
    )
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)

    api._sync_builtin_speech_capability()

    assert api._telemetry_view.local_advertised_capabilities == {"stt"}

@pytest.mark.anyio
async def test_runner_ready_event_resynchronizes_realtime_stt_advertisement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Normal loading-to-ready transition must advertise realtime capacity."""

    api, _, event_sender = _build_api()
    card = _realtime_card()
    state = _local_state(
        card, runner_status=RunnerLoading(layers_loaded=0, total_layers=1)
    )
    api.state = state
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == set()

    runner_id = next(iter(state.runners))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(api._apply_state)
        await event_sender.send(
            IndexedEvent(
                idx=0,
                event=RunnerStatusUpdated(
                    runner_id=runner_id,
                    runner_status=RunnerReady(),
                ),
            )
        )
        while not api._telemetry_view.local_advertised_capabilities:
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert api._telemetry_view.local_advertised_capabilities == {
        "stt",
        "stt.realtime",
    }


def test_realtime_stt_discovery_rejects_multi_host_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """API admission must match the master's single-host speech invariant."""

    api, _, _ = _build_api()
    state = _local_state(_realtime_card())
    instance_id, instance = next(iter(state.instances.items()))
    first_runner, shard = next(
        iter(instance.shard_assignments.runner_to_shard.items())
    )
    second_runner = RunnerId("speech-runner-two")
    assignments = instance.shard_assignments.model_copy(
        update={
            "runner_to_shard": {
                first_runner: shard,
                second_runner: shard,
            },
            "node_to_runner": {
                NodeId("api-node"): first_runner,
                NodeId("worker-node"): second_runner,
            },
        }
    )
    api.state = state.model_copy(
        update={
            "instances": {
                instance_id: instance.model_copy(
                    update={"shard_assignments": assignments}
                )
            },
            "runners": {
                first_runner: RunnerReady(),
                second_runner: RunnerReady(),
            },
        }
    )
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)

    api._sync_builtin_speech_capability()

    assert api._telemetry_view.local_advertised_capabilities == set()


@pytest.mark.anyio
async def test_node_timeout_withdraws_remote_realtime_stt_advertisement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pruning the last remote runner must withdraw stale discovery state."""

    api, _ = _build_remote_api()
    remote_node = NodeId("worker-node")
    api.state = _local_state(_realtime_card(), hosting_node=remote_node)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api._sync_builtin_speech_capability()
    assert api._telemetry_view.local_advertised_capabilities == {
        "stt",
        "stt.realtime",
    }
    event_sender = api.event_receiver.clone_sender()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(api._apply_state)
        await event_sender.send(
            IndexedEvent(idx=0, event=NodeTimedOut(node_id=remote_node))
        )
        with anyio.fail_after(1):
            while api._telemetry_view.local_advertised_capabilities:
                await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert api.state.instances == {}
    assert api._telemetry_view.local_advertised_capabilities == set()


@pytest.mark.anyio
async def test_realtime_stt_failed_input_preserves_failure_semantics() -> None:
    """A failed caller stream must not be rewritten as normal cancellation."""

    api, realtime_receiver, _ = _build_api()
    command_id = CommandId("failed-input")

    async def input_frames() -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id="failed-input-call",
            direction="caller_to_provider",
            sequence=0,
            kind="started",
        )
        yield CapabilityStreamFrame(
            call_id="failed-input-call",
            direction="caller_to_provider",
            sequence=1,
            kind="failed",
            error=CapabilityStreamError(code="timeout", message="input timed out"),
        )

    with pytest.raises(RuntimeError, match="input timed out"):
        await api._pump_builtin_realtime_stt_input(
            command_id=command_id,
            params=RealtimeAudioTranscriptionTaskParams(
                model=ModelId("mlx-community/voxtral-realtime-test"),
                input_sample_rate=16000,
            ),
            target_node=NodeId("api-node"),
            input_frames=input_frames(),
        )

    assert realtime_receiver.collect() == []


@pytest.mark.anyio
async def test_same_node_realtime_audio_uses_bounded_local_ingress() -> None:
    """Same-node PCM must backpressure on the bounded API-worker channel."""

    api, realtime_receiver, _ = _build_api()
    packet_sender, packet_receiver = channel[RealtimeAudioPacket](4)
    api._realtime_audio_packet_sender = packet_sender
    frame = RealtimeAudioInputFrame(
        command_id=CommandId("same-node-bounded-ingress"),
        sequence=1,
        kind="chunk",
        data=b"\x00\x00",
    )

    await api._send_realtime_audio_input(NodeId("api-node"), frame)

    assert await realtime_receiver.receive() == frame
    assert packet_receiver.collect() == []


@pytest.mark.anyio
async def test_same_node_realtime_audio_falls_back_after_worker_reset() -> None:
    """A closed old worker channel falls back to the recreated packet ingress."""

    api, realtime_receiver, _ = _build_api()
    packet_sender, packet_receiver = channel[RealtimeAudioPacket](4)
    api._realtime_audio_packet_sender = packet_sender
    realtime_receiver.close()
    frame = RealtimeAudioInputFrame(
        command_id=CommandId("same-node-reset-ingress"),
        sequence=1,
        kind="chunk",
        data=b"\x00\x00",
    )

    await api._send_realtime_audio_input(NodeId("api-node"), frame)

    packet = await packet_receiver.receive()
    assert packet.target_node == NodeId("api-node")
    assert packet.to_input_frame() == frame


@pytest.mark.anyio
async def test_realtime_stt_provider_forwards_pcm_and_streams_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, realtime_receiver, _ = _build_api()
    card = _realtime_card()
    api.state = _local_state(card)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
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
            assert output.statistics().max_buffer_size == 256
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


@pytest.mark.anyio
async def test_realtime_stt_provider_routes_pcm_to_remote_serving_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An API edge can stream PCM to a ready speech runner on another node."""

    api, packet_receiver = _build_remote_api()
    card = _realtime_card()
    api.state = _local_state(card, hosting_node=NodeId("worker-node"))
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api._sync_builtin_speech_capability()
    commands: list[RealtimeAudioTranscription] = []
    packets: list[RealtimeAudioPacket] = []

    async def send(command: object) -> None:
        if isinstance(command, RealtimeAudioTranscription):
            commands.append(command)

    async def emulate_remote_worker() -> None:
        while True:
            packet = await packet_receiver.receive()
            packets.append(packet)
            if packet.kind != "completed":
                continue
            command = commands[0]
            await api._audio_transcription_queues[command.command_id].send(
                TranscriptionChunk(
                    model=card.model_id,
                    text="remote transcript",
                    finish_reason="stop",
                )
            )
            return

    monkeypatch.setattr(api, "_send", send)
    frames: list[CapabilityStreamFrame] = []
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        task_group.start_soon(emulate_remote_worker)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            REALTIME_STT_CAPABILITY_DESCRIPTOR.id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(REALTIME_STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id), "sample_rate": 16000},
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

    assert commands[0].owner_node == NodeId("api-node")
    assert commands[0].target_instance_id == InstanceId("speech-instance")
    assert [packet.target_node for packet in packets] == [
        NodeId("worker-node"),
        NodeId("worker-node"),
    ]
    assert [packet.kind for packet in packets] == ["chunk", "completed"]
    assert packets[0].data == b"\x00\x00\x01\x00"
    assert frames[-1].kind == "completed"
    assert frames[-1].payload is not None
    assert frames[-1].payload["text"] == "remote transcript"


@pytest.mark.anyio
async def test_realtime_stt_admission_reserves_local_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concurrent opens cannot race before TaskCreated reaches API state."""

    api, _, _ = _build_api()
    card = _realtime_card()
    api.state = _local_state(card)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api._sync_builtin_speech_capability()

    async def send(_command: object) -> None:
        return None

    monkeypatch.setattr(api, "_send", send)
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        first = await api._extension_context.stream_capability(
            NodeId("api-node"),
            REALTIME_STT_CAPABILITY_DESCRIPTOR.id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(REALTIME_STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id), "sample_rate": 16000},
            timeout_seconds=2.0,
        )
        second = await api._extension_context.stream_capability(
            NodeId("api-node"),
            REALTIME_STT_CAPABILITY_DESCRIPTOR.id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(REALTIME_STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id), "sample_rate": 16000},
            timeout_seconds=2.0,
        )

        assert first.open_result.ok is True
        assert first.input is not None
        assert second.open_result.ok is False
        assert second.open_result.error is not None
        assert second.open_result.error.code == "overloaded"
        await first.input.cancel("test complete")
        _ = [frame async for frame in first.frames]
        task_group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_realtime_stt_admission_skips_busy_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Admission selects an idle runner when an earlier candidate is busy."""

    api, _, _ = _build_api()
    card = _realtime_card()
    first = _local_state(card)
    second = _local_state(
        card,
        runner_name="speech-runner-two",
        instance_name="speech-instance-two",
    )
    busy_task = RealtimeAudioTranscriptionTask(
        task_id=TaskId("busy-realtime-task"),
        instance_id=InstanceId("speech-instance"),
        command_id=CommandId("busy-realtime-command"),
        owner_node=NodeId("other-api"),
        task_status=TaskStatus.Running,
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=card.model_id,
            input_sample_rate=16000,
        ),
    )
    api.state = State(
        instances={**first.instances, **second.instances},
        runners={**first.runners, **second.runners},
        tasks={busy_task.task_id: busy_task},
    )
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api._sync_builtin_speech_capability()
    commands: list[RealtimeAudioTranscription] = []

    async def send(command: object) -> None:
        if isinstance(command, RealtimeAudioTranscription):
            commands.append(command)

    monkeypatch.setattr(api, "_send", send)
    async with api._tg as task_group:
        task_group.start_soon(api._apply_provider_data)
        session = await api._extension_context.stream_capability(
            NodeId("api-node"),
            REALTIME_STT_CAPABILITY_DESCRIPTOR.id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(REALTIME_STT_CAPABILITY_DESCRIPTOR),
            {"model": str(card.model_id), "sample_rate": 16000},
            timeout_seconds=2.0,
        )

        assert session.open_result.ok is True
        assert session.input is not None
        assert commands[0].target_instance_id == InstanceId("speech-instance-two")
        await session.input.cancel("test complete")
        _ = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_realtime_stt_runner_error_finishes_command_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A terminal runner error must use natural command finalization."""

    api, realtime_receiver, _ = _build_api()
    card = _realtime_card()
    api.state = _local_state(card)
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    _write_legacy_disabled_realtime_config(api, tmp_path)
    api._sync_builtin_speech_capability()
    commands: list[object] = []

    async def send(command: object) -> None:
        commands.append(command)

    async def emulate_worker() -> None:
        while True:
            frame = await realtime_receiver.receive()
            if frame.kind != "completed":
                continue
            started = next(
                command
                for command in commands
                if isinstance(command, RealtimeAudioTranscription)
            )
            await api._audio_transcription_queues[started.command_id].send(
                ErrorChunk(model=card.model_id, error_message="decode failed")
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
            {"model": str(card.model_id), "sample_rate": 16000},
            timeout_seconds=2.0,
        )
        assert session.input is not None
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert frames[-1].kind == "failed"
    assert any(isinstance(command, TaskFinished) for command in commands)
    assert not any(isinstance(command, TaskCancelled) for command in commands)


@pytest.mark.anyio
async def test_realtime_stt_output_overflow_cancels_only_its_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow realtime consumer cannot block DATA or grow without bound."""

    api, _, _ = _build_api()
    command_id = CommandId("realtime-output-overflow")
    sender, _ = channel[TranscriptionChunk | ErrorChunk](1)
    api._audio_transcription_queues[command_id] = sender
    api._realtime_audio_transcription_commands.add(command_id)
    cancelled: list[CommandId] = []

    async def cancel(target: CommandId) -> None:
        cancelled.append(target)

    monkeypatch.setattr(api, "_cancel_audio_transcription_command", cancel)
    chunk = TranscriptionChunk(
        model=ModelId("mlx-community/voxtral-realtime-test"),
        text="partial",
        is_partial=True,
    )

    await api._dispatch_generation_chunk(command_id, chunk)
    await api._dispatch_generation_chunk(command_id, chunk)

    assert cancelled == [command_id]
    assert command_id not in api._audio_transcription_queues


@pytest.mark.anyio
async def test_remote_realtime_transport_failure_cancels_only_its_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-routed PCM rejection fails and cancels its owning command."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    packet_sender, packet_receiver = channel[RealtimeAudioPacket](4)
    api = API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        telemetry_view=TelemetryView(),
        realtime_audio_packet_receiver=packet_receiver,
    )
    command_id = CommandId("remote-transport-failure")
    output_sender, output_receiver = channel[TranscriptionChunk | ErrorChunk](4)
    api._audio_transcription_queues[command_id] = output_sender
    api._realtime_audio_transcription_commands.add(command_id)
    cancelled: list[CommandId] = []

    async def cancel(target: CommandId) -> None:
        cancelled.append(target)

    monkeypatch.setattr(api, "_cancel_audio_transcription_command", cancel)
    chunk: TranscriptionChunk | ErrorChunk | None = None
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(api._apply_realtime_audio_transport)
        await packet_sender.send(
            RealtimeAudioPacket(
                source_node=NodeId("worker-node"),
                target_node=NodeId("api-node"),
                command_id=command_id,
                sequence=2,
                kind="transport_failed",
                error_message="remote ingress capacity exhausted",
            )
        )
        chunk = await output_receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(chunk, ErrorChunk)
    assert chunk.error_message == "remote ingress capacity exhausted"
    assert cancelled == [command_id]
