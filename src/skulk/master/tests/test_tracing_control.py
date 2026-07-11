"""Tests for master-side tracing control and task inheritance."""

import anyio
import pytest

from skulk.master.main import Master
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import (
    AudioCardConfig,
    AudioCardKind,
    AudioResponseFormat,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.audio import (
    AudioTranscriptionTaskParams,
    RealtimeAudioTranscriptionTaskParams,
    SpeechSynthesisTaskParams,
)
from skulk.shared.types.commands import (
    AudioTranscription,
    ForwarderCommand,
    ForwarderDownloadCommand,
    RealtimeAudioTranscription,
    SetTracingEnabled,
    SpeechSynthesis,
    TextGeneration,
)
from skulk.shared.types.common import CommandId, Host, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskCreated,
    TracingStateChanged,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import AudioTranscription as AudioTranscriptionTask
from skulk.shared.types.tasks import (
    RealtimeAudioTranscription as RealtimeAudioTranscriptionTask,
)
from skulk.shared.types.tasks import SpeechSynthesis as SpeechSynthesisTask
from skulk.shared.types.tasks import TextGeneration as TextGenerationTask
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver, Sender, channel


def _build_master() -> tuple[Master, NodeId, Sender[ForwarderCommand], Receiver[Event]]:
    """Create a master with in-memory channels for command-processor tests."""

    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    global_event_sender, _ = channel[GlobalForwarderEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    _, local_event_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    download_command_sender, _ = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_event_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_command_sender,
    )
    return master, node_id, command_sender, event_receiver


def _single_node_instance(node_id: NodeId) -> MlxRingInstance:
    instance_id = InstanceId("instance-1")
    runner_id = RunnerId("runner-1")
    model_card = ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        storage_size=Memory.from_mb(1024),
        n_layers=16,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=16,
        n_layers=16,
    )
    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard={runner_id: shard_metadata},
        node_to_runner={node_id: runner_id},
    )
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=shard_assignments,
        hosts_by_node={node_id: [Host(ip="0.0.0.0", port=58484)]},
        ephemeral_port=58484,
    )


def _single_node_speech_instance(node_id: NodeId) -> MlxRingInstance:
    instance_id = InstanceId("speech-instance-1")
    runner_id = RunnerId("speech-runner-1")
    model_card = ModelCard(
        model_id=ModelId("mlx-community/kokoro-test"),
        storage_size=Memory.from_mb(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToSpeech],
        capabilities=["tts"],
        audio=AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            response_formats=(AudioResponseFormat.Mp3, AudioResponseFormat.Wav),
        ),
    )
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard={runner_id: shard_metadata},
        node_to_runner={node_id: runner_id},
    )
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=shard_assignments,
        hosts_by_node={node_id: [Host(ip="0.0.0.0", port=58484)]},
        ephemeral_port=58484,
    )


def _single_node_transcription_instance(node_id: NodeId) -> MlxRingInstance:
    instance_id = InstanceId("transcription-instance-1")
    runner_id = RunnerId("transcription-runner-1")
    model_card = ModelCard(
        model_id=ModelId("mlx-community/whisper-test"),
        storage_size=Memory.from_mb(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.SpeechToText],
        capabilities=["stt"],
        audio=AudioCardConfig(kind=AudioCardKind.SpeechToText),
    )
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard={runner_id: shard_metadata},
        node_to_runner={node_id: runner_id},
    )
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=shard_assignments,
        hosts_by_node={node_id: [Host(ip="0.0.0.0", port=58485)]},
        ephemeral_port=58485,
    )


@pytest.mark.asyncio
async def test_master_emits_tracing_state_changed_for_toggle_command() -> None:
    """The master should translate SetTracingEnabled into TracingStateChanged."""

    master, _node_id, command_sender, event_receiver = _build_master()
    event: Event | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("API"),
                command=SetTracingEnabled(enabled=True),
            )
        )
        event = await event_receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(event, TracingStateChanged)
    assert event.enabled is True


@pytest.mark.asyncio
async def test_master_new_text_tasks_inherit_cluster_tracing_state() -> None:
    """New text tasks should inherit the cluster tracing toggle."""

    master, node_id, command_sender, event_receiver = _build_master()
    instance = _single_node_instance(node_id)
    event: Event | None = None
    master.state = master.state.model_copy(
        update={
            "tracing_enabled": True,
            "instances": {instance.instance_id: instance},
        }
    )

    command = TextGeneration(
        command_id=CommandId("cmd-1"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id,
            input=[InputMessage(role="user", content="hello")],
        ),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )
        event = await event_receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(event, TaskCreated)
    assert isinstance(event.task, TextGenerationTask)
    assert event.task.trace_enabled is True
    assert master.command_task_mapping[command.command_id] == event.task_id
    assert master._expected_ranks[event.task_id] == {0}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_master_new_speech_tasks_inherit_cluster_tracing_state() -> None:
    """New speech tasks should inherit tracing and preserve the owning API node."""

    master, node_id, command_sender, event_receiver = _build_master()
    instance = _single_node_speech_instance(node_id)
    event: Event | None = None
    master.state = master.state.model_copy(
        update={
            "tracing_enabled": True,
            "instances": {instance.instance_id: instance},
        }
    )

    command = SpeechSynthesis(
        command_id=CommandId("speech-cmd-1"),
        owner_node=NodeId("api-node"),
        task_params=SpeechSynthesisTaskParams(
            model=instance.shard_assignments.model_id,
            input_text="hello",
            response_format=AudioResponseFormat.Wav,
        ),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )
        event = await event_receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(event, TaskCreated)
    assert isinstance(event.task, SpeechSynthesisTask)
    assert event.task.owner_node == NodeId("api-node")
    assert event.task.task_params == command.task_params
    assert event.task.trace_enabled is True
    assert master.command_task_mapping[command.command_id] == event.task_id
    assert master._expected_ranks[event.task_id] == {0}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_master_new_transcription_tasks_inherit_cluster_tracing_state() -> None:
    """New STT tasks should inherit tracing and preserve the owning API node."""

    master, node_id, command_sender, event_receiver = _build_master()
    instance = _single_node_transcription_instance(node_id)
    event: Event | None = None
    master.state = master.state.model_copy(
        update={
            "tracing_enabled": True,
            "instances": {instance.instance_id: instance},
        }
    )

    command = AudioTranscription(
        command_id=CommandId("transcription-cmd-1"),
        owner_node=NodeId("api-node"),
        task_params=AudioTranscriptionTaskParams(
            model=instance.shard_assignments.model_id,
            total_input_chunks=1,
            audio_sha256="abc123",
        ),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )
        event = await event_receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(event, TaskCreated)
    assert isinstance(event.task, AudioTranscriptionTask)
    assert event.task.owner_node == NodeId("api-node")
    assert event.task.task_params == command.task_params
    assert event.task.trace_enabled is True
    assert master.command_task_mapping[command.command_id] == event.task_id
    assert master._expected_ranks[event.task_id] == {0}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_master_pins_realtime_transcription_without_trace_expectation() -> None:
    """Realtime STT pins locally without leaking unsupported trace ranks."""

    master, node_id, command_sender, event_receiver = _build_master()
    instance = _single_node_transcription_instance(node_id)
    master.state = master.state.model_copy(
        update={
            "tracing_enabled": True,
            "instances": {instance.instance_id: instance},
        }
    )
    command = RealtimeAudioTranscription(
        command_id=CommandId("realtime-transcription-cmd-1"),
        owner_node=node_id,
        target_instance_id=instance.instance_id,
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=instance.shard_assignments.model_id,
            input_sample_rate=16000,
            transcription_delay_ms=480,
        ),
    )
    event: Event | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )
        event = await event_receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(event, TaskCreated)
    assert isinstance(event.task, RealtimeAudioTranscriptionTask)
    assert event.task.instance_id == instance.instance_id
    assert event.task.owner_node == node_id
    assert event.task.task_params == command.task_params
    assert event.task.trace_enabled is False
    assert master.command_task_mapping[command.command_id] == event.task_id
    assert event.task_id not in master._expected_ranks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_realtime_stt_rejects_target_not_hosted_by_owner_node() -> None:
    """Realtime PCM ingress cannot target a runner on a different node."""

    master, node_id, command_sender, event_receiver = _build_master()
    instance = _single_node_transcription_instance(node_id)
    master.state = master.state.model_copy(
        update={"instances": {instance.instance_id: instance}}
    )
    command = RealtimeAudioTranscription(
        command_id=CommandId("remote-realtime-transcription"),
        owner_node=NodeId("different-api-node"),
        target_instance_id=instance.instance_id,
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=instance.shard_assignments.model_id,
            input_sample_rate=16000,
            transcription_delay_ms=480,
        ),
    )
    event: Event | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )
        with anyio.move_on_after(0.1):
            event = await event_receiver.receive()
        task_group.cancel_scope.cancel()

    assert event is None
    assert command.command_id not in master.command_task_mapping
