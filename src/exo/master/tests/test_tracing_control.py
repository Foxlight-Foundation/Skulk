"""Tests for master-side tracing control and task inheritance."""

import anyio
import pytest

from exo.master.main import Master
from exo.routing.router import get_node_id_keypair
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SetTracingEnabled,
    TextGeneration,
)
from exo.shared.types.common import CommandId, Host, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskCreated,
    TracingStateChanged,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state_sync import StateSyncMessage
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, Sender, channel


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
