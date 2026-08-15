from datetime import datetime, timezone

import anyio
import pytest

from skulk.master.main import Master, instance_failure_event
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import (
    FailInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    InstanceDeleted,
    InstanceFailureRecorded,
    LocalForwarderEvent,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def _instance() -> MlxRingInstance:
    model_id = ModelId("org/model")
    runner_id = RunnerId("runner-a")
    card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(128),
        n_layers=4,
        hidden_size=256,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )
    return MlxRingInstance(
        instance_id=InstanceId("failed-instance"),
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={NodeId("node-a"): runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_instance_failure_event_captures_truth_before_teardown() -> None:
    recorded_at = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)

    event = instance_failure_event(
        _instance(),
        error_code="runner_crashed",
        error_message="runner exited while serving the model",
        recorded_at=recorded_at,
    )

    assert event.failure.instance_id == InstanceId("failed-instance")
    assert event.failure.model_id == ModelId("org/model")
    assert event.failure.affected_node_ids == [NodeId("node-a")]
    assert event.failure.recorded_at == recorded_at
    assert event.failure.error_code == "runner_crashed"


@pytest.mark.asyncio
async def test_fail_instance_records_cause_before_normal_teardown() -> None:
    """The master must never collapse a terminal failure into a clean stop."""
    node_id = NodeId(get_node_id_keypair().to_node_id())
    instance = _instance()
    command_sender, command_receiver = channel[ForwarderCommand]()
    event_sender, event_receiver = channel[Event]()
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, local_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
    )
    master.state = State(instances={instance.instance_id: instance})
    emitted: list[Event] = []

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            master._command_processor  # pyright: ignore[reportPrivateUsage] - integration boundary under test
        )
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("worker-a"),
                command=FailInstance(
                    instance_id=instance.instance_id,
                    error_code="runner_crashed",
                    error_message="rank 1 exited during generation",
                ),
            )
        )
        with anyio.fail_after(2):
            emitted = await event_receiver.receive_at_least(2)
        task_group.cancel_scope.cancel()

    failure_event = emitted[0]
    deletion_event = emitted[1]
    assert isinstance(failure_event, InstanceFailureRecorded)
    assert failure_event.failure.error_message == "rank 1 exited during generation"
    assert isinstance(deletion_event, InstanceDeleted)
    assert deletion_event.instance_id == instance.instance_id
