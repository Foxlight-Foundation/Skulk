from datetime import datetime, timezone
from typing import cast

import anyio
import pytest
from pydantic import ValidationError

from skulk.master.main import (
    Master,
    dead_node_instance_failure_events,
    instance_failure_event,
)
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
from skulk.shared.types.worker.instances import (
    InstanceFailure,
    InstanceId,
    MlxRingInstance,
)
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
    assert event.failure.affected_node_ids == (NodeId("node-a"),)
    assert event.failure.recorded_at == recorded_at
    assert event.failure.error_code == "runner_crashed"


def test_instance_failure_event_bounds_large_placement_node_truth() -> None:
    """Diagnostic bounds must never make a valid large placement impossible to tear down."""
    instance = _instance()
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    node_to_runner = {
        NodeId(f"node-{index:03d}"): RunnerId(f"runner-{index:03d}")
        for index in range(70)
    }
    large_instance = instance.model_copy(
        update={
            "shard_assignments": ShardAssignments(
                model_id=instance.shard_assignments.model_id,
                node_to_runner=node_to_runner,
                runner_to_shard={runner_id: shard for runner_id in node_to_runner.values()},
            )
        }
    )

    event = instance_failure_event(
        large_instance,
        error_code="runner_crashed",
        error_message="runner exited while serving the model",
    )

    assert len(event.failure.affected_node_ids) == 64
    assert event.failure.affected_node_ids[0] == NodeId("node-000")
    assert event.failure.affected_node_ids[-1] == NodeId("node-063")


def test_timed_out_node_records_failure_while_still_in_topology() -> None:
    """Timeout truth must not depend on the stale topology edge disappearing."""
    instance = _instance()

    events = dead_node_instance_failure_events(
        State(instances={instance.instance_id: instance}),
        connected_node_ids={NodeId("node-a")},
        timed_out_node_ids={NodeId("node-a")},
    )

    assert len(events) == 1
    assert events[0].failure.instance_id == instance.instance_id
    assert events[0].failure.error_code == "node_unavailable"
    assert "timed out" in events[0].failure.error_message


def test_unavailable_node_identifier_cannot_overflow_failure_truth() -> None:
    """Untrusted explicit placement ids must not crash terminal teardown."""
    instance = _instance()
    oversized_node_id = NodeId("node-" + "x" * 4096)
    runner_id = next(iter(instance.shard_assignments.runner_to_shard))
    instance = instance.model_copy(
        update={
            "shard_assignments": instance.shard_assignments.model_copy(
                update={"node_to_runner": {oversized_node_id: runner_id}}
            )
        }
    )

    events = dead_node_instance_failure_events(
        State(instances={instance.instance_id: instance}),
        connected_node_ids=set(),
        timed_out_node_ids=set(),
    )

    assert len(events) == 1
    assert len(events[0].failure.error_message) == 2048
    assert events[0].failure.error_message.endswith(" left the live topology.")
    assert events[0].failure.affected_node_ids == (
        NodeId(
            "sha256:8fce4af2210bc59978059c60615ce8cbf07b835621d93ab544c6f7e01eb0867a"
        ),
    )
    assert len(events[0].model_dump_json()) < 4096


def test_retained_failure_bounds_all_operator_controlled_identities() -> None:
    """Oversized explicit placement identities cannot inflate replicated state."""
    failure = InstanceFailure(
        instance_id=InstanceId("instance-" + "i" * 4096),
        model_id=ModelId("model-" + "m" * 4096),
        error_code="placement_failed",
        error_message="placement failed",
        affected_node_ids=(NodeId("node-a"),),
        recorded_at=datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc),
    )

    assert str(failure.instance_id).startswith("sha256:")
    assert str(failure.model_id).startswith("sha256:")
    assert len(str(failure.instance_id)) == 71
    assert len(str(failure.model_id)) == 71
    assert failure.affected_node_ids == (NodeId("node-a"),)
    assert len(failure.model_dump_json()) < 1024


def test_retained_failure_rejects_non_string_node_identities() -> None:
    """Corrupt replicated values must fail instead of becoming invented node ids."""
    with pytest.raises(
        ValidationError, match="affected_node_ids entries must be strings"
    ):
        InstanceFailure.model_validate(
            {
                "instanceId": "instance-a",
                "modelId": "org/model",
                "errorCode": "placement_failed",
                "errorMessage": "placement failed",
                "affectedNodeIds": [42],
                "recordedAt": "2026-08-15T14:00:00+00:00",
            }
        )


def test_terminal_failure_command_and_event_are_immutable() -> None:
    """Authoritative failure identity cannot change after queue admission."""
    command = FailInstance(
        instance_id=InstanceId("failed-instance"),
        error_code="runner_crashed",
        error_message="runner crashed repeatedly",
    )
    event = instance_failure_event(
        _instance(),
        error_code="runner_crashed",
        error_message="runner crashed repeatedly",
    )

    with pytest.raises(ValidationError):
        command.error_message = "rewritten"
    with pytest.raises(ValidationError):
        event.failure = event.failure.model_copy()
    with pytest.raises(AttributeError):
        cast("list[NodeId]", cast(object, event.failure.affected_node_ids)).append(
            NodeId("node-b")
        )


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
