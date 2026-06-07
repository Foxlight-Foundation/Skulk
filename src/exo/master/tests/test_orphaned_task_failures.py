"""Tests for the master failing API tasks stranded by a lost instance (#223).

A node death mid-generation deletes the instance but used to leave the task
in state forever — the API never received a terminal chunk and the client
request hung until its own timeout. ``orphaned_task_failure_events`` is the
master-side half of the fix: it declares in-flight API tasks dead when their
instance is gone (or being torn down in the same plan pass).
"""

from exo.master.main import orphaned_task_failure_events
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    LoadModel,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    MlxRingInstance,
    RunnerId,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def _instance(instance_id: InstanceId) -> Instance:
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=16,
        storage_size=Memory.from_bytes(1024),
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    node_id = NodeId("test-node")
    runner_id = RunnerId()
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    start_layer=0,
                    end_layer=16,
                    n_layers=16,
                    model_card=model_card,
                    device_rank=0,
                    world_size=1,
                )
            },
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=12345,
    )


def _text_task(
    task_id: TaskId,
    instance_id: InstanceId,
    status: TaskStatus = TaskStatus.Running,
) -> TextGenerationTask:
    return TextGenerationTask(
        task_id=task_id,
        instance_id=instance_id,
        task_status=status,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content="hi")],
        ),
    )


def _state(
    tasks: dict[TaskId, Task],
    instances: dict[InstanceId, Instance],
) -> State:
    return State().model_copy(update={"tasks": tasks, "instances": instances})


def test_healthy_task_not_failed() -> None:
    instance_id = InstanceId()
    task_id = TaskId()
    state = _state(
        {task_id: _text_task(task_id, instance_id)},
        {instance_id: _instance(instance_id)},
    )
    assert orphaned_task_failure_events(state, frozenset()) == []


def test_task_with_missing_instance_is_failed() -> None:
    instance_id = InstanceId()
    task_id = TaskId()
    state = _state({task_id: _text_task(task_id, instance_id)}, {})
    events = orphaned_task_failure_events(state, frozenset())
    assert len(events) == 1
    assert events[0].task_id == task_id
    assert events[0].error_type == "instance_lost"


def test_task_on_dying_instance_is_failed_same_pass() -> None:
    """An instance whose InstanceDeleted was emitted this pass is still in
    state (the event has not round-tripped through apply yet) — the dying-set
    parameter covers it."""
    instance_id = InstanceId()
    task_id = TaskId()
    state = _state(
        {task_id: _text_task(task_id, instance_id)},
        {instance_id: _instance(instance_id)},
    )
    events = orphaned_task_failure_events(state, {instance_id})
    assert len(events) == 1
    assert events[0].task_id == task_id


def test_terminal_task_not_failed_again() -> None:
    """Failed/Complete tasks are skipped, making re-emission across plan
    passes idempotent once the first TaskFailed applies."""
    instance_id = InstanceId()
    for terminal in (TaskStatus.Failed, TaskStatus.Complete, TaskStatus.Cancelled):
        task_id = TaskId()
        state = _state(
            {task_id: _text_task(task_id, instance_id, status=terminal)}, {}
        )
        assert orphaned_task_failure_events(state, frozenset()) == []


def test_worker_lifecycle_task_not_failed() -> None:
    """Worker-emitted lifecycle tasks are reconciled by the worker's own plan
    loop; the master must not fail them from here."""
    instance_id = InstanceId()
    task_id = TaskId()
    state = _state(
        {
            task_id: LoadModel(
                task_id=task_id,
                instance_id=instance_id,
                task_status=TaskStatus.Running,
            )
        },
        {},
    )
    assert orphaned_task_failure_events(state, frozenset()) == []
