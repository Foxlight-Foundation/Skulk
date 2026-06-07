"""Tests for the master failing API tasks stranded by a lost instance (#223).

A node death mid-generation deletes the instance but used to leave the task
in state forever — the API never received a terminal chunk and the client
request hung until its own timeout. ``orphaned_task_failure_events`` is the
master-side half of the fix: it declares in-flight API tasks dead when their
instance is gone (or being torn down in the same plan pass).
"""

from skulk.master.main import instances_on_dead_nodes, orphaned_task_failure_events
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    LoadModel,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    MlxRingInstance,
    RunnerId,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata


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


def test_instance_on_timed_out_node_is_dying() -> None:
    """A node can exceed its last_seen timeout while still present in
    topology (wedged process, live TCP). NodeTimedOut removes the node's
    instances AND their tasks in one apply, so the sweep must treat those
    instances as dying in the same pass — a later pass can no longer see
    the tasks (#224 review catch)."""
    instance_id = InstanceId()
    node_id = NodeId("test-node")  # matches _instance's node
    state = _state({}, {instance_id: _instance(instance_id)})

    # connected and not timed out -> healthy
    assert (
        instances_on_dead_nodes(state, frozenset({node_id}), frozenset()) == set()
    )
    # connected but timed out -> dying
    assert instances_on_dead_nodes(
        state, frozenset({node_id}), frozenset({node_id})
    ) == {instance_id}
    # disconnected -> dying
    assert instances_on_dead_nodes(state, frozenset(), frozenset()) == {instance_id}


def test_sweep_is_idempotent_after_apply() -> None:
    """Applying the emitted TaskFailed must silence the next plan pass.

    Review catch on #224: apply_task_failed originally stored only the error
    fields without flipping task_status, so a task whose API never sent
    TaskFinished (e.g. the API restarted) was re-failed every 10 seconds
    forever — unbounded event-log growth.
    """
    from skulk.shared.apply import apply_task_failed

    instance_id = InstanceId()
    task_id = TaskId()
    state = _state({task_id: _text_task(task_id, instance_id)}, {})

    events = orphaned_task_failure_events(state, frozenset())
    assert len(events) == 1

    state_after = apply_task_failed(events[0], state)
    failed_task = state_after.tasks[task_id]
    assert isinstance(failed_task, TextGenerationTask)
    assert failed_task.task_status == TaskStatus.Failed
    assert failed_task.error_type == "instance_lost"
    assert orphaned_task_failure_events(state_after, frozenset()) == []


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
