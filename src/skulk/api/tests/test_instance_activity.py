"""Request activity per instance comes from task events, never from lifecycle work."""

from collections.abc import Mapping

from skulk.api.instance_activity import InstanceActivity, is_request
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import CommandId
from skulk.shared.types.events import (
    InstanceDeleted,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import Shutdown, Task, TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId


class Clock:
    """A settable wall clock in UTC seconds."""

    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def request(instance_id: InstanceId, status: TaskStatus) -> TextGeneration:
    return TextGeneration(
        command_id=CommandId(),
        instance_id=instance_id,
        task_status=status,
        task_params=TextGenerationTaskParams(
            model=ModelId("example/large"),
            input=[InputMessage(role="user", content="hi")],
        ),
    )


def test_requests_count_and_move_the_clock_but_lifecycle_work_does_not() -> None:
    clock = Clock()
    activity = InstanceActivity(clock)
    instance = InstanceId()
    before = activity.read(instance, {})
    assert before.in_flight == 0 and before.last_activity_at is None
    assert before.observed_since == 1_800_000_000

    shutdown = Shutdown(instance_id=instance, runner_id=RunnerId())
    assert not is_request(shutdown)
    clock.now += 60
    activity.observe(
        TaskCreated(task_id=shutdown.task_id, task=shutdown),
        {},
        {shutdown.task_id: shutdown},
    )
    assert (
        activity.read(instance, {shutdown.task_id: shutdown}).last_activity_at is None
    )

    running = request(instance, TaskStatus.Running)
    tasks: Mapping[TaskId, Task] = {running.task_id: running}
    clock.now += 60
    activity.observe(TaskCreated(task_id=running.task_id, task=running), {}, tasks)
    reading = activity.read(instance, tasks)
    assert reading.in_flight == 1 and reading.last_activity_at == 1_800_000_120

    done = running.model_copy(update={"task_status": TaskStatus.Complete})
    finished: Mapping[TaskId, Task] = {running.task_id: done}
    clock.now += 30
    activity.observe(
        TaskStatusUpdated(task_id=running.task_id, task_status=TaskStatus.Complete),
        tasks,
        finished,
    )
    reading = activity.read(instance, finished)
    assert reading.in_flight == 0 and reading.last_activity_at == 1_800_000_150

    clock.now += 30
    activity.observe(TaskDeleted(task_id=running.task_id), finished, {})
    assert activity.read(instance, {}).last_activity_at == 1_800_000_180
    # Another instance's requests are not this one's.
    assert activity.read(InstanceId(), {}).last_activity_at is None


def test_a_deleted_instance_leaves_no_history() -> None:
    clock = Clock()
    activity = InstanceActivity(clock)
    instance = InstanceId()
    running = request(instance, TaskStatus.Pending)
    activity.observe(
        TaskCreated(task_id=running.task_id, task=running),
        {},
        {running.task_id: running},
    )
    activity.observe(InstanceDeleted(instance_id=instance), {}, {})
    assert activity.read(instance, {}).last_activity_at is None
