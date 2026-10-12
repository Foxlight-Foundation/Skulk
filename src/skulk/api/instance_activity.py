"""Per-instance request activity, derived from the replicated task events.

Every node applies the same task events, so every API reads the same work in
flight. Times are this API's own receipt times, never a sender's clock. A
burst provider uses this read to release a rented machine once its model has
been idle long enough; nothing here acts on it.
"""

import time
from collections.abc import Callable, Mapping
from typing import final

from pydantic import Field

from skulk.shared.types.events import (
    Event,
    InstanceDeleted,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import (
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    InstallEngine,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.pydantic_ext import FrozenModel

_LIFECYCLE_TASKS = (
    CreateRunner,
    DownloadModel,
    InstallEngine,
    ConnectToGroup,
    LoadModel,
    StartWarmup,
    CancelTask,
    Shutdown,
)
_OPEN = frozenset((TaskStatus.Pending, TaskStatus.Running))


def is_request(task: Task) -> bool:
    """Whether a task is work a caller asked for, not the instance's own lifecycle."""
    return not isinstance(task, _LIFECYCLE_TASKS)


@final
class InstanceActivityReading(FrozenModel):
    """Requests in flight on one instance and when this API last saw one move."""

    instance_id: InstanceId = Field(description="The instance read.")
    in_flight: int = Field(
        ge=0, description="Requests created and not yet finished on the instance."
    )
    last_activity_at: int | None = Field(
        default=None,
        description=(
            "UTC Unix seconds when this API last saw a request start or finish "
            "on the instance; null when it has seen none since it started."
        ),
    )
    observed_since: int = Field(
        description=(
            "UTC Unix seconds since which this API has been watching; an idle "
            "period cannot be known to have started earlier."
        )
    )
    read_at: int = Field(description="UTC Unix seconds of this read.")


@final
class InstanceActivity:
    """Track the last request start or finish per instance, bounded by live instances."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        """Start watching now; earlier activity is unknown to this API."""
        self._clock = clock
        self._started = int(clock())
        self._last: dict[InstanceId, int] = {}

    def observe(
        self,
        event: Event,
        previous_tasks: Mapping[TaskId, Task],
        tasks: Mapping[TaskId, Task],
    ) -> None:
        """Record one applied event; ``previous_tasks`` is the state before it."""
        if isinstance(event, InstanceDeleted):
            self._last.pop(event.instance_id, None)
            return
        task: Task | None = None
        if isinstance(event, TaskCreated):
            task = event.task
        elif isinstance(event, TaskDeleted):
            task = previous_tasks.get(event.task_id)
        elif isinstance(event, (TaskStatusUpdated, TaskFailed)):
            task = tasks.get(event.task_id) or previous_tasks.get(event.task_id)
        if task is not None and is_request(task):
            self._last[task.instance_id] = int(self._clock())

    def read(
        self, instance_id: InstanceId, tasks: Mapping[TaskId, Task]
    ) -> InstanceActivityReading:
        """Count open requests from current state and report the last movement."""
        return InstanceActivityReading(
            instance_id=instance_id,
            in_flight=sum(
                1
                for task in tasks.values()
                if task.instance_id == instance_id
                and is_request(task)
                and task.task_status in _OPEN
            ),
            last_activity_at=self._last.get(instance_id),
            observed_since=self._started,
            read_at=int(self._clock()),
        )
