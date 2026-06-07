# pyright: reportPrivateUsage=false, reportAny=false
"""Tests for the API terminating open command streams on TaskFailed (#223).

The API half of the node-death fix: when the master declares a task dead,
the per-command chunk queue must receive a terminal ErrorChunk so streaming
responses close with an error and non-streaming handlers raise — instead of
the HTTP connection hanging until the client's own timeout.
"""

from typing import Any

import pytest

from exo.api.main import API
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import channel


def _make_api() -> Any:
    api = object.__new__(API)
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._embedding_queues = {}
    return api


def _failed_task(command_id: CommandId) -> TextGenerationTask:
    return TextGenerationTask(
        task_id=TaskId(),
        instance_id=InstanceId(),
        task_status=TaskStatus.Failed,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content="hi")],
        ),
        error_type="instance_lost",
        error_message="instance gone",
    )


async def test_task_failed_delivers_error_chunk() -> None:
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id)
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    sender, receiver = channel[Any]()
    api._text_generation_queues[command_id] = sender

    await api._terminate_command_stream(task.task_id, "instance gone")

    chunk = receiver.receive_nowait()
    assert isinstance(chunk, ErrorChunk)
    assert chunk.finish_reason == "error"
    assert chunk.error_message == "instance gone"
    assert chunk.model == ModelId("test-model")


async def test_task_failed_for_unknown_task_is_ignored() -> None:
    api = _make_api()
    api.state = State()
    # No queues registered, no task in state — must not raise.
    await api._terminate_command_stream(TaskId(), "y")


async def test_cancelled_status_delivers_error_chunk() -> None:
    """Operator instance deletion cancels in-flight tasks via
    TaskStatusUpdated(Cancelled); those requests must terminate too (#224
    review catch)."""
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id).model_copy(
        update={"task_status": TaskStatus.Cancelled}
    )
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    sender, receiver = channel[Any]()
    api._text_generation_queues[command_id] = sender

    await api._terminate_command_stream(
        task.task_id, "The request was cancelled because its instance was deleted"
    )

    chunk = receiver.receive_nowait()
    assert isinstance(chunk, ErrorChunk)
    assert "cancelled" in chunk.error_message


async def test_task_failed_with_closed_queue_drops_entry() -> None:
    """A request that disconnected concurrently leaves a broken queue; the
    handler must drop it rather than raise into the event-apply loop."""
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id)
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    sender, receiver = channel[Any]()
    receiver.close()
    api._text_generation_queues[command_id] = sender

    await api._terminate_command_stream(task.task_id, "y")
    assert command_id not in api._text_generation_queues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
