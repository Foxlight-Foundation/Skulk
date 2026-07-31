# pyright: reportPrivateUsage=false, reportAny=false
"""Tests for the API terminating open command streams on TaskFailed (#223).

The API half of the node-death fix: when the master declares a task dead,
the per-command chunk queue must receive a terminal ErrorChunk so streaming
responses close with an error and non-streaming handlers raise — instead of
the HTTP connection hanging until the client's own timeout.
"""

from typing import Any

import pytest

from skulk.api.main import API
from skulk.shared.types.audio import RealtimeAudioTranscriptionTaskParams
from skulk.shared.types.chunks import ErrorChunk
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.state import State
from skulk.shared.types.tasks import RealtimeAudioTranscription, TaskId, TaskStatus
from skulk.shared.types.tasks import TextGeneration as TextGenerationTask
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import channel


def _make_api() -> Any:
    api = object.__new__(API)
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._embedding_queues = {}
    api._audio_speech_queues = {}
    api._audio_transcription_queues = {}
    api._pending_stream_failures = {}
    api.node_id = NodeId("api-node")
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


async def test_realtime_task_failure_does_not_block_on_full_queue() -> None:
    """A stalled realtime consumer cannot block the API event apply loop."""

    api = _make_api()
    command_id = CommandId("realtime-failure-full")
    task = RealtimeAudioTranscription(
        task_id=TaskId("realtime-task"),
        instance_id=InstanceId(),
        command_id=command_id,
        owner_node=NodeId("api-node"),
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=ModelId("mlx-community/voxtral-realtime-test"),
            input_sample_rate=16000,
        ),
    )
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})
    sender, receiver = channel[Any](1)
    sender.send_nowait(object())
    api._audio_transcription_queues[command_id] = sender
    cancelled: list[CommandId] = []

    async def cancel(target: CommandId) -> None:
        cancelled.append(target)

    api._cancel_audio_transcription_command = cancel

    await api._terminate_command_stream(task.task_id, "runner exited")

    assert cancelled == [command_id]
    assert command_id not in api._audio_transcription_queues
    receiver.close()


async def test_session_reset_fails_open_streams() -> None:
    """Master failover starts a new session that cannot carry the old
    session's tasks; reset() used to replace the queue maps without closing
    the old senders, leaving open requests unreachable by dispatch, cancel,
    and the orphaned-task sweep — a guaranteed hang (#223 drill 2)."""
    from anyio import EndOfStream

    api = _make_api()
    command_id = CommandId()
    sender, receiver = channel[Any]()
    api._text_generation_queues[command_id] = sender

    api._fail_open_command_streams_for_session_reset()

    chunk = receiver.receive_nowait()
    assert isinstance(chunk, ErrorChunk)
    assert "session changed" in chunk.error_message
    with pytest.raises(EndOfStream):
        receiver.receive_nowait()


async def test_session_reset_with_already_closed_queue_is_silent() -> None:
    api = _make_api()
    command_id = CommandId()
    sender, receiver = channel[Any]()
    receiver.close()
    sender.close()
    api._text_generation_queues[command_id] = sender
    # Must not raise despite the dead channel.
    api._fail_open_command_streams_for_session_reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


async def test_owned_task_without_queue_buffers_terminal_chunk() -> None:
    """A fast local TaskFailed that beats queue registration must be buffered
    on the owning API so the late-registering stream still terminates."""
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id).model_copy(
        update={"owner_node": NodeId("api-node")}
    )
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    await api._terminate_command_stream(task.task_id, "instance gone")

    buffered = api._pending_stream_failures[command_id]
    assert isinstance(buffered, ErrorChunk)
    assert buffered.error_message == "instance gone"
    assert buffered.model == ModelId("test-model")


async def test_remote_owned_task_failure_is_not_buffered() -> None:
    """Only the owning API can ever register the command's queue; other nodes
    must not accumulate entries no consumer will drain."""
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id).model_copy(
        update={"owner_node": NodeId("some-other-node")}
    )
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    await api._terminate_command_stream(task.task_id, "instance gone")

    assert api._pending_stream_failures == {}


async def test_ownerless_task_failure_buffers_on_any_api() -> None:
    """owner_node None (legacy gossip fan-out) means any API may serve the
    stream, so every API buffers."""
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id)
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    await api._terminate_command_stream(task.task_id, "instance gone")

    assert command_id in api._pending_stream_failures


async def test_delivered_failure_is_not_also_buffered() -> None:
    api = _make_api()
    command_id = CommandId()
    task = _failed_task(command_id).model_copy(
        update={"owner_node": NodeId("api-node")}
    )
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})
    sender, receiver = channel[Any]()
    api._text_generation_queues[command_id] = sender

    await api._terminate_command_stream(task.task_id, "instance gone")

    assert isinstance(receiver.receive_nowait(), ErrorChunk)
    assert api._pending_stream_failures == {}


async def test_pending_failure_buffer_evicts_oldest_at_capacity() -> None:
    api = _make_api()
    stale = ErrorChunk(model=ModelId("m"), error_message="stale")
    for index in range(256):
        api._pending_stream_failures[CommandId(f"old-{index}")] = stale
    command_id = CommandId()
    task = _failed_task(command_id).model_copy(
        update={"owner_node": NodeId("api-node")}
    )
    api.state = State().model_copy(update={"tasks": {task.task_id: task}})

    await api._terminate_command_stream(task.task_id, "instance gone")

    assert len(api._pending_stream_failures) == 256
    assert CommandId("old-0") not in api._pending_stream_failures
    assert command_id in api._pending_stream_failures


async def test_token_stream_registration_drains_buffered_failure() -> None:
    """The real text-stream drain: a stream registering after its buffered
    failure yields exactly the terminal ErrorChunk and consumes the entry."""
    api = _make_api()
    command_id = CommandId()
    buffered = ErrorChunk(model=ModelId("test-model"), error_message="instance gone")
    api._pending_stream_failures[command_id] = buffered
    def no_vision_failure(_command_id: CommandId) -> None:
        return None

    api._take_vision_media_failure = no_vision_failure

    async def finalize(*_args: Any, **_kwargs: Any) -> None:
        return None

    api._finalize_command_stream = finalize

    chunks = [chunk async for chunk in api._token_chunk_stream(command_id)]

    assert chunks == [buffered]
    assert command_id not in api._pending_stream_failures
