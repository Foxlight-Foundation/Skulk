# pyright: reportPrivateUsage=false
"""Worker cleanup coverage for speech input upload buffers."""

from skulk.shared.types.audio import AudioTranscriptionTaskParams
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.events import TaskDeleted, TaskStatusUpdated
from skulk.shared.types.tasks import (
    AudioTranscription,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.main import _audio_input_cleanup_command_id


def _transcription_task(command_id: CommandId) -> AudioTranscription:
    return AudioTranscription(
        task_id=TaskId("audio-task"),
        instance_id=InstanceId("audio-instance"),
        task_status=TaskStatus.Running,
        command_id=command_id,
        task_params=AudioTranscriptionTaskParams(
            model=ModelId("mlx-community/whisper-test"),
            total_input_chunks=1,
            audio_sha256="abc123",
        ),
    )


def test_audio_input_cleanup_follows_terminal_transcription_status() -> None:
    """Any worker can release upload chunks once the STT task is terminal."""

    command_id = CommandId("audio-command")
    task = _transcription_task(command_id)

    assert (
        _audio_input_cleanup_command_id(
            TaskStatusUpdated(
                task_id=task.task_id,
                task_status=TaskStatus.Complete,
            ),
            {task.task_id: task},
            {
                task.task_id: task.model_copy(
                    update={"task_status": TaskStatus.Complete}
                )
            },
        )
        == command_id
    )


def test_audio_input_cleanup_follows_transcription_task_delete() -> None:
    """TaskDeleted still has the command id in the pre-apply task map."""

    command_id = CommandId("audio-command")
    task = _transcription_task(command_id)

    assert (
        _audio_input_cleanup_command_id(
            TaskDeleted(task_id=task.task_id),
            {task.task_id: task},
            {},
        )
        == command_id
    )


def test_audio_input_cleanup_ignores_non_transcription_tasks() -> None:
    """Text tasks must not clear unrelated audio upload buffers."""

    task_id = TaskId("text-task")
    task = TextGeneration(
        task_id=task_id,
        instance_id=InstanceId("text-instance"),
        task_status=TaskStatus.Running,
        command_id=CommandId("text-command"),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/text-test"),
            input=[InputMessage(role="user", content="hi")],
        ),
    )

    assert (
        _audio_input_cleanup_command_id(
            TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Complete),
            {task_id: task},
            {task_id: task.model_copy(update={"task_status": TaskStatus.Complete})},
        )
        is None
    )
