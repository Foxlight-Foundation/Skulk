"""New execution variants round-trip through actual replicated unions as metadata."""

import pytest
from pydantic import TypeAdapter, ValidationError

from skulk.shared.apply import apply_task_created, apply_task_failed
from skulk.shared.types.audio_separation import (
    AudioSeparationTaskParams,
    SeparationSource,
)
from skulk.shared.types.commands import AudioSeparation, Command
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import Event, TaskCreated, TaskFailed
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation as SeparationTask
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.worker.instances import InstanceId


def command() -> AudioSeparation:
    """Provide only bounded measured source facts, never a local path or WAV."""
    return AudioSeparation(command_id=CommandId("separate-one"), owner_node=NodeId("api"),
        task_params=AudioSeparationTaskParams(model="adefossez/HTDemucs", total_input_chunks=1,
            source=SeparationSource(sha256="a" * 64, size_bytes=16044, frames=8000,
                sample_rate=8000, channels=1, bit_depth=16, encoding="pcm")))


def test_execution_command_and_task_survive_union_state_and_terminal_replay() -> None:
    """The actual event reducer preserves owner/source correlation and termination."""
    original = command()
    commands = TypeAdapter[Command](Command)
    assert commands.validate_json(commands.dump_json(original)) == original
    task = SeparationTask(command_id=original.command_id, owner_node=original.owner_node,
                          task_params=original.task_params, instance_id=InstanceId("native"))
    events = TypeAdapter[Event](Event)
    created = TaskCreated(task_id=task.task_id, task=task)
    restored = events.validate_json(events.dump_json(created))
    assert isinstance(restored, TaskCreated)
    state = apply_task_created(restored, State())
    state = State.model_validate_json(state.model_dump_json())
    assert state.tasks[task.task_id] == task
    failed = TaskFailed(task_id=task.task_id, error_type="native_failed", error_message="Native job failed")
    state = apply_task_failed(failed, state)
    retained = state.tasks[task.task_id]
    assert isinstance(retained, SeparationTask) and retained.task_status == TaskStatus.Failed
    assert retained.command_id == original.command_id and retained.owner_node == original.owner_node
    assert retained.task_params.source == original.task_params.source
    serialized = state.model_dump_json()
    assert "source_path" not in serialized and "output_dir" not in serialized and "RIFF" not in serialized


def test_execution_metadata_rejects_mutation_paths_and_encoded_audio() -> None:
    """No arbitrary native option/path or WAV can enter the new control records."""
    original = command()
    with pytest.raises(ValidationError, match="frozen"):
        original.owner_node = NodeId("elsewhere")
    for field, value in (("source_path", "/tmp/source.wav"), ("audio", "base64"), ("device", "cuda")):
        fields = original.task_params.model_dump()
        fields[field] = value
        with pytest.raises(ValidationError):
            AudioSeparationTaskParams.model_validate(fields)
    with pytest.raises(ValidationError):
        SeparationTask(command_id=original.command_id, owner_node=original.owner_node,
                       task_params=original.task_params, instance_id=InstanceId("native"),
                       task_id=TaskId("task"), error_message="x" * 1025)
