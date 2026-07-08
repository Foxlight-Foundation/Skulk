# pyright: reportPrivateUsage=false, reportMissingParameterType=false
"""Unit coverage for the single-node speech runner."""

import base64
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from anyio import WouldBlock

from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.shared.types.audio import SpeechSynthesisTaskParams
from skulk.shared.types.chunks import AudioChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import SpeechSynthesis, TaskStatus
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from skulk.worker.runner.speech import runner as speech_runner
from skulk.worker.runner.speech.runner import Runner, _filter_kwargs


class _CaptureSender:
    """Stand-in for the runner's MpSender that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, item: Event) -> None:
        self.events.append(item)


class _OneShotReceiver:
    """Stand-in MpReceiver that yields a fixed task list once, then stops."""

    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __enter__(self):
        return iter(self._items)

    def __exit__(self, *_args: object) -> bool:
        return False


class _EmptyCancelReceiver:
    """Stand-in cancellation receiver with no pending cancellations."""

    def receive_nowait(self) -> object:
        raise WouldBlock


@dataclass
class _FakeSpeechResult:
    audio: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    sample_rate: int = 24000


class _FakeSpeechModel:
    """Small fake with an mlx-audio-like ``generate`` method."""

    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, float | None, bool]] = []

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
    ) -> list[_FakeSpeechResult]:
        self.calls.append((text, voice, speed, stream))
        return [_FakeSpeechResult()]


def _make_runner() -> tuple[Runner, _CaptureSender]:
    sender = _CaptureSender()
    bound = SimpleNamespace(
        instance=SimpleNamespace(),
        bound_runner_id=RunnerId("speech-runner-1"),
        bound_shard=SimpleNamespace(
            world_size=1,
            model_card=SimpleNamespace(model_id=ModelId("mlx-community/kokoro-test")),
            device_rank=0,
        ),
        bound_node_id=NodeId("node-1"),
    )
    runner = Runner(
        bound_instance=cast(BoundInstance, cast(object, bound)),
        event_sender=cast("object", sender),  # pyright: ignore[reportArgumentType]
        task_receiver=cast("object", _OneShotReceiver([])),  # pyright: ignore[reportArgumentType]
        cancel_receiver=cast("object", _EmptyCancelReceiver()),  # pyright: ignore[reportArgumentType]
    )
    return runner, sender


def test_filter_kwargs_drops_unsupported_and_none_values() -> None:
    """Model-specific generation functions should receive only supported args."""

    def _generate(
        text: str, *, voice: str | None = None, stream: bool = False
    ) -> list[_FakeSpeechResult]:
        del text, voice, stream
        return [_FakeSpeechResult()]

    assert _filter_kwargs(
        _generate,
        {
            "voice": "af_heart",
            "speed": 1.1,
            "stream": False,
            "reference_audio": None,
        },
    ) == {"voice": "af_heart", "stream": False}


def test_speech_synthesis_emits_audio_chunk_and_active_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TTS task should generate one terminal AudioChunk through the data plane."""

    runner, sender = _make_runner()
    model = _FakeSpeechModel()
    runner.model = model
    runner.current_status = RunnerReady()

    def _fake_encode(
        audio: np.ndarray,
        sample_rate: int,
        response_format: AudioResponseFormat,
    ) -> bytes:
        assert audio.tolist() == [0.1, 0.2, 0.3]
        assert sample_rate == 24000
        assert response_format == AudioResponseFormat.Wav
        return b"WAVDATA"

    monkeypatch.setattr(speech_runner, "_encode_audio", _fake_encode)

    command_id = CommandId("speech-command-1")
    task = SpeechSynthesis(
        instance_id=InstanceId("speech-instance-1"),
        command_id=command_id,
        task_params=SpeechSynthesisTaskParams(
            model=ModelId("mlx-community/kokoro-test"),
            input_text="hello world",
            response_format=AudioResponseFormat.Wav,
            voice="af_heart",
            speed=1.1,
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]

    runner.main()

    assert model.calls == [("hello world", "af_heart", 1.1, False)]
    assert any(
        isinstance(event, TaskAcknowledged) and event.task_id == task.task_id
        for event in sender.events
    )

    complete_index = next(
        index
        for index, event in enumerate(sender.events)
        if isinstance(event, TaskStatusUpdated)
        and event.task_status == TaskStatus.Complete
    )
    prior_statuses = [
        event.runner_status
        for event in sender.events[:complete_index]
        if isinstance(event, RunnerStatusUpdated)
    ]
    assert prior_statuses
    assert isinstance(prior_statuses[-1], RunnerRunning)

    generated: list[tuple[CommandId, AudioChunk]] = []
    for event in sender.events:
        if isinstance(event, ChunkGenerated) and isinstance(event.chunk, AudioChunk):
            generated.append((event.command_id, event.chunk))

    assert len(generated) == 1
    generated_command_id, chunk = generated[0]
    assert generated_command_id == command_id
    assert base64.b64decode(chunk.data.encode("ascii"), validate=True) == b"WAVDATA"
    assert chunk.format == AudioResponseFormat.Wav
    assert chunk.sample_rate == 24000
    assert chunk.finish_reason == "stop"
