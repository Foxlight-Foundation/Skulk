# pyright: reportPrivateUsage=false, reportMissingParameterType=false
"""Unit coverage for the single-node speech runner."""

import base64
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from anyio import WouldBlock

from skulk.shared.models.model_cards import AudioCardKind, AudioResponseFormat, ModelId
from skulk.shared.types.audio import (
    AudioTranscriptionTaskParams,
    SpeechSynthesisTaskParams,
)
from skulk.shared.types.chunks import AudioChunk, TranscriptionChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import AudioTranscription, SpeechSynthesis, TaskStatus
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from skulk.worker.runner.speech import runner as speech_runner
from skulk.worker.runner.speech.runner import (
    Runner,
    _filter_kwargs,
    _load_speech_model,
    _resolve_staged_voice_path,
)


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


class _FakeTranscriptionModel:
    """Small fake with an STT-style ``generate`` method."""

    def __init__(self, expected_audio: bytes) -> None:
        self.expected_audio = expected_audio
        self.calls: list[tuple[str, str | None, bool, bool]] = []

    def generate(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        stream: bool = False,
        verbose: bool = False,
    ) -> dict[str, object]:
        assert Path(audio_path).read_bytes() == self.expected_audio
        self.calls.append((Path(audio_path).suffix, language, stream, verbose))
        return {
            "text": "hello world",
            "language": language or "en",
            "segments": [
                {"id": 0, "text": "hello world", "start": 0.0, "end": 1.2}
            ],
        }


class _FakeWhisperTranscriptionModel:
    """Fake Whisper-like model with explicit aliases and loose decode options."""

    def __init__(self, expected_audio: bytes) -> None:
        self.expected_audio = expected_audio
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        audio_path: str,
        *,
        verbose: bool | None = None,
        language: str | None = None,
        chunk_duration: float = 1.0,
        stream: bool = False,
        temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        initial_prompt: str | None = None,
        return_timestamps: bool = True,
        word_timestamps: bool = False,
        **decode_options: object,
    ) -> dict[str, object]:
        assert Path(audio_path).read_bytes() == self.expected_audio
        self.calls.append(
            {
                "language": language,
                "chunk_duration": chunk_duration,
                "stream": stream,
                "temperature": temperature,
                "initial_prompt": initial_prompt,
                "return_timestamps": return_timestamps,
                "word_timestamps": word_timestamps,
                "verbose": verbose,
                "decode_options": decode_options,
            }
        )
        return {"text": "hello world", "language": language or "en"}


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


def _stub_mlx_audio_loader(
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    load_model: object,
) -> None:
    """Install an importable ``mlx_audio.<category>.utils`` stub."""
    root = ModuleType("mlx_audio")
    root.__dict__["__path__"] = []
    category_module = ModuleType(f"mlx_audio.{category}")
    category_module.__dict__["__path__"] = []
    utils_module = ModuleType(f"mlx_audio.{category}.utils")
    utils_module.__dict__["load_model"] = load_model

    monkeypatch.setitem(sys.modules, "mlx_audio", root)
    monkeypatch.setitem(sys.modules, f"mlx_audio.{category}", category_module)
    monkeypatch.setitem(sys.modules, f"mlx_audio.{category}.utils", utils_module)


def test_load_speech_model_uses_stt_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """STT cards should bypass upstream generic TTS/STT name inference."""
    calls: list[Path] = []

    def _fake_load_model(model_path: Path) -> object:
        calls.append(model_path)
        return "stt-model"

    _stub_mlx_audio_loader(monkeypatch, "stt", _fake_load_model)

    assert _load_speech_model(tmp_path, AudioCardKind.SpeechToText) == "stt-model"
    assert calls == [tmp_path]


def test_load_speech_model_uses_tts_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TTS cards should use the TTS loader directly."""
    calls: list[Path] = []

    def _fake_load_model(model_path: Path) -> object:
        calls.append(model_path)
        return "tts-model"

    _stub_mlx_audio_loader(monkeypatch, "tts", _fake_load_model)

    assert _load_speech_model(tmp_path, AudioCardKind.TextToSpeech) == "tts-model"
    assert calls == [tmp_path]


def test_resolve_staged_voice_path_uses_default_voice_from_model_store(
    tmp_path: Path,
) -> None:
    """Default Kokoro voice requests should stay inside the staged model."""

    voice_path = tmp_path / "voices" / "af_heart.safetensors"
    voice_path.parent.mkdir()
    voice_path.write_bytes(b"voice")

    assert _resolve_staged_voice_path(tmp_path, None) == str(voice_path)


def test_resolve_staged_voice_path_requires_named_staged_voice(
    tmp_path: Path,
) -> None:
    """Named Kokoro voices should fail locally instead of fetching elsewhere."""

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="bf_emma"):
        _resolve_staged_voice_path(tmp_path, "bf_emma")


def test_resolve_staged_voice_path_keeps_safetensors_inside_store(
    tmp_path: Path,
) -> None:
    """Explicit voice filenames should resolve under the staged voices dir."""

    voice_path = tmp_path / "voices" / "af_heart.safetensors"
    voice_path.parent.mkdir()
    voice_path.write_bytes(b"voice")

    assert _resolve_staged_voice_path(tmp_path, "af_heart.safetensors") == str(
        voice_path
    )


def test_resolve_staged_voice_path_rejects_path_components(tmp_path: Path) -> None:
    """Voice requests must not escape the staged voices directory."""

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    with pytest.raises(ValueError, match="path components"):
        _resolve_staged_voice_path(tmp_path, "../af_heart.safetensors")


def test_resolve_staged_voice_path_rejects_symlink_escape(tmp_path: Path) -> None:
    """Resolved voice paths must remain under the staged voices directory."""

    outside_voice = tmp_path / "outside.safetensors"
    outside_voice.write_bytes(b"voice")
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "escape.safetensors").symlink_to(outside_voice)

    with pytest.raises(ValueError, match="under voices"):
        _resolve_staged_voice_path(tmp_path, "escape")


def test_resolve_staged_voice_path_rejects_voices_dir_symlink_escape(
    tmp_path: Path,
) -> None:
    """The staged voices directory itself must not point outside the model."""

    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_dir.mkdir()
    (tmp_path / "voices").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="voices directory"):
        _resolve_staged_voice_path(tmp_path, "af_heart")


def test_resolve_staged_voice_path_requires_regular_file(tmp_path: Path) -> None:
    """Directories under voices/ are not valid staged voice assets."""

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "af_heart.safetensors").mkdir()

    with pytest.raises(FileNotFoundError, match="regular voice file"):
        _resolve_staged_voice_path(tmp_path, "af_heart")


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


def test_streaming_speech_synthesis_emits_partial_and_terminal_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming TTS should emit generated audio segments before final completion."""

    class _StreamingSpeechModel:
        sample_rate = 24000

        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None, bool, float | None]] = []

        def generate(
            self,
            text: str,
            *,
            voice: str | None = None,
            stream: bool = False,
            streaming_interval: float | None = None,
        ) -> list[_FakeSpeechResult]:
            self.calls.append((text, voice, stream, streaming_interval))
            return [
                _FakeSpeechResult(audio=[0.1]),
                _FakeSpeechResult(audio=[0.2]),
            ]

    runner, sender = _make_runner()
    model = _StreamingSpeechModel()
    runner.model = model
    runner.current_status = RunnerReady()
    encoded_calls: list[list[float]] = []

    def _fake_encode(
        audio: np.ndarray,
        sample_rate: int,
        response_format: AudioResponseFormat,
    ) -> bytes:
        assert sample_rate == 24000
        assert response_format == AudioResponseFormat.Mp3
        encoded_calls.append(cast(list[float], audio.tolist()))
        return f"mp3-{len(encoded_calls)}".encode("ascii")

    monkeypatch.setattr(speech_runner, "_encode_audio", _fake_encode)

    command_id = CommandId("speech-command-stream")
    task = SpeechSynthesis(
        instance_id=InstanceId("speech-instance-1"),
        command_id=command_id,
        task_params=SpeechSynthesisTaskParams(
            model=ModelId("mlx-community/fish-test"),
            input_text="hello streaming world",
            response_format=AudioResponseFormat.Mp3,
            voice="narrator",
            stream=True,
            streaming_interval=0.25,
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]

    runner.main()

    assert model.calls == [("hello streaming world", "narrator", True, 0.25)]
    assert encoded_calls == [[0.1], [0.2]]
    generated = [
        event.chunk
        for event in sender.events
        if isinstance(event, ChunkGenerated) and isinstance(event.chunk, AudioChunk)
    ]
    assert len(generated) == 2
    assert base64.b64decode(generated[0].data.encode("ascii"), validate=True) == b"mp3-1"
    assert generated[0].chunk_index == 0
    assert generated[0].total_chunks is None
    assert generated[0].is_partial is True
    assert generated[0].finish_reason is None
    assert base64.b64decode(generated[1].data.encode("ascii"), validate=True) == b"mp3-2"
    assert generated[1].chunk_index == 1
    assert generated[1].total_chunks is None
    assert generated[1].is_partial is False
    assert generated[1].finish_reason == "stop"


def test_speech_synthesis_uses_staged_voice_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Kokoro generation should receive a local voice file from the model store."""

    runner, _sender = _make_runner()
    model = _FakeSpeechModel()
    runner.model = model
    runner.local_model_path = tmp_path
    runner.current_status = RunnerReady()
    voice_path = tmp_path / "voices" / "af_heart.safetensors"
    voice_path.parent.mkdir()
    voice_path.write_bytes(b"voice")

    def _fake_encode(
        audio: np.ndarray,
        sample_rate: int,
        response_format: AudioResponseFormat,
    ) -> bytes:
        del audio, sample_rate, response_format
        return b"WAVDATA"

    monkeypatch.setattr(speech_runner, "_encode_audio", _fake_encode)

    task = SpeechSynthesis(
        instance_id=InstanceId("speech-instance-1"),
        command_id=CommandId("speech-command-1"),
        task_params=SpeechSynthesisTaskParams(
            model=ModelId("mlx-community/kokoro-test"),
            input_text="hello world",
            response_format=AudioResponseFormat.Wav,
            voice=None,
            speed=1.1,
        ),
    )

    encoded, sample_rate = runner._run_tts(task)

    assert encoded == b"WAVDATA"
    assert sample_rate == 24000
    assert model.calls == [("hello world", str(voice_path), 1.1, False)]


def test_audio_transcription_emits_terminal_transcription_chunk() -> None:
    """An STT task should decode uploaded audio and emit transcript output."""

    runner, sender = _make_runner()
    audio_bytes = b"RIFFtestWAVE"
    model = _FakeTranscriptionModel(audio_bytes)
    runner.model = model
    runner.current_status = RunnerReady()

    command_id = CommandId("transcription-command-1")
    task = AudioTranscription(
        instance_id=InstanceId("speech-instance-1"),
        command_id=command_id,
        task_params=AudioTranscriptionTaskParams(
            model=ModelId("mlx-community/whisper-test"),
            filename="sample.wav",
            content_type="audio/wav",
            total_input_chunks=1,
            audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            audio_data=base64.b64encode(audio_bytes).decode("ascii"),
            language="en",
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]

    runner.main()

    assert model.calls == [(".wav", "en", False, True)]
    generated: list[tuple[CommandId, TranscriptionChunk]] = []
    for event in sender.events:
        if isinstance(event, ChunkGenerated) and isinstance(
            event.chunk, TranscriptionChunk
        ):
            generated.append((event.command_id, event.chunk))

    assert len(generated) == 1
    generated_command_id, chunk = generated[0]
    assert generated_command_id == command_id
    assert chunk.text == "hello world"
    assert chunk.language == "en"
    assert chunk.segments[0]["start"] == 0.0
    assert chunk.finish_reason == "stop"


def test_audio_transcription_maps_whisper_aliases_without_decode_leak() -> None:
    """OpenAI STT aliases should not reach Whisper's loose decode kwargs."""

    runner, _sender = _make_runner()
    audio_bytes = b"RIFFtestWAVE"
    model = _FakeWhisperTranscriptionModel(audio_bytes)
    runner.model = model
    runner.current_status = RunnerReady()
    task = AudioTranscription(
        instance_id=InstanceId("speech-instance-1"),
        command_id=CommandId("transcription-command-1"),
        task_params=AudioTranscriptionTaskParams(
            model=ModelId("mlx-community/whisper-test"),
            filename="sample.wav",
            content_type="audio/wav",
            total_input_chunks=1,
            audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            audio_data=base64.b64encode(audio_bytes).decode("ascii"),
            language="en",
            prompt="domain vocabulary",
            context="fallback context",
            text="fallback text",
            temperature=0.2,
            max_tokens=32,
            chunk_duration=2.5,
            frame_threshold=12,
            prefill_step_size=3,
            timestamp_granularities=("word",),
        ),
    )

    text, language, segments = runner._run_stt(task)

    assert text == "hello world"
    assert language == "en"
    assert segments == []
    assert model.calls == [
        {
            "language": "en",
            "chunk_duration": 2.5,
            "stream": False,
            "temperature": 0.2,
            "initial_prompt": "domain vocabulary",
            "return_timestamps": True,
            "word_timestamps": True,
            "verbose": True,
            "decode_options": {"sample_len": 32},
        }
    ]
