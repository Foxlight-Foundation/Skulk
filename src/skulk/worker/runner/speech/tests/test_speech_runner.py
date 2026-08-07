# pyright: reportPrivateUsage=false, reportMissingParameterType=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false
"""Unit coverage for the single-node speech runner."""

import base64
import hashlib
import inspect
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from anyio import WouldBlock

from skulk.shared.models.model_cards import AudioCardKind, AudioResponseFormat, ModelId
from skulk.shared.models.reference_voices import ReferenceVoiceProfile
from skulk.shared.types.audio import (
    AudioTranscriptionTaskParams,
    RealtimeAudioInputFrame,
    RealtimeAudioTranscriptionTaskParams,
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
from skulk.shared.types.tasks import (
    AudioTranscription,
    RealtimeAudioTranscription,
    SpeechSynthesis,
    TaskStatus,
)
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from skulk.worker.runner.speech import runner as speech_runner
from skulk.worker.runner.speech.runner import (
    Runner,
    _encode_audio,
    _ensure_ffmpeg_available,
    _filter_kwargs,
    _install_attention_mask_dtype_compat,
    _install_canary_compatibility,
    _load_speech_model,
    _resolve_staged_voice_path,
    _stt_generate_kwargs,
)


def test_fish_s2_dependency_advances_generation_hidden_state() -> None:
    """The pinned Fish runtime must carry the upstream semantic-output fix."""

    pytest.importorskip("mlx_audio.tts.models.fish_qwen3_omni.fish_speech")
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import (  # pyright: ignore[reportMissingTypeStubs]
        Model,
    )

    source = inspect.getsource(Model._generate_codes_for_batch)

    assert "hidden_state = next_result.hidden_states[:, -1]" in source


def test_encode_audio_emits_little_endian_pcm16() -> None:
    """Raw PCM responses should be headerless, clipped signed 16-bit samples."""

    encoded = _encode_audio(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        24000,
        AudioResponseFormat.Pcm,
    )

    assert np.frombuffer(encoded, dtype="<i2").tolist() == [
        -32767,
        -32767,
        0,
        32767,
        32767,
    ]


def test_encode_audio_rejects_multi_channel_pcm() -> None:
    """Raw PCM must not silently interleave channels under a mono contract."""

    with pytest.raises(ValueError, match="must be mono"):
        _encode_audio(
            np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32),
            24000,
            AudioResponseFormat.Pcm,
        )


def test_ensure_ffmpeg_available_preserves_system_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A system encoder should remain authoritative without preparing a shim."""

    def _unexpected_bundled_lookup() -> Path:
        raise AssertionError("bundled encoder should not be resolved")

    monkeypatch.setattr(
        speech_runner, "_bundled_ffmpeg_executable", _unexpected_bundled_lookup
    )
    monkeypatch.setattr(
        speech_runner.shutil, "which", lambda _executable: "/usr/bin/ffmpeg"
    )

    assert _ensure_ffmpeg_available() == Path("/usr/bin/ffmpeg")


def test_ensure_ffmpeg_available_prepares_private_bundled_shim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh install should expose its packaged encoder through runner PATH."""

    executable = tmp_path / "ffmpeg-packaged"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    cache_home = tmp_path / "cache"
    (cache_home / "bin").mkdir(parents=True, mode=0o755)
    monkeypatch.setattr(speech_runner, "SKULK_CACHE_HOME", cache_home)
    monkeypatch.setattr(
        speech_runner, "_bundled_ffmpeg_executable", lambda: executable
    )
    monkeypatch.setenv("PATH", "")

    resolved = _ensure_ffmpeg_available()

    assert resolved == cache_home / "bin" / "ffmpeg"
    assert resolved.resolve() == executable
    assert (cache_home / "bin").stat().st_mode & 0o777 == 0o700
    assert os.environ["PATH"].split(os.pathsep)[0] == os.fspath(cache_home / "bin")


def test_encode_audio_mp3_uses_bundled_encoder_on_fresh_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MP3 encoding should work without a preinstalled system executable."""

    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("mlx_audio.audio_io")
    monkeypatch.setattr(speech_runner, "SKULK_CACHE_HOME", tmp_path / "cache")
    monkeypatch.setenv("PATH", "")

    encoded = _encode_audio(
        np.sin(np.linspace(0.0, np.pi * 2.0, 2400, dtype=np.float32)),
        24000,
        AudioResponseFormat.Mp3,
    )

    assert encoded.startswith((b"ID3", b"\xff"))


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


class _RealtimeFrameReceiver:
    """Blocking-receiver stand-in for one realtime task."""

    def __init__(self, frames: list[RealtimeAudioInputFrame]) -> None:
        self.frames = frames

    def receive_timeout(self, _timeout: float) -> RealtimeAudioInputFrame:
        if not self.frames:
            raise WouldBlock
        return self.frames.pop(0)


class _FakeRealtimeSession:
    input_sample_rate = 16000

    def __init__(self) -> None:
        self.done = False
        self.closed = False
        self.fed: list[np.ndarray] = []
        self._emitted_open_delta = False

    def feed(self, samples: np.ndarray) -> None:
        self.fed.append(samples)

    def step(self, *, max_decode_tokens: int) -> list[str]:
        if not self.closed and not self._emitted_open_delta:
            assert max_decode_tokens == 8
            self._emitted_open_delta = True
            return ["hel"]
        if self.closed and not self.done:
            assert max_decode_tokens == 16
            self.done = True
            return ["lo"]
        return []

    def close(self) -> None:
        self.closed = True


class _FakeRealtimeModel:
    def __init__(self) -> None:
        self.session = _FakeRealtimeSession()
        self.calls: list[tuple[float, int]] = []

    def create_streaming_session(
        self, *, temperature: float, transcription_delay_ms: int
    ) -> _FakeRealtimeSession:
        self.calls.append((temperature, transcription_delay_ms))
        return self.session


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


class _FakeStreamingTranscriptionModel:
    """STT fake that exposes model-generated text deltas."""

    def __init__(self, expected_audio: bytes) -> None:
        self.expected_audio = expected_audio
        self.stream_values: list[bool] = []

    def generate(self, audio_path: str, *, stream: bool = False) -> Iterator[str]:
        assert Path(audio_path).read_bytes() == self.expected_audio
        self.stream_values.append(stream)
        yield "hello "
        yield "world"


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


def test_attention_mask_compat_casts_mask_to_input_dtype() -> None:
    """The Canary adapter should cast only mismatched masks before attention."""

    class _Array:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

        def astype(self, dtype: str) -> "_Array":
            return _Array(dtype)

    class _Attention:
        seen_mask: _Array | None = None

        def __init__(self) -> None:
            self.q_proj = type(
                "_Projection",
                (),
                {"weight": _Array("bfloat16")},
            )()

        def __call__(
            self,
            x: _Array,
            mask: _Array | None = None,
            cache: object | None = None,
        ) -> tuple[_Array, object | None]:
            del x
            type(self).seen_mask = mask
            return _Array("bfloat16"), cache

    _install_attention_mask_dtype_compat(_Attention)
    result, cache = _Attention()(
        _Array("float32"),
        mask=_Array("float32"),
        cache="cache",
    )

    assert result.dtype == "bfloat16"
    assert cache == "cache"
    assert _Attention.seen_mask is not None
    assert _Attention.seen_mask.dtype == "bfloat16"


def test_canary_cross_attention_accepts_bfloat16_with_encoder_mask() -> None:
    """Canary cross-attention must keep its internally built mask in query dtype."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_audio.stt.models.canary.decoder")
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_audio.stt.models.canary.decoder import (  # pyright: ignore[reportMissingTypeStubs]
        MultiHeadCrossAttention,
    )

    _install_canary_compatibility()
    attention = MultiHeadCrossAttention(d_model=8, n_heads=2)
    attention.update(
        tree_map(
            lambda value: value.astype(mx.bfloat16),
            attention.parameters(),
        )
    )

    output, cache = attention(
        mx.zeros((1, 2, 8), dtype=mx.bfloat16),
        mx.zeros((1, 3, 8), dtype=mx.bfloat16),
        encoder_mask=mx.ones((1, 3), dtype=mx.float32),
    )
    mx.eval(output)

    assert output.dtype == mx.bfloat16
    assert cache is not None


def test_translation_kwargs_support_canary_contract() -> None:
    """Canary translation should receive explicit source and English target."""

    def generate(
        _path: str,
        *,
        source_lang: str,
        target_lang: str,
        use_pnc: bool,
        no_repeat_ngram_size: int,
    ) -> object:
        return object()

    params = AudioTranscriptionTaskParams(
        model=ModelId("mlx-community/canary-test"),
        audio_sha256="0" * 64,
        language="fr",
        translate_to_english=True,
    )

    kwargs = _filter_kwargs(generate, _stt_generate_kwargs(generate, params))

    assert kwargs == {
        "source_lang": "fr",
        "target_lang": "en",
        "use_pnc": True,
        "no_repeat_ngram_size": 3,
    }


def test_translation_kwargs_omit_ambiguous_language_alias() -> None:
    """Canary-style variadic APIs must not receive a target-overriding alias."""

    def generate(_path: str, **_kwargs: object) -> object:
        return object()

    params = AudioTranscriptionTaskParams(
        model=ModelId("mlx-community/canary-test"),
        audio_sha256="0" * 64,
        language="fr",
        translate_to_english=True,
    )

    kwargs = _filter_kwargs(generate, _stt_generate_kwargs(generate, params))

    assert "language" not in kwargs
    assert kwargs["source_lang"] == "fr"
    assert kwargs["target_lang"] == "en"


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
    seeded_requests: list[int | None] = []
    monkeypatch.setattr(
        speech_runner,
        "_seed_tts_sampling",
        lambda seed: seeded_requests.append(seed),
    )

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
            seed=42,
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]

    runner.main()

    assert model.calls == [("hello world", "af_heart", 1.1, False)]
    assert seeded_requests == [42]
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
    assert len(generated) == 3
    assert base64.b64decode(generated[0].data.encode("ascii"), validate=True) == b"mp3-1"
    assert generated[0].chunk_index == 0
    assert generated[0].total_chunks is None
    assert generated[0].is_partial is True
    assert generated[0].finish_reason is None
    assert base64.b64decode(generated[1].data.encode("ascii"), validate=True) == b"mp3-2"
    assert generated[1].chunk_index == 1
    assert generated[1].total_chunks is None
    assert generated[1].is_partial is True
    assert generated[1].finish_reason is None
    assert base64.b64decode(generated[2].data.encode("ascii"), validate=True) == b""
    assert generated[2].chunk_index == 2
    assert generated[2].total_chunks is None
    assert generated[2].is_partial is False
    assert generated[2].finish_reason == "stop"


def test_tts_reference_audio_temp_file_is_request_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner reference files exist during generation and are deleted afterward."""

    class _ReferenceSpeechModel:
        sample_rate = 24000

        def __init__(self) -> None:
            self.reference_audio: object | None = None

        def generate(
            self,
            text: str,
            *,
            ref_audio: object | None = None,
            ref_text: str | None = None,
        ) -> list[_FakeSpeechResult]:
            assert text == "hello"
            assert ref_text == "reference transcript"
            self.reference_audio = ref_audio
            return [_FakeSpeechResult()]

    runner, _ = _make_runner()
    model = _ReferenceSpeechModel()
    runner.model = model

    def _fake_encode(
        audio: np.ndarray,
        sample_rate: int,
        response_format: AudioResponseFormat,
    ) -> bytes:
        assert audio.size > 0
        assert sample_rate == 24000
        assert response_format == AudioResponseFormat.Wav
        return b"WAV"

    monkeypatch.setattr(speech_runner, "_encode_audio", _fake_encode)
    loaded_waveform = object()
    observed_reference_path: Path | None = None

    def _fake_load_reference_audio(audio_path: str, sample_rate: int) -> object:
        nonlocal observed_reference_path
        observed_reference_path = Path(audio_path)
        assert observed_reference_path.read_bytes() == b"RIFF-reference"
        assert sample_rate == 24000
        return loaded_waveform

    monkeypatch.setattr(
        speech_runner,
        "_load_tts_reference_audio",
        _fake_load_reference_audio,
    )

    encoded, sample_rate = runner._run_tts(
        SpeechSynthesis(
            instance_id=InstanceId("speech-instance-1"),
            command_id=CommandId("reference-command"),
            task_params=SpeechSynthesisTaskParams(
                model=ModelId("org/reference-tts"),
                input_text="hello",
                response_format=AudioResponseFormat.Wav,
                reference_text="reference transcript",
                reference_audio_present=True,
                reference_audio_filename="sample.wav",
                reference_audio_content_type="audio/wav",
                reference_audio_sha256=hashlib.sha256(
                    b"RIFF-reference"
                ).hexdigest(),
                reference_audio_data=b"RIFF-reference",
            ),
        )
    )

    assert encoded == b"WAV"
    assert sample_rate == 24000
    assert model.reference_audio is loaded_waveform
    assert observed_reference_path is not None
    assert not observed_reference_path.exists()


def test_bundled_reference_voice_is_resolved_inside_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Packaged voice media stays local and supplies exact conditioning text."""

    class _ReferenceSpeechModel:
        sample_rate = 24000

        def __init__(self) -> None:
            self.calls: list[dict[str, object | None]] = []

        def generate(
            self,
            text: str,
            *,
            voice: str | None = None,
            ref_audio: object | None = None,
            ref_text: str | None = None,
            guidance_method: str | None = None,
        ) -> list[_FakeSpeechResult]:
            assert text == "hello"
            self.calls.append(
                {
                    "voice": voice,
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                    "guidance_method": guidance_method,
                }
            )
            return [_FakeSpeechResult()]

    reference_path = tmp_path / "reference.mp3"
    reference_path.write_bytes(b"reference")
    profile = ReferenceVoiceProfile(
        id="angus",
        name="Angus",
        preferred_languages=("en",),
        audio_path=reference_path,
        transcript="Exact transcript.",
        sha256=hashlib.sha256(b"reference").hexdigest(),
    )
    runner, _ = _make_runner()
    runner.shard_metadata.model_card.family = "longcat_audiodit"
    model = _ReferenceSpeechModel()
    runner.model = model
    loaded_waveform = object()

    def _profile(profile_id: str) -> ReferenceVoiceProfile:
        assert profile_id == "angus"
        return profile

    def _load(audio_path: str, sample_rate: int) -> object:
        assert audio_path == str(reference_path)
        assert sample_rate == 24000
        return loaded_waveform

    monkeypatch.setattr(speech_runner, "bundled_reference_voice_profile", _profile)
    monkeypatch.setattr(speech_runner, "_load_tts_reference_audio", _load)
    monkeypatch.setattr(speech_runner, "_encode_audio", lambda *_args: b"WAV")

    encoded, sample_rate = runner._run_tts(
        SpeechSynthesis(
            instance_id=InstanceId("speech-instance-1"),
            command_id=CommandId("bundled-reference-command"),
            task_params=SpeechSynthesisTaskParams(
                model=ModelId("org/reference-tts"),
                input_text="hello",
                response_format=AudioResponseFormat.Wav,
                voice="angus",
                reference_voice_profile="angus",
            ),
        )
    )

    assert encoded == b"WAV"
    assert sample_rate == 24000
    assert model.calls == [
        {
            "voice": None,
            "ref_audio": loaded_waveform,
            "ref_text": "Exact transcript.",
            "guidance_method": "apg",
        }
    ]


def test_speech_synthesis_handles_single_tuple_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct ``(audio, sample_rate)`` model result is one TTS result."""

    class _TupleSpeechModel:
        def generate(self, text: str) -> tuple[list[float], int]:
            assert text == "hello tuple"
            return ([0.4, 0.5], 22050)

    runner, _sender = _make_runner()
    runner.model = _TupleSpeechModel()
    runner.current_status = RunnerReady()
    encoded_calls: list[tuple[list[float], int]] = []

    def _fake_encode(
        audio: np.ndarray,
        sample_rate: int,
        response_format: AudioResponseFormat,
    ) -> bytes:
        assert response_format == AudioResponseFormat.Mp3
        encoded_calls.append((cast(list[float], audio.tolist()), sample_rate))
        return b"tuple-audio"

    monkeypatch.setattr(speech_runner, "_encode_audio", _fake_encode)

    encoded, sample_rate = runner._run_tts(
        SpeechSynthesis(
            instance_id=InstanceId("speech-instance-1"),
            command_id=CommandId("speech-command-tuple"),
            task_params=SpeechSynthesisTaskParams(
                model=ModelId("mlx-community/fish-test"),
                input_text="hello tuple",
                response_format=AudioResponseFormat.Mp3,
            ),
        )
    )

    assert encoded == b"tuple-audio"
    assert sample_rate == 22050
    assert encoded_calls == [([0.4, 0.5], 22050)]


def test_speech_synthesis_handles_single_numpy_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct numpy audio array result uses the model sample-rate fallback."""

    class _ArraySpeechModel:
        sample_rate = 24000

        def generate(self, text: str) -> np.ndarray:
            assert text == "hello array"
            return np.array([0.7, 0.8])

    runner, _sender = _make_runner()
    runner.model = _ArraySpeechModel()
    runner.current_status = RunnerReady()
    encoded_calls: list[tuple[list[float], int]] = []

    def _fake_encode(
        audio: np.ndarray,
        sample_rate: int,
        response_format: AudioResponseFormat,
    ) -> bytes:
        assert response_format == AudioResponseFormat.Mp3
        encoded_calls.append((cast(list[float], audio.tolist()), sample_rate))
        return b"array-audio"

    monkeypatch.setattr(speech_runner, "_encode_audio", _fake_encode)

    encoded, sample_rate = runner._run_tts(
        SpeechSynthesis(
            instance_id=InstanceId("speech-instance-1"),
            command_id=CommandId("speech-command-array"),
            task_params=SpeechSynthesisTaskParams(
                model=ModelId("mlx-community/fish-test"),
                input_text="hello array",
                response_format=AudioResponseFormat.Mp3,
            ),
        )
    )

    assert encoded == b"array-audio"
    assert sample_rate == 24000
    assert encoded_calls == [([0.7, 0.8], 24000)]


def test_tts_generation_uses_safe_default_and_preserves_explicit_budget() -> None:
    """Omitted TTS limits must not inherit a model default that truncates speech."""

    class _BudgetSpeechModel:
        def __init__(self) -> None:
            self.max_token_calls: list[int] = []

        def generate(
            self,
            text: str,
            *,
            max_tokens: int = 1024,
        ) -> list[_FakeSpeechResult]:
            assert text == "hello budget"
            self.max_token_calls.append(max_tokens)
            return [_FakeSpeechResult()]

    runner, _sender = _make_runner()
    model = _BudgetSpeechModel()
    runner.model = model

    def task(max_tokens: int | None) -> SpeechSynthesis:
        return SpeechSynthesis(
            instance_id=InstanceId("speech-instance-1"),
            command_id=CommandId(f"speech-budget-{max_tokens}"),
            task_params=SpeechSynthesisTaskParams(
                model=ModelId("mlx-community/fish-test"),
                input_text="hello budget",
                response_format=AudioResponseFormat.Wav,
                max_tokens=max_tokens,
            ),
        )

    assert list(runner._iter_tts_results(task(None), stream=False))
    assert list(runner._iter_tts_results(task(2048), stream=False))
    assert model.max_token_calls == [4096, 2048]


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


def test_audio_transcription_emits_model_stream_deltas() -> None:
    """Uploaded streaming STT should preserve upstream delta boundaries."""

    runner, sender = _make_runner()
    audio_bytes = b"RIFFstreamWAVE"
    model = _FakeStreamingTranscriptionModel(audio_bytes)
    runner.model = model
    runner.current_status = RunnerReady()

    command_id = CommandId("transcription-command-stream")
    task = AudioTranscription(
        instance_id=InstanceId("speech-instance-1"),
        command_id=command_id,
        task_params=AudioTranscriptionTaskParams(
            model=ModelId("mlx-community/voxtral-realtime-test"),
            filename="sample.wav",
            content_type="audio/wav",
            total_input_chunks=1,
            audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            audio_data=base64.b64encode(audio_bytes).decode("ascii"),
            stream=True,
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]

    runner.main()

    assert model.stream_values == [True]
    generated = [
        event.chunk
        for event in sender.events
        if isinstance(event, ChunkGenerated)
        and isinstance(event.chunk, TranscriptionChunk)
    ]
    assert [chunk.text for chunk in generated] == ["hello ", "world", ""]
    assert [chunk.is_partial for chunk in generated] == [True, True, False]
    assert generated[-1].finish_reason == "stop"


def test_realtime_transcription_emits_partial_and_final_chunks() -> None:
    """A true streaming session should receive PCM and emit ordered deltas."""

    runner, sender = _make_runner()
    model = _FakeRealtimeModel()
    runner.model = model
    runner.current_status = RunnerReady()
    command_id = CommandId("realtime-transcription-command")
    task = RealtimeAudioTranscription(
        instance_id=InstanceId("speech-instance-1"),
        command_id=command_id,
        owner_node=NodeId("api-node"),
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=ModelId("mlx-community/voxtral-realtime-test"),
            input_sample_rate=16000,
            temperature=0.0,
            transcription_delay_ms=480,
        ),
    )
    pcm = np.asarray([0, 16384, -16384], dtype=np.int16).tobytes()
    runner.realtime_audio_receiver = cast(  # pyright: ignore[reportAttributeAccessIssue]
        "object",
        _RealtimeFrameReceiver(
            [
                RealtimeAudioInputFrame(
                    command_id=command_id,
                    sequence=1,
                    kind="chunk",
                    data=pcm,
                ),
                RealtimeAudioInputFrame(
                    command_id=command_id,
                    sequence=2,
                    kind="completed",
                ),
            ]
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]

    runner.main()

    assert model.calls == [(0.0, 480)]
    assert len(model.session.fed) == 1
    assert np.allclose(model.session.fed[0], [0.0, 0.5, -0.5])
    chunks = [
        event.chunk
        for event in sender.events
        if isinstance(event, ChunkGenerated)
        and isinstance(event.chunk, TranscriptionChunk)
    ]
    assert [(chunk.text, chunk.is_partial, chunk.finish_reason) for chunk in chunks] == [
        ("hel", True, None),
        ("lo", True, None),
        ("hello", False, "stop"),
    ]


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
