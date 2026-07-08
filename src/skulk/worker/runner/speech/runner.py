# pyright: reportAny=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Single-node speech runner backed by upstream ``mlx_audio``.

Supports non-streaming text-to-speech and speech-to-text. Realtime session
handling intentionally lands in a later phase so the first audio routes keep one
stable request/response contract.
"""

import base64
import binascii
import hashlib
import inspect
import io
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from anyio import WouldBlock

from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import AudioResponseFormat
from skulk.shared.tracing import (
    begin_trace_session,
    bind_trace_session,
    pop_trace_session,
    record_trace_marker,
    trace,
)
from skulk.shared.types.chunks import AudioChunk, ErrorChunk, TranscriptionChunk
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    AudioTranscription,
    LoadModel,
    Shutdown,
    SpeechSynthesis,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.bootstrap import logger

_DEFAULT_STAGED_TTS_VOICE = "af_heart"


@dataclass(frozen=True)
class _CallableParameters:
    """Parameter support discovered from a model-specific callable."""

    params: frozenset[str]
    accepts_var_kwargs: bool


def _callable_parameters(fn: Callable[..., Any]) -> _CallableParameters | None:
    """Return accepted parameter names for ``fn`` when introspection works."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = set[str]()
    has_var_kwargs = False
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
        else:
            params.add(name)
    return _CallableParameters(
        params=frozenset(params), accepts_var_kwargs=has_var_kwargs
    )


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` and parameters unsupported by a model-specific callable."""
    accepted = _callable_parameters(fn)
    if accepted is None or accepted.accepts_var_kwargs:
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if v is not None and k in accepted.params}


def _to_numpy_audio(audio: Any) -> np.ndarray:
    """Normalize model-emitted audio arrays to numpy for ``mlx_audio`` encoding."""
    if isinstance(audio, np.ndarray):
        return audio
    return np.array(audio)


def _result_audio_and_sample_rate(result: Any) -> tuple[Any | None, int | None]:
    """Extract audio payload and sample rate from common mlx-audio result shapes."""
    if isinstance(result, dict):
        audio = result.get("audio")
        sample_rate = result.get("sample_rate")
    elif isinstance(result, tuple) and len(result) >= 2:
        audio, sample_rate = result[0], result[1]
    else:
        audio = getattr(result, "audio", None)
        sample_rate = getattr(result, "sample_rate", None)
    return audio, sample_rate if isinstance(sample_rate, int) else None


def _encode_audio(
    audio: np.ndarray, sample_rate: int, response_format: AudioResponseFormat
) -> bytes:
    """Encode PCM audio into the requested response format."""
    from mlx_audio.audio_io import write as audio_write

    buffer = io.BytesIO()
    audio_write(buffer, audio, sample_rate, format=response_format.value)
    return buffer.getvalue()


def _load_speech_model(local_path: str) -> Any:
    """Load an ``mlx_audio`` model from the staged local model directory."""
    from mlx_audio.utils import load_model

    return load_model(local_path)


def _resolve_staged_voice_path(
    local_model_path: Path | None, requested_voice: str | None
) -> str | None:
    """Resolve named Kokoro voices to staged ``voices/*.safetensors`` assets."""
    if local_model_path is None:
        return requested_voice
    voices_dir = local_model_path / "voices"
    if not voices_dir.is_dir():
        return requested_voice
    voice = requested_voice or _DEFAULT_STAGED_TTS_VOICE
    if voice.endswith(".safetensors"):
        return voice
    staged_voice_path = voices_dir / f"{voice}.safetensors"
    if staged_voice_path.exists():
        return str(staged_voice_path)
    raise FileNotFoundError(
        f"Staged TTS voice {voice!r} was not found under {voices_dir}"
    )


def _emit_audio_chunks(
    *,
    event_sender: MpSender[Event],
    command_id: CommandId,
    model_id: ModelId,
    encoded_audio: bytes,
    response_format: AudioResponseFormat,
    sample_rate: int,
) -> None:
    """Emit encoded audio bytes as terminal data-plane chunks."""
    if not encoded_audio:
        raise ValueError("No audio generated")
    encoded_data = base64.b64encode(encoded_audio).decode("ascii")
    data_chunks = [
        encoded_data[index : index + SKULK_MAX_CHUNK_SIZE]
        for index in range(0, len(encoded_data), SKULK_MAX_CHUNK_SIZE)
    ]
    for chunk_index, chunk_data in enumerate(data_chunks):
        is_last = chunk_index == len(data_chunks) - 1
        event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=AudioChunk(
                    model=model_id,
                    data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=len(data_chunks),
                    format=response_format,
                    sample_rate=sample_rate,
                    finish_reason="stop" if is_last else None,
                ),
            )
        )


def _emit_transcription_chunk(
    *,
    event_sender: MpSender[Event],
    command_id: CommandId,
    model_id: ModelId,
    text: str,
    language: str | None,
    segments: list[dict[str, str | int | float | bool | None]],
) -> None:
    """Emit one terminal transcript chunk for a completed STT request."""
    event_sender.send(
        ChunkGenerated(
            command_id=command_id,
            chunk=TranscriptionChunk(
                model=model_id,
                text=text,
                language=language,
                segments=segments,
                finish_reason="stop",
            ),
        )
    )


def _transcription_text(result: Any) -> str:
    """Extract transcript text from common ``mlx_audio`` STT result shapes."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        text = result.get("text") or result.get("transcript")
        return text if isinstance(text, str) else ""
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    transcript = getattr(result, "transcript", None)
    return transcript if isinstance(transcript, str) else ""


def _transcription_language(result: Any) -> str | None:
    """Extract detected/requested language from a model result when present."""
    if isinstance(result, dict):
        language = result.get("language")
    else:
        language = getattr(result, "language", None)
    return language if isinstance(language, str) else None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _segment_value(
    value: Any,
) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_segment(
    raw_segment: Any,
    fallback_index: int,
) -> dict[str, str | int | float | bool | None]:
    """Normalize dict/object segment metadata without depending on one STT schema."""
    if isinstance(raw_segment, dict):
        raw_mapping = cast(dict[str, object], raw_segment)

        def get_value(name: str, default: object | None = None) -> object | None:
            return raw_mapping.get(name, default)
    else:

        def get_value(name: str, default: object | None = None) -> object | None:
            return getattr(raw_segment, name, default)

    segment_id = _coerce_int(
        get_value("id", get_value("segment_index", fallback_index))
    )
    text = get_value("text", "")
    start = _coerce_float(get_value("start", get_value("start_time")))
    end = _coerce_float(get_value("end", get_value("end_time")))
    language = get_value("language")
    segment: dict[str, str | int | float | bool | None] = {
        "id": segment_id if segment_id is not None else fallback_index,
        "text": text if isinstance(text, str) else "",
        "start": start,
        "end": end,
    }
    if isinstance(language, str):
        segment["language"] = language
    for key in ("confidence", "is_final"):
        value = get_value(key)
        if value is not None:
            segment[key] = _segment_value(value)
    return segment


def _extract_segments(
    result: Any,
    start_index: int,
) -> list[dict[str, str | int | float | bool | None]]:
    """Extract any segment list, or synthesize one from chunk timestamps."""
    if isinstance(result, dict):
        segments = result.get("segments")
        has_direct_timestamps = "start_time" in result or "start" in result
    else:
        segments = getattr(result, "segments", None)
        has_direct_timestamps = any(
            hasattr(result, name) for name in ("start_time", "start", "end_time", "end")
        )
    if isinstance(segments, Iterable) and not isinstance(segments, (str, bytes, dict)):
        return [
            _normalize_segment(segment, start_index + index)
            for index, segment in enumerate(segments)
        ]
    if has_direct_timestamps:
        return [_normalize_segment(result, start_index)]
    return []


def _iter_result_chunks(result: Any) -> Iterable[Any]:
    """Treat generator-style STT output as chunks without splitting strings/dicts."""
    if isinstance(result, (str, bytes, dict)):
        return (result,)
    if isinstance(result, Iterable):
        return result
    return (result,)


def _normalize_transcription_result(
    result: Any,
) -> tuple[str, str | None, list[dict[str, str | int | float | bool | None]]]:
    """Normalize STT result text, language, and segments for API formatting."""
    text_parts: list[str] = []
    segments: list[dict[str, str | int | float | bool | None]] = []
    language: str | None = None
    for chunk in _iter_result_chunks(result):
        chunk_text = _transcription_text(chunk)
        if chunk_text:
            text_parts.append(chunk_text)
        if language is None:
            language = _transcription_language(chunk)
        segments.extend(_extract_segments(chunk, len(segments)))

    text = "".join(text_parts).strip()
    if not text and segments:
        text = " ".join(
            segment_text
            for segment in segments
            if isinstance((segment_text := segment.get("text")), str)
        ).strip()
    return text, language, segments


def _audio_upload_suffix(filename: str | None, content_type: str | None) -> str:
    """Pick a stable temp-file suffix for audio decoders that inspect extensions."""
    if filename is not None:
        _, suffix = os.path.splitext(filename)
        if suffix:
            return suffix
    return {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/mp4": ".mp4",
        "audio/x-m4a": ".m4a",
    }.get(content_type or "", ".audio")


class Runner:
    """Runner process for single-node ``mlx_audio`` speech models."""

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ) -> None:
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            bound_instance.instance,
            bound_instance.bound_runner_id,
            bound_instance.bound_shard,
        )

        if self.shard_metadata.world_size != 1:
            raise RuntimeError(
                "Speech runner requires single-node placement, got "
                f"world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks = set[TaskId]()
        self.model: Any = None
        self.local_model_path: Path | None = None
        self.current_status: RunnerStatus = RunnerIdle()
        self.seen = set[TaskId]()
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus) -> None:
        """Publish the runner's current lifecycle status."""
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish a task status update."""
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task) -> None:
        """Tell the worker this runner accepted the task."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def _drain_cancellations(self) -> None:
        while True:
            try:
                cancelled = self.cancel_receiver.receive_nowait()
            except WouldBlock:
                break
            self.cancelled_tasks.add(cancelled)

    def _is_cancelled(self, task_id: TaskId) -> bool:
        self._drain_cancellations()
        return (
            task_id in self.cancelled_tasks or CANCEL_ALL_TASKS in self.cancelled_tasks
        )

    def main(self) -> None:
        """Run the speech task loop until shutdown."""
        with self.task_receiver as tasks:
            for task in tasks:
                if task.task_id in self.seen:
                    logger.warning("repeat speech task - potential error")
                self.seen.add(task.task_id)
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                self.send_task_status(task, TaskStatus.Running)
                self.handle_task(task)
                was_cancelled = self._is_cancelled(task.task_id)
                if was_cancelled:
                    self.send_task_status(task, TaskStatus.Cancelled)
                else:
                    self.send_task_status(task, TaskStatus.Complete)
                self.update_status(self.current_status)

                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_task(self, task: Task) -> None:
        """Execute one lifecycle or speech inference task."""
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case SpeechSynthesis() if isinstance(self.current_status, RunnerReady):
                self._synthesize(task)
            case AudioTranscription() if isinstance(self.current_status, RunnerReady):
                self._transcribe(task)
            case Shutdown():
                logger.info("speech runner shutting down")
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self.model = None
                self.current_status = RunnerShutdown()
            case _:
                raise RuntimeError(
                    "speech runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    def _load_model(self, task: LoadModel) -> None:
        """Load the local staged ``mlx_audio`` model."""
        logger.info("speech runner loading")
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)

        model_id = self.shard_metadata.model_card.model_id
        from skulk.download.download_utils import build_model_path
        from skulk.shared.types.common import ModelId

        local_path = build_model_path(ModelId(model_id))
        self.local_model_path = local_path
        logger.info(f"loading speech model from local path: {local_path}")
        self.model = _load_speech_model(str(local_path))
        self.current_status = RunnerReady()
        logger.info(
            f"speech runner ready in {time.time() - self.setup_start_time:.1f}s"
        )

    def _synthesize(self, task: SpeechSynthesis) -> None:
        """Run one non-streaming text-to-speech task and emit audio chunks."""
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert self.model is not None
        model_id = self.shard_metadata.model_card.model_id

        if task.trace_enabled:
            begin_trace_session(
                task.task_id,
                rank=self.shard_metadata.device_rank,
                node_id=str(self.bound_instance.bound_node_id),
                model_id=str(model_id),
                task_kind="speech",
                tags=["tts"],
            )
            record_trace_marker(
                "queued",
                self.shard_metadata.device_rank,
                task_id=task.task_id,
            )

        try:
            with bind_trace_session(task.task_id), trace(
                "tts_generate", self.shard_metadata.device_rank, "speech"
            ):
                encoded_audio, sample_rate = self._run_tts(task)
            if self._is_cancelled(task.task_id):
                return
            if task.trace_enabled:
                record_trace_marker(
                    "finish",
                    self.shard_metadata.device_rank,
                    task_id=task.task_id,
                )
            _emit_audio_chunks(
                event_sender=self.event_sender,
                command_id=task.command_id,
                model_id=model_id,
                encoded_audio=encoded_audio,
                response_format=task.task_params.response_format,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            if task.trace_enabled:
                record_trace_marker(
                    "error",
                    self.shard_metadata.device_rank,
                    task_id=task.task_id,
                    tags=["error"],
                    attrs={"message": str(exc)},
                )
            logger.opt(exception=exc).warning("speech synthesis failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=model_id,
                        error_message=str(exc),
                    ),
                )
            )
        finally:
            if task.trace_enabled:
                traces = pop_trace_session(task.task_id)
                self.event_sender.send(
                    TracesCollected(
                        task_id=task.task_id,
                        rank=self.shard_metadata.device_rank,
                        traces=[
                            TraceEventData(
                                name=trace_event.name,
                                start_us=trace_event.start_us,
                                duration_us=trace_event.duration_us,
                                rank=trace_event.rank,
                                category=trace_event.category,
                                node_id=trace_event.node_id,
                                model_id=trace_event.model_id,
                                task_kind=trace_event.task_kind,
                                tags=list(trace_event.tags),
                                attrs=trace_event.attrs,
                            )
                            for trace_event in traces
                        ],
                    )
                )
            self.current_status = RunnerReady()

    def _transcribe(self, task: AudioTranscription) -> None:
        """Run one non-streaming speech-to-text task and emit transcript output."""
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert self.model is not None
        model_id = self.shard_metadata.model_card.model_id

        if task.trace_enabled:
            begin_trace_session(
                task.task_id,
                rank=self.shard_metadata.device_rank,
                node_id=str(self.bound_instance.bound_node_id),
                model_id=str(model_id),
                task_kind="speech",
                tags=["stt"],
            )
            record_trace_marker(
                "queued",
                self.shard_metadata.device_rank,
                task_id=task.task_id,
            )

        try:
            with bind_trace_session(task.task_id), trace(
                "stt_generate", self.shard_metadata.device_rank, "speech"
            ):
                text, language, segments = self._run_stt(task)
            if self._is_cancelled(task.task_id):
                return
            if task.trace_enabled:
                record_trace_marker(
                    "finish",
                    self.shard_metadata.device_rank,
                    task_id=task.task_id,
                )
            _emit_transcription_chunk(
                event_sender=self.event_sender,
                command_id=task.command_id,
                model_id=model_id,
                text=text,
                language=language,
                segments=segments,
            )
        except Exception as exc:
            if task.trace_enabled:
                record_trace_marker(
                    "error",
                    self.shard_metadata.device_rank,
                    task_id=task.task_id,
                    tags=["error"],
                    attrs={"message": str(exc)},
                )
            logger.opt(exception=exc).warning("speech transcription failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=model_id,
                        error_message=str(exc),
                    ),
                )
            )
        finally:
            if task.trace_enabled:
                traces = pop_trace_session(task.task_id)
                self.event_sender.send(
                    TracesCollected(
                        task_id=task.task_id,
                        rank=self.shard_metadata.device_rank,
                        traces=[
                            TraceEventData(
                                name=trace_event.name,
                                start_us=trace_event.start_us,
                                duration_us=trace_event.duration_us,
                                rank=trace_event.rank,
                                category=trace_event.category,
                                node_id=trace_event.node_id,
                                model_id=trace_event.model_id,
                                task_kind=trace_event.task_kind,
                                tags=list(trace_event.tags),
                                attrs=trace_event.attrs,
                            )
                            for trace_event in traces
                        ],
                    )
                )
            self.current_status = RunnerReady()

    def _run_stt(
        self, task: AudioTranscription
    ) -> tuple[str, str | None, list[dict[str, str | int | float | bool | None]]]:
        """Write uploaded audio to a temp file and call the STT model."""
        assert self.model is not None
        params = task.task_params
        if params.audio_data is None:
            raise ValueError("No audio payload received")
        try:
            audio_bytes = base64.b64decode(
                params.audio_data.encode("ascii"), validate=True
            )
        except binascii.Error as exc:
            raise ValueError("Invalid base64 audio payload") from exc

        actual_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        if actual_sha256 != params.audio_sha256:
            raise ValueError("Audio payload checksum mismatch")

        tmp_path: str | None = None
        try:
            suffix = _audio_upload_suffix(params.filename, params.content_type)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            generate_kwargs = {
                "language": params.language,
                "prompt": params.prompt,
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "chunk_duration": params.chunk_duration,
                "frame_threshold": params.frame_threshold,
                "stream": False,
                "context": params.context,
                "prefill_step_size": params.prefill_step_size,
                "text": params.text or params.prompt,
                "word_timestamps": params.word_timestamps,
                "timestamp_granularities": list(params.timestamp_granularities)
                if params.timestamp_granularities
                else None,
                "verbose": True,
            }
            generate = self.model.generate
            filtered_kwargs = _filter_kwargs(generate, generate_kwargs)
            result = generate(tmp_path, **filtered_kwargs)
            return _normalize_transcription_result(result)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning(f"Failed to remove temporary speech upload {tmp_path}")

    def _run_tts(self, task: SpeechSynthesis) -> tuple[bytes, int]:
        """Generate and encode the complete TTS response."""
        assert self.model is not None
        params = task.task_params
        generate_kwargs = {
            "voice": _resolve_staged_voice_path(
                self.local_model_path, params.voice
            ),
            "speed": params.speed,
            "instruct": params.instruct,
            "lang_code": params.lang_code,
            "ref_audio": params.reference_audio,
            "ref_text": params.reference_text,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repetition_penalty": params.repetition_penalty,
            "stream": False,
            "max_tokens": params.max_tokens,
        }
        generate = self.model.generate
        filtered_kwargs = _filter_kwargs(generate, generate_kwargs)

        generated = generate(params.input_text, **filtered_kwargs)
        if not isinstance(generated, Iterable):
            generated = (generated,)

        audio_chunks: list[np.ndarray] = []
        sample_rate: int | None = None
        for result in generated:
            if self._is_cancelled(task.task_id):
                break
            result_audio, result_sample_rate = _result_audio_and_sample_rate(result)
            if result_audio is None:
                continue
            if result_sample_rate is None:
                model_sample_rate = getattr(self.model, "sample_rate", None)
                result_sample_rate = (
                    model_sample_rate if isinstance(model_sample_rate, int) else None
                )
            if result_sample_rate is None:
                raise ValueError("No audio sample rate returned")
            audio_chunks.append(_to_numpy_audio(result_audio))
            if sample_rate is None:
                sample_rate = result_sample_rate

        if self._is_cancelled(task.task_id):
            return b"", sample_rate or 1
        if not audio_chunks or sample_rate is None:
            raise ValueError("No audio generated")
        audio = (
            audio_chunks[0]
            if len(audio_chunks) == 1
            else np.concatenate(audio_chunks)
        )
        return _encode_audio(audio, sample_rate, params.response_format), sample_rate
