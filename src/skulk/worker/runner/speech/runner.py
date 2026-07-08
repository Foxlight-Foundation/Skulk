# pyright: reportAny=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Single-node speech runner backed by upstream ``mlx_audio``.

Phase 1 implements non-streaming text-to-speech. Speech-to-text and realtime
session handling intentionally land in later phases so the first audio route has
one stable runner contract.
"""

import base64
import inspect
import io
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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
from skulk.shared.types.chunks import AudioChunk, ErrorChunk
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
        """Execute one lifecycle or TTS task."""
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case SpeechSynthesis() if isinstance(self.current_status, RunnerReady):
                self._synthesize(task)
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

        local_path = str(build_model_path(ModelId(model_id)))
        logger.info(f"loading speech model from local path: {local_path}")
        self.model = _load_speech_model(local_path)
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

    def _run_tts(self, task: SpeechSynthesis) -> tuple[bytes, int]:
        """Generate and encode the complete TTS response."""
        assert self.model is not None
        params = task.task_params
        generate_kwargs = {
            "voice": params.voice,
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
