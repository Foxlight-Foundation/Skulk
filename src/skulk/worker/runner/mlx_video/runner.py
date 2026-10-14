"""Single-host native video runner using the existing admission/media lifecycle."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import final

from skulk.provisioning.mlx_video import prepared_runtime, probe_runtime
from skulk.shared.backends import ENGINE_UNAVAILABLE_FAILURE_MARKER
from skulk.shared.constants import (
    SKULK_CACHE_HOME,
    SKULK_VIDEO_INPUT_DIR,
    SKULK_VIDEO_OUTPUT_DIR,
)
from skulk.shared.types.chunks import ErrorChunk, VideoChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    VideoGeneration,
)
from skulk.shared.types.video import (
    VIDEO_OUTPUT_FILENAME,
    VIDEO_THUMBNAIL_FILENAME,
    VideoEngineSettings,
    VideoGenerationStats,
    VideoOutputManifest,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.mlx_video.plan import plan_native_job, selected_model_view
from skulk.worker.runner.mlx_video.process import NativeProgress, run_native_job
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch
from skulk.worker.runner.video_probe import probe_video_output


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def resolve_placed_runtime(backend: str | None, build: str | None) -> Path:
    """Resolve an exact installed build or terminate the placement with an explanation.

    Missing/corrupt runtime bytes cannot improve through runner restart, so this
    uses the worker's existing terminal engine-unavailable classification.
    """
    if backend != "mlx_video-metal" or build is None:
        raise RuntimeError(f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: native video placement lacks an exact GPU backend/build")
    interpreter = prepared_runtime(build)
    if interpreter is None:
        raise RuntimeError(f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: the exact published native video engine is not installed")
    probe_runtime(interpreter, build)
    return interpreter


@final
class Runner(ServedConcurrentDispatch):
    """Serialize GPU jobs while keeping admission and cancellation responsive."""

    _generation_kinds = (VideoGeneration,)

    def __init__(self, bound_instance: BoundInstance, event_sender: MpSender[Event], task_receiver: MpReceiver[Task], cancel_receiver: MpReceiver[TaskId]) -> None:
        """Bind an exact placement; construction never installs or downloads anything."""
        self.bound_instance = bound_instance
        self.shard_metadata = bound_instance.bound_shard
        self.card = self.shard_metadata.model_card
        self.model_id = self.card.model_id
        self.runner_id = bound_instance.bound_runner_id
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.current_status: RunnerStatus = RunnerIdle()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.interpreter: Path | None = None
        self.model_dir = SKULK_CACHE_HOME / "mlx-video" / str(self.runner_id) / "model"
        self._init_concurrent_dispatch(1, "mlx-video-render", queue_admitted=True)
        self.update_status(self.current_status)

    def update_status(self, status: RunnerStatus) -> None:
        """Publish a state transition to the existing supervisor."""
        self.current_status = status
        self.event_sender.send(RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status))

    def acknowledge_task(self, task: Task) -> None:
        """Acknowledge admission without waiting for an earlier GPU job."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish the terminal or running task state."""
        self.event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=status))

    def main(self) -> None:
        """Run the shared served-engine loop through shutdown and cleanup."""
        self.run_dispatch_loop()

    def _ensure_server_alive(self) -> None:
        # Native jobs have separate supervised children. Readiness here means
        # the already verified runtime remains available between serial jobs.
        if self.interpreter is not None and not self.interpreter.is_file():
            raise RuntimeError("the installed native video runtime disappeared")

    def _teardown_server(self) -> None:
        self.interpreter = None
        shutil.rmtree(self.model_dir.parent, ignore_errors=True)

    def handle_task(self, task: Task) -> None:
        """Load verified local capacity and enter the standard ready state."""
        match task:
            case ConnectToGroup() if isinstance(self.current_status, (RunnerIdle, RunnerFailed)):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.current_status = RunnerConnected()
            case LoadModel() if isinstance(self.current_status, (RunnerIdle, RunnerConnected)):
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)
                build = self.shard_metadata.resolved_engine_build
                interpreter = resolve_placed_runtime(self.shard_metadata.resolved_backend, build)
                shutil.rmtree(self.model_dir.parent, ignore_errors=True)
                selected_model_view(self.card, self.model_dir)
                self.interpreter = interpreter
                self.current_status = RunnerLoaded()
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                self._ensure_server_alive()
                self.current_status = RunnerReady()
            case _:
                raise ValueError(f"unexpected native video task {type(task).__name__}")

    def _generate(self, task: Task) -> None:
        assert isinstance(task, VideoGeneration)
        if self._is_cancelled(task.task_id):
            return
        output = SKULK_VIDEO_OUTPUT_DIR / str(task.command_id)
        try:
            assert self.interpreter is not None
            build = self.shard_metadata.resolved_engine_build
            if build is None or prepared_runtime(build) != self.interpreter:
                raise RuntimeError("native video runtime failed its integrity check")
            job, plan = plan_native_job(task.task_params, self.card, self.model_dir, output, SKULK_VIDEO_INPUT_DIR)
            base_steps = job.base_steps
            planned_steps = base_steps + 3

            def progress(message: NativeProgress) -> None:
                completed = (base_steps if message.sampling_stage == 2 else 0) + (message.step or 0)
                fraction = 0.05 if message.stage == "encoding" else 0.9 if message.stage == "decoding" else 0.98 if message.stage == "muxing" else 0.05 + 0.8 * min(1.0, completed / planned_steps)
                self.event_sender.send(ChunkGenerated(command_id=task.command_id, chunk=VideoChunk(model=self.model_id, stage=message.stage, step=completed if message.step is not None else None, total_steps=planned_steps if message.step is not None else None, progress=fraction)))

            result = run_native_job(self.interpreter, job, progress=progress, is_cancelled=lambda: self._is_cancelled(task.task_id))
            if result is None or self._is_cancelled(task.task_id):
                shutil.rmtree(output, ignore_errors=True)
                return
            video = self.card.video
            assert video is not None
            actual = probe_video_output(output / VIDEO_OUTPUT_FILENAME, plan, codec=job.codec, interpreter=self.interpreter, minimum_frames=video.frame_count_for_seconds(job.minimum_seconds) if job.duration == "auto" else None)
            identity, size = _file_identity(output / VIDEO_OUTPUT_FILENAME)
            thumbnail_identity, thumbnail_size = _file_identity(output / VIDEO_THUMBNAIL_FILENAME)
            manifest = VideoOutputManifest(sha256=identity, size_bytes=size, width=plan.width, height=plan.height, frame_count=actual.frame_count, fps=plan.fps, seconds=actual.seconds, audio_sample_rate=plan.sample_rate if plan.audio else None, audio_channels=plan.channels if plan.audio else None, thumbnail_sha256=thumbnail_identity, thumbnail_size_bytes=thumbnail_size)
            stats = VideoGenerationStats(steps=result.steps, seconds_per_step=result.sampling_seconds / result.steps, total_generation_time=result.total_generation_time, peak_memory_bytes=result.peak_memory_bytes, engine=VideoEngineSettings(sampler="res_2s" if job.pipeline == "hq_two_stage" else "euler_ancestral" if job.pipeline == "distilled_two_stage" else "euler", scheduler="ltx-2.5", steps=result.steps, seed=plan.seed, pipeline=job.pipeline, decoder=task.task_params.decoder or "convolutional", guidance=job.guidance, generated_keyframe_interval=job.generated_keyframe_interval, duration=task.task_params.duration, adapter="distilled_refinement" if job.pipeline != "distilled_two_stage" else None, adapter_strength=job.refinement_strength if job.pipeline != "distilled_two_stage" else None, width=plan.width, height=plan.height, frame_count=actual.frame_count, codec=job.codec))
            (output / "result.json").unlink()
            self.event_sender.send(ChunkGenerated(command_id=task.command_id, chunk=VideoChunk(model=self.model_id, stage="muxing", progress=1.0, output=manifest, stats=stats, finish_reason="stop")))
        except ValueError as error:
            shutil.rmtree(output, ignore_errors=True)
            self._mark_generation_failed(task.task_id)
            self.event_sender.send(ChunkGenerated(command_id=task.command_id, chunk=ErrorChunk(model=self.model_id, finish_reason="error", error_message=str(error))))
        except Exception:
            self._mark_generation_failed(task.task_id)
            shutil.rmtree(output, ignore_errors=True)
            raise
