"""Runner subprocess for the test video engine.

Mirrors the image runner's state machine so the supervisor, planner, and
worker treat it exactly like a real engine: it acknowledges each task,
walks ``RunnerIdle -> RunnerLoaded -> RunnerReady``, renders one
``VideoGeneration`` at a time while emitting progress ``VideoChunk`` frames,
and ends the render with the terminal chunk that carries the output
manifest the worker streams from.
"""

from __future__ import annotations

import os
import time
from typing import final

from anyio import WouldBlock
from loguru import logger

from skulk.shared.constants import SKULK_VIDEO_OUTPUT_DIR
from skulk.shared.models.model_cards import VideoCardConfig
from skulk.shared.types.chunks import ErrorChunk, VideoChunk
from skulk.shared.types.common import CommandId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    VideoGeneration,
)
from skulk.shared.types.video import VideoGenerationTaskParams, VideoStage
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.test_video.render import plan_render, render_clip

STEP_SECONDS_ENV = "SKULK_TEST_VIDEO_STEP_SECONDS"
"""Simulated wall time per sampling step; default keeps renders sub-second."""
_DEFAULT_STEP_SECONDS = 0.02


def _step_seconds() -> float:
    raw = os.environ.get(STEP_SECONDS_ENV, "").strip()
    try:
        return max(0.0, float(raw)) if raw else _DEFAULT_STEP_SECONDS
    except ValueError:
        return _DEFAULT_STEP_SECONDS


@final
class Runner:
    """Test video engine runner; same constructor shape as the image runner."""

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
        self.runner_id = bound_instance.bound_runner_id
        self.shard_metadata = bound_instance.bound_shard
        self.model_id = self.shard_metadata.model_card.model_id
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)
        self.video: VideoCardConfig | None = None
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.current_status: RunnerStatus = RunnerIdle()
        self.update_status(RunnerIdle())

    # --- events ---------------------------------------------------------------

    def update_status(self, status: RunnerStatus) -> None:
        """Record and publish a runner status."""
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish a task status transition."""
        self.event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=status))

    def acknowledge_task(self, task: Task) -> None:
        """Tell the supervisor the task was received."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    # --- cancellation ---------------------------------------------------------

    def _drain_cancellations(self) -> None:
        while True:
            try:
                cancelled = self.cancel_receiver.receive_nowait()
            except WouldBlock:
                return
            self.cancelled_tasks.add(cancelled)

    def _is_cancelled(self, task_id: TaskId) -> bool:
        self._drain_cancellations()
        return task_id in self.cancelled_tasks or CANCEL_ALL_TASKS in self.cancelled_tasks

    # --- main loop ------------------------------------------------------------

    def main(self) -> None:
        """Serve tasks until shutdown; the same loop shape as the image runner."""

        with self.task_receiver as tasks:
            for task in tasks:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                self.seen.add(task.task_id)
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                self.send_task_status(task, TaskStatus.Running)
                self.handle_task(task)
                if self._is_cancelled(task.task_id):
                    self.send_task_status(task, TaskStatus.Cancelled)
                else:
                    self.send_task_status(task, TaskStatus.Complete)
                self.update_status(self.current_status)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_task(self, task: Task) -> None:
        """Apply one task to the runner state machine."""

        match task:
            case ConnectToGroup() if isinstance(self.current_status, (RunnerIdle, RunnerFailed)):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                # Single-host by construction; there is no group to form.
                self.current_status = RunnerConnected()
            case LoadModel() if isinstance(self.current_status, (RunnerIdle, RunnerConnected)):
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)
                video = self.shard_metadata.model_card.video
                if video is None:
                    raise RuntimeError(f"{self.model_id} has no [video] section")
                self.video = video
                self.current_status = RunnerLoaded()
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                self.current_status = RunnerReady()
            case VideoGeneration(task_params=params, command_id=command_id) if isinstance(
                self.current_status, RunnerReady
            ):
                assert self.video is not None
                self.update_status(RunnerRunning())
                self.acknowledge_task(task)
                try:
                    self._render(task.task_id, command_id, params)
                except Exception as error:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                model=self.model_id,
                                finish_reason="error",
                                error_message=str(error),
                            ),
                        )
                    )
                    raise
                self.current_status = RunnerReady()
            case Shutdown():
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self.current_status = RunnerShutdown()
            case _:
                raise ValueError(
                    f"unexpected task {type(task).__name__} in state {self.current_status}"
                )

    def _render(
        self, task_id: TaskId, command_id: CommandId, params: VideoGenerationTaskParams
    ) -> None:
        assert self.video is not None
        plan = plan_render(params, self.video)
        logger.info(
            f"test video engine rendering {plan.frame_count} frames at "
            f"{plan.width}x{plan.height} over {plan.steps} steps for {command_id}"
        )

        def progress(
            stage: VideoStage, step: int | None, total_steps: int | None, fraction: float
        ) -> None:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=VideoChunk(
                        model=self.model_id,
                        stage=stage,
                        step=step,
                        total_steps=total_steps,
                        progress=fraction,
                    ),
                )
            )

        result = render_clip(
            plan,
            SKULK_VIDEO_OUTPUT_DIR / str(command_id),
            progress=progress,
            is_cancelled=lambda: self._is_cancelled(task_id),
            step_seconds=_step_seconds(),
        )
        if result is None:
            logger.info(f"test video engine render cancelled for {command_id}")
            return
        manifest, stats = result
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=VideoChunk(
                    model=self.model_id,
                    stage="muxing",
                    progress=1.0,
                    output=manifest,
                    stats=stats,
                    finish_reason="stop",
                ),
            )
        )
