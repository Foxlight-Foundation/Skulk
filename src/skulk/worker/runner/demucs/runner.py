"""Serial native separation lifecycle over the shared bounded runner dispatcher.

The bootstrap owner supplies an already validated model binding and private
source root. This runner cannot prepare a package or admit input by itself.
Only complete output facts are emitted; the worker owns addressed WAV delivery.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, final

from loguru import logger

from skulk.shared.types.audio_separation import (
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResult,
)
from skulk.shared.types.chunks import ErrorChunk, SeparationChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    AudioSeparation,
    ConnectToGroup,
    LoadModel,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
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
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.diagnostics import record_runner_phase
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch


class SeparationResult(Protocol):
    """Read only the complete native result; no local path enters a chunk."""

    @property
    def result(self) -> NativeSeparationResult:
        """Return independently verified complete source/stem metadata."""
        ...


class BoundSeparationModel(Protocol):
    """Worker-created model binding with a real retained resident lifetime."""

    @property
    def bound(self) -> BoundInstance:
        """Identify the complete authoritative placement bound to this model."""
        ...

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeSeparationReady:
        """Load the exact model and return its actual native ready receipt."""
        ...

    def is_ready(self) -> bool:
        """Observe current native liveness rather than a remembered receipt."""
        ...

    def separate(
        self,
        task: AudioSeparation,
        *,
        progress: Callable[[NativeSeparationProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 1800.0,
    ) -> SeparationResult | None:
        """Return verified complete facts, or drain cancelled/failed native work."""
        ...

    def close(self) -> None:
        """Drain native descendants before releasing lifetime/file ownership."""
        ...


@final
class Runner(ServedConcurrentDispatch):
    """Run one actual loaded separator with serial work and supervised recovery."""

    _generation_kinds = (AudioSeparation,)

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        model: BoundSeparationModel,
    ) -> None:
        """Bind worker-owned effects without importing native dependencies or loading."""
        if model.bound != bound_instance or not bound_instance.is_separation_model:
            raise ValueError("native model differs from runner placement")
        self.bound_instance = bound_instance
        self.event_sender, self.task_receiver = event_sender, task_receiver
        self.cancel_receiver = cancel_receiver
        self.shard_metadata = bound_instance.bound_shard
        self.runner_id = bound_instance.bound_runner_id
        self.model = model
        self.current_status: RunnerStatus = RunnerIdle()
        self.seen: set[TaskId] = set()
        self.cancelled_tasks: set[TaskId] = set()
        self._init_concurrent_dispatch(
            1, "native-separation", queue_admitted=True, max_admitted_queue=4
        )
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus) -> None:
        """Publish one observed runner lifecycle transition."""
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def acknowledge_task(self, task: Task) -> None:
        """Acknowledge bounded admission before long native work begins."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish the task lifecycle through the existing supervisor path."""
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def main(self) -> None:
        """Run the shared bounded dispatch, cancellation and shutdown lifecycle."""
        self.run_dispatch_loop()

    def _teardown_server(self) -> None:
        self.model.close()

    def _ensure_server_alive(self) -> None:
        with self._status_lock:
            active = self._inflight > 0
        if active:
            return  # The native call owns cancellation/failure and serial reload.
        if (
            isinstance(self.current_status, (RunnerReady, RunnerRunning, RunnerLoaded))
            and not self.model.is_ready()
        ):
            raise RuntimeError(
                "native separation process stopped; supervised restart is required"
            )

    def _load_model(self, is_cancelled: Callable[[], bool]) -> None:
        self.model.start(is_cancelled=is_cancelled, timeout_seconds=300)
        if not self.model.is_ready():
            raise RuntimeError("native separation model did not become ready")

    def handle_task(self, task: Task) -> None:
        """Connect, explicitly load and verify readiness before admitting inference."""
        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.current_status = RunnerConnected()
            case LoadModel() if isinstance(
                self.current_status, (RunnerIdle, RunnerConnected)
            ):
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)
                self._load_model(lambda: self._is_cancelled(task.task_id))
                self.current_status = RunnerLoaded()
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                if not self.model.is_ready():
                    raise RuntimeError(
                        "native separation process died before readiness"
                    )
                self.current_status = RunnerReady()
            case _:
                raise ValueError("unexpected native separation lifecycle task")

    def _generate(self, task: Task) -> None:
        assert isinstance(task, AudioSeparation)
        if self._is_cancelled(task.task_id):
            return

        def progress(update: NativeSeparationProgress) -> None:
            record_runner_phase(
                "decode_stream",
                event="separation_progress",
                task_id=task.task_id,
                command_id=str(task.command_id),
                attrs={"stage": update.stage, "frames": update.completed_frames},
            )

        succeeded = False
        try:
            generated = self.model.separate(
                task,
                progress=progress,
                is_cancelled=lambda: self._is_cancelled(task.task_id),
                timeout_seconds=1800,
            )
            if generated is None:
                if not self._is_cancelled(task.task_id):
                    raise RuntimeError(
                        "native separation ended without a complete result"
                    )
                return
            # Even a completed result racing cancellation belongs to the worker's
            # bounded output owner. Emit its facts before terminal task status so
            # that owner can retain/deliver or discard the files deliberately.
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=SeparationChunk(
                        model=self.shard_metadata.model_card.model_id,
                        finish_reason="stop",
                        output=generated.result.output,
                    ),
                )
            )
            succeeded = True
        except Exception as error:  # noqa: BLE001 - settle this admitted job
            logger.opt(exception=error).error("native separation task failed")
            self._mark_generation_failed(task.task_id)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.shard_metadata.model_card.model_id,
                        error_message="Native separation failed; inspect the runner diagnostics.",
                    ),
                )
            )
        finally:
            if not isinstance(
                self.current_status, (RunnerShuttingDown, RunnerShutdown)
            ):
                try:
                    if not self.model.is_ready():
                        self.model.close()
                        self._load_model(lambda: self._is_cancelled(CANCEL_ALL_TASKS))
                except Exception as error:  # noqa: BLE001 - preserve failed cleanup for supervisor
                    # Failure to restore the next model lifetime withdraws that
                    # runner. It must not retroactively fail a complete verified
                    # render already handed to the worker's output owner.
                    if not succeeded:
                        self._mark_generation_failed(task.task_id)
                    logger.opt(exception=error).error(
                        "native separation reload requires supervisor recovery"
                    )

    def _note_generation_finished(self) -> bool:
        # The shared loop publishes the task terminal before calling this hook.
        # Failed reload must become RunnerFailed then, never transient Ready,
        # while keeping the supervisor's Running -> task-terminal ordering.
        with self._status_lock:
            self._inflight = max(0, self._inflight - 1)
            self._publish_idle_status()
            return self._inflight == 0

    def _mark_ready_if_idle_after_waiter_terminal(self) -> None:
        # Cancelling the last queued job can make the runner idle after a
        # previous job's reload failed. It must use the same live readiness gate
        # as a finished generation rather than the mixin's unconditional Ready.
        with self._status_lock:
            self._publish_idle_status()

    def _publish_idle_status(self) -> None:
        # The caller owns _status_lock; do not reacquire it here.
        if (
            self._inflight == 0
            and self._dispatch_waiters == 0
            and isinstance(self.current_status, RunnerRunning)
        ):
            try:
                ready = self.model.is_ready()
            except Exception as error:  # noqa: BLE001 - liveness failure is not readiness
                ready = False
                logger.opt(exception=error).error(
                    "native separation liveness check failed"
                )
            self.update_status(
                RunnerReady()
                if ready
                else RunnerFailed(
                    error_message="Native separation requires supervised restart."
                )
            )
