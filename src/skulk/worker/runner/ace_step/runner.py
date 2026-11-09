"""Serial native music lifecycle over the shared bounded runner dispatcher.

The bootstrap owner supplies an already validated model binding and private
source root. Each replacement owns a new unloaded lifetime for that same binding.
This runner cannot prepare a package or admit input by itself.
Only complete output facts are emitted; the worker owns addressed WAV delivery.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol, final

from loguru import logger

from skulk.shared.constants import NATIVE_MUSIC_RENDER_SECONDS
from skulk.shared.types.chunks import ErrorChunk, MusicChunk, MusicEditingChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.music import MusicOutputManifest
from skulk.shared.types.music_edits import MusicEditingOutput
from skulk.shared.types.native_music import (
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicResult,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    LoadModel,
    MusicEditing,
    MusicGeneration,
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


class MusicResult(Protocol):
    """Read only the complete native result; no local path enters a chunk."""

    @property
    def result(self) -> NativeMusicResult:
        """Return independently verified complete output/provenance metadata."""
        ...


class BoundNativeMusicModel(Protocol):
    """Worker-created model binding with a real retained resident lifetime."""

    @property
    def bound(self) -> BoundInstance:
        """Identify the complete authoritative placement bound to this model."""
        ...

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeMusicReady:
        """Load the exact model and return its actual native ready receipt."""
        ...

    def is_ready(self) -> bool:
        """Observe current native liveness rather than a remembered receipt."""
        ...

    def run(
        self,
        task: MusicGeneration | MusicEditing,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = NATIVE_MUSIC_RENDER_SECONDS,
    ) -> MusicResult | None:
        """Return verified complete facts, or drain cancelled/failed native work."""
        ...

    def close(self) -> None:
        """Drain native descendants before releasing lifetime/file ownership."""
        ...


@final
class Runner(ServedConcurrentDispatch):
    """Run one actual loaded music model with serial work and supervised recovery."""

    _generation_kinds = (MusicGeneration, MusicEditing)

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        model: BoundNativeMusicModel,
        create_model: Callable[[], BoundNativeMusicModel],
    ) -> None:
        """Bind unloaded Core-owned effects without loading or preparing packages.

        The initial model and factory must retain the same authoritative binding.
        The factory creates a fresh exclusive lifetime after cancellation/fault;
        it may not reuse the retired model or take another workload's ownership.
        """
        card = bound_instance.bound_shard.model_card
        if (
            model.bound != bound_instance
            or not bound_instance.is_music_model
            or card.music is None
            or card.music.native is None
            or bound_instance.bound_shard.native_music_policy is None
            or model.is_ready()
        ):
            raise ValueError("native model differs from runner placement")
        self.bound_instance = bound_instance
        self.event_sender, self.task_receiver = event_sender, task_receiver
        self.cancel_receiver = cancel_receiver
        self.shard_metadata = bound_instance.bound_shard
        self.runner_id = bound_instance.bound_runner_id
        self.model = model
        self._create_model = create_model
        self._model_guard = threading.Lock()
        self._closing = threading.Event()
        self._restart_required = threading.Event()
        self.current_status: RunnerStatus = RunnerIdle()
        self.seen: set[TaskId] = set()
        self.cancelled_tasks: set[TaskId] = set()
        self._init_concurrent_dispatch(
            1, "native-music", queue_admitted=True, max_admitted_queue=4
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
        with self._model_guard:
            self._closing.set()
            model = self.model
        model.close()

    def _ensure_server_alive(self) -> None:
        with self._status_lock:
            active = self._inflight > 0
        if active:
            return  # The native call owns cancellation/failure and serial reload.
        if isinstance(
            self.current_status, (RunnerReady, RunnerRunning, RunnerLoaded)
        ) and (self._restart_required.is_set() or not self.model.is_ready()):
            raise RuntimeError(
                "native music process stopped; supervised restart is required"
            )

    def _load_model(self, is_cancelled: Callable[[], bool]) -> None:
        self.model.start(is_cancelled=is_cancelled, timeout_seconds=600)
        if not self.model.is_ready():
            raise RuntimeError("native music model did not become ready")

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
                    raise RuntimeError("native music process died before readiness")
                self.current_status = RunnerReady()
            case _:
                raise ValueError("unexpected native music lifecycle task")

    def _completion(
        self, task: MusicGeneration | MusicEditing, result: NativeMusicResult
    ) -> MusicChunk | MusicEditingChunk:
        policy = self.shard_metadata.native_music_policy
        operation = (
            "generate" if isinstance(task, MusicGeneration) else task.task_params.task
        )
        if (
            policy is None
            or result.request_id != str(task.command_id)
            or result.operation != operation
            or result.seed != task.task_params.seed
            or result.model_identity != policy.model_identity
            or result.payload_sha256 != policy.payload_sha256
        ):
            raise ValueError("native music completion differs from admitted task")
        if isinstance(task, MusicEditing):
            return MusicEditingChunk(
                model=self.shard_metadata.model_card.model_id,
                finish_reason="stop",
                output=MusicEditingOutput(
                    source=task.task_params.source, result=result
                ),
            )
        return MusicChunk(
            model=self.shard_metadata.model_card.model_id,
            finish_reason="stop",
            output=MusicOutputManifest(
                size_bytes=result.output.size_bytes,
                sha256=result.output.sha256,
                duration_seconds=result.output.frames / result.output.sample_rate,
                sample_rate=result.output.sample_rate,
                channels=result.output.channels,
            ),
        )

    def _recreate_model(self) -> None:
        # A retired music binding cannot be restarted. The bootstrap factory
        # supplies a new exclusive lifetime for the same prepared placement;
        # closing the old one drains descendants but retains journal-owned WAVs.
        self._restart_required.set()
        previous = self.model
        previous.close()
        if self._closing.is_set() or self._is_cancelled(CANCEL_ALL_TASKS):
            return
        replacement = self._create_model()
        if (
            replacement is previous
            or replacement.bound != self.bound_instance
            or replacement.is_ready()
        ):
            # A foreign or already-serving object has no proven ownership here.
            # Do not close it or change another workload in response to bad wiring.
            raise ValueError("native music factory requires a fresh unloaded binding")
        with self._model_guard:
            closing = self._closing.is_set()
            if not closing:
                self.model = replacement
        if closing:
            replacement.close()
            return
        self._load_model(
            lambda: self._closing.is_set() or self._is_cancelled(CANCEL_ALL_TASKS)
        )
        self._restart_required.clear()

    def _generate(self, task: Task) -> None:
        assert isinstance(task, (MusicGeneration, MusicEditing))
        if self._is_cancelled(task.task_id):
            return

        def progress(update: NativeMusicProgress) -> None:
            record_runner_phase(
                "decode_stream",
                event="native_music_progress",
                task_id=task.task_id,
                command_id=str(task.command_id),
                attrs={"stage": update.stage},
            )

        succeeded = False
        retire = False
        try:
            if self._restart_required.is_set():
                raise RuntimeError("native music requires supervised model recovery")
            generated = self.model.run(
                task,
                progress=progress,
                is_cancelled=lambda: self._is_cancelled(task.task_id),
                timeout_seconds=NATIVE_MUSIC_RENDER_SECONDS,
            )
            if generated is None:
                retire = True
                if not self._is_cancelled(task.task_id):
                    raise RuntimeError("native music ended without a complete result")
                return
            # Completion facts precede terminal task state even if cancellation
            # races them. The worker's independent media owner settles delivery.
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=self._completion(task, generated.result),
                )
            )
            succeeded = True
            retire = self._is_cancelled(task.task_id)
        except Exception as error:  # noqa: BLE001 - settle this admitted job
            retire = True
            logger.opt(exception=error).error("native music task failed")
            self._mark_generation_failed(task.task_id)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.shard_metadata.model_card.model_id,
                        error_message="Native music failed; inspect the runner diagnostics.",
                    ),
                )
            )
        finally:
            if not isinstance(
                self.current_status, (RunnerShuttingDown, RunnerShutdown)
            ):
                try:
                    if retire or not self.model.is_ready():
                        self._recreate_model()
                except Exception as error:  # noqa: BLE001 - retain cleanup for supervisor
                    self._restart_required.set()
                    # Preserve a completed render already handed to media custody.
                    # Withdrawing the failed replacement happens after task terminal.
                    if not succeeded:
                        self._mark_generation_failed(task.task_id)
                    logger.opt(exception=error).error(
                        "native music reload requires supervisor recovery"
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
                ready = (
                    not self._closing.is_set()
                    and not self._restart_required.is_set()
                    and self.model.is_ready()
                )
            except Exception as error:  # noqa: BLE001 - liveness failure is not readiness
                ready = False
                logger.opt(exception=error).error("native music liveness check failed")
            self.update_status(
                RunnerReady()
                if ready
                else RunnerFailed(
                    error_message="Native music requires supervised restart."
                )
            )
