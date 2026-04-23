import contextlib
import multiprocessing as mp
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    to_thread,
)
from loguru import logger

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.diagnostics import (
    RunnerLifecycleMilestone,
    RunnerSupervisorDiagnostics,
    RunnerTaskDiagnostics,
)
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ImageEdits,
    ImageGeneration,
    Task,
    TaskId,
    TaskStatus,
    TextEmbedding,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.utils.task_group import TaskGroup
from exo.worker.runner.bootstrap import entrypoint

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


def _now_utc_iso() -> str:
    """Return a compact UTC timestamp for live runner diagnostics."""

    return datetime.now(tz=timezone.utc).isoformat()


def _summarize_task(task: Task) -> str:
    """Return a compact task summary for logs."""
    if isinstance(task, TextGeneration):
        params = task.task_params
        return (
            "TextGeneration("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"model={params.model!r}, "
            f"input_messages={len(params.input)}, "
            f"chat_template_messages={len(params.chat_template_messages or [])}, "
            f"images={len(params.images)}, "
            f"cached_image_indices={sorted(params.image_hashes.keys())}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"image_count={params.image_count})"
        )
    return repr(task)


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: mp.Process
    initialize_timeout: float
    _ev_recv: MpReceiver[Event]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: MpSender[TaskId]
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    in_progress: dict[TaskId, Task] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )
    _status_since: str = field(default_factory=_now_utc_iso, init=False)
    _status_since_monotonic: float = field(default_factory=time.monotonic, init=False)
    _last_task_sent_at: str | None = field(default=None, init=False)
    _last_event_received_at: str | None = field(default=None, init=False)
    _last_event_type: str | None = field(default=None, init=False)
    _milestones: deque[RunnerLifecycleMilestone] = field(
        default_factory=lambda: deque(maxlen=32), init=False
    )

    def __post_init__(self) -> None:
        """Seed the lifecycle buffer with runner-supervisor construction."""

        self._record_milestone("supervisor_created", self.status.__class__.__name__)

    @classmethod
    def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        ev_send, ev_recv = mp_channel[Event]()
        task_sender, task_recv = mp_channel[Task]()
        cancel_sender, cancel_recv = mp_channel[TaskId]()

        runner_process = mp.Process(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                cancel_recv,
                logger,
            ),
            daemon=True,
        )

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
        )

        return self

    async def run(self):
        self._record_milestone("process_start_requested")
        self.runner_process.start()
        self._record_milestone(
            "process_started",
            f"pid={self.runner_process.pid}",
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._watch_runner)
                tg.start_soon(self._forward_events)
        finally:
            logger.info("Runner supervisor shutting down")
            if not self._cancel_watch_runner.cancel_called:
                self._cancel_watch_runner.cancel()
            with contextlib.suppress(ClosedResourceError):
                self._ev_recv.close()
            with contextlib.suppress(ClosedResourceError):
                self._task_sender.close()
            with contextlib.suppress(ClosedResourceError):
                self._event_sender.close()
            with contextlib.suppress(ClosedResourceError):
                self._cancel_sender.send(CANCEL_ALL_TASKS)
            with contextlib.suppress(ClosedResourceError):
                self._cancel_sender.close()

            await to_thread.run_sync(self.runner_process.join, 5)

            if self.runner_process.is_alive():
                logger.warning(
                    "Runner process didn't shutdown successfully, terminating"
                )
                self.runner_process.terminate()
                self.runner_process.join(timeout=5)
                # This is overkill but it's not technically bad, just unnecessary.
                if self.runner_process.is_alive():
                    logger.critical("Runner process didn't respond to SIGTERM, killing")
                    self.runner_process.kill()
                    self.runner_process.join(timeout=5)
            else:
                logger.info("Runner process successfully terminated")

            self.runner_process.close()

    def shutdown(self):
        self._record_milestone("shutdown_requested")
        self._tg.cancel_tasks()

    async def start_task(self, task: Task):
        if task.task_id in self.pending:
            logger.warning(
                f"Skipping invalid task {task} as it has already been submitted"
            )
            return
        if task.task_id in self.completed:
            logger.warning(
                f"Skipping invalid task {task} as it has already been completed"
            )
            return
        logger.info(f"Starting task {_summarize_task(task)}")
        event = anyio.Event()
        self.pending[task.task_id] = event
        self.in_progress[task.task_id] = task
        self._last_task_sent_at = _now_utc_iso()
        self._record_milestone(
            "task_sent",
            f"{task.__class__.__name__}:{task.task_id}",
        )
        try:
            await self._task_sender.send_async(task)
        except ClosedResourceError:
            self.in_progress.pop(task.task_id, None)
            logger.warning(f"Task {task} dropped, runner closed communication.")
            self._record_milestone(
                "task_send_failed",
                f"{task.__class__.__name__}:{task.task_id}",
            )
            return
        await event.wait()

    async def cancel_task(self, task_id: TaskId):
        if task_id in self.completed:
            logger.info(f"Unable to cancel {task_id} as it has been completed")
            self.cancelled.add(task_id)
            return
        self.cancelled.add(task_id)
        self._record_milestone("cancel_requested", str(task_id))
        with anyio.move_on_after(0.5) as scope:
            try:
                await self._cancel_sender.send_async(task_id)
            except ClosedResourceError:
                # typically occurs when trying to shut down a failed instance
                logger.warning(
                    f"Cancelling task {task_id} failed, runner closed communication"
                )
        if scope.cancel_called:
            logger.error("RunnerSupervisor cancel pipe blocked")
            await self._check_runner(TimeoutError("cancel pipe blocked"))

    async def _forward_events(self):
        try:
            with self._ev_recv as events:
                async for event in events:
                    self._last_event_received_at = _now_utc_iso()
                    self._last_event_type = event.__class__.__name__
                    self._record_milestone("event_received", self._last_event_type)
                    if isinstance(event, RunnerStatusUpdated):
                        self.status = event.runner_status
                        self._status_since = _now_utc_iso()
                        self._status_since_monotonic = time.monotonic()
                        self._record_milestone(
                            "status_changed",
                            event.runner_status.__class__.__name__,
                        )
                    if isinstance(event, TaskAcknowledged):
                        self.pending.pop(event.task_id).set()
                        self._record_milestone("task_acknowledged", str(event.task_id))
                        continue
                    if (
                        isinstance(event, TaskStatusUpdated)
                        and event.task_status == TaskStatus.Complete
                    ):
                        # If a task has just been completed, we should be working on it.
                        assert isinstance(
                            self.status,
                            (
                                RunnerRunning,
                                RunnerWarmingUp,
                                RunnerLoading,
                                RunnerConnecting,
                                RunnerShuttingDown,
                            ),
                        )
                        self.in_progress.pop(event.task_id, None)
                        self.completed.add(event.task_id)
                        self._record_milestone("task_completed", str(event.task_id))
                    await self._event_sender.send(event)
        except (ClosedResourceError, BrokenResourceError) as e:
            await self._check_runner(e)
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    async def _watch_runner(self) -> None:
        with self._cancel_watch_runner:
            while True:
                await anyio.sleep(5)
                if not self.runner_process.is_alive():
                    await self._check_runner(RuntimeError("Runner found to be dead"))

    async def _check_runner(self, e: Exception) -> None:
        self._record_milestone("runner_check", e.__class__.__name__)
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        logger.info("Checking runner's status")
        if self.runner_process.is_alive():
            logger.info("Runner was found to be alive, attempting to join process")
            await to_thread.run_sync(self.runner_process.join, 5)
        rc = self.runner_process.exitcode
        logger.info(f"Runner exited with exit code {rc}")
        if rc == 0:
            return

        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause = f"exitcode={rc}"

        logger.opt(exception=e).error(f"Runner terminated with {cause}")

        for task in self.in_progress.values():
            if isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
                with anyio.CancelScope(shield=True):
                    await self._event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ErrorChunk(
                                model=self.shard_metadata.model_card.model_id,
                                error_message=(
                                    "Runner shutdown before completing command "
                                    f"({cause})"
                                ),
                            ),
                        )
                    )

        try:
            self.status = RunnerFailed(error_message=f"Terminated ({cause})")
            with anyio.CancelScope(shield=True):
                await self._event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=self.bound_instance.bound_runner_id,
                        runner_status=RunnerFailed(
                            error_message=f"Terminated ({cause})"
                        ),
                    )
                )
        except (ClosedResourceError, BrokenResourceError):
            logger.warning(
                "Event sender already closed, unable to report runner failure"
            )
        self.shutdown()

    def _record_milestone(self, name: str, detail: str | None = None) -> None:
        """Append a bounded live milestone for diagnostics."""

        self._milestones.append(
            RunnerLifecycleMilestone(at=_now_utc_iso(), name=name, detail=detail)
        )

    def _task_diagnostics(self, task: Task) -> RunnerTaskDiagnostics:
        """Return a compact diagnostics view for a runner task."""

        command_id: str | None = None
        model_id = str(self.shard_metadata.model_card.model_id)
        if isinstance(
            task,
            (TextGeneration, ImageGeneration, ImageEdits, TextEmbedding),
        ):
            command_id = str(task.command_id)
            model_id = str(task.task_params.model)
        return RunnerTaskDiagnostics(
            task_id=str(task.task_id),
            task_kind=task.__class__.__name__,
            task_status=str(task.task_status.value),
            instance_id=str(task.instance_id),
            command_id=command_id,
            runner_id=str(self.bound_instance.bound_runner_id),
            model_id=model_id,
        )

    def diagnostics(self) -> RunnerSupervisorDiagnostics:
        """Return live read-only diagnostics for this runner supervisor."""

        return RunnerSupervisorDiagnostics(
            runner_id=str(self.bound_instance.bound_runner_id),
            instance_id=str(self.bound_instance.instance.instance_id),
            node_id=str(self.bound_instance.bound_node_id),
            model_id=str(self.shard_metadata.model_card.model_id),
            device_rank=self.shard_metadata.device_rank,
            world_size=self.shard_metadata.world_size,
            start_layer=self.shard_metadata.start_layer,
            end_layer=self.shard_metadata.end_layer,
            n_layers=self.shard_metadata.n_layers,
            pid=self.runner_process.pid,
            process_alive=self.runner_process.is_alive(),
            exit_code=self.runner_process.exitcode,
            status_kind=self.status.__class__.__name__,
            status_since=self._status_since,
            seconds_in_status=time.monotonic() - self._status_since_monotonic,
            pending_task_ids=[str(task_id) for task_id in self.pending],
            in_progress_tasks=[
                self._task_diagnostics(task) for task in self.in_progress.values()
            ],
            completed_task_count=len(self.completed),
            cancelled_task_ids=[str(task_id) for task_id in self.cancelled],
            last_task_sent_at=self._last_task_sent_at,
            last_event_received_at=self._last_event_received_at,
            last_event_type=self._last_event_type,
            milestones=list(self._milestones),
        )
