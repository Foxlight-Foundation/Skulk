# pyright: reportAny=false, reportUnknownMemberType=false
"""Shared concurrent-dispatch loop for the served-backend runners.

The served engines (``llama_server``, ``vllm``) proxy an external inference server
that does its own continuous batching. To realize that batching the runner must
keep several requests in flight at once instead of serving them one at a time.
This mixin owns that loop -- receive tasks, dispatch each ``TextGeneration`` to a
bounded thread pool (one streaming request per worker thread), serialize the
lifecycle tasks (``LoadModel``, ``Shutdown``) on the dispatch thread -- and the
concrete runner supplies the engine-specific pieces (``_generate``, the server
liveness/teardown hooks, ``handle_task`` for ``LoadModel``).

The logic here was proven on the vLLM runner (PR #580) across many correctness
passes: the lock-guarded in-flight status, ack-before-backpressure, the
semaphore that caps SUBMITTED jobs (a bare ``ThreadPoolExecutor`` has an
unbounded submit queue), the Ready-after-Complete ordering the supervisor
asserts, the submit-failure recovery, and the stale-``CANCEL_ALL`` cleanup on
drain. Extracting it lets ``llama_server`` gain concurrency without
re-implementing (and re-reviewing) any of that.
"""

import contextlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from anyio import ClosedResourceError, EndOfStream, WouldBlock

from skulk.api.types import GenerationStats
from skulk.shared.types.chunks import ErrorChunk
from skulk.shared.types.events import ChunkGenerated, Event
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import ShardMetadata
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import record_runner_phase

# Between tasks the loop wakes at this cadence to verify the server subprocess is
# still alive (a dead server between requests must crash the runner, not wedge it
# Ready) and to re-poll for a dispatch slot while saturated.
_LIVENESS_POLL_S: float = 2.0


class ServedConcurrentDispatch:
    """Mixin: a bounded concurrent task-dispatch loop for a served-backend runner.

    The concrete runner MUST call :meth:`_init_concurrent_dispatch` in ``__init__``
    and provide the attributes/methods declared below. ``_generate``,
    ``_ensure_server_alive``, ``_teardown_server`` and ``handle_task`` are
    engine-specific; the rest of the concurrent machinery lives here.
    """

    # --- supplied by the concrete runner --------------------------------------
    event_sender: MpSender[Event]
    task_receiver: MpReceiver[Task]
    cancel_receiver: MpReceiver[TaskId]
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    seen: set[TaskId]
    cancelled_tasks: set[TaskId]
    current_status: RunnerStatus

    def _generate(self, task: Task) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _ensure_server_alive(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _teardown_server(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def handle_task(self, task: Task) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def acknowledge_task(self, task: Task) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def send_task_status(
        self, task: Task, status: TaskStatus
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def update_status(self, status: RunnerStatus) -> None:  # pragma: no cover
        raise NotImplementedError

    # --- concurrency state ----------------------------------------------------

    def _init_concurrent_dispatch(
        self, max_concurrency: int, thread_name_prefix: str
    ) -> None:
        """Set up the concurrency state. Call once from the runner's ``__init__``.

        ``max_concurrency`` bounds both the pool width and the number of SUBMITTED
        generations (via the permit semaphore), so excess load backpressures the
        task receiver rather than queueing unbounded in-process.
        """
        self._max_concurrency = max_concurrency
        self._dispatch_thread_prefix = thread_name_prefix
        # Worker threads mutate the cancel set and the in-flight counter, so both
        # are lock-guarded; never hold both locks at once (no nested acquisition).
        self._status_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._inflight: int = 0
        self._dispatch_waiters: int = 0
        self._dispatch_permits = threading.Semaphore(max_concurrency)
        # In-flight count captured at each task's ADMISSION, on the single
        # dispatch-loop thread (#596). Sampling in the worker thread instead would
        # race the pool: under a burst every worker could observe the peak count
        # and file all envelope samples into one concurrency bucket, erasing the
        # 1..N throughput-vs-concurrency curve. Keyed by task; dropped when the
        # generation finishes. Written on the dispatch thread and popped/read on
        # worker threads (the Future done-callback and the engine's stamp), so it
        # is guarded by its own lock -- matching the discipline used for the other
        # cross-thread state above, and never held while acquiring another lock.
        self._admission_inflight: dict[TaskId, int] = {}
        self._admission_lock = threading.Lock()

    # --- cancellation ---------------------------------------------------------

    def _drain_cancellations(self) -> None:
        # Concurrent generation threads all poll cancellation, so serialize the
        # single-consumer cancel pipe and the shared set behind the cancel lock.
        with self._cancel_lock:
            while True:
                try:
                    cancelled = self.cancel_receiver.receive_nowait()
                except WouldBlock:
                    break
                self.cancelled_tasks.add(cancelled)

    def _is_cancelled(self, task_id: TaskId) -> bool:
        self._drain_cancellations()
        return self._was_cancelled(task_id)

    def _was_cancelled(self, task_id: TaskId) -> bool:
        """Whether ``task_id`` is cancelled, WITHOUT draining the pipe.

        Used to classify a finished generation's terminal status: draining here
        could pull in a cancellation for a *different, still-running* task and,
        combined with a stale ``CANCEL_ALL``, misreport this one.
        """
        with self._cancel_lock:
            return (
                task_id in self.cancelled_tasks
                or CANCEL_ALL_TASKS in self.cancelled_tasks
            )

    # --- the loop -------------------------------------------------------------

    def run_dispatch_loop(self) -> None:
        # One thread per in-flight generation. The pool caps ACTIVE threads; the
        # _dispatch_permits semaphore (acquired before each submit) caps SUBMITTED
        # jobs to the same bound, so the pool's otherwise-unbounded submit queue
        # never accumulates a backlog -- excess load backpressures the receiver.
        pool = ThreadPoolExecutor(
            max_workers=self._max_concurrency,
            thread_name_prefix=self._dispatch_thread_prefix,
        )
        try:
            with self.task_receiver as tasks:
                while True:
                    try:
                        task = tasks.receive_timeout(_LIVENESS_POLL_S)
                    except WouldBlock:
                        # No task within the poll window: verify the server
                        # subprocess is still alive. Without this a server that
                        # dies BETWEEN requests leaves the runner gossiping Ready
                        # forever while every future request fails.
                        self._ensure_server_alive()
                        continue
                    except (EndOfStream, ClosedResourceError):
                        break
                    if task.task_id in self.seen:
                        logger.warning("repeat task - potential error")
                        continue
                    self.seen.add(task.task_id)
                    match task:
                        case TextGeneration() if isinstance(
                            self.current_status, (RunnerReady, RunnerRunning)
                        ):
                            # Acknowledge acceptance NOW, before any backpressure
                            # block: the supervisor's start_task waits on the ack
                            # before the worker can plan again, so deferring it
                            # until a slot frees would stall dispatch of
                            # cancellations / shutdown behind the first
                            # over-capacity request. Ack means "accepted".
                            self.acknowledge_task(task)
                            # Backpressure: block until a dispatch slot frees so the
                            # runner never accumulates an unbounded backlog. Wake
                            # periodically while saturated so a dead server is still
                            # caught by the liveness check.
                            self._clear_stale_cancel_all_if_idle()
                            self._note_dispatch_waiter_started()
                            permit_acquired = False
                            cancelled_while_waiting = False
                            try:
                                while not permit_acquired:
                                    permit_acquired = self._dispatch_permits.acquire(
                                        timeout=_LIVENESS_POLL_S
                                    )
                                    if permit_acquired:
                                        break
                                    self._ensure_server_alive()
                                    if self._is_cancelled(task.task_id):
                                        cancelled_while_waiting = True
                                        break
                                if (
                                    not cancelled_while_waiting
                                    and self._is_cancelled(task.task_id)
                                ):
                                    cancelled_while_waiting = True
                            finally:
                                self._note_dispatch_waiter_finished()
                            if cancelled_while_waiting:
                                if permit_acquired:
                                    self._dispatch_permits.release()
                                self.send_task_status(task, TaskStatus.Cancelled)
                                self._mark_ready_if_idle_after_waiter_terminal()
                                continue
                            self._dispatch_generation(task, pool)
                        case Shutdown():
                            self._handle_shutdown(task, pool)
                            break
                        case _:
                            # Lifecycle (LoadModel) or an out-of-state task: run it
                            # inline. LoadModel only occurs once, before any
                            # generation, so serial handling is correct.
                            self.send_task_status(task, TaskStatus.Running)
                            self.handle_task(task)
                            # _load_model sets current_status = RunnerReady() by
                            # direct assignment (no broadcast); broadcast it here so
                            # the worker learns the runner is ready -- but ONLY AFTER
                            # the terminal Complete. Order is load-bearing:
                            # RunnerSupervisor._forward_events asserts the runner is
                            # in an active state (Loading/Running/...) when a terminal
                            # task status arrives, so Ready must not precede Complete
                            # (else Loading -> Ready -> Complete trips that assertion
                            # and aborts the forwarder). Matches the old serial loop.
                            self.send_task_status(task, TaskStatus.Complete)
                            self.update_status(self.current_status)
        finally:
            # Drain in-flight generations, then stop the server. Shutdown already
            # cancels them; this also covers the EndOfStream / crash exits.
            pool.shutdown(wait=True)
            self._teardown_server()

    # --- dispatch -------------------------------------------------------------

    def _dispatch_generation(
        self, task: TextGeneration, pool: ThreadPoolExecutor
    ) -> None:
        """Admit a generation and run it on the pool without blocking the loop."""
        self.send_task_status(task, TaskStatus.Running)
        # Capture the admission concurrency atomically with the increment (#596):
        # _note_generation_started returns the post-increment count from inside
        # _status_lock, so no peer can decrement between counting this request in
        # flight and recording its admission bucket. This is its true position in
        # a burst (1, 2, ... N as the loop admits them). Store under _admission_lock
        # (never nested with _status_lock, which is already released here).
        admitted = self._note_generation_started()
        with self._admission_lock:
            self._admission_inflight[task.task_id] = admitted
        try:
            future = pool.submit(self._run_one_generation, task)
        except RuntimeError as exc:
            # The pool rejected the job (already shut down / broken). The
            # done-callback that releases the permit and decrements the in-flight
            # count will never run, so undo them here and surface a terminal error
            # rather than leaking a slot and wedging RunnerRunning.
            logger.opt(exception=exc).error("served dispatch submit failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.shard_metadata.model_card.model_id,
                        error_message=f"runner could not dispatch generation: {exc}",
                    ),
                )
            )
            self.send_task_status(task, TaskStatus.Failed)
            self._note_generation_finished()
            with self._admission_lock:
                self._admission_inflight.pop(task.task_id, None)
            self._dispatch_permits.release()
            return
        future.add_done_callback(lambda f: self._finish_generation(task, f))

    def _run_one_generation(self, task: TextGeneration) -> None:
        """Pool-worker body: stream one generation on a worker thread.

        ``_generate`` catches its own errors and surfaces them as an ErrorChunk;
        this outer guard only covers an unexpected escape so a crashed worker
        never swallows the stream silently (its terminal status is still emitted
        by the done-callback).
        """
        try:
            self._generate(task)
        except Exception as exc:  # noqa: BLE001 - defensive; keep the pool alive
            logger.opt(exception=exc).error("served generation worker crashed")
            with contextlib.suppress(Exception):
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=ErrorChunk(
                            model=self.shard_metadata.model_card.model_id,
                            error_message=str(exc),
                        ),
                    )
                )

    def _finish_generation(self, task: TextGeneration, future: "Future[None]") -> None:
        """Done-callback: emit the terminal task status and drop the in-flight count."""
        try:
            was_cancelled = self._was_cancelled(task.task_id)
            self.send_task_status(
                task,
                TaskStatus.Cancelled if was_cancelled else TaskStatus.Complete,
            )
        finally:
            with self._admission_lock:
                self._admission_inflight.pop(task.task_id, None)
            drained_to_idle = self._note_generation_finished()
            # Clear a stale cluster-wide cancel the moment the last in-flight
            # generation drains, so a CANCEL_ALL that arrived while requests were in
            # flight can't linger and spuriously cancel a later request. Not under
            # _status_lock (lock ordering); skipped during shutdown, which sets
            # CANCEL_ALL deliberately to break the draining streams.
            if drained_to_idle and not self._has_dispatch_waiters():
                with self._cancel_lock:
                    if not isinstance(
                        self.current_status, (RunnerShuttingDown, RunnerShutdown)
                    ):
                        self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
            # Release the backpressure slot this generation held.
            self._dispatch_permits.release()

    def _note_generation_started(self) -> int:
        """Count a generation in flight; return the post-increment count (#596).

        The count is returned from inside the ``_status_lock`` hold so the caller
        can record this request's admission concurrency atomically with the
        increment. Reading it in a separate ``_inflight_count()`` acquisition would
        leave a window in which a peer's ``_note_generation_finished`` could
        decrement first, filing the sample into a too-low concurrency bucket.
        """
        with self._status_lock:
            self._inflight += 1
            if self._inflight == 1 and isinstance(self.current_status, RunnerReady):
                self.update_status(RunnerRunning())
            return self._inflight

    def _note_generation_finished(self) -> bool:
        """Drop the in-flight count; return True if this drained to idle (0)."""
        with self._status_lock:
            self._inflight = max(0, self._inflight - 1)
            if (
                self._inflight == 0
                and self._dispatch_waiters == 0
                and isinstance(self.current_status, RunnerRunning)
            ):
                self.update_status(RunnerReady())
            return self._inflight == 0

    def _inflight_count(self) -> int:
        with self._status_lock:
            return self._inflight

    def _note_dispatch_waiter_started(self) -> None:
        """Record that an accepted generation is waiting for a dispatch permit."""
        with self._status_lock:
            self._dispatch_waiters += 1

    def _note_dispatch_waiter_finished(self) -> None:
        """Drop the count of accepted generations waiting for a dispatch permit."""
        with self._status_lock:
            self._dispatch_waiters = max(0, self._dispatch_waiters - 1)

    def _has_dispatch_waiters(self) -> bool:
        """Whether any acknowledged generation is blocked behind backpressure."""
        with self._status_lock:
            return self._dispatch_waiters > 0

    def _clear_stale_cancel_all_if_idle(self) -> None:
        """Drop old cancel-all markers before accepting fresh idle work.

        A cancel-all that arrives while another acknowledged task is waiting is
        still live and must not be cleared by the generation that drains first.
        """
        if self._inflight_count() == 0 and not self._has_dispatch_waiters():
            with self._cancel_lock:
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)

    def _mark_ready_if_idle_after_waiter_terminal(self) -> None:
        """Return to Ready after a never-dispatched waiter emits its terminal status."""
        with self._status_lock:
            if (
                self._inflight == 0
                and self._dispatch_waiters == 0
                and isinstance(self.current_status, RunnerRunning)
            ):
                self.update_status(RunnerReady())

    def _admission_concurrency(self, task_id: TaskId) -> int:
        """In-flight count captured when ``task_id`` was admitted (#596).

        Read by the engine-specific ``_generate`` when stamping its stats. Falls
        back to the live in-flight count if the admission capture is missing
        (defensive; should not happen for a dispatched task), so a stamp is never
        keyed to 0. The map read is under ``_admission_lock``; the fallback
        ``_inflight_count()`` (which takes ``_status_lock``) runs outside it so
        the two locks are never nested.
        """
        with self._admission_lock:
            captured = self._admission_inflight.get(task_id)
        return captured if captured is not None else self._inflight_count()

    def _set_admission_concurrency(self, task_id: TaskId, concurrency: int) -> None:
        """Replace a task's stamp with engine-resource-active concurrency.

        The generic dispatch count is correct when the external server admits
        every submitted request immediately. An engine with an additional
        resource budget may refine it after that gate opens so envelope samples
        count active competitors rather than requests still waiting for budget.

        Args:
            task_id: The admitted generation whose stamp should be refined.
            concurrency: Positive count of resource-active generations.
        """
        if concurrency < 1:
            raise ValueError("admission concurrency must be positive")
        with self._admission_lock:
            if task_id in self._admission_inflight:
                self._admission_inflight[task_id] = concurrency

    def stamp_runner_stats(
        self, stats: GenerationStats, in_flight_at_admission: int
    ) -> GenerationStats:
        """Stamp runner ground truth onto a generation's stats (#596).

        The performance-envelope tap on the API attributes each generation to the
        serving instance using these fields, so the envelope reflects the true
        per-instance in-flight concurrency (immune to which API node dispatched
        the request) and the correct batching classification.
        ``in_flight_at_admission`` is the count captured on the dispatch loop when
        this generation was admitted (see ``_admission_concurrency``), so a burst
        of N requests yields the true 1..N spread rather than all landing on N; a
        serial served config reports 1, a batching one reports up to its
        parallelism.
        """
        return stats.model_copy(
            update={
                "serving_node": str(self.bound_instance.bound_node_id),
                "serving_backend": self.shard_metadata.resolved_backend,
                "in_flight_at_admission": max(1, in_flight_at_admission),
                "serving_batches": self._max_concurrency > 1,
            }
        )

    def _handle_shutdown(self, task: Task, pool: ThreadPoolExecutor) -> None:
        """Cancel in-flight generations, drain the pool, then tear down the server."""
        logger.info("served runner shutting down")
        # Emit Running before the terminal Complete: the worker's task-lifecycle
        # contract expects Running -> Complete for every task, shutdown included.
        self.send_task_status(task, TaskStatus.Running)
        record_runner_phase(
            "shutdown_cleanup",
            event="runner_shutdown_requested",
            task_id=task.task_id,
        )
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        # Break every in-flight stream: their loops poll _is_cancelled.
        with self._cancel_lock:
            self.cancelled_tasks.add(CANCEL_ALL_TASKS)
        pool.shutdown(wait=True)
        self._teardown_server()
        record_runner_phase(
            "shutdown_cleanup",
            event="server_teardown_complete",
            task_id=task.task_id,
        )
        self.current_status = RunnerShutdown()
        self.send_task_status(task, TaskStatus.Complete)
        self.update_status(RunnerShutdown())
