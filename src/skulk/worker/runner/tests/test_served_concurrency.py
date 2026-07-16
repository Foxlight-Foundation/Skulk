# pyright: reportPrivateUsage=false, reportAny=false
"""Tests for the shared served-runner concurrent-dispatch mixin.

Drives the real ``ServedConcurrentDispatch`` loop over genuine mp channels with a
fake host that stubs the engine-specific hooks (``_generate``, server
liveness/teardown, ``handle_task``). Covers the properties that matter for both
served runners: non-blocking concurrent dispatch, the Ready<->Running status
transitions, the Ready-after-Complete ordering the supervisor asserts, semaphore
backpressure, and cancellation.
"""

import threading
import time
from typing import Any, cast

from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    LoadModel,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
)
from skulk.utils.channels import mp_channel
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch


class _FakeHost(ServedConcurrentDispatch):
    """Minimal host runner: real mixin, stubbed engine hooks + status plumbing."""

    def __init__(self, max_concurrency: int) -> None:
        self.runner_id = RunnerId("fake")
        self.seen: set[TaskId] = set()
        self.cancelled_tasks: set[TaskId] = set()
        self.current_status: RunnerStatus = RunnerIdle()
        self.events: list[Event] = []
        self._events_lock = threading.Lock()
        self.generate_gate: threading.Event | None = None
        self.started = threading.Semaphore(0)
        self.peak_inflight = 0
        self.admission_samples: list[int] = []
        self._admission_lock = threading.Lock()
        evt_s, self._evt_r = mp_channel[Event]()
        task_s, task_r = mp_channel[Task]()
        cancel_s, cancel_r = mp_channel[TaskId]()
        self.event_sender = evt_s
        self.task_receiver = task_r
        self.cancel_receiver = cancel_r
        self._task_sender = task_s
        self._cancel_sender = cancel_s
        # shard_metadata is only touched on the crash path; a duck-typed stub is
        # enough for these tests.
        self.shard_metadata: Any = type(
            "S", (), {"model_card": type("C", (), {"model_id": ModelId("m")})()}
        )()
        self._init_concurrent_dispatch(max_concurrency, "fake-gen")

    # engine hooks
    def _generate(self, task: Task) -> None:
        # Record the ADMISSION concurrency (#596): read from the dispatch-loop
        # capture, not the live count, mirroring what the real served runners do.
        with self._admission_lock:
            self.admission_samples.append(self._admission_concurrency(task.task_id))
        self.started.release()
        self.peak_inflight = max(self.peak_inflight, self._inflight_count())
        if self.generate_gate is not None:
            # Generous hang-guard only: every test sets the gate. A short timeout
            # would let the earliest generation return (and drop the in-flight
            # count) before the test asserts on it under full-suite CPU load.
            self.generate_gate.wait(60)
        # A real _generate polls _is_cancelled per streamed line (which drains the
        # cancel pipe into the shared set); mirror that so cancellation classifies.
        self._is_cancelled(task.task_id)

    def _ensure_server_alive(self) -> None:
        pass

    def _teardown_server(self) -> None:
        pass

    def _load_model(self, _task: Task) -> None:
        self.current_status = RunnerReady()

    def handle_task(self, task: Task) -> None:
        if isinstance(task, LoadModel):
            self._load_model(task)

    # status plumbing (records events)
    def update_status(self, status: RunnerStatus) -> None:
        self.current_status = status
        with self._events_lock:
            self.events.append(RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status))

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        with self._events_lock:
            self.events.append(TaskStatusUpdated(task_id=task.task_id, task_status=status))

    def acknowledge_task(self, task: Task) -> None:
        pass

    # test helpers
    def send(self, task: Task) -> None:
        self._task_sender.send(task)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run_dispatch_loop, daemon=True)
        t.start()
        return t

    def task_statuses(self, status: TaskStatus) -> int:
        with self._events_lock:
            return sum(
                1
                for e in self.events
                if isinstance(e, TaskStatusUpdated) and e.task_status is status
            )


def _iid() -> Any:
    from skulk.shared.types.worker.instances import InstanceId

    return InstanceId("i")


def _gen() -> TextGeneration:
    return TextGeneration(
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(model=ModelId("m"), input=[]),
        instance_id=_iid(),
    )


def _load_ready(host: _FakeHost) -> None:
    host.current_status = RunnerReady()


def _wait_inflight(host: _FakeHost, target: int, timeout: float = 5.0) -> None:
    """Poll the in-flight count to a target (robust to CPU contention under load)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and host._inflight_count() != target:
        time.sleep(0.02)


def test_dispatch_runs_generations_concurrently() -> None:
    host = _FakeHost(max_concurrency=4)
    _load_ready(host)
    host.generate_gate = threading.Event()
    t = host.start()
    try:
        for _ in range(3):
            host.send(_gen())
        for _ in range(3):
            assert host.started.acquire(timeout=10)
        _wait_inflight(host, 3)
        assert host._inflight_count() == 3
        assert isinstance(host.current_status, RunnerRunning)
        host.generate_gate.set()
        # Wait for all three generations to reach terminal Complete BEFORE
        # shutting down. Shutdown sets CANCEL_ALL, which correctly reclassifies a
        # still-finishing generation as Cancelled; sending it immediately races
        # the Completes we assert on here (a test-only race, seen under heavy
        # CPU load, not a product bug).
        deadline = time.monotonic() + 10
        while (
            time.monotonic() < deadline
            and host.task_statuses(TaskStatus.Complete) < 3
        ):
            time.sleep(0.02)
        assert host.task_statuses(TaskStatus.Complete) >= 3
    finally:
        host.send(Shutdown(instance_id=_iid(), runner_id=host.runner_id))
        t.join(timeout=5)


def test_admission_concurrency_captures_true_burst_spread() -> None:
    """A burst of N yields the 1..N admission spread, not all-N (#596).

    The dispatch loop captures each task's in-flight count at admission, so the
    envelope buckets the true concurrency curve. Sampling live in the worker
    thread would race the pool and collapse every sample onto N.
    """
    host = _FakeHost(max_concurrency=4)
    _load_ready(host)
    host.generate_gate = threading.Event()  # hold every generation in _generate
    t = host.start()
    try:
        for _ in range(4):
            host.send(_gen())
        # All four admitted and blocked in _generate before any finishes.
        for _ in range(4):
            assert host.started.acquire(timeout=10)
        _wait_inflight(host, 4)
        with host._admission_lock:
            samples = sorted(host.admission_samples)
        # Distinct 1..4: each task was counted in-flight at its own admission.
        assert samples == [1, 2, 3, 4], samples
        host.generate_gate.set()
        # Each generation pops its admission entry before it decrements the
        # in-flight count, so draining to 0 guarantees the map is empty (no leak).
        _wait_inflight(host, 0, timeout=10)
        assert host._admission_inflight == {}
    finally:
        host.generate_gate.set()
        host.send(Shutdown(instance_id=_iid(), runner_id=host.runner_id))
        t.join(timeout=5)


def test_backpressure_caps_submitted() -> None:
    host = _FakeHost(max_concurrency=2)
    _load_ready(host)
    host.generate_gate = threading.Event()
    t = host.start()
    try:
        for _ in range(4):
            host.send(_gen())
        assert host.started.acquire(timeout=10)
        assert host.started.acquire(timeout=10)
        _wait_inflight(host, 2)
        assert not host.started.acquire(timeout=1), "3rd ran despite the 2-permit cap"
        assert host._inflight_count() == 2
        host.generate_gate.set()
        assert host.started.acquire(timeout=10)
        assert host.started.acquire(timeout=10)
        assert host.peak_inflight == 2
    finally:
        host.generate_gate.set()
        host.send(Shutdown(instance_id=_iid(), runner_id=host.runner_id))
        t.join(timeout=5)


def test_load_broadcasts_ready_after_complete() -> None:
    host = _FakeHost(max_concurrency=2)
    host.current_status = RunnerIdle()
    t = host.start()
    try:
        host.send(LoadModel(instance_id=_iid()))
        complete_at: int | None = None
        ready_at: int | None = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and ready_at is None:
            time.sleep(0.05)
            with host._events_lock:
                for i, e in enumerate(host.events):
                    if (
                        isinstance(e, TaskStatusUpdated)
                        and e.task_status is TaskStatus.Complete
                        and complete_at is None
                    ):
                        complete_at = i
                    if isinstance(e, RunnerStatusUpdated) and isinstance(
                        e.runner_status, RunnerReady
                    ):
                        ready_at = i
        assert ready_at is not None and complete_at is not None
        # Ready must be broadcast AFTER the terminal Complete (supervisor assertion).
        assert complete_at < ready_at
    finally:
        host.send(Shutdown(instance_id=_iid(), runner_id=host.runner_id))
        t.join(timeout=5)


def test_cancel_marks_cancelled() -> None:
    host = _FakeHost(max_concurrency=2)
    _load_ready(host)
    host.generate_gate = threading.Event()
    t = host.start()
    try:
        gen = _gen()
        host.send(gen)
        assert host.started.acquire(timeout=5)
        host._cancel_sender.send(gen.task_id)
        # let the loop drain the cancel pipe via a peer poll, then release
        time.sleep(0.3)
        host.generate_gate.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if host.task_statuses(TaskStatus.Cancelled) >= 1:
                break
            time.sleep(0.05)
        assert host.task_statuses(TaskStatus.Cancelled) >= 1
    finally:
        host.generate_gate.set()
        host.send(Shutdown(instance_id=_iid(), runner_id=host.runner_id))
        t.join(timeout=5)


def test_stale_cancel_all_cleared_on_drain() -> None:
    host = _FakeHost(max_concurrency=2)
    _load_ready(host)
    with host._cancel_lock:
        host.cancelled_tasks.add(CANCEL_ALL_TASKS)
    host._inflight = 1
    host.current_status = RunnerRunning()
    # a generation finishing drains to idle and must clear the stale CANCEL_ALL
    gen = _gen()
    from concurrent.futures import Future

    fut: Future[None] = Future()
    fut.set_result(None)
    host._finish_generation(gen, fut)
    assert CANCEL_ALL_TASKS not in host.cancelled_tasks
    assert host._inflight == 0
    assert isinstance(host.current_status, RunnerReady)


def _base_stats() -> Any:
    from skulk.api.types import GenerationStats
    from skulk.shared.types.memory import Memory

    return GenerationStats(
        prompt_tps=1.0,
        generation_tps=2.0,
        prompt_tokens=3,
        generation_tokens=4,
        peak_memory_usage=Memory(in_bytes=0),
    )


def test_stamp_runner_stats_records_serving_ground_truth() -> None:
    """A batching served runner stamps its node, backend, in-flight, and mode."""
    from skulk.shared.types.common import NodeId

    host = _FakeHost(max_concurrency=4)
    host.bound_instance = cast(
        Any, type("B", (), {"bound_node_id": NodeId("node-7")})()
    )
    host.shard_metadata = cast(Any, type("S", (), {"resolved_backend": "vllm-cuda"})())

    stamped = host.stamp_runner_stats(_base_stats(), in_flight_at_admission=3)

    assert stamped.serving_node == "node-7"
    assert stamped.serving_backend == "vllm-cuda"
    assert stamped.in_flight_at_admission == 3
    assert stamped.serving_batches is True  # max_concurrency 4 > 1
    # The engine's own measurements survive the stamp.
    assert stamped.generation_tps == 2.0


def test_stamp_runner_stats_serial_reports_not_batching_and_floors_inflight() -> None:
    """A serial served runner reports batches=False and floors in-flight to 1."""
    from skulk.shared.types.common import NodeId

    host = _FakeHost(max_concurrency=1)
    host.bound_instance = cast(
        Any, type("B", (), {"bound_node_id": NodeId("node-1")})()
    )
    host.shard_metadata = cast(
        Any, type("S", (), {"resolved_backend": "llama_server-vulkan"})()
    )

    stamped = host.stamp_runner_stats(_base_stats(), in_flight_at_admission=0)

    assert stamped.serving_batches is False  # max_concurrency 1
    assert stamped.in_flight_at_admission == 1  # floored to >= 1
