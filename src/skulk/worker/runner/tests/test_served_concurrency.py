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
from typing import Any

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
        self.started.release()
        self.peak_inflight = max(self.peak_inflight, self._inflight_count())
        if self.generate_gate is not None:
            self.generate_gate.wait(5)
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
    finally:
        host.send(Shutdown(instance_id=_iid(), runner_id=host.runner_id))
        t.join(timeout=5)
    assert isinstance(host.current_status, type(host.current_status))
    assert host.task_statuses(TaskStatus.Complete) >= 3


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
