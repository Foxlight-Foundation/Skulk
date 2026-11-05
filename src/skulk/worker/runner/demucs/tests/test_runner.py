# pyright: reportPrivateUsage=false
"""Real dispatch channels/threads with model effects; not native qualification."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from skulk.master.tests.test_separation_admission import state
from skulk.shared.types.audio_separation import (
    SEPARATION_STEMS,
    AudioSeparationOutput,
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResult,
    SeparatedStem,
)
from skulk.shared.types.chunks import SeparationChunk
from skulk.shared.types.common import NodeId
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
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from skulk.utils.channels import mp_channel
from skulk.worker.runner.demucs.runner import Runner
from skulk.worker.runner.demucs.tests.test_model import fixture as model_fixture
from skulk.worker.tests.test_separation_input import fixture as input_fixture


@dataclass(frozen=True)
class Result:
    """Return complete typed metadata only; fake effects do not generate WAVs."""

    result: NativeSeparationResult


class Model:
    """Control native effects while exercising actual shared serial dispatch."""

    def __init__(self, bound: BoundInstance) -> None:
        self.bound = bound
        self.ready = False
        self.starts = 0
        self.closes = 0
        self.runs = 0
        self.peak = 0
        self.active = 0
        self.guard = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.mode: Literal[
            "success", "cancel", "failure", "empty", "completed_dead"
        ] = "success"
        self.fail_reload = False

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeSeparationReady:
        self.starts += 1
        if is_cancelled() or self.fail_reload and self.starts > 1:
            raise RuntimeError("fixture load refused")
        self.ready = True
        return NativeSeparationReady(
            payload_sha256="a" * 64,
            weights_sha256="b" * 64,
            device="cpu",
            threads=2,
            load_seconds=0.0,
            sample_rate=44100,
            audio_channels=2,
            stems=("drums", "bass", "other", "vocals"),
        )

    def is_ready(self) -> bool:
        return self.ready

    def separate(
        self,
        task: AudioSeparation,
        *,
        progress: Callable[[NativeSeparationProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 1800.0,
    ) -> Result | None:
        with self.guard:
            self.runs += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.entered.set()
        try:
            deadline = time.monotonic() + 5
            while not self.release.wait(0.005):
                if is_cancelled():
                    self.ready = False
                    return None
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixture generation release missing")
            if self.mode == "failure":
                self.ready = False
                raise RuntimeError("fixture native crash")
            if self.mode == "empty":
                return None
            if self.mode == "completed_dead":
                self.ready = False
            output = AudioSeparationOutput(
                source=task.task_params.source,
                stems=tuple(
                    SeparatedStem(
                        name=name,
                        sample_rate=44100,
                        frames=44100,
                        size_bytes=264644,
                        sha256="c" * 64,
                        peak=0.1,
                    )
                    for name in SEPARATION_STEMS
                ),
                common_gain=1.0,
                pre_gain_peak=0.1,
            )
            return Result(
                NativeSeparationResult(
                    output=output,
                    device="cpu",
                    load_seconds=0.0,
                    separation_seconds=0.01,
                    total_seconds=0.02,
                    process_peak_rss_bytes=1,
                )
            )
        finally:
            with self.guard:
                self.active -= 1

    def close(self) -> None:
        self.closes += 1
        self.ready = False


class Harness:
    """Drive the actual main loop and retain every observed event in order."""

    def __init__(self) -> None:
        current, _ = input_fixture()
        self.task = current.task
        snapshot = state(RunnerReady())
        instance = snapshot.instances[InstanceId("instance-0")]
        self.bound = BoundInstance(
            instance=instance,
            bound_node_id=NodeId("node-0"),
            bound_runner_id=RunnerId("runner-0"),
        )
        self.model = Model(self.bound)
        events, self.events = mp_channel[Event]()
        self.tasks, tasks = mp_channel[Task]()
        self.cancels, cancels = mp_channel[TaskId]()
        self.runner = Runner(self.bound, events, tasks, cancels, self.model)
        self.observed: list[Event] = []
        self.thread = threading.Thread(target=self.runner.main, daemon=True)
        self.thread.start()
        self.warmup = StartWarmup(instance_id=self.task.instance_id)
        for task in (
            ConnectToGroup(instance_id=self.task.instance_id),
            LoadModel(instance_id=self.task.instance_id),
            self.warmup,
        ):
            self.tasks.send(task)
        self.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )

    def until(self, predicate: Callable[[Event], bool]) -> Event:
        """Wait on genuine events rather than assuming a generation has finished."""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            event = self.events.receive_timeout(max(0.01, deadline - time.monotonic()))
            self.observed.append(event)
            if predicate(event):
                return event
        raise AssertionError("fixture runner did not publish expected event")

    def terminal(self, task: AudioSeparation) -> TaskStatus:
        """Return this job's actual dispatcher terminal status."""
        event = self.until(
            lambda event: isinstance(event, TaskStatusUpdated)
            and event.task_id == task.task_id
            and event.task_status
            in (TaskStatus.Complete, TaskStatus.Failed, TaskStatus.Cancelled)
        )
        assert isinstance(event, TaskStatusUpdated)
        return event.task_status

    def close(self) -> None:
        """Always drain active fake work and stop/join the actual dispatcher thread."""
        self.model.release.set()
        self.cancels.send(CANCEL_ALL_TASKS)
        self.tasks.send(
            Shutdown(
                instance_id=self.task.instance_id, runner_id=self.bound.bound_runner_id
            )
        )
        self.thread.join(5)
        assert not self.thread.is_alive()
        self.tasks.close()
        self.cancels.close()
        self.events.close()


def test_native_load_ready_order_and_complete_metadata_dispatch() -> None:
    """Installed libraries are not model readiness; real load/warmup precede Ready."""
    harness = Harness()
    try:
        terminal = next(
            index
            for index, event in enumerate(harness.observed)
            if isinstance(event, TaskStatusUpdated)
            and event.task_id == harness.warmup.task_id
            and event.task_status is TaskStatus.Complete
        )
        ready = next(
            index
            for index, event in enumerate(harness.observed)
            if isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        assert terminal < ready and harness.model.starts == 1
        harness.tasks.send(harness.task)
        assert harness.terminal(harness.task) is TaskStatus.Complete
        output = [
            event.chunk
            for event in harness.observed
            if isinstance(event, ChunkGenerated)
        ]
        assert len(output) == 1 and isinstance(output[0], SeparationChunk)
        assert output[0].output is not None and len(output[0].output.stems) == 4
        assert harness.model.peak == 1
    finally:
        harness.close()


def test_production_model_binding_satisfies_runner_contract_without_implicit_load(
    tmp_path: Path,
) -> None:
    """Use the actual model binding with injected resident effects at construction."""
    model, factory, _, _ = model_fixture(tmp_path)
    events, receiver = mp_channel[Event]()
    sender, tasks = mp_channel[Task]()
    cancels, cancel_receiver = mp_channel[TaskId]()
    try:
        runner = Runner(model.bound, events, tasks, cancel_receiver, model)
        assert runner.model is model and not model.is_ready() and not factory.calls
    finally:
        sender.close()
        tasks.close()
        cancels.close()
        cancel_receiver.close()
        events.close()
        receiver.close()
        model.close()


@pytest.mark.parametrize("mode", ["failure", "empty"])
def test_native_failure_or_empty_success_is_failed_not_complete(
    mode: Literal["failure", "empty"],
) -> None:
    """A renderer exception or absent result must not inherit generic Complete."""
    harness = Harness()
    try:
        harness.model.mode = mode
        harness.tasks.send(harness.task)
        assert harness.terminal(harness.task) is TaskStatus.Failed
        harness.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        assert harness.model.ready
    finally:
        harness.close()


def test_native_cancellation_drains_and_reloads_before_ready() -> None:
    """Cancellation goes through the serial native owner and normal task ordering."""
    harness = Harness()
    try:
        harness.model.release.clear()
        harness.tasks.send(harness.task)
        assert harness.model.entered.wait(2)
        harness.cancels.send(harness.task.task_id)
        assert harness.terminal(harness.task) is TaskStatus.Cancelled
        harness.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        assert harness.model.starts == 2 and harness.model.active == 0
    finally:
        harness.close()


def test_valid_completed_render_survives_failed_next_lifetime_reload() -> None:
    """Preserve verified output while withdrawing failed mounted-model readiness."""
    harness = Harness()
    try:
        harness.model.mode = "completed_dead"
        harness.model.fail_reload = True
        harness.tasks.send(harness.task)
        assert harness.terminal(harness.task) is TaskStatus.Complete
        harness.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerFailed)
        )
        assert not harness.model.ready
        assert any(
            isinstance(event, ChunkGenerated)
            and isinstance(event.chunk, SeparationChunk)
            for event in harness.observed
        )
    finally:
        harness.close()


def test_queued_native_generation_is_acknowledged_and_cancelled_without_running() -> (
    None
):
    """The shared dispatcher remains responsive while one native generation owns compute."""
    harness = Harness()
    try:
        harness.model.release.clear()
        harness.tasks.send(harness.task)
        assert harness.model.entered.wait(2)
        second = harness.task.model_copy(update={"task_id": TaskId("second")})
        harness.tasks.send(second)
        harness.until(
            lambda event: isinstance(event, TaskAcknowledged)
            and event.task_id == second.task_id
        )
        harness.cancels.send(second.task_id)
        assert harness.terminal(second) is TaskStatus.Cancelled
        assert harness.model.runs == harness.model.peak == 1
        harness.model.release.set()
        assert harness.terminal(harness.task) is TaskStatus.Complete
    finally:
        harness.close()


@pytest.mark.parametrize("ready", [True, False])
def test_last_queued_terminal_observes_native_liveness(ready: bool) -> None:
    """A cancelled final waiter must not restore a failed reload to Ready."""
    snapshot = state(RunnerReady())
    instance = snapshot.instances[InstanceId("instance-0")]
    bound = BoundInstance(
        instance=instance,
        bound_node_id=NodeId("node-0"),
        bound_runner_id=RunnerId("runner-0"),
    )
    model = Model(bound)
    events, receiver = mp_channel[Event]()
    sender, tasks = mp_channel[Task]()
    cancels, cancel_receiver = mp_channel[TaskId]()
    try:
        runner = Runner(bound, events, tasks, cancel_receiver, model)
        receiver.receive_timeout(1)  # Constructor's idle observation.
        model.ready = ready
        runner.current_status = RunnerRunning()
        # This is the state after the active job's terminal, followed by the
        # final queued job's cancellation and waiter decrement.
        runner._mark_ready_if_idle_after_waiter_terminal()
        event = receiver.receive_timeout(1)
        assert isinstance(event, RunnerStatusUpdated)
        assert isinstance(event.runner_status, RunnerReady if ready else RunnerFailed)
    finally:
        sender.close()
        tasks.close()
        cancels.close()
        cancel_receiver.close()
        events.close()
        receiver.close()
