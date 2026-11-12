"""Actual dispatch/cancellation threads with controlled native model effects.

These cases prove Core lifecycle and provenance; they load no music model.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from skulk.shared.types.chunks import MusicChunk, MusicEditingChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.native_music import (
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicWav,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    LoadModel,
    MusicEditing,
    MusicGeneration,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerFailed, RunnerReady
from skulk.utils.channels import mp_channel
from skulk.worker.runner.ace_step.model import NativeMusicModel
from skulk.worker.runner.ace_step.result_custody import music_result_directory
from skulk.worker.runner.ace_step.runner import Runner
from skulk.worker.runner.ace_step.tests.test_model import (
    Factory,
    edit_task,
)
from skulk.worker.runner.ace_step.tests.test_model import (
    fixture as model_fixture,
)


@dataclass(frozen=True)
class Result:
    """Typed completion facts; controlled effects do not create real audio."""

    result: NativeMusicResult


class Model:
    """Observe real dispatcher admission while controlling load, wait and failure."""

    def __init__(self, bound: BoundInstance) -> None:
        self.bound = bound
        self.ready = False
        self.starts = 0
        self.closes = 0
        self.runs = 0
        self.active = 0
        self.peak = 0
        self.guard = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.mode: Literal[
            "success", "failure", "empty", "foreign", "completed_dead"
        ] = "success"
        self.fail_close = False

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeMusicReady:
        """Return the admitted policy without native imports or model execution."""
        assert timeout_seconds > 0
        if is_cancelled():
            raise InterruptedError("fixture load cancelled")
        policy = self.bound.bound_shard.native_music_policy
        assert policy is not None
        self.starts += 1
        self.ready = True
        return NativeMusicReady(
            model_identity=policy.model_identity,
            payload_sha256=policy.payload_sha256,
            execution=policy.execution,
            load_seconds=0.01,
        )

    def is_ready(self) -> bool:
        """Observe controlled native liveness without changing it."""
        return self.ready

    def run(
        self,
        task: MusicGeneration | MusicEditing,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 7200.0,
    ) -> Result | None:
        """Emit exact task provenance while allowing cancellation during a real wait."""
        assert timeout_seconds > 0 and self.ready
        with self.guard:
            self.active += 1
            self.runs += 1
            self.peak = max(self.peak, self.active)
        self.entered.set()
        try:
            deadline = time.monotonic() + 5
            while not self.release.wait(0.005):
                if is_cancelled():
                    self.ready = False
                    return None
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixture release missing")
            if self.mode == "failure":
                self.ready = False
                raise RuntimeError("controlled native crash")
            if self.mode == "empty":
                return None
            policy = self.bound.bound_shard.native_music_policy
            assert policy is not None and task.task_params.seed is not None
            editing = isinstance(task, MusicEditing)
            frames = (
                task.task_params.target_frames
                if isinstance(task, MusicEditing)
                else task.task_params.seconds * 48000
            )
            result = NativeMusicResult(
                request_id="foreign"
                if self.mode == "foreign"
                else str(task.command_id),
                model_identity=policy.model_identity,
                payload_sha256=policy.payload_sha256,
                operation=task.task_params.task
                if isinstance(task, MusicEditing)
                else "generate",
                seed=task.task_params.seed,
                source_sha256=task.task_params.source.sha256
                if isinstance(task, MusicEditing)
                else None,
                conditioning_sha256="a" * 64 if editing else None,
                conditioning_gain=1.0 if editing else None,
                conditioning_padding_frames=(-frames % 1920) if editing else 0,
                execution=policy.execution,
                output=NativeMusicWav(
                    sha256="c" * 64,
                    size_bytes=frames * 6 + 44,
                    frames=frames,
                    peak=0.1,
                    gain=1.0,
                ),
                generation_seconds=0.01,
            )
            if self.mode == "completed_dead":
                self.ready = False
            progress(
                NativeMusicProgress(request_id=str(task.command_id), stage="writing")
            )
            return Result(result)
        finally:
            with self.guard:
                self.active -= 1

    def close(self) -> None:
        """Count owned drain attempts, optionally leaving an ambiguous live process."""
        self.closes += 1
        if self.fail_close:
            raise RuntimeError("controlled cleanup failure")
        self.ready = False


class Harness:
    """Own actual channels/dispatcher and a fresh-binding factory for one placement."""

    def __init__(self, root: Path) -> None:
        self.real, _, self.task = model_fixture(root)
        self.bound = self.real.bound
        self.models = [Model(self.bound)]
        self.fail_factory = False
        self.return_existing = False
        self.foreign: Model | None = None
        self.errors: list[Exception] = []
        events, self.events = mp_channel[Event]()
        self.tasks, tasks = mp_channel[Task]()
        self.cancels, cancels = mp_channel[TaskId]()
        self.runner = Runner(
            self.bound, events, tasks, cancels, self.models[0], self.replace
        )
        self.observed: list[Event] = []
        self.thread = threading.Thread(target=self.run, daemon=True)
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

    def replace(self) -> Model:
        """Create only a new unloaded binding, with explicit fault injection."""
        if self.fail_factory:
            raise RuntimeError("controlled replacement failure")
        if self.foreign is not None:
            return self.foreign
        if self.return_existing:
            return self.models[-1]
        model = Model(self.bound)
        self.models.append(model)
        return model

    def run(self) -> None:
        """Retain unexpected dispatcher errors for the fixture's final assertion."""
        try:
            self.runner.main()
        except Exception as error:
            self.errors.append(error)

    def until(self, predicate: Callable[[Event], bool]) -> Event:
        """Observe actual channel events instead of sleeping past a transition."""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            event = self.events.receive_timeout(max(0.01, deadline - time.monotonic()))
            self.observed.append(event)
            if predicate(event):
                return event
        raise AssertionError("native music runner did not publish expected event")

    def terminal(self, task: MusicGeneration | MusicEditing) -> TaskStatus:
        """Read the real shared dispatcher's terminal for this exact task."""
        event = self.until(
            lambda event: isinstance(event, TaskStatusUpdated)
            and event.task_id == task.task_id
            and event.task_status
            in (TaskStatus.Complete, TaskStatus.Failed, TaskStatus.Cancelled)
        )
        assert isinstance(event, TaskStatusUpdated)
        return event.task_status

    def close(self) -> None:
        """Release controlled waits and join all owned dispatch work before returning."""
        for model in self.models:
            model.fail_close = False
            model.release.set()
        self.cancels.send(CANCEL_ALL_TASKS)
        self.tasks.send(
            Shutdown(
                instance_id=self.task.instance_id, runner_id=self.bound.bound_runner_id
            )
        )
        self.thread.join(5)
        assert not self.thread.is_alive() and not self.errors
        self.tasks.close()
        self.cancels.close()
        self.events.close()
        self.real.close()


def test_generation_uses_real_load_order_and_only_measured_wav_metadata(
    tmp_path: Path,
) -> None:
    """Packages/bindings alone do not publish model readiness or completed content."""
    harness = Harness(tmp_path)
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
        assert terminal < ready and harness.models[0].starts == 1
        harness.tasks.send(harness.task)
        assert harness.terminal(harness.task) is TaskStatus.Complete
        chunks = [
            event.chunk
            for event in harness.observed
            if isinstance(event, ChunkGenerated)
        ]
        assert len(chunks) == 1 and isinstance(chunks[0], MusicChunk)
        assert chunks[0].output is not None and chunks[0].output.duration_seconds == 5
        assert "directory" not in chunks[0].model_dump_json()
        assert len(harness.models) == 1 and harness.models[0].peak == 1
    finally:
        harness.close()


@pytest.mark.parametrize("mode", ["add_part", "repaint", "continue"])
def test_edit_task_enters_shared_dispatch_and_retains_original_lineage(
    tmp_path: Path, mode: str
) -> None:
    """All three actual resolved edit types dispatch, rather than falling into lifecycle handling."""
    harness = Harness(tmp_path)
    try:
        task = edit_task(harness.real, mode)
        harness.tasks.send(task)
        assert harness.terminal(task) is TaskStatus.Complete
        chunks = [
            event.chunk
            for event in harness.observed
            if isinstance(event, ChunkGenerated)
        ]
        assert len(chunks) == 1 and isinstance(chunks[0], MusicEditingChunk)
        assert chunks[0].output is not None
        assert chunks[0].output.source == task.task_params.source
        assert chunks[0].output.result.operation == mode
        assert chunks[0].output.result.seed == task.task_params.seed
        assert "original.wav" not in chunks[0].model_dump_json()
    finally:
        harness.close()


@pytest.mark.parametrize("mode", ["failure", "empty", "foreign"])
def test_failure_or_foreign_completion_retires_binding_before_ready(
    tmp_path: Path, mode: Literal["failure", "empty", "foreign"]
) -> None:
    """An exception, empty success or mismatched receipt cannot become completion."""
    harness = Harness(tmp_path)
    try:
        old = harness.models[0]
        old.mode = mode
        harness.tasks.send(harness.task)
        assert harness.terminal(harness.task) is TaskStatus.Failed
        harness.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        assert old.closes == 1 and old.starts == 1 and not old.ready
        assert len(harness.models) == 2 and harness.models[1].starts == 1
        assert not any(
            isinstance(event, ChunkGenerated)
            and isinstance(event.chunk, (MusicChunk, MusicEditingChunk))
            for event in harness.observed
        )
    finally:
        harness.close()


def test_cancel_drains_old_binding_and_constructs_fresh_lifetime(
    tmp_path: Path,
) -> None:
    """Cancellation reaches the active native call and never restarts its retired object."""
    harness = Harness(tmp_path)
    try:
        old = harness.models[0]
        old.release.clear()
        harness.tasks.send(harness.task)
        assert old.entered.wait(2)
        harness.cancels.send(harness.task.task_id)
        assert harness.terminal(harness.task) is TaskStatus.Cancelled
        harness.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        assert old.active == 0 and old.closes == 1 and old.starts == 1
        assert len(harness.models) == 2 and harness.models[1].ready
    finally:
        harness.close()


@pytest.mark.parametrize("fault", ["factory", "reuse", "cleanup"])
def test_failed_replacement_withdraws_readiness_even_if_old_process_reports_live(
    tmp_path: Path, fault: str
) -> None:
    """A failed close/factory cannot transiently report Ready after the task terminal."""
    harness = Harness(tmp_path)
    try:
        old = harness.models[0]
        old.mode = "foreign"
        harness.fail_factory = fault == "factory"
        harness.return_existing = fault == "reuse"
        old.fail_close = fault == "cleanup"
        harness.tasks.send(harness.task)
        assert harness.terminal(harness.task) is TaskStatus.Failed
        harness.until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerFailed)
        )
        after_terminal = next(
            index
            for index, event in enumerate(harness.observed)
            if isinstance(event, TaskStatusUpdated)
            and event.task_id == harness.task.task_id
            and event.task_status is TaskStatus.Failed
        )
        assert not any(
            isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
            for event in harness.observed[after_terminal + 1 :]
        )
        assert len(harness.models) == 1
    finally:
        harness.close()


def test_generation_and_edit_share_one_slot_and_acknowledge_queued_work(
    tmp_path: Path,
) -> None:
    """Source editing cannot run concurrently with generation on the same model."""
    harness = Harness(tmp_path)
    try:
        old = harness.models[0]
        old.release.clear()
        harness.tasks.send(harness.task)
        assert old.entered.wait(2)
        editing = edit_task(harness.real, "add_part")
        harness.tasks.send(editing)
        harness.until(
            lambda event: isinstance(event, TaskAcknowledged)
            and event.task_id == editing.task_id
        )
        assert old.runs == 1 and old.active == 1
        old.release.set()
        assert harness.terminal(harness.task) is TaskStatus.Complete
        assert harness.terminal(editing) is TaskStatus.Complete
        assert old.runs == 2 and old.peak == 1
    finally:
        harness.close()


def test_actual_model_binding_recreates_without_deleting_completed_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual artifact/workspace/WAV custody survives replacing a dead resident.

    Native inference alone is controlled; dispatcher, model adapter, verified
    tiny artifact files and result storage are the production implementations.
    """
    model, factory, task = model_fixture(tmp_path)
    replacements: list[NativeMusicModel] = []

    def replacement() -> NativeMusicModel:
        native = NativeMusicModel(
            model.bound,
            model.runtime,
            model.sources,
            tmp_path / f"replacement-{len(replacements)}",
            model.source_root,
            create_resident=Factory(model.bound),
        )
        replacements.append(native)
        return native

    events, receiver = mp_channel[Event]()
    sender, tasks = mp_channel[Task]()
    cancels, cancel_receiver = mp_channel[TaskId]()
    runner = Runner(model.bound, events, tasks, cancel_receiver, model, replacement)
    errors: list[Exception] = []

    def run() -> None:
        try:
            runner.main()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def until(predicate: Callable[[Event], bool]) -> Event:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            event = receiver.receive_timeout(max(0.01, deadline - time.monotonic()))
            if predicate(event):
                return event
        raise AssertionError("actual native model dispatch did not settle")

    try:
        for instruction in (
            ConnectToGroup(instance_id=task.instance_id),
            LoadModel(instance_id=task.instance_id),
            StartWarmup(instance_id=task.instance_id),
        ):
            sender.send(instruction)
        until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        resident = factory.resident
        assert resident is not None
        original = resident.run

        def completed_then_died(
            request: NativeMusicRequest,
            *,
            progress: Callable[[NativeMusicProgress], None],
            is_cancelled: Callable[[], bool],
            timeout_seconds: float = 7200.0,
        ) -> NativeMusicResult | None:
            result = original(
                request,
                progress=progress,
                is_cancelled=is_cancelled,
                timeout_seconds=timeout_seconds,
            )
            resident.live = False
            return result

        monkeypatch.setattr(resident, "run", completed_then_died)
        sender.send(task)
        completed = until(
            lambda event: isinstance(event, TaskStatusUpdated)
            and event.task_id == task.task_id
            and event.task_status in (TaskStatus.Complete, TaskStatus.Failed)
        )
        assert (
            isinstance(completed, TaskStatusUpdated)
            and completed.task_status is TaskStatus.Complete
        )
        until(
            lambda event: isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerReady)
        )
        assert len(replacements) == 1 and replacements[0].is_ready()
        assert not model.is_ready()
        assert (
            music_result_directory(model.root, task.command_id) / "result.wav"
        ).is_file()
    finally:
        cancels.send(CANCEL_ALL_TASKS)
        sender.send(
            Shutdown(
                instance_id=task.instance_id, runner_id=model.bound.bound_runner_id
            )
        )
        thread.join(5)
        assert not thread.is_alive() and not errors
        sender.close()
        cancels.close()
        receiver.close()
        model.close()
        for native in replacements:
            native.close()
