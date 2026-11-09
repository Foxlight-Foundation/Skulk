"""Prepared native bootstrap selection and private reader handover boundaries."""

import io
from collections.abc import Callable
from pathlib import Path

import loguru
import pytest

from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.chunks import DataChunk, MusicChunk, MusicEditingChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import Event, RunnerStatusUpdated
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerFailed
from skulk.utils.channels import MpReceiver, MpSender, channel, mp_channel
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.runner import bootstrap
from skulk.worker.runner.ace_step import runner as runner_module
from skulk.worker.runner.ace_step.context import NativeMusicRunnerContext
from skulk.worker.runner.ace_step.runner import BoundNativeMusicModel
from skulk.worker.runner.ace_step.tests.test_context import resource_guard
from skulk.worker.runner.ace_step.tests.test_model import fixture
from skulk.worker.runner.ace_step.tests.test_resident import (
    create_resident_interpreter,
    process_alive,
    resident,
)
from skulk.worker.runner.demucs.source_fence import SourceReadFence
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


def test_actual_startup_failure_publishes_no_private_native_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real bootstrap publishes a failed native load without its private notes."""
    actual, _factory, _task = fixture(tmp_path / "binding")
    interpreter = create_resident_interpreter(tmp_path / "interpreter")
    native_root = tmp_path / "resident"
    native = resident(native_root, interpreter, mode="stderr-load-exit")
    owner = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )
    events: list[Event] = []
    event_sender, event_receiver = mp_channel[Event]()
    diagnostic_sender, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    audio_sender, audio_receiver = mp_channel[RealtimeAudioInputFrame]()
    private_log = io.StringIO()
    sink = loguru.logger.add(private_log, format="{message}", diagnose=False)

    def fail_native_load(
        bound: BoundInstance,
        _events: MpSender[Event],
        _tasks: MpReceiver[Task],
        _cancels: MpReceiver[TaskId],
        prepared: NativeMusicRunnerContext,
    ) -> None:
        """Exercise a real failing child rather than constructing a test exception."""
        assert bound == actual.bound and prepared is context
        native.start(is_cancelled=lambda: False, timeout_seconds=5)

    def skip_process_setup(*_arguments: object, **_keywords: object) -> None:
        """Keep process-wide setup out of this in-process bootstrap regression."""

    monkeypatch.setattr(event_sender, "send", events.append)
    monkeypatch.setattr(bootstrap, "run_native_music", fail_native_load)
    monkeypatch.setattr(bootstrap.signal, "signal", skip_process_setup)
    monkeypatch.setattr(bootstrap.resource, "setrlimit", skip_process_setup)
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0")
    monkeypatch.setattr(bootstrap, "logger", bootstrap.logger)
    for name in (
        "install_stack_dump_signal",
        "_install_parent_death_watchdog",
        "configure_runner_diagnostics",
        "record_runner_phase",
        "_release_metal_resources",
    ):
        monkeypatch.setattr(bootstrap, name, skip_process_setup)
    try:
        bootstrap.entrypoint(
            actual.bound,
            event_sender,
            diagnostic_sender,
            task_receiver,
            cancel_receiver,
            audio_receiver,
            loguru.logger,
            native_music_context=context,
        )
        assert len(events) == 1 and isinstance(events[0], RunnerStatusUpdated)
        assert isinstance(events[0].runner_status, RunnerFailed)
        serialized = events[0].model_dump_json()
        for private in (
            "/native/private/checkpoints",
            "native_stderr_tail=",
            "native_exit_before_cleanup=",
            "device allocation ceiling",
        ):
            assert private not in serialized
            assert private in private_log.getvalue()
        assert not native.is_ready()
        assert not process_alive(int((native_root / "native.pid").read_text()))
        assert not tuple((native_root / "workspace/loader").iterdir())
    finally:
        loguru.logger.remove(sink)
        native.close()
        fence.close()
        event_sender.close()
        diagnostic_sender.close()
        task_sender.close()
        cancel_sender.close()
        audio_sender.close()
        event_receiver.close()
        diagnostic_receiver.close()
        task_receiver.close()
        cancel_receiver.close()
        audio_receiver.close()


@pytest.mark.parametrize("fails", [False, True])
def test_bootstrap_uses_fresh_unloaded_models_and_releases_its_reader_on_exit(
    tmp_path: Path, fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal and exceptional native runner return both settle bootstrap ownership."""
    actual, factory, _task = fixture(tmp_path)
    owner = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )
    models: list[BoundNativeMusicModel] = []
    event_sender, event_receiver = mp_channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()

    class ControlledRunner:
        """Observe the actual unloaded wrappers without starting inference."""

        def __init__(
            self,
            bound: BoundInstance,
            events: MpSender[Event],
            tasks: MpReceiver[Task],
            cancels: MpReceiver[TaskId],
            model: BoundNativeMusicModel,
            create_model: Callable[[], BoundNativeMusicModel],
        ) -> None:
            assert bound == actual.bound
            assert events is event_sender and tasks is task_receiver
            assert cancels is cancel_receiver and model.bound == actual.bound
            models.append(model)
            self.create_model = create_model

        def main(self) -> None:
            """Exercise bootstrap's replacement factory and exceptional cleanup."""
            replacement = self.create_model()
            models.append(replacement)
            assert replacement is not models[0] and replacement.bound == actual.bound
            assert not any(model.is_ready() for model in models)
            assert fence.descriptor >= 0 and factory.calls == 0
            replacement.close()
            if fails:
                raise RuntimeError("controlled native runner failure")

    monkeypatch.setattr(runner_module, "Runner", ControlledRunner)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="controlled native runner failure"):
                bootstrap.run_native_music(
                    actual.bound, event_sender, task_receiver, cancel_receiver, context
                )
        else:
            bootstrap.run_native_music(
                actual.bound, event_sender, task_receiver, cancel_receiver, context
            )
        assert fence.descriptor < 0 and len(models) == 2 and factory.calls == 0
        with pytest.raises(RuntimeError, match="new unloaded lifetime"):
            models[0].start(is_cancelled=lambda: False)
    finally:
        for model in models:
            model.close()
        fence.close()
        event_sender.close()
        event_receiver.close()
        task_sender.close()
        task_receiver.close()
        cancel_sender.close()
        cancel_receiver.close()


@pytest.mark.parametrize("missing", ["data", "generate", "edit"])
def test_supervisor_refuses_native_creation_without_complete_output_delivery(
    tmp_path: Path, missing: str
) -> None:
    """Prepared libraries cannot start a native runner with undeliverable results."""
    actual, factory, _task = fixture(tmp_path)
    owner = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )
    events, event_receiver = channel[Event](10)
    data, data_receiver = channel[DataChunk](10)

    def generation(
        _command: CommandId, _owner: NodeId | None, _chunk: MusicChunk
    ) -> None:
        pytest.fail("a rejected supervisor cannot deliver generation output")

    def editing(
        _command: CommandId, _owner: NodeId | None, _chunk: MusicEditingChunk
    ) -> None:
        pytest.fail("a rejected supervisor cannot deliver edit output")

    try:
        with pytest.raises(ValueError, match="generation and edit media delivery"):
            RunnerSupervisor.create(
                bound_instance=actual.bound,
                event_sender=events,
                data_sender=None if missing == "data" else data,
                on_music_output=None if missing == "generate" else generation,
                on_music_edit_output=None if missing == "edit" else editing,
                native_music_context=context,
            )
        assert factory.calls == 0 and fence.descriptor >= 0
    finally:
        fence.close()
        events.close()
        event_receiver.close()
        data.close()
        data_receiver.close()
