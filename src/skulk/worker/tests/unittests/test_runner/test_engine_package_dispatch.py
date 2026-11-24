"""A placement on a held engine package reaches the generic runner; a refused load is given up."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anyio
import loguru
import pytest

from skulk.engines.load_results import LoadObservation
from skulk.engines.tests.package_fixtures import install_test_package
from skulk.shared.backends import (
    ENGINE_PACKAGES_DISABLE_ENV,
    ENGINE_UNKNOWN_FAILURE_MARKER,
    engine_load_refusal,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.common import NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.engine_packages import NodeEngineLoadResult
from skulk.shared.types.events import Event, RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoading,
    RunnerReady,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel, mp_channel
from skulk.worker.main import (
    _engine_load_refused_instance_failure_message,  # pyright: ignore[reportPrivateUsage]
    engine_load_refused_live_instances,
    engine_package_load_results,
)
from skulk.worker.runner import bootstrap
from skulk.worker.runner.engine_package import runner as runner_module
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import get_mlx_ring_instance

NODE = NodeId("node-a")
RUNNER = RunnerId("runner-a")
MODEL = ModelId("org/model")
BUILD = "a" * 64


def _bound(build: str = BUILD) -> BoundInstance:
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=MODEL,
            storage_size=Memory.from_mb(1000),
            n_layers=4,
            hidden_size=64,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
        resolved_backend="tensorfold-rocm",
        resolved_engine_build=build,
    )
    instance = get_mlx_ring_instance(
        InstanceId("instance-a"), MODEL, {NODE: RUNNER}, {RUNNER: shard}
    )
    return BoundInstance(instance=instance, bound_runner_id=RUNNER, bound_node_id=NODE)


def _skip_process_setup(*_arguments: object, **_keywords: object) -> None:
    """Keep signal, limit, watchdog and diagnostic effects out of an in-process run."""


def _run_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    bound: BoundInstance,
    lookup: object,
    runner_class: type,
) -> list[Event]:
    """Run the runner entry point in process and return the events it sent.

    ``lookup`` stands in for the installed-package lookup and ``runner_class``
    for the generic runner. Reaching built-in engine resolution fails the test.
    """
    events: list[Event] = []

    def built_in_reached(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("a package placement reached built-in engine resolution")

    event_sender, event_receiver = mp_channel[Event]()
    diagnostic_sender, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    audio_sender, audio_receiver = mp_channel[RealtimeAudioInputFrame]()
    monkeypatch.setattr(event_sender, "send", events.append)
    monkeypatch.setattr(bootstrap.signal, "signal", _skip_process_setup)
    monkeypatch.setattr(bootstrap.resource, "setrlimit", _skip_process_setup)
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0")
    monkeypatch.setattr(bootstrap, "logger", bootstrap.logger)
    for name in (
        "install_stack_dump_signal",
        "_install_parent_death_watchdog",
        "configure_runner_diagnostics",
        "record_runner_phase",
        "_release_metal_resources",
        "require_remote_code_approval",
    ):
        monkeypatch.setattr(bootstrap, name, _skip_process_setup)
    monkeypatch.setattr(bootstrap, "installed_engine_package", lookup)
    monkeypatch.setattr(bootstrap, "_resolve_text_engine", built_in_reached)
    monkeypatch.setattr(runner_module, "Runner", runner_class)
    try:
        bootstrap.entrypoint(
            bound,
            event_sender,
            diagnostic_sender,
            task_receiver,
            cancel_receiver,
            audio_receiver,
            loguru.logger,
            context_token_limit=4096,
        )
    finally:
        for channel_end in (
            event_sender,
            diagnostic_sender,
            task_sender,
            cancel_sender,
            audio_sender,
            event_receiver,
            diagnostic_receiver,
            task_receiver,
            cancel_receiver,
            audio_receiver,
        ):
            channel_end.close()
    return events


def test_a_placement_on_a_held_package_reaches_the_generic_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamped backend and build select the package; no built-in runner is consulted."""
    package = object()
    looked_up: list[tuple[str | None, str | None]] = []
    started: list[tuple[object, int | None]] = []

    def lookup(backend: str | None, build: str | None) -> object:
        looked_up.append((backend, build))
        return package

    class RecordingRunner:
        """Stand in for the generic runner; record what it was built with."""

        def __init__(
            self,
            _bound_instance: BoundInstance,
            _events: object,
            _tasks: object,
            _cancels: object,
            installed: object,
            context_token_limit: int | None = None,
        ) -> None:
            self.installed = installed
            self.context_token_limit = context_token_limit

        def main(self) -> None:
            started.append((self.installed, self.context_token_limit))

    events = _run_entrypoint(monkeypatch, _bound(), lookup, RecordingRunner)

    assert looked_up == [("tensorfold-rocm", BUILD)]
    assert started == [(package, 4096)]
    assert not [
        event
        for event in events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerFailed)
    ]


def test_packages_turned_off_refuse_a_placement_on_a_held_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The off switch makes a node that holds the package act as one that does not."""
    packages = tmp_path / "packages"
    installed = install_test_package(tmp_path, packages)
    tag = "tensorfold-rocm"
    real_lookup = bootstrap.installed_engine_package

    def lookup(backend: str | None, build: str | None) -> object:
        return real_lookup(backend, build, packages_root=packages)

    class UnreachableRunner:
        """The generic runner must not be built while packages are off."""

        def __init__(self, *_arguments: object, **_keywords: object) -> None:
            raise AssertionError("a package ran while engine packages were turned off")

    held = real_lookup(tag, installed.digest, packages_root=packages)
    assert held is not None and held.digest == installed.digest

    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    assert real_lookup(tag, installed.digest, packages_root=packages) is None
    # A stamp for a generation that is gone is a missing-engine error while
    # packages are on; turned off, the node does not look at all.
    assert real_lookup(tag, "f" * 64, packages_root=packages) is None
    events = _run_entrypoint(
        monkeypatch, _bound(installed.digest), lookup, UnreachableRunner
    )

    failures = [
        event.runner_status
        for event in events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerFailed)
    ]
    assert len(failures) == 1
    message = failures[0].error_message or ""
    assert message.startswith(f"{ENGINE_UNKNOWN_FAILURE_MARKER}: ")
    assert f"{ENGINE_PACKAGES_DISABLE_ENV}=1" in message


def test_the_worker_gives_up_a_refused_load_with_the_code_only() -> None:
    """The engine's own words may name local paths, so only its code is kept."""

    def supervisor(instance_id: str, message: str) -> object:
        return SimpleNamespace(
            bound_instance=SimpleNamespace(
                instance=SimpleNamespace(instance_id=instance_id)
            ),
            shard_metadata=SimpleNamespace(model_card=SimpleNamespace(model_id=MODEL)),
            status=RunnerFailed(error_message=message),
        )

    refused = engine_load_refusal("load_refused", "cannot map /private/weights.bin")
    runners = cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(
            object,
            {
                "runner-a": supervisor("instance-a", refused),
                "runner-b": supervisor("instance-b", "ordinary crash"),
                "runner-c": supervisor("instance-gone", refused),
            },
        ),
    )

    found = engine_load_refused_live_instances(
        runners, {InstanceId("instance-a"), InstanceId("instance-b")}
    )

    assert found == [("instance-a", MODEL, "load_refused", refused)]
    durable = _engine_load_refused_instance_failure_message(MODEL, found[0][2])
    assert "load_refused" in durable and "/private" not in durable


def test_the_worker_reads_load_results_only_from_engine_package_runners() -> None:
    """A built-in engine's load is not part of the engines report."""

    def supervisor(
        tag: str | None, build: str | None, observed: LoadObservation | None
    ) -> object:
        return SimpleNamespace(
            shard_metadata=SimpleNamespace(
                model_card=SimpleNamespace(model_id=MODEL),
                resolved_backend=tag,
                resolved_engine_build=build,
            ),
            load_observation=observed,
        )

    loaded = LoadObservation("loaded", None, 1000)
    refused = LoadObservation("refused", "load_refused", 2000)
    runners = cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(
            object,
            {
                "runner-a": supervisor("tensorfold-rocm", BUILD, loaded),
                "runner-b": supervisor("tensorfold-cuda", "b" * 64, refused),
                "runner-still-loading": supervisor("tensorfold-rocm", BUILD, None),
                "runner-built-in": supervisor("llama_server-vulkan", BUILD, loaded),
                "runner-unstamped": supervisor(None, None, loaded),
                "runner-no-build": supervisor("tensorfold-rocm", None, loaded),
                "runner-odd-build": supervisor("tensorfold-rocm", "not-a-digest", loaded),
            },
        ),
    )

    assert engine_package_load_results(runners) == [
        (
            "tensorfold-rocm",
            NodeEngineLoadResult(
                build=BUILD, outcome="loaded", model_id=str(MODEL), at_unix_ms=1000
            ),
        ),
        (
            "tensorfold-cuda",
            NodeEngineLoadResult(
                build="b" * 64,
                outcome="refused",
                code="load_refused",
                model_id=str(MODEL),
                at_unix_ms=2000,
            ),
        ),
    ]


class _ExitedProcess:
    """A runner process that has already gone, for a supervisor nothing starts."""

    pid = 1
    exitcode = 0

    def is_alive(self) -> bool:
        return False

    def join(self, _timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ending", "outcome", "code"),
    [
        (RunnerReady(), "loaded", None),
        (
            RunnerFailed(
                error_message=engine_load_refusal("load_refused", "no /private/w.bin")
            ),
            "refused",
            "load_refused",
        ),
        (RunnerFailed(error_message="adapter exited"), "failed", None),
    ],
)
async def test_a_supervisor_keeps_how_its_runners_load_ended(
    ending: RunnerStatus, outcome: str, code: str | None
) -> None:
    """The load's end is read from the runner's ordered statuses and survives later ones."""
    event_sender, event_receiver = channel[Event](16)
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()
    bound = _bound()
    supervisor = RunnerSupervisor(
        shard_metadata=bound.bound_shard,
        bound_instance=bound,
        runner_process=cast("mp.Process", cast(object, _ExitedProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    assert supervisor.load_observation is None
    statuses: list[RunnerStatus] = [
        RunnerLoading(),
        ending,
        # What follows a load's end must not rewrite it.
        RunnerFailed(error_message="Terminated (exitcode=1)"),
    ]

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            supervisor._forward_events  # pyright: ignore[reportPrivateUsage]
        )
        for status in statuses:
            ev_send.send(RunnerStatusUpdated(runner_id=RUNNER, runner_status=status))
            with anyio.fail_after(5):
                await event_receiver.receive()
            if status is statuses[0]:
                assert supervisor.load_observation is None
        task_group.cancel_scope.cancel()

    ev_send.close()
    observed = supervisor.load_observation
    assert observed is not None
    assert (observed.outcome, observed.code) == (outcome, code)
    assert engine_package_load_results({RUNNER: supervisor}) == [
        (
            "tensorfold-rocm",
            NodeEngineLoadResult.model_validate(
                {
                    "build": BUILD,
                    "outcome": outcome,
                    "code": code,
                    "modelId": str(MODEL),
                    "atUnixMs": observed.at_unix_ms,
                }
            ),
        )
    ]
