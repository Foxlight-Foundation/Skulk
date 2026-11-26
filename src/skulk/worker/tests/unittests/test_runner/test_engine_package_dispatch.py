"""A placement on a held engine package reaches the generic runner; a refused load is given up.

So does a placement whose package the worker is still installing: the runner
starts first, so the model downloads meanwhile, and reads the package at load.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
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
from skulk.shared.types.tasks import CreateRunner, Task, TaskId
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
    Worker,
    _engine_load_refused_instance_failure_message,  # pyright: ignore[reportPrivateUsage]
    engine_load_refused_live_instances,
    engine_package_load_results,
)
from skulk.worker.runner import bootstrap, runner_supervisor
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
    *,
    awaited: bool = False,
) -> list[Event]:
    """Run the runner entry point in process and return the events it sent.

    ``lookup`` stands in for the installed-package lookup and ``runner_class``
    for the generic runner. Reaching built-in engine resolution fails the test.
    ``awaited`` is what the worker says when it is installing the package.
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
            engine_package_awaited=awaited,
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
            find_installed: object = None,
        ) -> None:
            self.installed = installed
            self.context_token_limit = context_token_limit

        def main(self) -> None:
            started.append((self.installed, self.context_token_limit))

    events = _run_entrypoint(monkeypatch, _bound(), lookup, RecordingRunner)

    assert looked_up == [("tensorfold-rocm", BUILD)]
    assert started == [(package, 4096)]
    assert not _failures(events)


def _failures(events: list[Event]) -> list[RunnerFailed]:
    return [
        event.runner_status
        for event in events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerFailed)
    ]


class _UnreachableRunner:
    """The generic runner must not be built for a placement that is refused."""

    def __init__(self, *_arguments: object, **_keywords: object) -> None:
        raise AssertionError("the generic runner was built for a refused placement")


def test_a_runner_started_while_its_package_installs_finds_it_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker starts the runner, then installs: the start must not refuse the engine."""
    package = object()
    on_disk: list[object] = []
    seen: list[tuple[object, object]] = []

    def lookup(_backend: str | None, _build: str | None) -> object | None:
        return on_disk[0] if on_disk else None

    class WaitingRunner:
        """Stand in for the generic runner; the install lands before its load."""

        def __init__(
            self,
            _bound_instance: BoundInstance,
            _events: object,
            _tasks: object,
            _cancels: object,
            installed: object,
            context_token_limit: int | None = None,
            find_installed: object = None,
        ) -> None:
            self.installed = installed
            self.find_installed = cast("Callable[[], object]", find_installed)

        def main(self) -> None:
            on_disk.append(package)
            seen.append((self.installed, self.find_installed()))

    events = _run_entrypoint(
        monkeypatch, _bound(), lookup, WaitingRunner, awaited=True
    )

    assert seen == [(None, package)]
    assert not _failures(events)


def test_a_package_nobody_is_installing_is_refused_at_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the worker's word, a missing package is an unknown engine, at once."""

    def lookup(_backend: str | None, _build: str | None) -> None:
        return None

    events = _run_entrypoint(monkeypatch, _bound(), lookup, _UnreachableRunner)

    failures = _failures(events)
    assert len(failures) == 1
    assert (failures[0].error_message or "").startswith(
        f"{ENGINE_UNKNOWN_FAILURE_MARKER}: "
    )


def test_packages_turned_off_refuse_a_package_that_was_being_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The off switch wins over an install the worker began before it was set."""

    def lookup(_backend: str | None, _build: str | None) -> None:
        return None

    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    events = _run_entrypoint(
        monkeypatch, _bound(), lookup, _UnreachableRunner, awaited=True
    )

    failures = _failures(events)
    assert len(failures) == 1
    assert f"{ENGINE_PACKAGES_DISABLE_ENV}=1" in (failures[0].error_message or "")


class _RecordedProcess:
    """Stand in for the runner process: keep what it would be started with."""

    def __init__(self, **started_with: object) -> None:
        self.started_with = started_with


def _process_keywords(
    monkeypatch: pytest.MonkeyPatch, *, awaited: bool | None
) -> object:
    """The keyword arguments a supervisor gives its runner's entry point."""
    monkeypatch.setattr(runner_supervisor.mp, "Process", _RecordedProcess)
    event_sender, _ = channel[Event](1)
    bound = _bound()
    supervisor = (
        RunnerSupervisor.create(bound_instance=bound, event_sender=event_sender)
        if awaited is None
        else RunnerSupervisor.create(
            bound_instance=bound,
            event_sender=event_sender,
            engine_package_awaited=awaited,
        )
    )
    process = cast("_RecordedProcess", cast(object, supervisor.runner_process))
    assert process.started_with["target"] is runner_supervisor.entrypoint
    return process.started_with["kwargs"]


@pytest.mark.asyncio
async def test_a_supervisor_tells_its_runner_the_package_is_on_its_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _process_keywords(monkeypatch, awaited=True) == {
        "engine_package_awaited": True
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("awaited", [None, False])
async def test_any_other_runner_is_started_with_the_arguments_it_always_had(
    monkeypatch: pytest.MonkeyPatch, awaited: bool | None
) -> None:
    assert _process_keywords(monkeypatch, awaited=awaited) == {}


@pytest.mark.parametrize(
    ("on_demand", "awaited"),
    [
        (frozenset({"tensorfold-rocm"}), True),
        # Held already, or not offered here at all: nothing is on its way.
        (frozenset[str](), False),
        (frozenset({"tensorfold-cuda", "llama_server-vulkan"}), False),
    ],
)
def test_the_worker_says_so_only_for_a_package_this_node_still_has_to_install(
    monkeypatch: pytest.MonkeyPatch, on_demand: frozenset[str], awaited: bool
) -> None:
    created: list[dict[str, object]] = []

    def create(**keywords: object) -> object:
        created.append(keywords)
        return SimpleNamespace(run=None)

    monkeypatch.setattr(RunnerSupervisor, "create", create)
    bound = _bound()
    event_sender, _ = channel[Event](1)
    runners: dict[RunnerId, object] = {}
    worker = SimpleNamespace(
        event_sender=event_sender,
        runners=runners,
        _data_sender=None,
        _trace_data_sender=None,
        _on_demand_backends=on_demand,
        _schedule_video_output_transfer=None,
        _schedule_music_output_transfer=None,
        _schedule_music_edit_output_transfer=None,
        _schedule_separation_output_transfer=None,
        _tg=SimpleNamespace(start_soon=_skip_process_setup),
    )

    _ = Worker._create_supervisor(  # pyright: ignore[reportPrivateUsage]
        cast("Worker", cast(object, worker)),
        CreateRunner(instance_id=bound.instance.instance_id, bound_instance=bound),
    )

    assert [made["engine_package_awaited"] for made in created] == [awaited]
    assert list(runners) == [RUNNER]


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

    held = real_lookup(tag, installed.digest, packages_root=packages)
    assert held is not None and held.digest == installed.digest

    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    assert real_lookup(tag, installed.digest, packages_root=packages) is None
    # A stamp for a generation that is gone is a missing-engine error while
    # packages are on; turned off, the node does not look at all.
    assert real_lookup(tag, "f" * 64, packages_root=packages) is None
    events = _run_entrypoint(
        monkeypatch, _bound(installed.digest), lookup, _UnreachableRunner
    )

    failures = _failures(events)
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
