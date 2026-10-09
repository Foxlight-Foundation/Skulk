# pyright: reportPrivateUsage=false
"""On-demand engine installs: one shared job, each waiter released or failed once."""

import sys
import threading
import time
from pathlib import Path
from typing import cast

import anyio
import pytest
from anyio import to_thread

import skulk.facts as facts_module
import skulk.provisioning.comfy as comfy_module
from skulk.facts.derive import BackendDerivation
from skulk.facts.testing import NVIDIA_A40, make_facts
from skulk.routing.router import get_node_id_keypair
from skulk.shared.types.commands import (
    FailInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import Event, IndexedEvent, TaskStatusUpdated
from skulk.shared.types.state import State
from skulk.shared.types.tasks import InstallEngine, TaskId, TaskStatus
from skulk.shared.types.worker.instances import Instance, InstanceId
from skulk.shared.types.worker.runners import RunnerId
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.utils.channels import Receiver, channel
from skulk.utils.task_group import TaskGroup
from skulk.worker.main import Worker, _EngineInstallJob
from skulk.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

_MODEL = ModelId("org/video")
_FIRST = InstanceId("instance-first")
_SECOND = InstanceId("instance-second")


def _worker(
    tmp_path: Path,
) -> tuple[Worker, Receiver[Event], Receiver[ForwarderCommand]]:
    _, indexed_receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId(get_node_id_keypair().to_node_id()),
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        store_client=ModelStoreClient(store_host="store.local"),
        staging_config=StagingNodeConfig(node_cache_path=str(tmp_path)),
    )
    instances: dict[InstanceId, Instance] = {}
    for instance_id in (_FIRST, _SECOND):
        runner_id = RunnerId(f"runner-{instance_id}")
        shard = get_pipeline_shard_metadata(model_id=_MODEL, device_rank=0, world_size=1)
        instances[instance_id] = get_mlx_ring_instance(
            instance_id=instance_id,
            model_id=_MODEL,
            node_to_runner={worker.node_id: runner_id},
            runner_to_shard={runner_id: shard},
        )
    worker.state = State(instances=instances)
    worker._on_demand_backends = frozenset({"comfy", "comfy-cuda"})
    return worker, event_receiver, command_receiver


class _StartRecorder:
    """Records what the worker would start in its task group."""

    def __init__(self) -> None:
        self.started: list[object] = []

    def start_soon(self, function: object, *args: object) -> None:
        self.started.append((function, args))


def _record_starts(worker: Worker) -> _StartRecorder:
    recorder = _StartRecorder()
    worker._tg = cast(TaskGroup, cast(object, recorder))
    return recorder


def _statuses(events: Receiver[Event]) -> list[tuple[TaskId, TaskStatus]]:
    return [
        (event.task_id, event.task_status)
        for event in events.collect()
        if isinstance(event, TaskStatusUpdated)
    ]


def _derivation_after_install(*, installed: bool) -> BackendDerivation:
    tags = frozenset({"comfy", "comfy-cuda"})
    return BackendDerivation(
        backends=tags, on_demand_backends=frozenset() if installed else tags
    )


@pytest.fixture
def node_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = make_facts(gpus=(NVIDIA_A40,), comfy_on_demand_variants=("cuda",))
    monkeypatch.setattr(facts_module, "current_node_facts", lambda: facts)
    monkeypatch.setattr(facts_module, "refresh_node_facts", lambda: facts)


async def test_waiting_instances_share_one_install(tmp_path: Path) -> None:
    worker, events, _ = _worker(tmp_path)
    recorder = _record_starts(worker)
    first = InstallEngine(instance_id=_FIRST, engine="comfy")
    second = InstallEngine(instance_id=_SECOND, engine="comfy")

    await worker._start_engine_install(first)
    await worker._start_engine_install(second)

    assert len(recorder.started) == 1
    job = worker._engine_install_jobs["comfy"]
    assert job.waiters == {_FIRST: first.task_id, _SECOND: second.task_id}
    assert _statuses(events) == [
        (first.task_id, TaskStatus.Running),
        (second.task_id, TaskStatus.Running),
    ]


@pytest.mark.usefixtures("node_facts")
async def test_a_finished_install_releases_every_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, events, commands = _worker(tmp_path)
    offline_requests: list[bool] = []

    def install(_facts: object, *, offline: bool, **_kwargs: object) -> Path:
        offline_requests.append(offline)
        return tmp_path

    monkeypatch.setattr(comfy_module, "install_comfy_on_demand", install)
    monkeypatch.setattr(
        facts_module,
        "current_backend_derivation",
        lambda: _derivation_after_install(installed=True),
    )
    first, second = TaskId("task-first"), TaskId("task-second")
    job = _EngineInstallJob(engine="comfy", waiters={_FIRST: first, _SECOND: second})

    await worker._run_engine_install(job)

    assert job.finished and job.succeeded and offline_requests == [False]
    assert worker._on_demand_backends == frozenset()
    assert _statuses(events) == [(first, TaskStatus.Complete), (second, TaskStatus.Complete)]
    assert commands.collect() == []

    # A placement that arrives after the install is released at once.
    recorder = _record_starts(worker)
    worker._engine_install_jobs["comfy"] = job
    late = InstallEngine(instance_id=InstanceId("instance-late"), engine="comfy")
    await worker._start_engine_install(late)
    assert recorder.started == []
    assert _statuses(events) == [
        (late.task_id, TaskStatus.Running),
        (late.task_id, TaskStatus.Complete),
    ]


@pytest.mark.usefixtures("node_facts")
async def test_a_failed_install_gives_each_waiter_up_with_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, events, commands = _worker(tmp_path)

    def refuse(_facts: object, *, offline: bool, **_kwargs: object) -> Path:
        raise comfy_module.ComfyInstallError("the disk-space check", "2.0 GB free")

    monkeypatch.setattr(comfy_module, "install_comfy_on_demand", refuse)
    first, second = TaskId("task-first"), TaskId("task-second")
    job = _EngineInstallJob(engine="comfy", waiters={_FIRST: first, _SECOND: second})

    await worker._run_engine_install(job)

    assert job.finished and not job.succeeded
    assert _statuses(events) == [(first, TaskStatus.Failed), (second, TaskStatus.Failed)]
    failures = [
        forwarded.command
        for forwarded in commands.collect()
        if isinstance(forwarded.command, FailInstance)
    ]
    assert [failure.instance_id for failure in failures] == [_FIRST, _SECOND]
    for failure in failures:
        assert failure.error_code == "engine_install_failed"
        assert "the disk-space check failed" in failure.error_message
        # Installer detail stays in the log; durable state gets guidance only.
        assert "2.0 GB free" not in failure.error_message
    assert worker._on_demand_backends == frozenset({"comfy", "comfy-cuda"})

    # A new placement gets a fresh attempt rather than the old failure.
    recorder = _record_starts(worker)
    worker._engine_install_jobs["comfy"] = job
    await worker._start_engine_install(
        InstallEngine(instance_id=InstanceId("instance-retry"), engine="comfy")
    )
    assert len(recorder.started) == 1
    assert worker._engine_install_jobs["comfy"] is not job


@pytest.mark.usefixtures("node_facts")
async def test_an_install_the_node_does_not_report_as_usable_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, events, commands = _worker(tmp_path)
    def install(_facts: object, *, offline: bool, **_kwargs: object) -> Path:
        return tmp_path

    monkeypatch.setattr(comfy_module, "install_comfy_on_demand", install)
    monkeypatch.setattr(
        facts_module,
        "current_backend_derivation",
        lambda: _derivation_after_install(installed=False),
    )
    task = TaskId("task-first")
    job = _EngineInstallJob(engine="comfy", waiters={_FIRST: task})

    await worker._run_engine_install(job)

    assert not job.succeeded
    assert _statuses(events) == [(task, TaskStatus.Failed)]
    failures = [forwarded.command for forwarded in commands.collect()]
    assert len(failures) == 1 and isinstance(failures[0], FailInstance)
    assert "the engine check failed" in failures[0].error_message


def test_a_runner_that_finds_no_engine_is_terminal_for_its_instance() -> None:
    from types import SimpleNamespace

    from skulk.shared.backends import ENGINE_UNAVAILABLE_FAILURE_MARKER
    from skulk.shared.types.worker.runners import RunnerFailed
    from skulk.worker.main import engine_unavailable_live_instances
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    def supervisor(instance_id: str, message: str) -> object:
        return SimpleNamespace(
            bound_instance=SimpleNamespace(instance=SimpleNamespace(instance_id=instance_id)),
            shard_metadata=SimpleNamespace(model_card=SimpleNamespace(model_id="org/video")),
            status=RunnerFailed(error_message=message),
        )

    missing = f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: no managed install"
    runners = cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(
            object,
            {
                "runner-a": supervisor("instance-a", missing),
                "runner-b": supervisor("instance-b", "ordinary crash"),
                "runner-c": supervisor("instance-gone", missing),
            },
        ),
    )
    assert engine_unavailable_live_instances(
        runners, {InstanceId("instance-a"), InstanceId("instance-b")}
    ) == [("instance-a", "org/video", missing)]


@pytest.mark.usefixtures("node_facts")
async def test_a_worker_shutting_down_ends_the_install_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A master change replaces the worker; the install must not hold that up."""
    worker, events, commands = _worker(tmp_path)
    started = threading.Event()
    finished = threading.Event()
    outcome: list[str] = []

    def install(
        _facts: object, *, offline: bool, run: comfy_module.Runner, **_kwargs: object
    ) -> Path:
        started.set()
        try:
            run(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
                env=None,
            )
            outcome.append("finished")
        except comfy_module.ComfyInstallCancelledError:
            outcome.append("stopped")
            raise
        finally:
            finished.set()
        return tmp_path

    monkeypatch.setattr(comfy_module, "install_comfy_on_demand", install)
    job = _EngineInstallJob(engine="comfy", waiters={_FIRST: TaskId("task-first")})

    began = time.monotonic()
    async with anyio.create_task_group() as group:
        group.start_soon(worker._run_engine_install, job)
        assert await to_thread.run_sync(started.wait, 30)
        group.cancel_scope.cancel()
    assert time.monotonic() - began < 30
    assert await to_thread.run_sync(finished.wait, 30)
    assert outcome == ["stopped"]
    # The waiters are left for the next worker, which plans the install again.
    assert not job.finished
    assert events.collect() == [] and commands.collect() == []
