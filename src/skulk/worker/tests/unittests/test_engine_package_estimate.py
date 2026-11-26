# pyright: reportPrivateUsage=false
"""The worker asks an engine package what a load needs before dispatching it.

These drive the worker's real ``_start_runner_task`` and its real fit guard.
Only the adapter call and the node's memory readings are replaced, so each
case shows what reaches the runner and what the worker reports.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import skulk.engines.estimate as estimate_module
import skulk.worker.main as worker_main
import skulk.worker.runner.bootstrap as bootstrap_module
import skulk.worker.runner.engine_package.request as request_module
from skulk import facts as facts_module
from skulk.engines.estimate import EngineNeed
from skulk.engines.protocol import LoadSettings, ModelIdentity
from skulk.facts.testing import make_facts
from skulk.provisioning.engine_install import InstalledEngine
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.engine_memory_settings import engine_package_memory_settings
from skulk.shared.models.tests.test_engine_memory_settings import memory_card
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import (
    EnginePackageNeedStated,
    Event,
    IndexedEvent,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state import State
from skulk.shared.types.tasks import LoadModel, Task, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.utils.channels import Receiver, channel
from skulk.worker.main import Worker
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import get_mlx_ring_instance

_INSTANCE = InstanceId("instance-package")
_RUNNER = RunnerId("runner-package")
_PACKAGE_TAG = "tensorfold-rocm"
_OVERHEAD = 2 * 1024**3
_USABLE_VRAM = Memory.from_gb(64)

_MODEL = ModelIdentity(
    model_id="test/cache-policy", source_revision=None, directory="/models/example"
)
_SETTINGS = LoadSettings(
    context_tokens=8192, parallel_streams=1, cache_precision="default", drafting=False
)


_KEPT_OPEN: list[object] = []


class _Asked:
    """What the worker asked the engine, and the answer it is given."""

    def __init__(self, answer: EngineNeed | None) -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []
        self.requested: list[tuple[object, ...]] = []

    def __call__(self, installed: object, **request: object) -> EngineNeed | None:
        self.calls.append({"installed": installed, **request})
        return self.answer


class _Runner:
    """Stands in for the supervisor: records the tasks dispatched to it."""

    def __init__(self) -> None:
        self.status = None
        self.started: list[Task] = []

    async def start_task(self, task: Task) -> None:
        self.started.append(task)


def _shard(backend: str) -> PipelineShardMetadata:
    packaged = backend == _PACKAGE_TAG
    return PipelineShardMetadata(
        model_card=memory_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
        resolved_backend=backend,
        memory_settings=engine_package_memory_settings(_OVERHEAD) if packaged else None,
    )


def _worker(
    tmp_path: Path, backend: str = _PACKAGE_TAG
) -> tuple[Worker, _Runner, Receiver[Event]]:
    indexed_sender, indexed_receiver = channel[IndexedEvent]()
    _KEPT_OPEN.append(indexed_sender)
    event_sender, event_receiver = channel[Event]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    # A refusal sends a command; its receiving end must outlive this function.
    _KEPT_OPEN.extend((indexed_receiver, command_receiver, download_receiver))
    worker = Worker(
        node_id=NodeId(get_node_id_keypair().to_node_id()),
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        store_client=ModelStoreClient(store_host="store.local"),
        staging_config=StagingNodeConfig(node_cache_path=str(tmp_path)),
    )
    instance = get_mlx_ring_instance(
        instance_id=_INSTANCE,
        model_id=ModelId("test/cache-policy"),
        node_to_runner={worker.node_id: _RUNNER},
        runner_to_shard={_RUNNER: _shard(backend)},
    )
    worker.state = State(instances={_INSTANCE: instance})
    runner = _Runner()
    worker.runners[_RUNNER] = cast(RunnerSupervisor, cast(object, runner))
    return worker, runner, event_receiver


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Asked:
    """A node holding the package, with 64 GB of usable VRAM and a model on disk."""
    recorder = _Asked(None)

    def accepted(*_arguments: object, **_keywords: object) -> None:
        return None

    async def installed_artifact(*_arguments: object, **_keywords: object) -> None:
        return None

    def model_path(*_arguments: object, **_keywords: object) -> Path:
        return tmp_path

    def held(*_arguments: object, **_keywords: object) -> InstalledEngine:
        # The held build's manifest says it drafts.
        return cast(
            InstalledEngine,
            cast(
                object,
                SimpleNamespace(digest="a" * 64, claims=SimpleNamespace(drafting=True)),
            ),
        )

    def request(*arguments: object) -> tuple[ModelIdentity, LoadSettings]:
        recorder.requested.append(arguments)
        return _MODEL, _SETTINGS

    def wireable(_cls: type[MemoryUsage]) -> MemoryUsage:
        return MemoryUsage.from_bytes(
            ram_total=Memory.from_gb(128).in_bytes,
            ram_available=Memory.from_gb(100).in_bytes,
            swap_total=0,
            swap_available=0,
        )

    monkeypatch.setattr(
        worker_main, "_require_worker_model_execution_identity", accepted
    )
    monkeypatch.setattr(
        worker_main, "_require_installed_artifact_for_load", installed_artifact
    )
    monkeypatch.setattr(worker_main, "build_model_path", model_path)
    monkeypatch.setattr(worker_main, "_local_usable_vram", lambda: _USABLE_VRAM)
    monkeypatch.setattr(worker_main, "_local_unified_memory_gpu", lambda: False)
    monkeypatch.setattr(MemoryUsage, "from_local_gpu_wireable", classmethod(wireable))
    monkeypatch.setattr(
        facts_module,
        "_facts",
        make_facts(platform="linux").model_copy(
            update={"engine_package_overhead_bytes": {_PACKAGE_TAG: _OVERHEAD}}
        ),
    )
    monkeypatch.setattr(bootstrap_module, "installed_engine_package", held)
    monkeypatch.setattr(request_module, "package_load_request", request)
    monkeypatch.setattr(estimate_module, "ask_engine_need", recorder)
    return recorder


def _task_statuses(events: Receiver[Event]) -> list[TaskStatus]:
    return [
        event.task_status
        for event in events.collect()
        if isinstance(event, TaskStatusUpdated)
    ]


def _failed(events: Receiver[Event]) -> bool:
    return TaskStatus.Failed in _task_statuses(events)


def _stated(events: list[Event]) -> list[tuple[RunnerId, int]]:
    """Every figure the worker sent to the master, in order."""
    return [
        (event.runner_id, event.need_bytes)
        for event in events
        if isinstance(event, EnginePackageNeedStated)
    ]


async def test_a_load_the_engine_says_fits_is_dispatched(
    tmp_path: Path, asked: _Asked
) -> None:
    """The engine is asked about this instance's exact load, then the load goes out."""
    asked.answer = EngineNeed(pool="gpu", bytes=Memory.from_gb(30).in_bytes)
    worker, runner, events = _worker(tmp_path)
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert runner.started == [task]
    assert not _failed(events)
    assert len(asked.calls) == 1
    assert asked.calls[0]["instance_id"] == str(_INSTANCE)
    assert asked.calls[0]["model"] == _MODEL
    assert asked.calls[0]["settings"] == _SETTINGS
    # The request is built with what the held build's manifest says about
    # drafting, so the engine prices the load the runner will then state.
    assert [arguments[2] for arguments in asked.requested] == [True]


async def test_a_load_the_engine_says_this_node_cannot_hold_is_refused(
    tmp_path: Path, asked: _Asked
) -> None:
    """The admitted figure fits 64 GB; the engine's own 80 GB does not."""
    asked.answer = EngineNeed(pool="gpu", bytes=Memory.from_gb(80).in_bytes)
    worker, runner, events = _worker(tmp_path)

    await worker._start_runner_task(LoadModel(instance_id=_INSTANCE))

    assert runner.started == []
    assert _failed(events)


@pytest.mark.usefixtures("asked")
def test_the_refusal_names_the_engines_figure(tmp_path: Path) -> None:
    """The operator reads what the engine asked for, not the admitted figure."""
    worker, _, _ = _worker(tmp_path)
    shard = _shard(_PACKAGE_TAG)

    refusal = worker._local_shard_fit_error(shard, engine_need=Memory.from_gb(80))

    assert refusal is not None and "80.0GB needed" in refusal
    # A figure below the admitted one never lowers the check.
    assert worker._local_shard_fit_error(shard, engine_need=Memory.from_gb(1)) is None
    assert worker._local_shard_fit_error(shard) is None


async def test_an_engine_that_does_not_know_loads_on_the_admitted_figure(
    tmp_path: Path, asked: _Asked
) -> None:
    """Unknown is a legal answer and changes nothing."""
    worker, runner, events = _worker(tmp_path)
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert len(asked.calls) == 1
    assert runner.started == [task]
    assert not _failed(events)


async def test_a_figure_in_another_pool_is_not_compared(
    tmp_path: Path, asked: _Asked
) -> None:
    """A ROCm lane is admitted against VRAM; a host-RAM figure cannot be checked there."""
    asked.answer = EngineNeed(pool="host", bytes=Memory.from_gb(80).in_bytes)
    worker, runner, events = _worker(tmp_path)
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert runner.started == [task]
    assert not _failed(events)


async def test_a_built_in_engine_is_never_asked(tmp_path: Path, asked: _Asked) -> None:
    """With no package on the placement the load path is what it was."""
    asked.answer = EngineNeed(pool="gpu", bytes=Memory.from_gb(80).in_bytes)
    worker, runner, events = _worker(tmp_path, backend="llama_server-rocm")
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert asked.calls == []
    assert runner.started == [task]
    assert not _failed(events)


async def test_a_placement_removed_while_the_engine_was_asked_is_not_loaded(
    tmp_path: Path, asked: _Asked, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer can take seconds; the load goes only to a placement still there."""
    worker, runner, events = _worker(tmp_path)

    def removed_meanwhile(installed: object, **request: object) -> EngineNeed | None:
        worker.state = State()
        return asked(installed, **request)

    monkeypatch.setattr(estimate_module, "ask_engine_need", removed_meanwhile)

    await worker._start_runner_task(LoadModel(instance_id=_INSTANCE))

    assert len(asked.calls) == 1
    assert runner.started == []
    # The task was created before dispatch, so it is closed, not left open.
    assert _task_statuses(events) == [TaskStatus.Cancelled]


async def test_the_master_is_told_the_engines_figure_once_the_load_goes_ahead(
    tmp_path: Path, asked: _Asked
) -> None:
    """The master reserved on its own estimate; the engine's figure corrects it."""
    need = Memory.from_gb(30).in_bytes
    asked.answer = EngineNeed(pool="gpu", bytes=need)
    worker, runner, events = _worker(tmp_path)
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert runner.started == [task]
    assert _stated(events.collect()) == [(_RUNNER, need)]


async def test_a_figure_the_master_already_holds_is_not_sent_again(
    tmp_path: Path, asked: _Asked
) -> None:
    """A reload of the same placement must not grow the event log."""
    need = Memory.from_gb(30).in_bytes
    asked.answer = EngineNeed(pool="gpu", bytes=need)
    worker, runner, events = _worker(tmp_path)
    worker.state = worker.state.model_copy(
        update={"engine_package_needs": {_RUNNER: need}}
    )
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert runner.started == [task]
    assert _stated(events.collect()) == []


@pytest.mark.parametrize(
    "answer",
    [
        None,
        EngineNeed(pool="host", bytes=Memory.from_gb(30).in_bytes),
        EngineNeed(pool="gpu", bytes=Memory.from_gb(80).in_bytes),
    ],
    ids=["unknown", "other-pool", "refused"],
)
async def test_nothing_is_sent_without_a_figure_for_a_load_that_goes_ahead(
    tmp_path: Path, asked: _Asked, answer: EngineNeed | None
) -> None:
    """No figure, a figure for another pool, or a refused load reserve nothing new."""
    asked.answer = answer
    worker, _, events = _worker(tmp_path)

    await worker._start_runner_task(LoadModel(instance_id=_INSTANCE))

    assert _stated(events.collect()) == []


async def test_a_built_in_engines_load_sends_no_figure(
    tmp_path: Path, asked: _Asked
) -> None:
    asked.answer = EngineNeed(pool="gpu", bytes=Memory.from_gb(30).in_bytes)
    worker, runner, events = _worker(tmp_path, backend="llama_server-rocm")
    task = LoadModel(instance_id=_INSTANCE)

    await worker._start_runner_task(task)

    assert runner.started == [task]
    assert _stated(events.collect()) == []
