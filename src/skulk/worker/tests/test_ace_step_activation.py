# pyright: reportPrivateUsage=false
"""Actual Worker, sealed cache and ordered resource bridge with injected execution."""

import multiprocessing
import time
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.provisioning.tests import test_ace_step_runtime as fixtures
from skulk.provisioning.tests.test_ace_step_manager import manager
from skulk.shared.models import native_music_admission
from skulk.shared.models.tests.test_native_music_admission import (
    instance as native_instance,
)
from skulk.shared.models.tests.test_native_music_admission import (
    resources as native_resources,
)
from skulk.shared.types.chunks import DataChunk
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import (
    AceStepPreparationCompleted,
    AceStepPreparationRequested,
    Event,
    IndexedEvent,
    TaskFailed,
    TaskStatusUpdated,
)
from skulk.shared.types.native_music import (
    NativeMusicDevice,
    NativeMusicLibraryProbe,
    NativeMusicMemoryObservation,
)
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.tasks import CreateRunner, TaskStatus
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.instances import BoundInstance
from skulk.utils.channels import Receiver, channel
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker import main as worker_module
from skulk.worker.ace_step_preparation import AceStepPreparationController
from skulk.worker.main import Worker
from skulk.worker.native_music_media import NativeMusicMediaOwner
from skulk.worker.runner.ace_step.resources import NativeMusicHostGuard
from skulk.worker.runner.runner_supervisor import RunnerSupervisor

interpreter_fixture = fixtures.interpreter_fixture
installed = fixtures.installed
pytestmark = pytest.mark.parametrize("variant", ("cpu", "cuda", "rocm"))


def fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
    *,
    offline: bool = False,
) -> tuple[Worker, Receiver[Event], Receiver[NodeTelemetry]]:
    """Keep real Worker/cache verification; replace only network and native execution."""
    _, indexed = channel[IndexedEvent]()
    events, results = channel[Event]()
    commands, _ = channel[ForwarderCommand]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    telemetry, observations = channel[NodeTelemetry]()
    worker = Worker(
        NodeId("worker"),
        event_receiver=indexed,
        event_sender=events,
        command_sender=commands,
        download_command_sender=downloads,
        telemetry_sender=telemetry,
        offline=offline,
    )
    worker._demucs_platforms = frozenset()
    worker._ace_step_platforms = installed.wheel.required_platform_tags
    worker._ace_step_manager = manager(root / "cache", installed)
    worker._ace_step_preparation = AceStepPreparationController(
        worker._ace_step_manager, offline=offline
    )

    def select(
        platforms: AbstractSet[str], *, device: NativeMusicDevice
    ) -> AceStepWheel:
        if (
            installed.wheel.required_platform_tags.issubset(platforms)
            and device == installed.device
        ):
            return installed.wheel
        raise ValueError("No injected package implements this ABI/device")

    monkeypatch.setattr(worker_module, "select_ace_step_package", select)
    monkeypatch.setattr(
        NodeResources,
        "gather",
        AsyncMock(
            return_value=NodeResources(
                backends=frozenset({"audio_cpp-cpu", "ace_step-cpu"}),
                engine_builds={
                    "audio_cpp-cpu": "existing-music-build",
                    "ace_step-cpu": "unverified",
                },
                on_demand_backends=frozenset({"ace_step-cpu"}),
            )
        ),
    )
    return worker, results, observations


def request(
    worker: Worker, installed: fixtures.NativeInstall, *, digest: str | None = None
) -> AceStepPreparationRequested:
    """Issue one bounded request using the selected exact package and device."""
    return AceStepPreparationRequested(
        request_id=CommandId(),
        target_node=worker.node_id,
        owner_node=NodeId("api"),
        package_sha256=digest or installed.wheel.sha256,
        device=installed.device,
        expires_at=time.time() + 30,
    )


@pytest.mark.parametrize("corrupt_archive", [False, True])
async def test_cold_worker_recovers_only_freshly_verified_cached_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall, corrupt_archive: bool,
) -> None:
    """A fresh worker owner restores readiness without downloading, including online mode."""
    worker, _, observations = fixture(tmp_path, monkeypatch, installed, offline=False)
    try:
        await worker._prepare_ace_step_engine(request(worker, installed))
        await observations.receive()
        await worker._ace_step_preparation.close()
        worker._ace_step_manager = manager(tmp_path / "cache", installed)
        worker._ace_step_preparation = AceStepPreparationController(
            worker._ace_step_manager, offline=False
        )
        if corrupt_archive:
            archive = tmp_path / "cache" / installed.wheel.sha256 / "objects" / installed.wheel.filename
            archive.write_bytes(b"corrupted retained original")
        assert worker._ace_step_preparation.engine_builds() == {}
        await worker._restore_cached_native_runtimes()
        resources = (await observations.receive()).info
        assert isinstance(resources, NodeResources)
        assert resources.engine_builds["audio_cpp-cpu"] == "existing-music-build"
        backend = "ace_step-" + installed.device
        if corrupt_archive:
            assert backend not in resources.backends
            assert resources.native_music_probes == {}
        else:
            assert resources.engine_builds[backend] == installed.wheel.build_identity
            assert resources.native_music_probes[backend].payload_sha256 == installed.wheel.payload_sha256
            assert installed.probes >= 2
        assert installed.downloads == installed.installs == 1
    finally:
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


async def test_inventory_observation_has_no_install_or_probe_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """Actual resource gathering never promotes a candidate or preserves unverified lanes."""
    worker, _, _ = fixture(tmp_path, monkeypatch, installed)
    try:
        resources = await worker._gather_node_resources()
        assert resources.backends == frozenset({"audio_cpp-cpu"})
        assert resources.engine_builds == {"audio_cpp-cpu": "existing-music-build"}
        assert resources.native_music_probes == {} and not resources.on_demand_backends
        assert resources.installable_engine_builds == {
            "ace_step-" + installed.device: installed.wheel.build_identity
        }
        assert installed.downloads == installed.installs == installed.probes == 0
    finally:
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


async def test_actual_worker_publishes_same_verified_snapshot_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """The ordered barrier carries real cache-verified bytes and the requested lane."""
    worker, output, observations = fixture(tmp_path, monkeypatch, installed)
    try:
        command = request(worker, installed)
        await worker._prepare_ace_step_engine(command)
        event = await output.receive()
        assert isinstance(event, AceStepPreparationCompleted) and event.success
        assert event.request_id == command.request_id
        assert event.package_sha256 == installed.wheel.sha256
        assert event.device == installed.device
        resources = event.resources
        assert resources is not None
        assert (await observations.receive()).info == resources
        backend = "ace_step-" + installed.device
        assert resources.engine_builds[backend] == installed.wheel.build_identity
        assert (
            resources.native_music_probes[backend].payload_sha256
            == installed.wheel.payload_sha256
        )
        assert (
            NodeResources.model_validate_json(resources.model_dump_json()) == resources
        )
        assert installed.downloads == installed.installs == installed.probes == 1
        assert (
            await worker._gather_node_resources()
        ).native_music_probes == resources.native_music_probes
    finally:
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()
    assert (await worker._gather_node_resources()).native_music_probes == {}


async def test_failed_revalidation_immediately_withdraws_old_library_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """A failed probe cannot leave the last successful build advertised."""
    worker, output, observations = fixture(tmp_path, monkeypatch, installed)
    try:
        await worker._prepare_ace_step_engine(request(worker, installed))
        accepted = await output.receive()
        assert isinstance(accepted, AceStepPreparationCompleted) and accepted.success
        assert (await observations.receive()).info == accepted.resources
        installed.foreign_payload = True
        await worker._prepare_ace_step_engine(request(worker, installed))
        failed = await output.receive()
        assert isinstance(failed, AceStepPreparationCompleted) and not failed.success
        withdrawn = (await observations.receive()).info
        assert isinstance(withdrawn, NodeResources)
        assert not withdrawn.native_music_probes
        assert withdrawn.engine_builds == {"audio_cpp-cpu": "existing-music-build"}
    finally:
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


@pytest.mark.parametrize(
    "fault", ["unknown-package", "offline-empty", "foreign-payload"]
)
async def test_failed_preparation_preserves_worker_and_other_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
    fault: str,
) -> None:
    """Actionable mount failure cannot publish false native capacity or install unknown bytes."""
    worker, output, _ = fixture(
        tmp_path, monkeypatch, installed, offline=fault == "offline-empty"
    )
    installed.foreign_payload = fault == "foreign-payload"
    try:
        await worker._prepare_ace_step_engine(
            request(
                worker,
                installed,
                digest="0" * 64 if fault == "unknown-package" else None,
            )
        )
        event = await output.receive()
        assert isinstance(event, AceStepPreparationCompleted) and not event.success
        assert event.resources is None and event.error is not None
        assert "inspect worker diagnostics" in event.error
        resources = await worker._gather_node_resources()
        assert resources.native_music_probes == {}
        assert resources.engine_builds == {"audio_cpp-cpu": "existing-music-build"}
        if fault != "foreign-payload":
            assert installed.downloads == installed.installs == installed.probes == 0
    finally:
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


def runner_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
    *,
    free_bytes: int = 64 * 1024**3,
) -> tuple[Worker, CreateRunner, Receiver[Event]]:
    """Use real cache, Worker admission and supervisor construction, without inference.

    Only executed library observations and the supervisor's process loop are
    controlled. No model files exist, so early weight resolution would fail.
    """
    worker, output, _ = fixture(root, monkeypatch, installed)
    observed = native_resources(installed.device)
    backend = next(iter(observed.backends))
    expected = observed.native_music_probes[backend]
    assert expected.memory_capacity is not None
    capacity = expected.memory_capacity.model_copy(
        update={
            "sampled_at_unix_ms": int(time.time() * 1000),
            "host_total_bytes": 64 * 1024**3,
            "host_available_bytes": free_bytes,
            "device_available_bytes": free_bytes if installed.device != "cpu" else None,
            "shared_available_bytes": free_bytes
            if installed.device == "rocm"
            else None,
        }
    )
    expected = expected.model_copy(
        update={
            "payload_sha256": installed.wheel.payload_sha256,
            "memory_capacity": capacity,
        }
    )
    observed = NodeResources(
        backends=frozenset({backend}),
        engine_builds={backend: installed.wheel.build_identity},
        native_music_probes={backend: expected},
    )
    owner = native_instance(observed)
    worker.node_id = next(iter(owner.shard_assignments.node_to_runner))
    worker.state = State().model_copy(update={"instances": {owner.instance_id: owner}})
    task = CreateRunner(
        instance_id=owner.instance_id,
        bound_instance=BoundInstance(
            instance=owner,
            bound_node_id=worker.node_id,
            bound_runner_id=owner.shard_assignments.node_to_runner[worker.node_id],
        ),
    )
    original_probe = installed.probe

    def measured(
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        actual = NativeMusicLibraryProbe.model_validate_json(
            original_probe(
                arguments,
                environment=environment,
                timeout_seconds=timeout_seconds,
                capture=capture,
            )
        )
        return actual.model_copy(update={"memory_capacity": capacity}).model_dump_json()

    async def no_inference(_supervisor: RunnerSupervisor) -> None:
        """The actual supervisor is constructed; native load/generation is separate proof."""

    monkeypatch.setattr(installed, "probe", measured)
    monkeypatch.setattr(native_music_admission, "ACE_STEP_PACKAGES", (installed.wheel,))
    monkeypatch.setattr(RunnerSupervisor, "run", no_inference)
    monkeypatch.setattr(worker, "_give_up_on_instance", AsyncMock())
    worker._data_sender, _ = channel[DataChunk]()
    worker._native_music_media = NativeMusicMediaOwner(
        worker.node_id,
        lambda: worker.state,
        root / "sources",
        root / "readers",
        worker._send_native_music_source_packet,
        worker._send_separation_output_packet,
        worker._fail_native_music,
        worker._native_music_reader_drained,
    )
    return worker, task, output


async def test_actual_worker_constructs_lazy_guarded_context_and_transfers_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """A first mount creates a real supervisor before weights exist, never generic fit."""
    worker, task, output = runner_fixture(tmp_path, monkeypatch, installed)

    def generic_fit(*_args: object) -> str:
        return "Generic fit must not run"

    monkeypatch.setattr(worker, "_local_shard_fit_error", generic_fit)
    context = None
    try:
        async with worker._tg:
            await worker._execute_create_runner(task)
            # A second planner tick must not enqueue the same creation again.
            await worker._execute_create_runner(task)
            event = await output.receive()
            assert (
                isinstance(event, TaskStatusUpdated)
                and event.task_status == TaskStatus.Complete
            )
            supervisor = worker.runners[task.bound_instance.bound_runner_id]
            context = supervisor._native_music_context
            assert context is not None and isinstance(
                context.resource_guard, NativeMusicHostGuard
            )
            assert context.resource_guard.host_total_bytes == 64 * 1024**3
            assert context.resource_guard.unified == (installed.device == "rocm")
            assert context.runtime.wheel == installed.wheel
            assert not context.model(task.bound_instance).is_ready()
            assert context.source_root == tmp_path / "sources"
            with pytest.raises(BlockingIOError):
                RuntimeLock(context.root, name="attachment.lock")
        assert (
            not worker._native_music_creations
            and not worker._native_music_creation_requests
        )
        assert installed.downloads == installed.installs == installed.probes == 1
    finally:
        if context is not None:
            context.source_fence.close()
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


@pytest.mark.parametrize(
    "fault",
    [
        "valid",
        "withdrawn-library",
        "foreign-owner",
        "foreign-payload",
        "stale",
        "changed-pool",
        "dead-runner",
    ],
)
async def test_live_native_samples_refresh_only_owned_current_resource_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
    fault: str,
) -> None:
    """Real Worker/supervisor custody refreshes one ephemeral sample, never history or credits.

    A real controlled child supplies runner liveness/identity; native capacity is
    injected. Resident-frame verification and diagnostic IPC have separate real
    boundary tests. This is not native model or hardware qualification.
    """
    worker, task, output = runner_fixture(tmp_path, monkeypatch, installed)
    context = None
    child = multiprocessing.Process(target=time.sleep, args=(30,), daemon=True)
    try:
        async with worker._tg:
            await worker._execute_create_runner(task)
            event = await output.receive()
            assert (
                isinstance(event, TaskStatusUpdated)
                and event.task_status == TaskStatus.Complete
            )
        supervisor = worker.runners[task.bound_instance.bound_runner_id]
        context = supervisor._native_music_context
        assert context is not None
        supervisor.runner_process.close()
        child.start()
        supervisor.runner_process = child
        observed = context.runtime.probe
        original = observed.memory_capacity
        assert original is not None and child.pid is not None
        updates: dict[str, object] = {
            "sampled_at_unix_ms": max(
                original.sampled_at_unix_ms + 1, time.time_ns() // 1_000_000
            ),
            "host_available_bytes": original.host_available_bytes - 1024**3,
        }
        if fault == "stale":
            updates["sampled_at_unix_ms"] = time.time_ns() // 1_000_000 - 10_000
        if fault == "changed-pool":
            updates["host_total_bytes"] = original.host_total_bytes + 1024**3
        capacity = original.model_copy(update=updates)
        observation = NativeMusicMemoryObservation(
            process_id=child.pid,
            device=observed.device,
            payload_sha256="f" * 64
            if fault == "foreign-payload"
            else observed.payload_sha256,
            capacity=capacity,
            host_resident_bytes=1,
            device_allocated_bytes=0,
        )
        identity = supervisor._diagnostic_context()
        if fault == "foreign-owner":
            identity = identity.model_copy(update={"runner_id": "other-runner"})
        entry = RunnerDiagnosticUpdate(
            at="2026-11-05T00:00:00Z",
            phase="idle",
            event="native_music_memory",
            context=identity,
            native_music_memory=observation,
        )
        phase, history = supervisor._phase, len(supervisor._flight_recorder)
        supervisor._apply_diagnostic_update(entry)
        assert (
            supervisor._phase == phase and len(supervisor._flight_recorder) == history
        )
        if fault == "withdrawn-library":
            await worker._ace_step_preparation.close()
        if fault == "dead-runner":
            child.terminate()
            child.join(timeout=5)
        refreshed = await worker._gather_node_resources()
        lane = "ace_step-" + installed.device
        if fault == "withdrawn-library":
            assert not refreshed.native_music_probes and lane not in refreshed.backends
        else:
            actual = refreshed.native_music_probes[lane].memory_capacity
            assert actual == (capacity if fault == "valid" else original)
            assert refreshed.engine_builds[lane] == installed.wheel.build_identity
        assert refreshed.engine_builds["audio_cpp-cpu"] == "existing-music-build"
    finally:
        if child.pid is not None:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)
        child.close()
        if context is not None:
            context.source_fence.close()
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


@pytest.mark.parametrize("fault", ["memory", "partial-owner"])
async def test_actual_worker_rejects_invalid_admission_and_releases_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
    fault: str,
) -> None:
    """A valid package cannot override low memory or an incomplete restored owner."""
    worker, task, output = runner_fixture(
        tmp_path,
        monkeypatch,
        installed,
        free_bytes=1 if fault == "memory" else 64 * 1024**3,
    )
    if fault == "partial-owner":
        bound = task.bound_instance
        assignments = bound.instance.shard_assignments
        runner = bound.bound_runner_id
        shard = bound.bound_shard
        changed = bound.instance.model_copy(
            update={
                "shard_assignments": assignments.model_copy(
                    update={
                        "runner_to_shard": {
                            runner: shard.model_copy(
                                update={"n_layers": shard.n_layers + 1}
                            )
                        },
                    }
                )
            }
        )
        worker.state = worker.state.model_copy(
            update={"instances": {changed.instance_id: changed}}
        )
        task = task.model_copy(
            update={"bound_instance": bound.model_copy(update={"instance": changed})}
        )
    try:
        async with worker._tg:
            await worker._execute_create_runner(task)
            event = await output.receive()
            assert (
                isinstance(event, TaskFailed)
                and event.error_type == "native_music_unavailable"
            )
        assert not worker.runners and not worker._native_music_creations
        for root in (tmp_path / "readers").iterdir():
            if root.is_dir():
                fence = RuntimeLock(root, name="attachment.lock")
                fence.close()
        assert installed.probes == 1
    finally:
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()


async def test_placement_change_during_reader_preparation_cannot_spawn_or_retain_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: fixtures.NativeInstall,
) -> None:
    """Final staging recheck cancels a removed owner and closes newly acquired custody."""
    worker, task, output = runner_fixture(tmp_path, monkeypatch, installed)
    prepare_reader = worker._native_music_media.prepare_reader

    async def removed(root: Path):
        fence = await prepare_reader(root)
        worker.state = State()
        return fence

    monkeypatch.setattr(worker._native_music_media, "prepare_reader", removed)
    try:
        async with worker._tg:
            await worker._execute_create_runner(task)
            event = await output.receive()
            assert (
                isinstance(event, TaskStatusUpdated)
                and event.task_status == TaskStatus.Cancelled
            )
        assert not worker.runners and not worker._native_music_creations
        for root in (tmp_path / "readers").iterdir():
            if root.is_dir():
                fence = RuntimeLock(root, name="attachment.lock")
                fence.close()
    finally:
        await worker._native_music_media.detach()
        await worker._ace_step_preparation.close()
        await worker._demucs_preparation.close()
