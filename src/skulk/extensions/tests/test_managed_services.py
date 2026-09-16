"""Late local setup and live manager membership without API process restart."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from skulk.extensions import CapabilityDescriptor, LoadedExtensions
from skulk.extensions.managed import ManagedConnection, ManagedOwner
from skulk.extensions.managed_attachment import ManagedAttachment
from skulk.extensions.managed_services import ManagedServices
from skulk.extensions.runtime_artifacts import QualifiedHost
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_controller import LifecycleRequest
from skulk.extensions.runtime_download import ReleaseSource
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    InventoryRequest,
    ReleaseRequest,
    RuntimeManager,
    SubmitRequest,
    manager_request,
)
from skulk.extensions.service_setup import SetupOperation
from skulk.extensions.tests.test_managed import Dynamic
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE, running
from skulk.extensions.tests.test_steward_tools import context

PROFILE = "1" * 32
HOST = QualifiedHost("macos-arm64", "3.13.13", "1.5.2", "a" * 64)


def manager_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeManager:
    """Create an isolated real local manager, without any private SDK or provider call."""
    write_private(
        root / "host.json",
        HostSettings(transport_node_id="old-peer", profile_id=PROFILE)
        .model_dump_json()
        .encode(),
    )
    monkeypatch.setattr("skulk.extensions.runtime_manager.measure_host", lambda: HOST)
    monkeypatch.setattr(
        "skulk.extensions.managed_attachment.measure_host", lambda: HOST
    )
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: HOST)
    return RuntimeManager(root)


def connect(path: Path, root: Path) -> None:
    """Publish the exact generated local setup connection atomically."""
    write_private(
        path,
        ServiceConnection(manager_root=str(root), profile_id=PROFILE)
        .model_dump_json()
        .encode(),
    )


async def test_late_setup_activation_outage_and_shutdown_preserve_independent_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Membership changes reach existing Fabric/configuration lookup without a restart."""
    root, path = (
        tmp_path / "manager",
        tmp_path / "config/managed-service/connection.json",
    )
    manager = manager_fixture(root, monkeypatch)
    await manager.start()
    services = ManagedServices(path)
    registry = LoadedExtensions([], managed_services=services).with_builtin_extensions(
        []
    )
    host_context = replace(context(), skulk_version="1.5.2")
    registry.run_startup_hooks(host_context)
    descriptor = CapabilityDescriptor(
        id="fixture",
        version="1.0.0",
        title="Fixture",
        description="Inert fixture",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    async def description(
        self: ManagedOwner, message: dict[str, JsonValue], *, timeout: float = 30
    ) -> dict[str, JsonValue]:
        return {
            "transport_node_id": "test-node",
            "nodes": [
                {
                    "node_id": "durable-node",
                    "bundle_id": "fixture",
                    "version": "1.0.0",
                    "status": "ready",
                    "configurable": True,
                    "descriptors": [descriptor.model_dump(mode="json")],
                }
            ],
        }

    monkeypatch.setattr(ManagedOwner, "_request", description)
    try:
        assert not registry.configuration_providers
        with pytest.raises(FileNotFoundError):
            await services.refresh()
        connect(path, root)
        await services.request(
            InstallationRequest(action="register", plugin_id="managed.fixture")
        )
        await services.refresh()
        assert registry.names == ["managed.fixture"]
        owner = services.owners["managed.fixture"]
        assert registry.configuration_providers["managed.fixture"] is owner
        await owner.refresh()
        assert owner.manager_enabled is None
        assert (
            not registry.capability_descriptors
        )  # Empty registration is not admission.
        controller = manager.controllers["managed.fixture"]
        metadata, trust, _ = artifacts(tmp_path / "source", owner_source=OWNER_SOURCE)
        write_private(
            controller.root / "publisher-trust.json", trust.model_dump_json().encode()
        )
        staged = await controller.selector.installer.stage(
            metadata, tmp_path / "source"
        )
        await services.request(
            SubmitRequest(
                plugin_id="managed.fixture",
                request=LifecycleRequest(
                    operation_id="2" * 32,
                    action="activate",
                    expected_revision=0,
                    runtime_digest=staged.runtime_digest,
                ),
            )
        )
        assert controller.work is not None
        await controller.work
        assert controller.service is not None
        await running(controller.service)
        await services.refresh()
        await owner.refresh()
        assert registry.capability_descriptors == (descriptor,)
        assert registry.call_handler(descriptor.qualified_id) is not None
        assert owner.manager_enabled is True
        replacement = Dynamic("replacement")
        replacement.snapshot = (descriptor,)
        competing = LoadedExtensions([replacement], managed_services=services)
        assert not competing.capability_descriptors
        await manager.close()
        with pytest.raises((OSError, ValueError)):
            await services.refresh()
        assert owner.manager_enabled is None
        assert not competing.capability_descriptors
        owner.available = (
            True  # Even a late successful child observation cannot restore admission.
        )
        assert not registry.capability_descriptors
        manager = RuntimeManager(root)
        await manager.start()
        assert manager.boot is not None
        await manager.boot
        controller = manager.controllers["managed.fixture"]
        assert controller.service is not None
        await running(controller.service)
        await services.refresh()
        await owner.refresh()
        assert services.owners["managed.fixture"] is owner
        assert owner.manager_enabled is True
        assert registry.capability_descriptors == (descriptor,)
        cached_nodes = owner.nodes
        await services.request(
            SubmitRequest(
                plugin_id="managed.fixture",
                request=LifecycleRequest(
                    operation_id="3" * 32,
                    action="disable",
                    expected_revision=1,
                ),
            )
        )
        assert controller.work is not None
        await controller.work
        await services.refresh()
        assert owner.manager_enabled is False
        assert owner.nodes == cached_nodes
        assert registry.configuration_providers["managed.fixture"] is owner
        assert not registry.capability_descriptors
        assert competing.capability_descriptors == (descriptor,)
        entry = competing.call_handler(descriptor.qualified_id)
        assert entry is not None and entry[1] is replacement
        selection_path = controller.root / "runtime-selection.json"
        original_selection = read_private(selection_path)
        for missing in (False, True):
            if missing:
                selection_path.unlink()
            else:
                write_private(selection_path, b"invalid selection")
            try:
                await services.refresh()
                assert owner.manager_enabled is None
                assert owner.nodes == cached_nodes
                assert not competing.capability_descriptors
                assert competing.call_handler(descriptor.qualified_id) is None
            finally:
                write_private(selection_path, original_selection)
            await services.refresh()
            assert owner.manager_enabled is False
            assert competing.capability_descriptors == (descriptor,)
        # A lost manager observation cannot authorize transferring ownership.
        await manager.close()
        with pytest.raises((OSError, ValueError)):
            await services.refresh()
        assert owner.manager_enabled is None
        assert not competing.capability_descriptors
        manager = RuntimeManager(root)
        await manager.start()
        assert manager.boot is not None
        await manager.boot
        await services.refresh()
        assert owner.manager_enabled is False
        assert competing.capability_descriptors == (descriptor,)
        await registry.run_shutdown_hooks()
        assert not registry.capability_descriptors
        assert "result" in await manager_request(root, InventoryRequest())
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "manager.lock")
        RuntimeLock(root, "attachment.lock").close()
        assert (
            HostSettings.model_validate_json(
                read_private(root / "host.json")
            ).transport_node_id
            == "test-node"
        )
    finally:
        await registry.run_shutdown_hooks()
        await manager.close()


async def test_shared_adapter_concurrent_shutdown_releases_attachment_once(
    tmp_path: Path,
) -> None:
    """Legacy and manager registry shutdown cannot underflow their shared bridge."""
    attachment = ManagedAttachment(tmp_path, PROFILE)
    owner = ManagedOwner(
        ManagedConnection(plugin_id="managed.fixture", state_root=str(tmp_path)),
        attachment=attachment,
    )
    attachment.retain("test-node")
    attachment.lock = RuntimeLock(tmp_path, "attachment.lock")

    async def observer() -> None:
        await asyncio.Event().wait()

    owner.poll_task = asyncio.create_task(observer())
    await asyncio.gather(owner.on_stop(), owner.on_stop())
    assert attachment.users == 0
    assert attachment.lock is None
    RuntimeLock(tmp_path, "attachment.lock").close()


async def test_slow_release_inspection_does_not_block_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both API attachment and manager inventory remain responsive during source HTTPS."""
    root, path = tmp_path / "manager", tmp_path / "connection.json"
    manager = manager_fixture(root, monkeypatch)
    await manager.start()
    connect(path, root)
    services = ManagedServices(path)
    services.on_start(replace(context(), skulk_version="1.5.2"))
    started, release = asyncio.Event(), asyncio.Event()
    operation: asyncio.Task[dict[str, JsonValue]] | None = None
    try:
        await services.request(
            InstallationRequest(action="register", plugin_id="managed.fixture")
        )
        downloads = manager.downloads["managed.fixture"]
        metadata, trust, _ = artifacts(tmp_path / "source")
        write_private(
            downloads.root / "publisher-trust.json", trust.model_dump_json().encode()
        )
        write_private(
            downloads.root / "release-source.json",
            ReleaseSource(
                revision=1,
                base_url="https://release.example.test/",
                metadata_filename="runtime.json",
            )
            .model_dump_json()
            .encode(),
        )

        async def delayed(_: httpx.Request) -> httpx.Response:
            started.set()
            await release.wait()
            return httpx.Response(200, content=metadata)

        downloads.transport = httpx.MockTransport(delayed)
        operation = asyncio.create_task(
            services.request(
                ReleaseRequest(action="inspect_release", plugin_id="managed.fixture")
            )
        )
        async with asyncio.timeout(5):
            await started.wait()
            assert services.attachment is not None
            services.attachment.observed = 0
            inventory = await services.refresh()
            assert len(inventory.installations) == 1
            assert not operation.done()
        release.set()
        assert "runtime_digest" in await operation
    finally:
        release.set()
        if operation is not None:
            await asyncio.gather(operation, return_exceptions=True)
        await services.on_stop()
        await manager.close()


def test_the_api_side_rebuilds_the_typed_refusal_from_the_fixed_vocabulary() -> None:
    from skulk.extensions.managed_services import protocol_refusal
    from skulk.extensions.runtime_artifacts import ProtocolUnsupportedError

    refused = protocol_refusal(
        {
            "error": "release_protocol_unsupported",
            "kind": "runtime",
            "offered": 3,
            "accepted": [2],
        }
    )
    assert isinstance(refused, ProtocolUnsupportedError)
    assert (refused.kind, refused.offered, refused.accepted) == ("runtime", 3, (2,))
    assert protocol_refusal({"error": "manager_operation_refused"}) is None
    assert (
        protocol_refusal({"error": "release_protocol_unsupported", "accepted": ["2"]})
        is None
    )
    assert (
        protocol_refusal(
            {
                "error": "release_protocol_unsupported",
                "kind": "runtime",
                "offered": True,
                "accepted": [2],
            }
        )
        is None
    )


async def test_a_build_mismatch_stages_a_matching_runtime_and_asks_for_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host refreshes the manager on its own after a Skulk update, once."""
    from skulk.extensions.runtime_manager import (
        ManagerBuildMismatchError,
        ReloadRuntimeRequest,
    )
    from skulk.extensions.service_snapshot import ServiceSnapshot

    staged: list[Path] = []
    sent: list[object] = []

    async def stage(root: Path) -> ServiceSnapshot:
        staged.append(root)
        return ServiceSnapshot(
            generation="d" * 32,
            manifest_sha256="e" * 64,
            skulk_build_sha256="f" * 64,
            copied_files=1,
            copied_bytes=1,
        )

    async def request(root: Path, request: object) -> dict[str, JsonValue]:
        sent.append(request)
        if isinstance(request, InventoryRequest):
            return {"result": {"installations": [], "reload_runtime": True}}
        return {"result": {"generation": "d" * 32, "restarting": True}}

    monkeypatch.setattr(
        "skulk.extensions.managed_services.stage_service_runtime", stage
    )
    monkeypatch.setattr("skulk.extensions.managed_services.manager_request", request)
    monkeypatch.setattr(
        "skulk.extensions.managed_services.service_source_identity", lambda: "9" * 64
    )
    write_private(
        tmp_path / "setup.json",
        SetupOperation(
            operation_id=PROFILE,
            profile_id=PROFILE,
            skulk_build_sha256="a" * 64,
            source_sha256="c" * 64,
            configuration_directory=str(tmp_path),
            phase="ready",
            snapshot=ServiceSnapshot(
                generation="0" * 32,
                manifest_sha256="e" * 64,
                skulk_build_sha256="a" * 64,
                copied_files=1,
                copied_bytes=1,
            ),
        )
        .model_dump_json()
        .encode(),
    )
    services = ManagedServices(tmp_path / "connection.json")
    services.connection = ServiceConnection(
        manager_root=str(tmp_path), profile_id=PROFILE
    )
    differs = ManagerBuildMismatchError("a" * 64, "b" * 64)
    services._schedule_runtime_refresh(differs)  # pyright: ignore[reportPrivateUsage]
    services._schedule_runtime_refresh(differs)  # pyright: ignore[reportPrivateUsage]
    assert services.runtime_refresh is not None
    # Detaching drains the refresh instead of cancelling it: its selection
    # runs in a thread that cancellation would leave running.
    await services._settle_runtime_refresh()  # pyright: ignore[reportPrivateUsage]
    assert services.runtime_refresh is None
    assert staged == [tmp_path]
    assert [type(request) for request in sent] == [
        InventoryRequest,
        ReloadRuntimeRequest,
    ]
    assert isinstance(sent[1], ReloadRuntimeRequest)
    assert sent[1].generation == "d" * 32
    recorded = SetupOperation.model_validate_json(read_private(tmp_path / "setup.json"))
    assert recorded.snapshot is not None
    assert recorded.snapshot.generation == "d" * 32
    assert recorded.skulk_build_sha256 == "f" * 64
    assert recorded.source_sha256 == "9" * 64
    assert recorded.phase == "registered"


async def test_a_generic_refusal_after_a_finished_selection_is_still_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manager's handler deadline answers generically while its selection lands."""
    from skulk.extensions import managed_services
    from skulk.extensions.runtime_manager import (
        ManagerBuildMismatchError,
        ReloadRuntimeRequest,
    )
    from skulk.extensions.service_snapshot import ServiceSnapshot

    snapshot = ServiceSnapshot(
        generation="d" * 32,
        manifest_sha256="e" * 64,
        skulk_build_sha256="f" * 64,
        copied_files=1,
        copied_bytes=1,
    )

    async def stage(root: Path) -> ServiceSnapshot:
        return snapshot

    async def request(root: Path, request: object) -> dict[str, JsonValue]:
        if isinstance(request, InventoryRequest):
            return {"result": {"installations": [], "reload_runtime": True}}
        assert isinstance(request, ReloadRuntimeRequest)
        write_private(
            root / "core-runtime.json",
            json.dumps(
                {"generation": request.generation, "manifest_sha256": "e" * 64}
            ).encode(),
        )
        return {"error": "manager_operation_refused"}

    monkeypatch.setattr(managed_services, "stage_service_runtime", stage)
    monkeypatch.setattr(managed_services, "manager_request", request)
    monkeypatch.setattr(managed_services, "service_source_identity", lambda: "9" * 64)
    monkeypatch.setattr(managed_services, "_SELECTION_WAIT_SECONDS", 1.0)
    write_private(
        tmp_path / "setup.json",
        SetupOperation(
            operation_id=PROFILE,
            profile_id=PROFILE,
            skulk_build_sha256="a" * 64,
            source_sha256="c" * 64,
            configuration_directory=str(tmp_path),
            phase="ready",
            snapshot=ServiceSnapshot(
                generation="0" * 32,
                manifest_sha256="e" * 64,
                skulk_build_sha256="a" * 64,
                copied_files=1,
                copied_bytes=1,
            ),
        )
        .model_dump_json()
        .encode(),
    )
    services = ManagedServices(tmp_path / "connection.json")
    services.connection = ServiceConnection(
        manager_root=str(tmp_path), profile_id=PROFILE
    )
    services._schedule_runtime_refresh(  # pyright: ignore[reportPrivateUsage]
        ManagerBuildMismatchError("a" * 64, "f" * 64)
    )
    assert services.runtime_refresh is not None
    await services.runtime_refresh
    recorded = SetupOperation.model_validate_json(read_private(tmp_path / "setup.json"))
    assert recorded.snapshot == snapshot
    assert (tmp_path / "core-runtimes" / snapshot.generation).exists() is False


async def test_a_reload_that_outlives_its_request_is_still_recorded_once_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manager that selects after the request deadline still updates setup state."""
    from skulk.extensions import managed_services
    from skulk.extensions.runtime_manager import (
        ManagerBuildMismatchError,
        ReloadRuntimeRequest,
    )
    from skulk.extensions.service_snapshot import ServiceSnapshot

    snapshot = ServiceSnapshot(
        generation="d" * 32,
        manifest_sha256="e" * 64,
        skulk_build_sha256="f" * 64,
        copied_files=1,
        copied_bytes=1,
    )

    async def stage(root: Path) -> ServiceSnapshot:
        return snapshot

    async def request(root: Path, request: object) -> dict[str, JsonValue]:
        if isinstance(request, InventoryRequest):
            return {"result": {"installations": [], "reload_runtime": True}}
        assert isinstance(request, ReloadRuntimeRequest)
        # The manager selected the generation while the client gave up.
        write_private(
            root / "core-runtime.json",
            json.dumps(
                {"generation": request.generation, "manifest_sha256": "e" * 64}
            ).encode(),
        )
        raise TimeoutError

    monkeypatch.setattr(managed_services, "stage_service_runtime", stage)
    monkeypatch.setattr(managed_services, "manager_request", request)
    monkeypatch.setattr(managed_services, "service_source_identity", lambda: "9" * 64)
    monkeypatch.setattr(managed_services, "_SELECTION_WAIT_SECONDS", 1.0)
    write_private(
        tmp_path / "setup.json",
        SetupOperation(
            operation_id=PROFILE,
            profile_id=PROFILE,
            skulk_build_sha256="a" * 64,
            source_sha256="c" * 64,
            configuration_directory=str(tmp_path),
            phase="ready",
            snapshot=ServiceSnapshot(
                generation="0" * 32,
                manifest_sha256="e" * 64,
                skulk_build_sha256="a" * 64,
                copied_files=1,
                copied_bytes=1,
            ),
        )
        .model_dump_json()
        .encode(),
    )
    services = ManagedServices(tmp_path / "connection.json")
    services.connection = ServiceConnection(
        manager_root=str(tmp_path), profile_id=PROFILE
    )
    services._schedule_runtime_refresh(  # pyright: ignore[reportPrivateUsage]
        ManagerBuildMismatchError("a" * 64, "f" * 64)
    )
    assert services.runtime_refresh is not None
    await services.runtime_refresh
    recorded = SetupOperation.model_validate_json(read_private(tmp_path / "setup.json"))
    assert recorded.snapshot == snapshot
    assert recorded.source_sha256 == "9" * 64
    assert recorded.phase == "registered"


async def test_a_refused_reload_removes_the_unselected_candidate_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused generation the manager refuses does not pile up on the volume."""
    from skulk.extensions.runtime_manager import (
        ManagerBuildMismatchError,
        ReloadRuntimeRequest,
    )
    from skulk.extensions.service_snapshot import ServiceSnapshot

    build = "f" * 64
    reused = ServiceSnapshot(
        generation="d" * 32,
        manifest_sha256="e" * 64,
        skulk_build_sha256=build,
        copied_files=1,
        copied_bytes=1,
    )
    generation = tmp_path / "core-runtimes" / reused.generation
    private_directory(generation)
    write_private(generation / "staged.json", reused.model_dump_json().encode())
    selected = tmp_path / "core-runtimes" / ("1" * 32)
    private_directory(selected)
    write_private(
        tmp_path / "core-runtime.json",
        json.dumps({"generation": "1" * 32, "manifest_sha256": "2" * 64}).encode(),
    )
    sent: list[object] = []

    async def stage(root: Path) -> ServiceSnapshot:
        raise AssertionError("a matching generation was already staged")

    async def request(root: Path, request: object) -> dict[str, JsonValue]:
        sent.append(request)
        if isinstance(request, InventoryRequest):
            return {"result": {"installations": [], "reload_runtime": True}}
        return {"error": "staged service identity differs"}

    monkeypatch.setattr(
        "skulk.extensions.managed_services.stage_service_runtime", stage
    )
    monkeypatch.setattr("skulk.extensions.managed_services.manager_request", request)
    services = ManagedServices(tmp_path / "connection.json")
    services.connection = ServiceConnection(
        manager_root=str(tmp_path), profile_id=PROFILE
    )
    services._schedule_runtime_refresh(  # pyright: ignore[reportPrivateUsage]
        ManagerBuildMismatchError("a" * 64, build)
    )
    assert services.runtime_refresh is not None
    await services.runtime_refresh
    assert isinstance(sent[-1], ReloadRuntimeRequest)
    assert not generation.exists()
    assert selected.exists()


async def test_a_legacy_manager_under_another_interpreter_is_left_for_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy path never stops a service this interpreter could not restart."""
    from skulk.extensions import managed_services
    from skulk.extensions.runtime_manager import ManagerBuildMismatchError
    from skulk.extensions.service_snapshot import ServiceSnapshot

    signalled: list[int] = []

    async def stage(root: Path) -> ServiceSnapshot:
        raise AssertionError("nothing is staged for a service setup must repair")

    async def request(root: Path, request: object) -> dict[str, JsonValue]:
        assert isinstance(request, InventoryRequest)
        return {"result": {"installations": []}}

    def record(pid: int, signal_number: int) -> None:
        signalled.append(pid)

    def another_interpreter(base: Path) -> bool:
        return False

    def running(root: Path) -> list[int]:
        return [os.getpid()]

    monkeypatch.setattr(managed_services, "stage_service_runtime", stage)
    monkeypatch.setattr(managed_services, "manager_request", request)
    monkeypatch.setattr(
        managed_services, "registered_unit_names_base", another_interpreter
    )
    monkeypatch.setattr(managed_services, "manager_pids", running)
    monkeypatch.setattr(managed_services.os, "kill", record)
    services = ManagedServices(tmp_path / "connection.json")
    services.connection = ServiceConnection(
        manager_root=str(tmp_path), profile_id=PROFILE
    )
    services._schedule_runtime_refresh(  # pyright: ignore[reportPrivateUsage]
        ManagerBuildMismatchError("a" * 64, "f" * 64)
    )
    assert services.runtime_refresh is not None
    await services.runtime_refresh
    assert signalled == []
    assert not (tmp_path / "core-runtimes").exists()


async def test_a_staging_failure_ends_automatic_attempts_until_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An environment that cannot be staged does not grow a copy every attempt."""
    from skulk.extensions import managed_services
    from skulk.extensions.runtime_manager import ManagerBuildMismatchError
    from skulk.extensions.service_snapshot import ServiceSnapshot

    attempts: list[Path] = []

    async def stage(root: Path) -> ServiceSnapshot:
        attempts.append(root)
        raise ValueError("copied service identity differs")

    async def request(root: Path, request: object) -> dict[str, JsonValue]:
        return {"result": {"installations": [], "reload_runtime": True}}

    monkeypatch.setattr(managed_services, "stage_service_runtime", stage)
    monkeypatch.setattr(managed_services, "manager_request", request)
    services = ManagedServices(tmp_path / "connection.json")
    services.connection = ServiceConnection(
        manager_root=str(tmp_path), profile_id=PROFILE
    )
    differs = ManagerBuildMismatchError("a" * 64, "f" * 64)
    services._schedule_runtime_refresh(differs)  # pyright: ignore[reportPrivateUsage]
    assert services.runtime_refresh is not None
    await services.runtime_refresh
    assert services.runtime_refreshed is not None
    services.runtime_refreshed = None
    services._schedule_runtime_refresh(differs)  # pyright: ignore[reportPrivateUsage]
    assert services.runtime_refresh.done()
    assert attempts == [tmp_path]
    assert services.runtime_refresh_exhausted


def test_a_generation_staged_for_the_live_build_is_reused_not_restaged(
    tmp_path: Path,
) -> None:
    from skulk.extensions.managed_services import staged_generation_for
    from skulk.extensions.runtime_files import write_private
    from skulk.extensions.service_snapshot import ServiceSnapshot

    generations = tmp_path / "core-runtimes"
    generations.mkdir(mode=0o700)
    for name, build in (("a" * 32, "f" * 64), ("b" * 32, "9" * 64)):
        (generations / name).mkdir(mode=0o700)
        write_private(
            generations / name / "staged.json",
            ServiceSnapshot(
                generation=name,
                manifest_sha256="e" * 64,
                skulk_build_sha256=build,
                copied_files=1,
                copied_bytes=1,
            )
            .model_dump_json()
            .encode(),
        )
    found = staged_generation_for(tmp_path, "f" * 64)
    assert found is not None and found.generation == "a" * 32
    # The selected generation is never offered again, and an unknown build is not found.
    write_private(
        tmp_path / "core-runtime.json",
        b'{"generation": "' + b"a" * 32 + b'", "manifest_sha256": "x"}',
    )
    assert staged_generation_for(tmp_path, "f" * 64) is None
    assert staged_generation_for(tmp_path, "0" * 64) is None


def test_a_legacy_manager_is_detected_from_its_selected_generation(
    tmp_path: Path,
) -> None:
    from skulk.extensions.managed_attachment import selected_manager_build
    from skulk.extensions.runtime_files import write_private

    assert selected_manager_build(tmp_path) is None
    generations = tmp_path / "core-runtimes"
    generations.mkdir(mode=0o700)
    (generations / ("a" * 32)).mkdir(mode=0o700)
    write_private(
        generations / ("a" * 32) / "staged.json",
        b'{"skulk_build_sha256": "' + b"f" * 64 + b'"}',
    )
    write_private(
        tmp_path / "core-runtime.json",
        b'{"generation": "' + b"a" * 32 + b'", "manifest_sha256": "x"}',
    )
    assert selected_manager_build(tmp_path) == "f" * 64


def test_a_legacy_manager_is_stopped_and_the_generation_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skulk.extensions.managed_services import reload_legacy_manager
    from skulk.extensions.runtime_files import read_private
    from skulk.extensions.tests.test_service_snapshot import staged

    def no_manager(root: Path) -> list[int]:
        return []

    monkeypatch.setattr("skulk.extensions.managed_services.manager_pids", no_manager)
    snapshot = staged(tmp_path)
    reload_legacy_manager(tmp_path, snapshot)
    assert snapshot.generation.encode() in read_private(tmp_path / "core-runtime.json")


def test_a_legacy_manager_that_keeps_running_is_not_selected_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selection waits for the stopped manager's exit and reports a stuck one."""
    from skulk.extensions import managed_services
    from skulk.extensions.managed_services import reload_legacy_manager
    from skulk.extensions.tests.test_service_snapshot import staged

    signalled: list[int] = []

    def still_running(root: Path) -> list[int]:
        return [os.getpid()]

    def record(pid: int, signal_number: int) -> None:
        signalled.append(pid)

    monkeypatch.setattr(managed_services, "manager_pids", still_running)
    monkeypatch.setattr(managed_services.os, "kill", record)
    monkeypatch.setattr(managed_services, "_MANAGER_STOP_SECONDS", 0.6)
    snapshot = staged(tmp_path)
    with pytest.raises(OSError, match="has not exited"):
        reload_legacy_manager(tmp_path, snapshot)
    assert signalled == [os.getpid()]
    assert not (tmp_path / "core-runtime.json").exists()
