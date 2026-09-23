"""Real local manager transport, client disconnect and unavailable-plugin management."""

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.runtime_controller import LifecycleRequest
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import (
    HostSettings,
    InstallationRequest,
    InventoryRequest,
    OperationRequest,
    RuntimeManager,
    SubmitRequest,
    manager_request,
    manager_socket,
)
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE, running


@pytest.mark.parametrize("action", ["disable", "uninstall"])
async def test_socket_registration_activation_disconnect_and_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: Literal["disable", "uninstall"],
) -> None:
    """The shared terminal/API socket persists accepted work after the client closes."""
    private_directory(tmp_path)
    write_private(
        tmp_path / "host.json",
        HostSettings(transport_node_id="fixture-peer").model_dump_json().encode(),
    )
    manager = RuntimeManager(tmp_path)
    await manager.start()
    identifier = "managed.fixture"
    try:
        assert await manager_request(tmp_path, InventoryRequest()) == {
            "result": {"installations": [], "reload_runtime": True}
        }
        await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id=identifier)
        )
        controller = manager.controllers[identifier]
        assert (
            HostSettings.model_validate_json(
                read_private(controller.root / "owner.json")
            )
            == manager.settings
        )
        metadata, trust, host = artifacts(
            tmp_path / "source", owner_source=OWNER_SOURCE
        )
        monkeypatch.setattr(
            "skulk.extensions.runtime_install.measure_host", lambda: host
        )
        write_private(
            controller.root / "publisher-trust.json", trust.model_dump_json().encode()
        )
        staged = await controller.selector.installer.stage(
            metadata, tmp_path / "source"
        )
        activate = SubmitRequest(
            plugin_id=identifier,
            request=LifecycleRequest(
                operation_id="1" * 32,
                action="activate",
                expected_revision=0,
                runtime_digest=staged.runtime_digest,
            ),
        )
        response = await manager_request(tmp_path, activate)
        assert "result" in response
        assert controller.work is not None
        await controller.work
        assert controller.service is not None
        await running(controller.service)
        process = controller.service.process
        assert process is not None
        disable = SubmitRequest(
            plugin_id=identifier,
            request=LifecycleRequest(
                operation_id="2" * 32, action=action, expected_revision=1
            ),
        )
        reader, writer = await asyncio.open_unix_connection(manager_socket(tmp_path))
        writer.write(disable.model_dump_json().encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        async with asyncio.timeout(10):
            while True:
                result = await manager_request(
                    tmp_path,
                    OperationRequest(plugin_id=identifier, operation_id="2" * 32),
                )
                if '"state": "complete"' in json.dumps(result):
                    break
                await asyncio.sleep(0.01)
        assert process.returncode == 0
        assert await manager_request(tmp_path, disable) == result
        current = controller.selector.current()
        assert current is not None and current.revision == 2
        inventory = await manager_request(tmp_path, InventoryRequest())
        assert json.loads(json.dumps(inventory))["result"]["installations"][0][
            "uninstalled"
        ] == (action == "uninstall")
        assert await reader.read() == b""
        inspection = await manager_request(
            tmp_path, InstallationRequest(action="get", plugin_id=identifier)
        )
        encoded = json.dumps(inspection)
        assert str(tmp_path) not in encoded and "sensitive fixture" not in encoded
        # Purge ends the retention of an uninstall and nothing else: a disabled
        # installation is refused unchanged, an uninstalled one leaves with its
        # directory, and a second purge finds nothing.
        purge = InstallationRequest(action="purge", plugin_id=identifier)
        reply = await manager_request(tmp_path, purge)
        if action == "uninstall":
            assert reply == {"result": {"plugin_id": identifier, "purged": True}}
            assert not (tmp_path / "installations" / identifier).exists()
            assert identifier not in manager.controllers
            assert identifier not in manager.downloads
            assert await manager_request(tmp_path, InventoryRequest()) == {
                "result": {"installations": [], "reload_runtime": True}
            }
            assert "error" in await manager_request(tmp_path, purge)
        else:
            assert "error" in reply
            assert (tmp_path / "installations" / identifier).exists()
            assert manager.controllers[identifier] is controller
    finally:
        await manager.close()
    assert not manager.path.exists()
    RuntimeLock(tmp_path, "manager.lock").close()


async def test_purge_removes_an_installation_that_never_selected_a_release(
    tmp_path: Path,
) -> None:
    """A registration that went nowhere can be removed without an uninstall first."""
    private_directory(tmp_path)
    write_private(
        tmp_path / "host.json",
        HostSettings(transport_node_id="fixture-peer").model_dump_json().encode(),
    )
    manager = RuntimeManager(tmp_path)
    await manager.start()
    identifier = "managed.fixture"
    try:
        await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id=identifier)
        )
        assert (tmp_path / "installations" / identifier).is_dir()
        # Release work under way holds the removal off: the source's guard is
        # what an inspection or a download holds outside the manager guard.
        async with manager.downloads[identifier].guard:
            assert "error" in await manager_request(
                tmp_path, InstallationRequest(action="purge", plugin_id=identifier)
            )
        assert (tmp_path / "installations" / identifier).is_dir()
        assert await manager_request(
            tmp_path, InstallationRequest(action="purge", plugin_id=identifier)
        ) == {"result": {"plugin_id": identifier, "purged": True}}
        assert not (tmp_path / "installations" / identifier).exists()
        assert await manager_request(tmp_path, InventoryRequest()) == {
            "result": {"installations": [], "reload_runtime": True}
        }
        # Registering the same identity again starts clean.
        await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id=identifier)
        )
        assert identifier in manager.controllers
    finally:
        await manager.close()
    RuntimeLock(tmp_path, "manager.lock").close()


async def test_finder_metadata_among_installations_refuses_nothing(
    tmp_path: Path,
) -> None:
    """A folder browsed in Finder still lists and attaches its installations."""
    private_directory(tmp_path)
    write_private(
        tmp_path / "host.json",
        HostSettings(transport_node_id="fixture-peer").model_dump_json().encode(),
    )
    manager = RuntimeManager(tmp_path)
    await manager.start()
    identifier = "managed.fixture"
    try:
        await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id=identifier)
        )
        write_private(tmp_path / "installations" / ".DS_Store", b"finder")
        write_private(tmp_path / "installations" / "._managed.fixture", b"appledouble")
        inventory = await manager_request(tmp_path, InventoryRequest())
        assert "error" not in inventory
        listed = inventory["result"]
        assert isinstance(listed, dict)
        installations = listed["installations"]
        assert isinstance(installations, list)
        assert [
            item["plugin_id"] for item in installations if isinstance(item, dict)
        ] == [identifier]
    finally:
        await manager.close()
    RuntimeLock(tmp_path, "manager.lock").close()


async def test_bad_host_binding_does_not_remove_management(
    tmp_path: Path,
) -> None:
    """A copied or broken installation stays visible without silently changing identity."""
    private_directory(tmp_path)
    write_private(
        tmp_path / "host.json",
        HostSettings(transport_node_id="local-peer").model_dump_json().encode(),
    )
    root = tmp_path / "installations/managed.foreign"
    private_directory(root.parent)
    private_directory(root)
    write_private(
        root / "owner.json",
        HostSettings(transport_node_id="foreign-peer").model_dump_json().encode(),
    )
    manager = RuntimeManager(tmp_path)
    await manager.start()
    try:
        assert manager.boot is not None
        await manager.boot
        inventory = await manager_request(tmp_path, InventoryRequest())
        assert "installation_unavailable" in json.dumps(inventory)
        await manager_request(
            tmp_path,
            InstallationRequest(action="register", plugin_id="managed.foreign"),
        )
        assert (
            HostSettings.model_validate_json(
                read_private(root / "owner.json")
            ).transport_node_id
            == "foreign-peer"
        )
        assert "result" in await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id="managed.good")
        )
        reader, writer = await asyncio.open_unix_connection(manager_socket(tmp_path))
        try:
            writer.write(
                b'{"action":"register","plugin_id":"../escape","executable":"sensitive-token"}\n'
            )
            await writer.drain()
            assert await reader.readline() == b'{"error":"manager_operation_refused"}\n'
        finally:
            writer.close()
            await writer.wait_closed()
        assert not (tmp_path / "escape").exists()
    finally:
        await manager.close()


async def test_socket_removal_failure_still_releases_manager_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local cleanup I/O error cannot strand an already stopped manager's lock."""
    private_directory(tmp_path)
    write_private(
        tmp_path / "host.json",
        HostSettings(transport_node_id="local-peer").model_dump_json().encode(),
    )
    manager = RuntimeManager(tmp_path)
    await manager.start()
    unlink = Path.unlink

    def failed_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == manager.path:
            raise OSError("synthetic socket removal failure")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "unlink", failed_unlink)
        with pytest.raises(OSError, match="synthetic"):
            await manager.close()
    RuntimeLock(tmp_path, "manager.lock").close()
    manager.path.unlink(missing_ok=True)


def test_a_protocol_refusal_is_a_fixed_vocabulary_of_a_code_and_integers() -> None:
    """The manager names exactly one refusal, and names it with numbers only."""
    from skulk.extensions.runtime_artifacts import ProtocolUnsupportedError
    from skulk.extensions.runtime_manager import PROTOCOL_UNSUPPORTED, refusal_payload

    payload = TypeAdapter(dict[str, JsonValue]).validate_json(
        refusal_payload(ProtocolUnsupportedError("release", 3, (1, 2)))
    )
    assert payload == {
        "error": PROTOCOL_UNSUPPORTED,
        "kind": "release",
        "offered": 3,
        "accepted": [1, 2],
    }


async def test_reload_runtime_selects_the_staged_generation_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The running manager activates its own successor and asks its loop to stop."""
    from pydantic import JsonValue, TypeAdapter

    from skulk.extensions.runtime_files import read_private
    from skulk.extensions.runtime_manager import (
        MANAGER_BUILD_DIFFERS,
        ManagerBuildMismatchError,
        ReloadRuntimeRequest,
        build_mismatch_payload,
    )
    from skulk.extensions.tests.test_managed_services import manager_fixture
    from skulk.extensions.tests.test_service_snapshot import staged

    stops: list[bool] = []
    manager = manager_fixture(tmp_path, monkeypatch)
    manager.request_stop = lambda: stops.append(True)
    snapshot = staged(tmp_path)
    await manager.start()
    try:
        result = await manager_request(
            tmp_path,
            ReloadRuntimeRequest(
                generation=snapshot.generation, manifest_sha256=snapshot.manifest_sha256
            ),
        )
        assert result == {
            "result": {"generation": snapshot.generation, "restarting": True}
        }
        assert TypeAdapter(dict[str, JsonValue]).validate_json(
            read_private(tmp_path / "core-runtime.json")
        ) == {
            "generation": snapshot.generation,
            "manifest_sha256": snapshot.manifest_sha256,
        }
        await asyncio.sleep(0.8)
        assert stops == [True]
        refused = await manager_request(
            tmp_path,
            ReloadRuntimeRequest(generation="b" * 32, manifest_sha256="c" * 64),
        )
        assert refused == {"error": "manager_operation_refused"}
    finally:
        await manager.close()
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(
        build_mismatch_payload(ManagerBuildMismatchError("a" * 64, "b" * 64))
    )
    assert payload == {
        "error": MANAGER_BUILD_DIFFERS,
        "manager": "a" * 64,
        "live": "b" * 64,
    }
