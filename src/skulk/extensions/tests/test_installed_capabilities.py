"""Installed declarations survive disabled children without creating effect authority."""

import json
import time
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from pydantic import JsonValue

from skulk.api.plugins import create_plugins_router
from skulk.api.steward import StewardHarness, steward_tool_definitions
from skulk.api.tests.test_operator_gateway import paired_service
from skulk.extensions import CapabilityDescriptor, LoadedExtensions
from skulk.extensions.capabilities import (
    InstalledCapabilityRegistration,
    RegisteredCapabilitySummary,
    descriptor_revision,
)
from skulk.extensions.installed_capabilities import (
    read_installed_capabilities,
    read_installed_contract,
    steward_installed_result,
)
from skulk.extensions.managed import ManagedConnection, ManagedNode, ManagedOwner
from skulk.extensions.managed_services import (
    ManagedInstallation,
    ManagedInventory,
    ManagedServices,
)
from skulk.extensions.runtime_files import write_private
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    InstalledContractRequest,
    installed_release,
)
from skulk.extensions.runtime_service import RuntimeServiceStatus
from skulk.extensions.tests.test_managed_services import connect, manager_fixture
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.operator.pairing import PluginGrantUpdate


def declared_operation(
    description: str = "Estimate offers; acquisition requires owner policy.",
) -> CapabilityDescriptor:
    """Declare an operation independently of its child or actual capacity."""
    return CapabilityDescriptor(
        id="capacity",
        version="1.0.0",
        title="Capacity planning",
        description=description,
        input_schema={
            "type": "object",
            "properties": {"operation": {"enum": ["estimate", "acquire", "release"]}},
        },
    )


async def test_disabled_registered_contract_http_read_is_scoped_and_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real manager serves selected schemas without attachment or child startup."""
    manager = manager_fixture(tmp_path / "manager", monkeypatch)
    await manager.start()
    identifier = "managed.fixture"
    path = tmp_path / "connection.json"
    connect(path, manager.root)
    services = ManagedServices(path)
    descriptor = declared_operation()
    try:
        await manager.dispatch(
            InstallationRequest(action="register", plugin_id=identifier)
        )
        controller = manager.controllers[identifier]
        metadata, trust, _ = artifacts(
            tmp_path / "source",
            manifest_extra={
                "descriptors": [descriptor.model_dump(mode="json")],
                "credential_inputs": [
                    {"credential_id": "provider_key", "required": True}
                ],
                "configuration_schema": {"type": "object"},
            },
        )
        write_private(
            controller.root / "publisher-trust.json", trust.model_dump_json().encode()
        )
        staged = await controller.selector.installer.stage(
            metadata, tmp_path / "source"
        )
        await controller.selector.activate(
            staged.runtime_digest, expected_revision=0, enabled=False
        )

        async def refuse_attachment() -> None:
            raise AssertionError("metadata read attempted an attachment")

        monkeypatch.setattr(services, "_connect", refuse_attachment)
        inventory = await read_installed_capabilities(services)
        row = inventory.installations[0]
        assert not row.installation.enabled
        assert row.reachable is None and row.ready_capabilities == ()
        release = row.installation.release
        assert release is not None and release.registration is not None
        assert (
            release.registration.capabilities[0].qualified_id == descriptor.qualified_id
        )
        assert release.registration.required_credential_ids == ("provider_key",)
        assert release.registration.configuration_required
        serialized = inventory.model_dump_json()
        assert "bundle.pyz" not in serialized and str(tmp_path) not in serialized
        selected = InstalledContractRequest(
            plugin_id=identifier,
            qualified_id=descriptor.qualified_id,
            revision=descriptor_revision(descriptor),
        )
        contract = await read_installed_contract(services, selected)
        assert contract.descriptor == descriptor
        with pytest.raises(ValueError, match="unavailable"):
            await read_installed_contract(
                services, selected.model_copy(update={"revision": "f" * 16})
            )
        assert controller.service is None or controller.service.process is None

        pairing, exchange = paired_service(tmp_path / "pairing")
        app = FastAPI()
        app.include_router(
            create_plugins_router(
                LoadedExtensions([], managed_services=services), pairing
            )
        )
        headers = {"Authorization": f"Bearer {exchange.access_token}"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 52000)),
            base_url="https://localhost",
        ) as client:
            prefix = "/v1/plugins/managed/capabilities"
            assert (await client.get(prefix, headers=headers)).status_code == 403
            pairing.set_plugin_grant(
                exchange.device_id,
                PluginGrantUpdate(expected_revision=0, scopes=("plugins:read",)),
            )
            result = await client.get(prefix, headers=headers)
            assert (
                result.status_code == 200
                and result.headers["Cache-Control"] == "no-store"
            )
            full = await client.get(
                f"/v1/plugins/managed/{identifier}/capabilities/{descriptor.qualified_id}",
                headers=headers,
                params={"revision": selected.revision},
            )
            assert full.status_code == 200
            assert full.json()["descriptor"]["input_schema"] == descriptor.input_schema
            pairing.set_plugin_grant(
                exchange.device_id, PluginGrantUpdate(expected_revision=1, scopes=())
            )
            assert (await client.get(prefix, headers=headers)).status_code == 403
        assert controller.service is None or controller.service.process is None
    finally:
        await manager.close()


def test_malformed_declarations_preserve_installed_identity(tmp_path: Path) -> None:
    """Invalid optional registration cannot turn an installed package into absence."""
    path = tmp_path / "staged.json"
    write_private(
        path,
        json.dumps(
            {
                "runtime": {
                    "release": {
                        "publisher": "fixture",
                        "sequence": 1,
                        "manifest": {
                            "bundle_id": "example.plugin",
                            "bundle_version": "1.0.0",
                            "descriptors": "invalid",
                        },
                    }
                }
            }
        ).encode(),
    )
    release = installed_release(path)
    assert release is not None and release.bundle_id == "example.plugin"
    assert release.registration is None


def test_large_contract_preserves_selection_without_truncated_schema() -> None:
    """The bounded model result does not pretend an omitted schema is usable."""
    from skulk.extensions.installed_capabilities import InstalledContract

    descriptor = declared_operation("large purpose " * 1500)
    contract = InstalledContract(
        plugin_id="managed.fixture",
        selected_digest="a" * 64,
        descriptor=descriptor,
        revision=descriptor_revision(descriptor),
    )
    result = steward_installed_result(contract, 6000)
    assert result["schemaIncluded"] is False
    assert result["revision"] == descriptor_revision(descriptor)
    assert "descriptor" not in result
    assert len(json.dumps(result)) < 6000


async def test_steward_installation_tool_requires_separate_reader() -> None:
    """A forged tool call cannot use general Steward access to inspect installations."""
    from skulk.api.main import API

    assert "get_installed_capabilities" not in json.dumps(steward_tool_definitions())
    assert "get_installed_capabilities" in json.dumps(
        steward_tool_definitions(include_installations=True)
    )
    harness = StewardHarness(cast("API", object()))
    assert "plugins:read" in await harness.execute_tool(
        "get_installed_capabilities", {}
    )
    calls: list[tuple[str | None, str | None, str | None]] = []

    async def reader(
        plugin: str | None, capability: str | None, revision: str | None
    ) -> dict[str, object]:
        calls.append((plugin, capability, revision))
        raise OSError("private-path-never-disclose")

    allowed = StewardHarness(cast("API", object()), installation_reader=reader)
    assert "together" in await allowed.execute_tool(
        "get_installed_capabilities", {"plugin_id": "managed.fixture"}
    )
    assert not calls
    refused = await allowed.execute_tool("get_installed_capabilities", {})
    assert "unavailable" in refused and "private-path" not in refused
    assert calls == [(None, None, None)]


@pytest.mark.parametrize(
    "condition",
    [
        "fresh",
        "stale",
        "future",
        "changed",
        "disabled",
        "uninstalled",
        "generation_changed",
    ],
)
async def test_installed_identity_alone_never_establishes_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    """Readiness requires matching current child observations and selected runtime."""
    descriptor = declared_operation()
    identifier = "managed.fixture"
    from skulk.extensions.runtime_manager import InstalledRelease, InventoryRequest

    registration = InstalledCapabilityRegistration(
        capabilities=(
            RegisteredCapabilitySummary(
                qualified_id=descriptor.qualified_id,
                revision=descriptor_revision(descriptor),
                title=descriptor.title,
                purpose=descriptor.description,
                io_mode="unary",
            ),
        ),
        skulk_requires=">=2",
        python_requires=">=3.12",
        platforms=("linux",),
    )
    release = InstalledRelease(
        bundle_id="example.plugin",
        bundle_version="1.0.0",
        publisher="fixture",
        sequence=1,
        registration=registration,
    )
    installation = ManagedInstallation(
        plugin_id=identifier,
        release=release,
        selected_digest="a" * 64,
        enabled=condition != "disabled",
        uninstalled=condition == "uninstalled",
        stale=False,
        service=RuntimeServiceStatus(
            service_instance="b" * 32,
            observed_at=time.time(),
            state="running",
            selected_digest="a" * 64,
            active_digest=("c" if condition == "generation_changed" else "a") * 64,
            output_bytes=0,
        ),
    )
    owner = ManagedOwner(
        ManagedConnection(plugin_id=identifier, state_root=str(tmp_path / "owner"))
    )
    owner.available = owner.manager_available = True
    owner.manager_enabled = True
    owner.observed = time.monotonic() + (
        120 if condition == "future" else -120 if condition == "stale" else 0
    )
    observed_descriptor = (
        descriptor.model_copy(update={"description": "changed"})
        if condition == "changed"
        else descriptor
    )
    owner.nodes = (
        ManagedNode(
            node_id="fixture",
            bundle_id=release.bundle_id,
            version=release.bundle_version,
            status="ready",
            configurable=False,
            descriptors=(observed_descriptor,),
        ),
    )
    services = ManagedServices(tmp_path / "unopened-connection.json")
    services.owners[identifier] = owner

    async def read(
        _request: InventoryRequest | InstalledContractRequest,
    ) -> dict[str, JsonValue]:
        return ManagedInventory(installations=(installation,)).model_dump(mode="json")

    monkeypatch.setattr(services, "read_installed_metadata", read)
    result = await read_installed_capabilities(services)
    assert result.installations[0].ready_capabilities == (
        (descriptor.qualified_id,) if condition == "fresh" else ()
    )
