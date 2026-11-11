"""Native selected declarations retain full identity without granting execution."""

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.plugins import create_plugins_router
from skulk.api.tests.test_operator_gateway import paired_service
from skulk.extensions.installed_capabilities import (
    read_installed_capabilities,
    read_installed_contract,
)
from skulk.extensions.loader import LoadedExtensions
from skulk.extensions.managed_services import ManagedServices
from skulk.extensions.native_contracts import NativeDescriptor
from skulk.extensions.runtime_artifacts import (
    QualifiedHost,
    VerifiedRuntime,
    verify_runtime,
)
from skulk.extensions.runtime_files import private_directory, write_private
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    InstalledContractRequest,
)
from skulk.extensions.runtime_selection import RuntimeSelection, RuntimeSelector
from skulk.extensions.tests.test_managed_services import connect, manager_fixture
from skulk.extensions.tests.test_native_runtime import OBJECT, native_fixture, sign
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.operator.pairing import PluginGrantUpdate


async def test_native_installed_metadata_round_trips_without_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real metadata RPC/HTTP path preserves native schemas and full revisions.

    Only retained metadata is provisioned: this test deliberately installs no
    interpreter or executable. A display read must not launch or attach one.
    Actual offline installation has a separate signed-artifact qualification.
    """
    manager = manager_fixture(tmp_path / "manager", monkeypatch)
    await manager.start()
    path = tmp_path / "connection.json"
    connect(path, manager.root)
    services = ManagedServices(path)
    identifier = "managed.native-metadata"
    try:
        await manager.dispatch(
            InstallationRequest(action="register", plugin_id=identifier)
        )
        controller = manager.controllers[identifier]
        payload, key, trust, host = native_fixture(tmp_path / "source")
        metadata = sign(payload, key)
        runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
        manifest = runtime.native_manifest
        assert manifest is not None
        contracts = OBJECT.validate_python(manifest.value()["contracts"], strict=True)
        descriptors = contracts["descriptors"]
        assert isinstance(descriptors, list)
        descriptor = NativeDescriptor.model_validate_json(json.dumps(descriptors[0]))
        release = runtime.claims.release
        selection = RuntimeSelection(
            revision=1,
            operation_id="d" * 32,
            runtime_digest=runtime.digest,
            enabled=False,
            bundle_id=manifest.bundle_id,
            sequence=release.sequence,
            highest_sequence=release.sequence,
            state_schema=release.state_schema,
            permissions=release.permissions,
            configuration_schema=manifest.configuration_schema,
            platform=host.platform,
            python_version=host.python_version,
            skulk_version=host.skulk_version,
            skulk_build_sha256=host.skulk_build_sha256,
        )
        generation = controller.root / "generations" / runtime.digest
        private_directory(generation)
        write_private(generation / "staged.json", metadata)
        write_private(
            controller.root / "runtime-selection.json",
            selection.model_dump_json().encode(),
        )

        async def refuse_attachment() -> None:
            raise AssertionError("native metadata attempted a live owner attachment")

        monkeypatch.setattr(services, "_connect", refuse_attachment)
        inventory = await read_installed_capabilities(services)
        row = inventory.installations[0]
        assert row.ready_capabilities == () and row.reachable is None
        assert row.installation.release is not None
        registration = row.installation.release.registration
        assert registration is not None and registration.configuration_required
        declared = registration.capabilities[0]
        assert declared.qualified_id == "UpperCase:Echo@native@1.0.0"
        assert declared.revision == descriptor.revision and len(declared.revision) == 64
        request = InstalledContractRequest(
            plugin_id=identifier,
            qualified_id=declared.qualified_id,
            revision=declared.revision,
        )
        result = await read_installed_contract(services, request)
        assert (
            result.descriptor == descriptor and result.revision == descriptor.revision
        )
        assert result.descriptor.input_schema is True
        assert controller.service is None or controller.service.process is None

        pairing, exchange = paired_service(tmp_path / "pairing")
        app = FastAPI()
        app.include_router(
            create_plugins_router(
                LoadedExtensions([], managed_services=services), pairing
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 52000)),
            base_url="https://localhost",
        ) as client:
            endpoint = (
                f"/v1/plugins/managed/{identifier}/capabilities/{declared.qualified_id}"
            )
            headers = {"Authorization": f"Bearer {exchange.access_token}"}
            refused = await client.get(
                endpoint, params={"revision": declared.revision}, headers=headers
            )
            assert refused.status_code == 403
            pairing.set_plugin_grant(
                exchange.device_id,
                PluginGrantUpdate(expected_revision=0, scopes=("plugins:read",)),
            )
            response = await client.get(
                endpoint, params={"revision": declared.revision}, headers=headers
            )
            assert response.status_code == 200
            observed = OBJECT.validate_json(response.content)
            assert observed["revision"] == declared.revision
            assert (
                OBJECT.validate_python(observed["descriptor"])["input_schema"] is True
            )
            for changed in ("f" * 64, declared.revision[:16]):
                response = await client.get(
                    endpoint, params={"revision": changed}, headers=headers
                )
                assert response.status_code == 409
        assert not (generation / "runtime").exists()
    finally:
        await manager.close()


async def test_normal_selection_refuses_implicit_profile_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal state/schema labels cannot relabel retained data under new semantics."""
    payload, key, trust, host = native_fixture(tmp_path / "native-source")
    candidate = verify_runtime(sign(payload, key), trust, host, now=int(time.time()))
    legacy_metadata, legacy_trust, legacy_host = artifacts(
        tmp_path / "legacy-source",
        signing_key=key,
        bundle_id=candidate.claims.release.manifest.bundle_id,
        manifest_extra={"configuration_schema": True},
    )
    legacy = verify_runtime(
        legacy_metadata, legacy_trust, legacy_host, now=int(time.time())
    )
    selector = RuntimeSelector(tmp_path / "installation")
    selected = RuntimeSelection(
        revision=1,
        operation_id="a" * 32,
        runtime_digest=legacy.digest,
        enabled=False,
        bundle_id=legacy.claims.release.manifest.bundle_id,
        sequence=1,
        highest_sequence=1,
        state_schema=legacy.claims.release.state_schema,
        permissions=legacy.claims.release.permissions,
        configuration_schema=True,
        platform=legacy_host.platform,
        python_version=legacy_host.python_version,
        skulk_version=legacy_host.skulk_version,
        skulk_build_sha256=legacy_host.skulk_build_sha256,
    )
    generation = selector.root / "generations" / legacy.digest
    private_directory(generation)
    write_private(generation / "staged.json", legacy_metadata)
    before = selected.model_dump_json().encode()
    write_private(selector.root / "runtime-selection.json", before)
    retained = selector.root / "protected-state-canary"
    write_private(retained, b"Existing opaque settings and grants.")

    @asynccontextmanager
    async def verified_candidate(
        _digest: str,
    ) -> AsyncIterator[tuple[VerifiedRuntime, QualifiedHost]]:
        # Only installation I/O is injected. Both interpretation records above
        # were admitted by their real signature/native readers.
        yield candidate, host

    monkeypatch.setattr(selector.installer, "locked_generation", verified_candidate)
    with pytest.raises(ValueError, match="profile change requires explicit migration"):
        await selector.preview(
            candidate.digest,
            expected_revision=1,
            operation_id="b" * 32,
            rollback=False,
            accept_permissions=False,
        )
    assert (selector.root / "runtime-selection.json").read_bytes() == before
    assert retained.read_bytes() == b"Existing opaque settings and grants."
