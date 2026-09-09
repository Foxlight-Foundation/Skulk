"""Scoped HTTP lifecycle operations backed by the real independent local manager."""

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.plugins import create_plugins_router
from skulk.api.tests.test_operator_gateway import paired_service
from skulk.extensions import LoadedExtensions
from skulk.extensions.managed_services import ManagedInventory, ManagedServices
from skulk.extensions.runtime_controller import LifecycleOperation, LifecycleRequest
from skulk.extensions.runtime_files import read_private, write_private
from skulk.extensions.runtime_manager import OperationRequest, manager_request
from skulk.extensions.tests.test_managed_services import connect, manager_fixture
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE, running
from skulk.extensions.tests.test_steward_tools import context
from skulk.operator.pairing import PluginGrantUpdate


async def test_http_scopes_lifecycle_status_and_api_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only explicit grants reach durable local operations, which outlive the API."""
    manager = manager_fixture(tmp_path / "manager", monkeypatch)
    await manager.start()
    path = tmp_path / "config/managed-service/connection.json"
    connect(path, manager.root)
    services = ManagedServices(path)
    extensions = LoadedExtensions([], managed_services=services)
    extensions.run_startup_hooks(replace(context(), skulk_version="1.5.2"))
    pairing, exchange = paired_service(tmp_path / "pairing")
    app = FastAPI()
    app.include_router(create_plugins_router(extensions, pairing))
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 52000))
    bearer = {"Authorization": f"Bearer {exchange.access_token}"}
    owner = {"X-Skulk-Dashboard": "pairing-v1", "Origin": "https://localhost"}
    prefix = "/v1/plugins/managed"
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://localhost"
        ) as client:
            assert (await client.get(prefix, headers=bearer)).status_code == 403
            assert (
                await client.post(
                    prefix + "/installations",
                    headers=bearer,
                    json={"plugin_id": "managed.fixture"},
                )
            ).status_code == 403
            assert not manager.controllers
            assert (
                await client.get(
                    prefix, headers={**owner, "Origin": "https://foreign.example"}
                )
            ).status_code == 403
            assert (await client.get(prefix, headers=owner)).status_code == 200
            pairing.set_plugin_grant(
                exchange.device_id,
                PluginGrantUpdate(expected_revision=0, scopes=("plugins:read",)),
            )
            inventory = await client.get(prefix, headers=bearer)
            assert (
                inventory.status_code == 200
                and inventory.headers["Cache-Control"] == "no-store"
            )
            assert (
                ManagedInventory.model_validate_json(inventory.content).installations
                == ()
            )
            assert (
                await client.post(
                    prefix + "/installations",
                    headers=bearer,
                    json={"plugin_id": "managed.fixture"},
                )
            ).status_code == 403
            pairing.set_plugin_grant(
                exchange.device_id,
                PluginGrantUpdate(
                    expected_revision=1, scopes=("plugins:read", "plugins:manage")
                ),
            )
            refused = await client.post(
                prefix + "/installations",
                headers=bearer,
                json={
                    "plugin_id": "managed.fixture",
                    "manager_root": "never-disclose",
                    "executable": "ignored",
                },
            )
            assert refused.status_code == 422 and not manager.controllers
            assert (
                await client.post(
                    prefix + "/installations",
                    headers=bearer,
                    json={"plugin_id": "managed.fixture"},
                )
            ).status_code == 200
            assert (
                await client.get(
                    prefix + "/installations/managed.fixture", headers=bearer
                )
            ).status_code == 200
            controller = manager.controllers["managed.fixture"]
            metadata, trust, _ = artifacts(
                tmp_path / "source", owner_source=OWNER_SOURCE
            )
            write_private(
                controller.root / "publisher-trust.json",
                trust.model_dump_json().encode(),
            )
            staged = await controller.selector.installer.stage(
                metadata, tmp_path / "source"
            )
            activate = LifecycleRequest(
                operation_id="2" * 32,
                action="activate",
                expected_revision=0,
                runtime_digest=staged.runtime_digest,
            )
            operations = prefix + "/installations/managed.fixture/operations"
            response = await client.post(
                operations,
                headers={**bearer, "Content-Type": "application/json"},
                content=activate.model_dump_json(),
            )
            assert response.status_code == 200
            assert (
                LifecycleOperation.model_validate_json(response.content).request
                == activate
            )
            assert controller.work is not None
            await controller.work
            assert controller.service is not None
            await running(controller.service)
            process = controller.service.process
            assert process is not None
            record = await client.get(
                operations + "/" + activate.operation_id, headers=bearer
            )
            assert record.status_code == 200
            assert (
                LifecycleOperation.model_validate_json(record.content).state
                == "complete"
            )
            replay = await client.post(
                operations,
                headers={**bearer, "Content-Type": "application/json"},
                content=activate.model_dump_json(),
            )
            assert replay.content == record.content and process.returncode is None
            stale = LifecycleRequest(
                operation_id="3" * 32, action="disable", expected_revision=0
            )
            denied = await client.post(
                operations,
                headers={**bearer, "Content-Type": "application/json"},
                content=stale.model_dump_json(),
            )
            assert denied.status_code == 409 and process.returncode is None
            assert str(tmp_path) not in denied.text
            assert (
                await client.post(
                    operations,
                    headers=bearer,
                    json={
                        "operation_id": "4" * 32,
                        "action": "approve",
                        "expected_revision": 1,
                    },
                )
            ).status_code == 422
            recovery = await client.post(
                operations + "/" + activate.operation_id + "/recover", headers=bearer
            )
            assert recovery.status_code == 200 and recovery.content == record.content
            disable = LifecycleRequest(
                operation_id="5" * 32, action="disable", expected_revision=1
            )
            accepted = await client.post(
                operations,
                headers={**bearer, "Content-Type": "application/json"},
                content=disable.model_dump_json(),
            )
            assert accepted.status_code == 200
            await extensions.run_shutdown_hooks()
            assert controller.work is not None
            await controller.work
            retained = await manager_request(
                manager.root,
                OperationRequest(
                    plugin_id="managed.fixture", operation_id=disable.operation_id
                ),
            )
            assert (
                '"state":"complete"'
                in read_private(
                    controller.records / (disable.operation_id + ".json")
                ).decode()
            )
            assert "result" in retained and process.returncode == 0
            pairing.revoke_device(exchange.access_token, exchange.device_id)
            assert (await client.get(prefix, headers=bearer)).status_code == 401
            assert (
                await client.post(
                    prefix + "/installations",
                    headers=bearer,
                    json={"plugin_id": "managed.extra"},
                )
            ).status_code == 401
            assert "managed.extra" not in manager.controllers
    finally:
        await extensions.run_shutdown_hooks()
        await manager.close()
