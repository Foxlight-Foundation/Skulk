"""Periodic disk scans preserve owner responsiveness and download command order."""

# pyright: reportPrivateUsage=false

import asyncio
import contextlib
import json
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.download import coordinator as coordinator_module
from skulk.download.tests.test_explicit_retry import _Harness, _harness
from skulk.extensions import CapabilityDescriptor, LoadedExtensions, descriptor_revision
from skulk.extensions.managed import ManagedConnection, ManagedNode, ManagedOwner
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    PurgeStagingCache,
    StartDownload,
)
from skulk.shared.types.common import SystemId
from skulk.shared.types.worker.downloads import DownloadCompleted


@asynccontextmanager
async def _blocked_scan(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
    phase: Literal["resolve", "companions"] = "resolve",
) -> AsyncIterator[threading.Event]:
    """Hold real periodic discovery in one bounded synchronous filesystem probe."""
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()

    def block() -> None:
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(timeout=5), "scan starved the task that releases it"
        finally:
            finished.set()

    async def cards() -> list[ModelCard]:
        return [harness.shard.model_card]

    def resolve(
        model_id: ModelId, revision: str | None = None, **kwargs: object
    ) -> Path:
        assert model_id == harness.shard.model_card.model_id
        assert revision == harness.shard.model_card.source_revision
        assert kwargs["expected_card"] == harness.shard.model_card
        if phase == "resolve":
            block()
        return directory

    def companions(card: ModelCard, *, required_only: bool) -> bool:
        assert card == harness.shard.model_card
        assert required_only == harness.coordinator.offline
        if phase == "companions":
            block()
        return True

    monkeypatch.setattr(coordinator_module, "get_model_cards", cards)
    monkeypatch.setattr(coordinator_module, "resolve_model_in_path", resolve)
    monkeypatch.setattr(coordinator_module, "model_companions_present_on_disk", companions)
    task = asyncio.create_task(harness.coordinator._emit_existing_download_progress())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not finished.is_set(), "filesystem work blocked the Core event loop"
        yield release
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("phase", ["resolve", "companions"])
async def test_scan_preserves_managed_refresh_and_pinned_public_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["resolve", "companions"],
) -> None:
    """A stalled disk probe cannot withdraw a healthy real-socket managed owner."""
    harness = _harness(tmp_path, monkeypatch)
    owner = ManagedOwner(
        ManagedConnection(plugin_id="managed.scan-fixture", state_root=str(tmp_path))
    )
    owner.context = harness.api._extension_context
    descriptor = CapabilityDescriptor(
        id="fixture.read",
        version="1.0.0",
        title="Read project",
        description="Read an unchanged project during periodic artifact discovery.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    node = ManagedNode(
        node_id="scan-fixture",
        bundle_id="scan-fixture",
        version="1.0.0",
        status="ready",
        configurable=False,
        descriptors=(descriptor,),
    )
    harness.api._extensions = LoadedExtensions([owner])
    decoder = TypeAdapter(dict[str, JsonValue])

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            message = decoder.validate_json(await reader.readline())
            if message["operation"] in ("describe", "describe-extended"):
                result = {
                    "transport_node_id": str(harness.coordinator.node_id),
                    "protocol": 4,
                    "nodes": [node.model_dump(mode="json")],
                }
            else:
                assert message["operation"] == "invoke"
                invocation = message["invoke"]
                assert isinstance(invocation, dict)
                assert invocation["descriptor_revision"] == descriptor_revision(descriptor)
                result = {"project": "preserved"}
            writer.write(json.dumps({"result": result}).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    # Unix socket paths need a short private directory independent of pytest's
    # deliberately long fixture paths and the user's actual managed runtime.
    with tempfile.TemporaryDirectory(prefix="skulk-scan-", dir="/tmp") as directory:
        socket = Path(directory) / "control.sock"
        server = await asyncio.start_unix_server(serve, path=socket)
        def fixture_socket(_name: str) -> Path:
            return socket

        monkeypatch.setattr(owner, "_socket_path", fixture_socket)
        try:
            await owner.refresh()
            async with _blocked_scan(harness, monkeypatch, tmp_path, phase) as release:
                # Use the production one-second refresh deadline, real socket
                # framing and public admission rather than a heartbeat mock.
                await owner.refresh()
                transport = httpx.ASGITransport(
                    app=harness.api.app, client=("127.0.0.1", 50000)
                )
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://localhost"
                ) as client:
                    catalog = await client.get("/v1/capabilities")
                    assert catalog.status_code == 200
                    assert catalog.json()["revisions"][descriptor.qualified_id] == (
                        descriptor_revision(descriptor)
                    )
                    response = await client.post(
                        "/v1/capabilities/call",
                        json={
                            "call_id": "scan-read",
                            "capability_id": descriptor.id,
                            "version": descriptor.version,
                            "descriptor_revision": descriptor_revision(descriptor),
                            "caller_node": str(harness.coordinator.node_id),
                            "target_node": str(harness.coordinator.node_id),
                            "timeout_seconds": 1.0,
                            "payload": {},
                        },
                    )
                    assert response.status_code == 200
                    assert response.json()["ok"] is True
                    assert response.json()["result"] == {"project": "preserved"}
                assert owner.capability_ready(descriptor.qualified_id)
                assert not release.is_set()
        finally:
            await owner.on_stop()
            server.close()
            await server.wait_closed()


@pytest.mark.parametrize("operation", ["start", "cancel", "delete", "purge"])
async def test_scan_publication_is_ordered_before_waiting_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: Literal["start", "cancel", "delete", "purge"],
) -> None:
    """A disk lookup must not publish stale completion over a newer mutation."""
    harness = _harness(tmp_path, monkeypatch)
    invoked = asyncio.Event()

    async def mutate(*_args: object, **_kwargs: object) -> None:
        # Publication precedes releasing admission. No late disk result can
        # arrive after the command has taken ownership of this model.
        assert isinstance(
            harness.coordinator.download_status.get(harness.shard.model_card.model_id),
            DownloadCompleted,
        )
        invoked.set()

    methods = {
        "start": "_start_download",
        "cancel": "_cancel_download",
        "delete": "_delete_download",
        "purge": "_purge_staging_cache",
    }
    monkeypatch.setattr(harness.coordinator, methods[operation], mutate)
    match operation:
        case "start":
            command = StartDownload(
                target_node_id=harness.coordinator.node_id, shard_metadata=harness.shard
            )
        case "cancel":
            command = CancelDownload(
                target_node_id=harness.coordinator.node_id,
                model_id=harness.shard.model_card.model_id,
            )
        case "delete":
            command = DeleteDownload(
                target_node_id=harness.coordinator.node_id,
                model_id=harness.shard.model_card.model_id,
            )
        case "purge":
            command = PurgeStagingCache(model_id=harness.shard.model_card.model_id)
    processor = asyncio.create_task(harness.coordinator._command_processor())
    try:
        async with _blocked_scan(harness, monkeypatch, tmp_path) as release:
            await harness.commands.send(
                ForwarderDownloadCommand(origin=SystemId(), command=command)
            )
            await asyncio.sleep(0)
            assert not invoked.is_set()
            release.set()
            await asyncio.wait_for(invoked.wait(), timeout=1)
    finally:
        processor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await processor
