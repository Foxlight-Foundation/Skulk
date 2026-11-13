"""Signed discovery over real role sockets; only physical host measurement is injected.

The separate installed-app gate exercises the actual measured full Core and
normal publisher/installer/owner/author processes. These peers isolate discovery
failure timing and signed metadata admission without invoking author code.
"""

import asyncio
import hashlib
import json
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.managed_limits import MAX_RUNTIME_METADATA_BYTES
from skulk.extensions.native_calls import (
    NativeCalls,
    NativeListing,
    PolicyUpdate,
    Target,
)
from skulk.extensions.native_contracts import NativeDescriptor
from skulk.extensions.native_host_binding import provision_native_host
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    HostBinding,
    RequestFrame,
    admit_core,
)
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.runtime_artifacts import verify_runtime
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_selection import RuntimeSelection
from skulk.extensions.tests.test_call_runtime import call_fixture
from skulk.extensions.tests.test_native_runtime import OBJECT, sign
from skulk.shared.types.common import NodeId


@dataclass(frozen=True)
class Owner:
    """Synthetic signed installation and actual host-role observations."""

    listing: NativeListing
    events: list[str]


@asynccontextmanager
async def broker(root: Path) -> AsyncIterator[NativeCalls]:
    """Create a private independent host profile without account or fleet state."""
    settings = HostSettings(profile_id="a" * 32, transport_node_id="discovery-core")
    write_private(root / "host.json", settings.model_dump_json().encode())
    connection = ServiceConnection(manager_root=str(root), profile_id="a" * 32)
    calls = NativeCalls(connection, "discovery-core")
    try:
        yield calls
    finally:
        await calls.close()
        assert calls.budget.used == 0


@asynccontextmanager
async def owner(
    calls: NativeCalls,
    identifier: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stall: Literal["none", "handshake", "describe"] = "none",
    metadata_size: int = 0,
    malformed: Literal["none", "children", "node"] = "none",
) -> AsyncIterator[Owner]:
    """Use real signature/trust/selection admission and the actual mutual proof."""
    payload, key, trust, host = call_fixture(calls.root / "fixture" / identifier)
    metadata = sign(payload, key, b"skulk.runtime/4\0")
    metadata += b" " * max(0, metadata_size - len(metadata))
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    manifest = runtime.native_manifest
    assert manifest is not None
    root = calls.root / "installations" / identifier
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    selected = RuntimeSelection(
        revision=1,
        operation_id=secrets.token_hex(16),
        runtime_digest=runtime.digest,
        enabled=True,
        bundle_id=manifest.bundle_id,
        sequence=1,
        highest_sequence=1,
        state_schema=runtime.claims.release.state_schema,
        permissions=runtime.claims.release.permissions,
        configuration_schema=True,
        platform=host.platform,
        python_version=host.python_version,
        skulk_version=host.skulk_version,
        skulk_build_sha256=host.skulk_build_sha256,
    )
    write_private(root / "runtime-selection.json", selected.model_dump_json().encode())
    write_private(root / "publisher-trust.json", trust.model_dump_json().encode())
    write_private(root / "generations" / runtime.digest / "staged.json", metadata)
    provision_native_host(root, runtime.digest, secrets.token_hex(16))
    binding = HostBinding.model_validate_json(read_private(root / "core-host.json"))
    contracts = OBJECT.validate_python(
        manifest.value()["contracts"], strict=True
    )
    descriptors = TypeAdapter(tuple[NativeDescriptor, ...]).validate_json(
        json.dumps(contracts["descriptors"])
    )
    listing = NativeListing(
        plugin_id=identifier,
        runtime_digest=runtime.digest,
        incarnation=secrets.token_hex(16),
        target=Target(
            node_id=identifier,
            capability_id=descriptors[0].id,
            version=descriptors[0].version,
            descriptor_revision=descriptors[0].revision,
        ),
        descriptor=descriptors[0],
    )
    events: list[str] = []
    peers: list[Peer] = []
    tasks: set[asyncio.Task[None]] = set()
    budgets: list[NativePayloadBudget] = []

    async def respond(_peer: Peer, request: RequestFrame, _body: bytes) -> Reply:
        events.append(request.operation)
        if request.operation == "describe":
            if stall == "describe":
                await asyncio.Future[None]()
            if malformed == "children":
                return Reply("complete", {})
            if malformed == "node":
                return Reply("complete", {"children": [{}]})
            child: dict[str, JsonValue] = {
                "node_id": identifier,
                "incarnation": listing.incarnation,
                "manifest_digest": manifest.digest,
                "descriptors": [d.model_dump(mode="json") for d in descriptors],
            }
            return Reply("complete", {"children": [child]})
        return Reply("complete", {}, b"{}")

    async def accept(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        events.append("connected")
        budget = NativePayloadBudget()
        budgets.append(budget)
        try:
            if stall == "handshake":
                await asyncio.Future[None]()
            keys = await admit_core(reader, writer, binding, lambda: None)
            peer = Peer(
                AuthenticatedStream(reader, writer, keys, budget),
                respond,
                frozenset({"describe", "invoke"}),
                lambda: None,
            )
            peers.append(peer)
            await peer.pump
        except (OSError, ValueError, EOFError):
            pass
        finally:
            writer.close()
            tasks.discard(task)

    name = hashlib.sha256(str(root.absolute()).encode()).hexdigest()[:24]
    socket_root = Path("/tmp") / f"skulk-control-{os.getuid()}-{name}"
    private_directory(socket_root)
    socket = socket_root / "native-host.sock"
    server = await asyncio.start_unix_server(accept, socket)
    try:
        yield Owner(listing, events)
    finally:
        server.close()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        await asyncio.gather(*(p.close() for p in peers))
        await server.wait_closed()
        socket.unlink(missing_ok=True)
        socket_root.rmdir()
        assert all(b.used == 0 for b in budgets)


@pytest.mark.parametrize("stall", ["handshake", "describe"])
async def test_four_unresponsive_owners_cannot_exhaust_a_healthy_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stall: Literal["handshake", "describe"],
) -> None:
    """Discovery, enrollment and a healthy root survive four actual stalled peers."""
    async with broker(tmp_path) as calls, AsyncExitStack() as stack:
        unavailable = [
            await stack.enter_async_context(
                owner(calls, f"managed.aaa-{n}", monkeypatch, stall=stall)
            )
            for n in range(4)
        ]
        healthy = await stack.enter_async_context(
            owner(calls, "managed.zzz-healthy", monkeypatch)
        )
        started = asyncio.get_running_loop().time()
        async with asyncio.timeout(11):
            await calls.update_policy(
                PolicyUpdate(
                    expected_revision="0" * 64,
                    enabled=True,
                    grants=(healthy.listing.grant(),),
                )
            )
            admitted = AuthenticatedOwner(
                identifier="direct-owner", issuer_node_id=NodeId("discovery-core")
            )
            response = await calls.invoke(
                admitted,
                healthy.listing.plugin_id,
                healthy.listing.target.capability_id,
                healthy.listing.target.version,
                b"{}",
                10,
            )
            try:
                assert response.payload == b"{}"
            finally:
                response.release()
        assert asyncio.get_running_loop().time() - started < 10
        assert all("connected" in p.events for p in unavailable)
        if stall == "describe":
            assert all("describe" in p.events for p in unavailable)
        assert "invoke" in healthy.events


async def test_valid_large_signed_metadata_is_discovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted signed envelope above the old directory bound stays usable."""
    async with (
        broker(tmp_path) as calls,
        owner(calls, "managed.large", monkeypatch, metadata_size=300_000) as installed,
    ):
        assert 262144 < 300_000 <= MAX_RUNTIME_METADATA_BYTES
        assert await calls.directory() == (installed.listing,)


async def test_concurrent_scans_share_one_connection_per_owner_and_stable_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel discovery neither multiplies role channels nor permutes targets."""
    async with (
        broker(tmp_path) as calls,
        owner(calls, "managed.second", monkeypatch) as second,
        owner(calls, "managed.first", monkeypatch) as first,
    ):
        results = await asyncio.gather(*(calls.directory() for _ in range(8)))
        assert all(result == (first.listing, second.listing) for result in results)
        assert first.events.count("connected") == second.events.count("connected") == 1
        assert calls.budget.used == 0


@pytest.mark.parametrize("malformed", ["children", "node"])
async def test_malformed_owner_reply_cannot_withdraw_a_healthy_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: Literal["children", "node"],
) -> None:
    """Missing untrusted reply fields invalidate only the owner that sent them."""
    async with (
        broker(tmp_path) as calls,
        owner(calls, "managed.broken", monkeypatch, malformed=malformed) as broken,
        owner(calls, "managed.healthy", monkeypatch) as healthy,
    ):
        assert await calls.directory() == (healthy.listing,)
        assert "describe" in broken.events
