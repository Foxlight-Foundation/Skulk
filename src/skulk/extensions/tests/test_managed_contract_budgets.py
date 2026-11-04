"""Real owner sockets and signed large-record staging across declaration bounds."""

import asyncio
import contextlib
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.managed import ManagedConnection, ManagedNode, ManagedOwner
from skulk.extensions.managed_limits import (
    MAX_MANAGED_DISCOVERY_BYTES,
    MAX_RUNTIME_METADATA_BYTES,
)
from skulk.extensions.runtime_artifacts import verify_runtime
from skulk.extensions.runtime_download import (
    InstallRequest,
    ReleaseSource,
    RuntimeDownloads,
)
from skulk.extensions.runtime_files import read_private, write_private
from skulk.extensions.runtime_manager import installed_release
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_steward_tools import context

_JSON = TypeAdapter(dict[str, JsonValue])


def declarations(count: int = 16) -> tuple[CapabilityDescriptor, ...]:
    """Use rich individual schemas, not one unbounded blob or a mocked decoder."""
    return tuple(
        CapabilityDescriptor(
            id=f"fixture.read{index}",
            version="1.0.0",
            title=f"Read fixture {index}",
            description="Synthetic declaration evidence.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "description": "x" * 12000,
            },
            output_schema={"type": "object", "additionalProperties": False},
        )
        for index in range(count)
    )


def test_managed_node_accepts_sixteen_and_refuses_a_seventeenth() -> None:
    """The node budget applies independently of the control-reader byte ceiling."""
    node = ManagedNode(
        node_id="node",
        bundle_id="fixture",
        version="1.0.0",
        status="ready",
        configurable=True,
        descriptors=declarations(),
    )
    assert len(node.descriptors) == 16
    with pytest.raises(ValidationError):
        ManagedNode.model_validate_json(
            node.model_copy(
                update={
                    "descriptors": declarations(17),
                }
            ).model_dump_json()
        )


async def test_large_discovery_keeps_ordinary_settings_bound_and_withdraws_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual Core reader handles rich discovery and keeps unrelated reads bounded."""
    owner = ManagedOwner(
        ManagedConnection(plugin_id="managed.fixture", state_root=str(tmp_path))
    )
    host_context = context()
    owner.context = host_context
    node = ManagedNode(
        node_id="node",
        bundle_id="fixture",
        version="1.0.0",
        status="ready",
        configurable=True,
        descriptors=declarations(),
    )
    overflow = False

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            message = _JSON.validate_json(await reader.readline())
            reply: dict[str, JsonValue]
            if message["operation"] in ("describe", "describe-extended"):
                selected = node.model_dump(mode="json")
                if overflow:
                    selected["descriptors"] = [
                        {
                            **node.descriptors[0].model_dump(mode="json"),
                            "input_schema": {
                                "description": "x" * (MAX_MANAGED_DISCOVERY_BYTES + 1)
                            },
                        }
                    ]
                reply = {
                    "transport_node_id": str(host_context.node_id),
                    "protocol": 4,
                    "nodes": [selected],
                }
            else:
                # This is a structurally valid settings response. It must be
                # refused by the ordinary byte limit before model decoding.
                reply = {
                    "schema": {"type": "object", "description": "x" * 150000},
                    "settings": {
                        "revision": 1,
                        "enabled": True,
                        "schema_digest": "0" * 64,
                        "values": {},
                    },
                }
            writer.write(json.dumps({"result": reply}).encode() + b"\n")
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    # Keep the fixture socket below the platform's AF_UNIX pathname bound.
    with tempfile.TemporaryDirectory(prefix="skulk-contract-", dir="/tmp") as directory:
        socket = Path(directory) / "control.sock"
        server = await asyncio.start_unix_server(handle, path=socket)
        socket.chmod(0o600)

        def fixture_socket(_name: str) -> Path:
            return socket

        monkeypatch.setattr(owner, "_socket_path", fixture_socket)
        try:
            await owner.refresh()
            assert owner.available and len(owner.nodes[0].descriptors) == 16
            assert len(owner.dynamic_capabilities()) == 16
            with pytest.raises(ValueError):
                await owner.node_configuration("node")
            overflow = True
            with pytest.raises(RuntimeError, match="managed owner unavailable"):
                await owner.refresh()
            assert not owner.available
            # Cached identities survive a transient outage; none remains ready.
            assert len(owner.dynamic_capabilities()) == 16
            assert all(
                not owner.capability_ready(descriptor.qualified_id)
                for descriptor in node.descriptors
            )
        finally:
            await owner.on_stop()
            server.close()
            await server.wait_closed()


async def test_large_signed_metadata_download_stage_reopen_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real signatures, bounded HTTP and offline pip retain one exact large record."""
    source = tmp_path / "source"
    extra: dict[str, JsonValue] = {
        "descriptors": [
            _JSON.validate_json(value.model_dump_json()) for value in declarations()
        ],
        "skulk_requires": ">=2.0.3,<3",
    }
    metadata, trust, host = artifacts(source, manifest_extra=extra)
    host = replace(host, skulk_version="2.0.3")
    assert 131072 < len(metadata) < MAX_RUNTIME_METADATA_BYTES
    verified = verify_runtime(metadata, trust, host, now=int(time.time()))
    with pytest.raises(ValueError, match="metadata exceeds"):
        verify_runtime(
            b" " * (MAX_RUNTIME_METADATA_BYTES + 1), trust, host, now=int(time.time())
        )
    root = tmp_path / "installation"
    write_private(root / "publisher-trust.json", trust.model_dump_json().encode())
    write_private(
        root / "release-source.json",
        ReleaseSource(
            revision=1,
            base_url="https://fixture.example.test/",
            metadata_filename="release.json",
        )
        .model_dump_json()
        .encode(),
    )
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[1]
        paths.append(name)
        return httpx.Response(
            200,
            content=metadata if name == "release.json" else read_private(source / name),
        )

    downloads = RuntimeDownloads(root, transport=httpx.MockTransport(respond))
    try:
        review = await downloads.inspect()
        assert review.runtime_digest == verified.digest and paths == ["release.json"]
        request = InstallRequest(
            operation_id="b" * 32,
            runtime_digest=review.runtime_digest,
            expected_source_revision=1,
        )
        await downloads.submit(request)
        assert downloads.work is not None
        await downloads.work
        assert downloads.operation(request.operation_id).state == "staged"
        async with downloads.installer.locked_generation(verified.digest) as (reread, _):
            assert reread.metadata == metadata
        staged = root / "generations" / verified.digest / "staged.json"
        installed = installed_release(staged)
        assert installed is not None and installed.registration is not None
        assert len(installed.registration.capabilities) == 16
        assert (await downloads.submit(request)).state == "staged"
        assert not (root / "runtime-selection.json").exists()
    finally:
        await downloads.close()
