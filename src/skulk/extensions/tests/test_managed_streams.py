"""Managed stream cleanup must survive Fabric's level-triggered cancellation."""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anyio
import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.extensions import CapabilityCall, CapabilityDescriptor, CapabilityStreamFrame
from skulk.extensions.capabilities import descriptor_revision
from skulk.extensions.managed import ManagedConnection, ManagedNode, ManagedOwner
from skulk.extensions.managed_streams import (
    ManagedStreamEnd,
    read_managed_frame,
    write_managed_frame,
)
from skulk.extensions.tests.test_steward_tools import context
from skulk.extensions.types import ExtensionContext


@asynccontextmanager
async def _stream_fixture(
    handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]],
    timeout: float = 2.0,
    mode: Literal["server_streaming", "bidirectional"] = "server_streaming",
    on_request: Callable[[bytes], None] | None = None,
) -> AsyncIterator[tuple[ManagedOwner, ExtensionContext, CapabilityCall]]:
    with tempfile.TemporaryDirectory(prefix="managed-frames-", dir="/tmp") as name:
        root = Path(name)
        digest = hashlib.sha256(str(root.absolute()).encode()).hexdigest()[:24]
        directory = Path("/tmp") / f"skulk-control-{os.getuid()}-{digest}"
        directory.mkdir(mode=0o700)
        path = directory / "control.sock"
        clients: set[asyncio.Task[None]] = set()

        async def serve(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            task = asyncio.current_task()
            assert task is not None
            clients.add(task)
            try:
                raw = await reader.readline()
                if on_request is not None:
                    on_request(raw)
                await handler(reader, writer)
            finally:
                writer.close()
                await writer.wait_closed()
                clients.discard(task)

        server = await asyncio.start_unix_server(serve, path=path)
        path.chmod(0o600)
        descriptor = CapabilityDescriptor(
            id="fixture",
            version="1.0.0",
            title="Fixture",
            description="Boundary fixture",
            io_mode=mode,
            input_chunk_schema={} if mode == "bidirectional" else None,
            input_schema={},
            output_chunk_schema={},
        )
        owner = ManagedOwner(
            ManagedConnection(plugin_id="managed.fixture", state_root=str(root))
        )
        host_context = context()
        owner.context = host_context
        owner.protocol = 4
        owner.available = True
        owner.observed = time.monotonic()
        owner.nodes = (
            ManagedNode(
                node_id="installed",
                bundle_id="fixture",
                version="1.0.0",
                status="ready",
                configurable=False,
                descriptors=(descriptor,),
            ),
        )
        call = CapabilityCall(
            call_id="call",
            capability_id=descriptor.id,
            version=descriptor.version,
            descriptor_revision=descriptor_revision(descriptor),
            caller_node="caller",
            target_node="test-node",
            timeout_seconds=timeout,
            payload={},
        )
        try:
            yield owner, host_context, call
        finally:
            server.close()
            server.close_clients()
            for task in tuple(clients):
                task.cancel()
            await asyncio.gather(*tuple(clients), return_exceptions=True)
            await server.wait_closed()
            path.unlink(missing_ok=True)
            directory.rmdir()


def _ack(call_id: str = "call") -> bytes:
    header = json.dumps(
        {"protocol": 4, "frame": {"kind": "end", "call_id": call_id}, "media_size": 0}
    ).encode()
    return len(header).to_bytes(4, "big") + header


@pytest.mark.parametrize("waiting_mode", ["unary", "stream"])
@pytest.mark.parametrize("outcome", ["success", "cancel", "deadline", "withdrawn", "reassigned"])
async def test_owner_invocations_share_one_deadline_bound_slot(
    waiting_mode: Literal["unary", "stream"],
    outcome: Literal["success", "cancel", "deadline", "withdrawn", "reassigned"],
) -> None:
    """Queue behind active media without blocking health or admitting withdrawn work."""
    requests: asyncio.Queue[bytes] = asyncio.Queue()
    release = asyncio.Event()
    queued_contacted_owner = asyncio.Event()

    async def handle(
        _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request = TypeAdapter(dict[str, JsonValue]).validate_json(await requests.get())
        if request["operation"] == "describe-extended":
            writer.write(
                json.dumps(
                    {
                        "result": {
                            "transport_node_id": str(host_context.node_id),
                            "protocol": 4,
                            "nodes": [node.model_dump(mode="json") for node in owner.nodes],
                        }
                    }
                ).encode()
                + b"\n"
            )
            await writer.drain()
            return
        invocation = request["invoke"]
        assert isinstance(invocation, dict)
        call_id = invocation["call_id"]
        assert isinstance(call_id, str)
        if call_id == "queued":
            queued_contacted_owner.set()
        if request["operation"] == "invoke":
            writer.write(b'{"result":{"value":"cached"}}\n')
            await writer.drain()
            return
        assert request["operation"] == "stream"
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id=call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={},
            ),
        )
        await release.wait()
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id=call_id,
                direction="provider_to_caller",
                sequence=2,
                kind="completed",
            ),
        )
        writer.write(_ack(call_id))
        await writer.drain()

    async with _stream_fixture(handle, on_request=requests.put_nowait) as (
        owner,
        host_context,
        call,
    ):
        catalog = CapabilityDescriptor(
            id="catalog",
            version="1.0.0",
            title="Cached catalog",
            description="A unary read sharing the installed node's invocation slot.",
            input_schema={},
            output_schema={},
        )
        owner.nodes = (
            owner.nodes[0].model_copy(
                update={"descriptors": (*owner.nodes[0].descriptors, catalog)}
            ),
        )
        queued_call = call.model_copy(
            update={
                "call_id": "queued",
                "capability_id": catalog.id if waiting_mode == "unary" else call.capability_id,
                "descriptor_revision": descriptor_revision(catalog)
                if waiting_mode == "unary"
                else call.descriptor_revision,
                "timeout_seconds": 0.1 if outcome == "deadline" else 2.0,
            }
        )

        async def waiting_call() -> dict[str, object] | list[CapabilityStreamFrame]:
            if waiting_mode == "unary":
                return await owner.handle_call(host_context, queued_call)
            return [
                frame async for frame in owner.handle_stream(host_context, queued_call)
            ]

        stream = owner.handle_stream(host_context, call)
        queued: asyncio.Task[dict[str, object] | list[CapabilityStreamFrame]] | None = None
        try:
            assert (await anext(stream)).kind == "chunk"
            queued = asyncio.create_task(waiting_call())
            await asyncio.sleep(0.02)
            assert not queued_contacted_owner.is_set()
            async with asyncio.timeout(0.5):
                await owner.refresh()
            assert owner.available
            if outcome == "cancel":
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
            elif outcome == "deadline":
                with pytest.raises(TimeoutError):
                    await queued
            elif outcome == "withdrawn":
                owner.disabled = True
            elif outcome == "reassigned":
                owner.nodes = (owner.nodes[0].model_copy(update={"node_id": "replacement"}),)
            assert not queued_contacted_owner.is_set()
            release.set()
            assert [frame.kind async for frame in stream] == ["completed"]
            if outcome == "success":
                result = await queued
                if waiting_mode == "unary":
                    assert result == {"value": "cached"}
                else:
                    assert isinstance(result, list) and result[-1].kind == "completed"
                assert queued_contacted_owner.is_set()
            elif outcome in {"withdrawn", "reassigned"}:
                with pytest.raises(RuntimeError, match="unavailable"):
                    await queued
                assert not queued_contacted_owner.is_set()
        finally:
            release.set()
            if queued is not None:
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
            await stream.aclose()
        assert owner.invocation_slots == {}


async def test_sibling_nodes_have_independent_invocation_slots() -> None:
    """A streaming child must not block a sibling child's unary invocation."""
    requests: asyncio.Queue[bytes] = asyncio.Queue()
    release = asyncio.Event()

    async def handle(
        _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request = TypeAdapter(dict[str, JsonValue]).validate_json(await requests.get())
        if request["operation"] == "invoke":
            assert request["node_id"] == "sibling"
            writer.write(b'{"result":{"independent":true}}\n')
            await writer.drain()
            return
        assert request["node_id"] == "installed"
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id="call", direction="provider_to_caller",
                sequence=1, kind="chunk", payload={},
            ),
        )
        await release.wait()
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id="call", direction="provider_to_caller",
                sequence=2, kind="completed",
            ),
        )
        writer.write(_ack())
        await writer.drain()

    async with _stream_fixture(handle, on_request=requests.put_nowait) as (
        owner, host_context, call,
    ):
        descriptor = CapabilityDescriptor(
            id="sibling", version="1.0.0", title="Sibling", description="Independent child",
            input_schema={}, output_schema={},
        )
        owner.nodes = (*owner.nodes, owner.nodes[0].model_copy(update={
            "node_id": "sibling", "descriptors": (descriptor,),
        }))
        stream = owner.handle_stream(host_context, call)
        try:
            assert (await anext(stream)).kind == "chunk"
            sibling = call.model_copy(update={
                "capability_id": descriptor.id,
                "descriptor_revision": descriptor_revision(descriptor),
                "timeout_seconds": 0.2,
            })
            assert await owner.handle_call(host_context, sibling) == {"independent": True}
            release.set()
            assert [frame.kind async for frame in stream] == ["completed"]
        finally:
            release.set()
            await stream.aclose()
        assert owner.invocation_slots == {}


@pytest.mark.parametrize("invalid_input", [False, True])
async def test_provider_completion_stops_open_input_before_socket_close(
    invalid_input: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sender racing EOF must not replace a valid provider terminal with failure."""
    terminal_read = asyncio.Event()
    input_stopped = asyncio.Event()

    async def read(
        reader: asyncio.StreamReader,
    ) -> CapabilityStreamFrame | ManagedStreamEnd:
        frame = await read_managed_frame(reader)
        if isinstance(frame, CapabilityStreamFrame) and frame.is_terminal:
            terminal_read.set()
        return frame

    monkeypatch.setattr("skulk.extensions.managed.read_managed_frame", read)

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        assert isinstance(await read_managed_frame(reader), CapabilityStreamFrame)
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id="call",
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            ),
        )
        writer.write(_ack())
        await writer.drain()

    async def inputs() -> AsyncIterator[CapabilityStreamFrame]:
        try:
            yield CapabilityStreamFrame(
                call_id="call",
                direction="caller_to_provider",
                sequence=0,
                kind="started",
            )
            await terminal_read.wait()
            if invalid_input:
                raise ValueError("invalid input frame")
            raise ConnectionResetError("input write raced provider EOF")
        finally:
            input_stopped.set()

    async with _stream_fixture(handle, mode="bidirectional") as (
        owner,
        host_context,
        call,
    ):
        if invalid_input:
            with pytest.raises(ValueError, match="invalid input"):
                await anext(owner.handle_input_stream(host_context, call, inputs()))
        else:
            output = [
                frame
                async for frame in owner.handle_input_stream(
                    host_context, call, inputs()
                )
            ]
            assert len(output) == 1 and output[0].kind == "completed"
        assert input_stopped.is_set()


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "sequence",
        "direction",
        "synthetic",
        "early_end",
        "duplicate_terminal",
        "wrong_end",
        "trailing",
    ],
)
async def test_owner_rejects_invalid_output_and_cleanup(fault: str) -> None:
    async def handle(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if fault == "early_end":
            writer.write(_ack())
        elif fault in ("identity", "sequence", "direction", "synthetic"):
            await write_managed_frame(
                writer,
                CapabilityStreamFrame(
                    call_id="different" if fault == "identity" else "call",
                    direction="caller_to_provider"
                    if fault == "direction"
                    else "provider_to_caller",
                    sequence=2 if fault == "sequence" else 1,
                    kind="chunk",
                    payload={},
                    synthetic=fault == "synthetic",
                ),
            )
        else:
            terminal = CapabilityStreamFrame(
                call_id="call",
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            )
            await write_managed_frame(writer, terminal)
            if fault == "duplicate_terminal":
                await write_managed_frame(writer, terminal)
            else:
                writer.write(_ack("different" if fault == "wrong_end" else "call"))
                if fault == "trailing":
                    writer.write(b"unexpected")
        await writer.drain()

    async with _stream_fixture(handle) as (owner, host_context, call):
        with pytest.raises(ValueError):
            await anext(owner.handle_stream(host_context, call))


async def test_terminal_is_withheld_until_cleanup_and_eof() -> None:
    terminal_sent = asyncio.Event()
    release = asyncio.Event()

    async def handle(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id="call",
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            ),
        )
        terminal_sent.set()
        await release.wait()
        writer.write(_ack())
        await writer.drain()

    async with _stream_fixture(handle) as (owner, host_context, call):
        stream = owner.handle_stream(host_context, call)
        result = asyncio.create_task(anext(stream))
        try:
            await terminal_sent.wait()
            await asyncio.sleep(0.02)
            assert not result.done()
            release.set()
            assert (await result).kind == "completed"
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        finally:
            result.cancel()
            await asyncio.gather(result, return_exceptions=True)
            await stream.aclose()


async def test_timeout_before_cleanup_closes_owner_socket() -> None:
    disconnected = asyncio.Event()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await write_managed_frame(
            writer,
            CapabilityStreamFrame(
                call_id="call",
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            ),
        )
        assert await reader.read() == b""
        disconnected.set()

    async with _stream_fixture(handle, timeout=0.05) as (owner, host_context, call):
        with pytest.raises(TimeoutError):
            await anext(owner.handle_stream(host_context, call))
        async with asyncio.timeout(1):
            await disconnected.wait()


@pytest.mark.parametrize(
    "mode,open_input",
    [
        ("server_streaming", False),
        ("bidirectional", False),
        ("bidirectional", True),
    ],
)
async def test_fabric_cancellation_closes_the_managed_socket_explicitly(
    mode: Literal["server_streaming", "bidirectional"],
    open_input: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep writer references alive so garbage collection cannot hide a leaked call."""
    with tempfile.TemporaryDirectory(prefix="managed-cancel-", dir="/tmp") as name:
        root = Path(name)
        digest = hashlib.sha256(str(root.absolute()).encode()).hexdigest()[:24]
        directory = Path("/tmp") / f"skulk-control-{os.getuid()}-{digest}"
        directory.mkdir(mode=0o700)
        path = directory / "control.sock"
        disconnected = asyncio.Event()
        writers: list[asyncio.StreamWriter] = []
        original = asyncio.open_unix_connection

        async def connect(
            path: str | Path, *, limit: int
        ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            reader, writer = await original(path, limit=limit)
            writers.append(writer)
            return reader, writer

        monkeypatch.setattr(asyncio, "open_unix_connection", connect)
        descriptor = CapabilityDescriptor(
            id="fixture",
            version="1.0.0",
            title="Fixture",
            description="Cancellation fixture",
            input_schema={},
            io_mode=mode,
            input_chunk_schema={} if mode == "bidirectional" else None,
            output_chunk_schema={},
        )
        call = CapabilityCall(
            call_id="call",
            capability_id=descriptor.id,
            version=descriptor.version,
            descriptor_revision=descriptor_revision(descriptor),
            caller_node="caller",
            target_node="test-node",
            timeout_seconds=5.0,
            payload={},
        )

        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await reader.readline()
                await write_managed_frame(
                    writer,
                    CapabilityStreamFrame(
                        call_id=call.call_id,
                        direction="provider_to_caller",
                        sequence=1,
                        kind="chunk",
                        payload={},
                    ),
                )
                # Drain possible caller lifecycle packets until explicit EOF.
                while await reader.read(65536):
                    pass
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=path)
        path.chmod(0o600)
        owner = ManagedOwner(
            ManagedConnection(plugin_id="managed.fixture", state_root=str(root))
        )
        host_context = context()
        owner.context = host_context
        owner.protocol = 4
        owner.available = True
        owner.observed = time.monotonic()
        owner.nodes = (
            ManagedNode(
                node_id="installed",
                bundle_id="fixture",
                version="1.0.0",
                status="ready",
                configurable=False,
                descriptors=(descriptor,),
            ),
        )

        async def inputs() -> AsyncIterator[CapabilityStreamFrame]:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="caller_to_provider",
                sequence=0,
                kind="started",
            )
            if open_input:
                await asyncio.Event().wait()
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="caller_to_provider",
                sequence=1,
                kind="completed",
            )

        try:
            stream = (
                owner.handle_stream(host_context, call)
                if mode == "server_streaming"
                else owner.handle_input_stream(host_context, call, inputs())
            )
            with anyio.CancelScope() as scope:
                assert (await anext(stream)).kind == "chunk"
                scope.cancel()
                await anext(stream)
            assert writers and all(writer.is_closing() for writer in writers)
            with anyio.fail_after(1):
                await disconnected.wait()
        finally:
            for writer in writers:
                writer.close()
            server.close()
            server.close_clients()
            await server.wait_closed()
            path.unlink(missing_ok=True)
            directory.rmdir()
