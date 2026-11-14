"""Compiled output-schema enforcement over actual authenticated component sockets.

These peers isolate Core's independent frame admission. Normal installed caller,
role, scope and application authority are qualified separately.
"""

import asyncio
import json
from typing import cast

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    RequestFrame,
    ResponseFrame,
    SessionKeys,
)
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.native_inference_contracts import (
    InferenceDescriptor,
    InferenceProfiles,
    public_inference_kernel,
)
from skulk.extensions.native_inference_streams import NativeInferenceStream


@pytest.mark.parametrize(
    "schema", [False, {"type": "object", "required": ["result"]}, {"type": "null"}]
)
@pytest.mark.parametrize("omit", [True, False])
async def test_core_independently_enforces_absent_terminal_payload_schema(
    schema: dict[str, JsonValue] | bool, omit: bool
) -> None:
    """A native-valid null terminal requires an explicitly accepting result schema."""
    native = public_inference_kernel()
    descriptor = InferenceDescriptor.model_validate_json(
        native.portable_v2_contract_json(
            "Descriptor",
            json.dumps(
                {
                    "contract_profile": "skulk.portable/2",
                    "profile_digest": InferenceProfiles.qualified().portable,
                    "id": "qualification.schema",
                    "version": "1.0.0",
                    "title": "Result schema",
                    "description": "Component output admission",
                    "behavior": "inference",
                    "processing_domain": "external",
                    "io_mode": "server_streaming",
                    "input_schema": {"type": "object"},
                    "output_schema": schema,
                    "output_chunk_schema": {"type": "object"},
                }
            ),
        )
    )
    frame: dict[str, JsonValue] = {
        "call_id": "call",
        "direction": "provider_to_caller",
        "sequence": 1,
        "kind": "completed",
    }
    if not omit:
        frame["payload"] = None
    packets = [
        native.encode_transport_v2_packet(json.dumps(frame), b""),
        native.encode_transport_v2_packet('{"kind":"end","call_id":"call"}', b""),
    ]
    incoming: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.Queue()
    )
    server = await asyncio.start_server(
        lambda r, w: incoming.put_nowait((r, w)), "127.0.0.1", 0
    )
    address = TypeAdapter(tuple[str, int]).validate_python(
        cast(object, server.sockets[0].getsockname()), strict=True
    )
    reader, writer = await asyncio.open_connection(*address)
    owner_reader, owner_writer = await incoming.get()
    budget, owner_budget = NativePayloadBudget(), NativePayloadBudget()

    async def refuse(_peer: Peer, _request: RequestFrame, _body: bytes) -> Reply:
        return Reply("pre_dispatch", {})

    async def respond(_peer: Peer, request: RequestFrame, _body: bytes) -> Reply:
        if request.operation == "infer_stream":
            return Reply("complete", {}, b"{}")
        assert request.operation == "stream_next" and packets
        return Reply("complete", {}, packets.pop(0))

    client = Peer(
        AuthenticatedStream(
            reader, writer, SessionKeys(outbound=b"a" * 32, inbound=b"b" * 32), budget
        ),
        refuse,
        frozenset(),
        lambda: None,
    )
    owner = Peer(
        AuthenticatedStream(
            owner_reader,
            owner_writer,
            SessionKeys(outbound=b"b" * 32, inbound=b"a" * 32),
            owner_budget,
        ),
        respond,
        frozenset({"infer_stream", "stream_next"}),
        lambda: None,
    )
    control = asyncio.create_task(client.request("infer_stream", {}))
    port = NativeInferenceStream(
        client, "call", descriptor, control, lambda: None, lambda: None
    )
    try:
        if schema == {"type": "null"}:
            value = await port.next_output()
            assert value is not None and value.kind == "completed" and port.finished
            value.release()
            assert await port.next_output() is None
        else:
            with pytest.raises(ValueError, match="schema"):
                await port.next_output()
            assert port.closed and len(packets) == 1
    finally:
        port.close()
        await control
        control.result().release()
        await client.close()
        await owner.close()
        server.close()
        await server.wait_closed()
        assert budget.used == owner_budget.used == 0


@pytest.mark.parametrize("operation", ["stream_ready", "stream_input", "stream_next"])
async def test_stream_rpc_cancellation_respects_explicit_cleanup_grace(
    operation: str,
) -> None:
    """Valid slow cancellation acknowledgement cannot fence an unrelated caller peer."""
    native = public_inference_kernel()
    descriptor = InferenceDescriptor.model_validate_json(
        native.portable_v2_contract_json(
            "Descriptor",
            json.dumps(
                {
                    "contract_profile": "skulk.portable/2",
                    "profile_digest": InferenceProfiles.qualified().portable,
                    "id": "qualification.grace",
                    "version": "1.0.0",
                    "title": "Cleanup grace",
                    "description": "Component cancellation resource",
                    "behavior": "inference",
                    "processing_domain": "external",
                    "io_mode": "client_streaming",
                    "input_schema": {"type": "object"},
                    "input_chunk_schema": {"type": "object"},
                    "output_schema": None,
                    "output_chunk_schema": None,
                }
            ),
        )
    )
    incoming: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.Queue()
    )
    server = await asyncio.start_server(
        lambda r, w: incoming.put_nowait((r, w)), "127.0.0.1", 0
    )
    address = TypeAdapter(tuple[str, int]).validate_python(
        cast(object, server.sockets[0].getsockname()), strict=True
    )
    reader, writer = await asyncio.open_connection(*address)
    owner_reader, owner_writer = await incoming.get()
    budget, owner_budget = NativePayloadBudget(), NativePayloadBudget()
    entered, finish_control = asyncio.Event(), asyncio.Event()

    async def refuse(_peer: Peer, _request: RequestFrame, _body: bytes) -> Reply:
        return Reply("pre_dispatch", {})

    async def respond(_peer: Peer, request: RequestFrame, _body: bytes) -> Reply:
        if request.operation == "infer_stream":
            await finish_control.wait()
            return Reply("complete", {}, b"{}")
        if request.parameters.get("call_id") == "call":
            entered.set()
            try:
                await asyncio.Future[None]()
            finally:
                # The peer default is three seconds. A signed five-second
                # cleanup resource must permit this valid acknowledgement.
                await asyncio.sleep(3.1)
        return Reply("complete", {"ready": False})

    client = Peer(
        AuthenticatedStream(
            reader, writer, SessionKeys(outbound=b"a" * 32, inbound=b"b" * 32), budget
        ),
        refuse,
        frozenset(),
        lambda: None,
    )
    owner = Peer(
        AuthenticatedStream(
            owner_reader,
            owner_writer,
            SessionKeys(outbound=b"b" * 32, inbound=b"a" * 32),
            owner_budget,
        ),
        respond,
        frozenset({"infer_stream", "stream_ready", "stream_input", "stream_next"}),
        lambda: None,
    )
    control = asyncio.create_task(client.request("infer_stream", {}))
    port = NativeInferenceStream(
        client,
        "call",
        descriptor,
        control,
        lambda: None,
        lambda: None,
        cleanup_seconds=5,
    )
    if operation == "stream_ready":
        pending = asyncio.create_task(port.ready(10))
    elif operation == "stream_input":
        wire = native.encode_transport_v2_packet(
            '{"call_id":"call","direction":"caller_to_provider","sequence":0,"kind":"started"}',
            b"",
        )
        pending = asyncio.create_task(port.send_input(wire))
    else:
        pending = asyncio.create_task(port.next_output())
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            client.check()
            owner.check()
            sibling = await client.request("stream_ready", {"call_id": "other"})
            assert isinstance(sibling.frame, ResponseFrame)
            assert sibling.frame.outcome == "complete"
            sibling.release()
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        port.close()
        finish_control.set()
        await control
        control.result().release()
        await client.close()
        await owner.close()
        server.close()
        await server.wait_closed()
        assert budget.used == owner_budget.used == 0
