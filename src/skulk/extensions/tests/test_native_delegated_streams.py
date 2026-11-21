"""Core forwarding over protected peers and real current caller authority.

Signed discovery observations and role keys are explicit synthetic fixtures.
These tests are not manager enrollment or installed application qualification.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.api.tests.test_inference_access import application
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_calls import NativeCalls, NativeHostEndpoint
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    HostBinding,
    RequestFrame,
    ResponseFrame,
    SessionKeys,
)
from skulk.extensions.native_host_rpc import Peer, Received, Reply
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    PrincipalAccess,
)
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_contracts import (
    InferenceDescriptor,
    InferenceProfiles,
)
from skulk.extensions.native_inference_hosts import InferenceHosts, InferenceListing
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import write_private

_OBJECT = TypeAdapter(dict[str, JsonValue])


def listing(name: str, mode: str, ceiling: float | None = None) -> InferenceListing:
    """Compile independent descriptors; a listing supplies observations, not grants."""
    descriptor = InferenceDescriptor.model_validate_json(
        json.dumps(
            {
                "contract_profile": "skulk.portable/2",
                "profile_digest": InferenceProfiles.qualified().portable,
                "id": "delegation." + name,
                "version": "1.0.0",
                "title": name,
                "description": "Independent delegated stream witness",
                "behavior": "inference",
                "processing_domain": "external",
                "io_mode": mode,
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object", "required": ["output"]},
                "input_chunk_schema": {"type": "object", "required": ["input"]}
                if mode in ("client_streaming", "bidirectional")
                else None,
                "output_chunk_schema": {"type": "object", "required": ["output"]}
                if mode in ("server_streaming", "bidirectional")
                else None,
            }
        )
    )
    return InferenceListing.model_validate_json(
        json.dumps(
            {
                "plugin_id": "managed." + name,
                "runtime_digest": ("b" if name == "source" else "c") * 64,
                "incarnation": "d" * 32,
                "target": {
                    "node_id": name,
                    "capability_id": descriptor.id,
                    "version": descriptor.version,
                    "descriptor_revision": descriptor.revision,
                    "behavior": descriptor.behavior,
                    "processing_domain": descriptor.processing_domain,
                },
                "descriptor": descriptor.model_dump(mode="json"),
                "inference_seconds": ceiling,
            }
        )
    )


@dataclass
class ProviderCall:
    """Own synthetic receiver behavior after actual Core dispatch admission."""

    context: dict[str, JsonValue]
    body: bytes
    packets: list[bytes]
    inputs: list[bytes] = field(default_factory=list)
    input_done: asyncio.Event = field(default_factory=asyncio.Event)
    end: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Witness:
    """Observe current caller admission, exact media and independently held cleanup."""

    broker: NativeInferenceCalls
    caller: AuthenticatedInferenceCaller
    client_id: str
    source: InferenceListing
    receiver: InferenceListing
    observed: list[ProviderCall] = field(default_factory=list)
    target_started: asyncio.Event = field(default_factory=asyncio.Event)
    target_cleaned: asyncio.Event = field(default_factory=asyncio.Event)
    produce: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup: asyncio.Event = field(default_factory=asyncio.Event)
    retiring: asyncio.Event = field(default_factory=asyncio.Event)
    retire: asyncio.Event = field(default_factory=asyncio.Event)
    ready: list[dict[str, JsonValue]] = field(default_factory=list)
    output: list[bytes] = field(default_factory=list)
    tamper_body: bool = False
    allowed: bool = True

    async def invoke(self) -> Received:
        """Start a real source root with its enrolled client credential."""
        return await self.broker.invoke(
            self.caller,
            self.source.plugin_id,
            self.source.target.capability_id,
            "1.0.0",
            b'{"input":"root"}',
        )


@asynccontextmanager
async def connected(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    ceiling: float | None = None,
) -> AsyncIterator[Witness]:
    """Use actual HMAC framing, policy files, compiled attenuation and task custody."""
    application(root)
    broker = NativeInferenceCalls(
        ServiceConnection(manager_root=str(root), profile_id="a" * 32), "access-core"
    )
    client, token = broker.callers.create(
        broker.callers.inventory().revision, "Synthetic caller", None
    )
    caller = broker.callers.authenticate(token)
    source, receiver = listing("source", "unary"), listing("receiver", mode, ceiling)
    witness = Witness(broker, caller, client.identifier, source, receiver)
    witness.produce.set()
    witness.cleanup.set()
    witness.retire.set()
    write_private(
        root / "native-inference-policy.json",
        InferenceAccessPolicy(
            revision="e" * 64,
            enabled=True,
            callers=(
                PrincipalAccess(
                    principal=caller.principal,
                    grants=(source.grant(), receiver.grant()),
                ),
            ),
        )
        .model_dump_json()
        .encode(),
    )

    async def directory(
        _hosts: InferenceHosts, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        return (source, receiver) if streaming else (source,)

    def owner(_hosts: NativeCalls, plugin: str, runtime: str, peer: Peer) -> None:
        assert plugin in ("managed.source", "managed.receiver")
        assert runtime == ("b" if plugin == "managed.source" else "c") * 64
        peer.check()

    monkeypatch.setattr(InferenceHosts, "directory", directory)
    monkeypatch.setattr(NativeCalls, "check_owner", owner)
    incoming: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.Queue()
    )
    server = await asyncio.start_server(
        lambda r, w: incoming.put_nowait((r, w)), "127.0.0.1", 0
    )
    peers: list[Peer] = []
    budgets: list[NativePayloadBudget] = []
    calls: dict[str, ProviderCall] = {}
    kernel = broker.kernel

    async def core_handler(peer: Peer, request: RequestFrame, body: bytes) -> Reply:
        handler = broker.hosts.inference_handler
        assert handler is not None
        return await handler(peer, request, body)

    async def receiver_handler(
        _peer: Peer, request: RequestFrame, body: bytes
    ) -> Reply:
        identifier = TypeAdapter(str).validate_python(
            request.parameters["call_id"], strict=True
        )
        if request.operation == "infer_stream":
            context = _OBJECT.validate_python(request.parameters["context"])
            packets: list[bytes] = []
            sequence = 1
            if mode != "client_streaming":
                packets.append(
                    kernel.encode_transport_v2_packet(
                        json.dumps(
                            {
                                "call_id": identifier,
                                "direction": "provider_to_caller",
                                "sequence": sequence,
                                "kind": "chunk",
                                "payload": {"output": "chunk"},
                                "media": {"kind": "inline", "media_type": "audio/pcm"},
                            }
                        ),
                        b"\xfe\x00",
                    )
                )
                sequence += 1
            packets.append(
                kernel.encode_transport_v2_packet(
                    json.dumps(
                        {
                            "call_id": identifier,
                            "direction": "provider_to_caller",
                            "sequence": sequence,
                            "kind": "completed",
                            "payload": {"output": "complete"},
                        }
                    ),
                    b"",
                )
            )
            packets.append(
                kernel.encode_transport_v2_packet(
                    json.dumps({"call_id": identifier, "kind": "end"}), b""
                )
            )
            call = ProviderCall(context, body, packets)
            calls[identifier] = call
            witness.observed.append(call)
            if mode == "server_streaming":
                call.input_done.set()
            witness.target_started.set()
            try:
                await call.end.wait()
                await witness.cleanup.wait()
                return Reply("complete", {}, b"{}")
            finally:
                witness.retiring.set()
                await witness.retire.wait()
                witness.target_cleaned.set()
        call = calls.get(identifier)
        if request.operation == "stream_ready":
            return Reply("complete", {"ready": call is not None})
        assert call is not None
        if request.operation == "stream_input":
            projection, _ = kernel.decode_transport_v2_packet(body)
            call.inputs.append(body)
            if _OBJECT.validate_json(projection)["kind"] == "completed":
                call.input_done.set()
            return Reply("complete", {})
        assert request.operation == "stream_next"
        await call.input_done.wait()
        await witness.produce.wait()
        value = call.packets.pop(0)
        if not call.packets:
            call.end.set()
        return Reply("complete", {}, value)

    async def source_handler(peer: Peer, frame: RequestFrame, _body: bytes) -> Reply:
        assert frame.operation == "infer"
        parent_id = TypeAdapter(str).validate_python(
            frame.parameters["call_id"], strict=True
        )
        context = _OBJECT.validate_python(frame.parameters["context"])
        description = await peer.request(
            "describe_inference",
            {
                "parent_call_id": parent_id,
                "destination": receiver.target.model_dump(mode="json"),
            },
        )
        try:
            assert isinstance(description.frame, ResponseFrame)
            if description.frame.outcome != "complete":
                assert not witness.allowed
                return Reply("complete", {}, b'{"output":"denied"}')
            assert description.frame.parameters == {
                "descriptor": receiver.descriptor.model_dump(mode="json")
            }
        finally:
            description.release()
        child_body = b'{ "input": "exact child", "number": 1.0 }'
        request: dict[str, JsonValue] = {
            "protocol": 3,
            "contract_profile": "skulk.composition/3",
            "profile_digest": kernel.composition_v3_profile_digest(),
            "kind": "delegate",
            "request_id": f"{1:064x}",
            "parent_scope_id": context["scope_id"],
            "parent_request_sha256": context["request_sha256"],
            "request_sha256": hashlib.sha256(child_body).hexdigest(),
            "destination": receiver.target.model_dump(mode="json"),
            "deadline_unix_ms": context["deadline_unix_ms"],
        }
        policy = {
            key: context[key]
            for key in [
                "protocol",
                "contract_profile",
                "profile_digest",
                "caller_id",
                "caller_kind",
                "issuer_node_id",
                "policy_revision",
                "allowed_targets",
                "allowed_processing_domains",
                "deadline_unix_ms",
            ]
        }
        policy["kind"] = "policy"
        admitted = kernel.CompositionV3Contract.from_json(
            "Context", json.dumps(context)
        ).admit(
            kernel.CompositionV3Contract.from_json(
                "DelegateRequest", json.dumps(request)
            ),
            kernel.CompositionV3Contract.from_json("Policy", json.dumps(policy)),
            source.target.model_dump_json(),
            child_body,
            time.time_ns() // 1_000_000,
            "2" * 64,
        )
        primary = asyncio.create_task(
            peer.request(
                "delegate_inference_stream",
                {
                    "parent_call_id": parent_id,
                    "request": request,
                    "context": _OBJECT.validate_json(admitted.canonical()),
                    "remaining_seconds": frame.parameters["remaining_seconds"],
                },
                b'{"changed":true}' if witness.tamper_body else child_body,
            )
        )
        try:
            async with asyncio.timeout(5):
                while not primary.done():
                    ready = await peer.request("stream_ready", {"call_id": "2" * 64})
                    try:
                        assert isinstance(ready.frame, ResponseFrame)
                        if ready.frame.outcome == "complete":
                            witness.ready.append(dict(ready.frame.parameters))
                            break
                        assert ready.frame.outcome == "pre_dispatch"
                    finally:
                        ready.release()
                    await asyncio.sleep(0.005)
                else:
                    response = primary.result()
                    assert (
                        isinstance(response.frame, ResponseFrame)
                        and response.frame.outcome != "complete"
                    )
                    return Reply("unknown", {})
                if mode != "server_streaming":
                    for sequence, kind in enumerate(("started", "chunk", "completed")):
                        value: dict[str, JsonValue] = {
                            "call_id": "2" * 64,
                            "direction": "caller_to_provider",
                            "sequence": sequence,
                            "kind": kind,
                        }
                        media = b""
                        if kind == "chunk":
                            value.update(
                                payload={"input": "chunk"},
                                media={"kind": "inline", "media_type": "audio/pcm"},
                            )
                            media = b"\x00\xff"
                        result = await peer.request(
                            "stream_input",
                            {"call_id": "2" * 64},
                            kernel.encode_transport_v2_packet(json.dumps(value), media),
                        )
                        try:
                            assert (
                                isinstance(result.frame, ResponseFrame)
                                and result.frame.outcome == "complete"
                            )
                        finally:
                            result.release()
                while True:
                    result = await peer.request("stream_next", {"call_id": "2" * 64})
                    try:
                        assert (
                            isinstance(result.frame, ResponseFrame)
                            and result.frame.outcome == "complete"
                        )
                        projection, _ = kernel.decode_transport_v2_packet(
                            result.payload
                        )
                        kind = _OBJECT.validate_json(projection)["kind"]
                        assert result.frame.parameters == {"kind": kind}
                        witness.output.append(result.payload)
                    finally:
                        result.release()
                    if kind != "chunk":
                        break
                response = await asyncio.shield(primary)
                assert (
                    isinstance(response.frame, ResponseFrame)
                    and response.frame.outcome == "complete"
                )
                assert response.payload == b"{}"
                return Reply("complete", {}, b'{"output":"root complete"}')
        finally:
            if not primary.done():
                primary.cancel()
            await asyncio.gather(primary, return_exceptions=True)
            if not primary.cancelled() and primary.exception() is None:
                primary.result().release()

    try:
        for item, handler in [(source, source_handler), (receiver, receiver_handler)]:
            port = TypeAdapter(int).validate_python(
                server.sockets[0].getsockname()[1], strict=True
            )
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            other_reader, other_writer = await incoming.get()
            core_budget, owner_budget = NativePayloadBudget(), NativePayloadBudget()
            budgets.extend((core_budget, owner_budget))
            core = Peer(
                AuthenticatedStream(
                    reader,
                    writer,
                    SessionKeys(outbound=b"a" * 32, inbound=b"b" * 32),
                    core_budget,
                ),
                core_handler,
                frozenset(
                    {
                        "describe_inference",
                        "delegate_inference",
                        "delegate_inference_stream",
                        "stream_ready",
                        "stream_input",
                        "stream_next",
                    }
                ),
                lambda: None,
            )
            owner_peer = Peer(
                AuthenticatedStream(
                    other_reader,
                    other_writer,
                    SessionKeys(outbound=b"b" * 32, inbound=b"a" * 32),
                    owner_budget,
                ),
                handler,
                frozenset(
                    {
                        "infer",
                        "infer_stream",
                        "stream_ready",
                        "stream_input",
                        "stream_next",
                    }
                ),
                lambda: None,
            )
            peers.extend((core, owner_peer))
            broker.hosts.endpoints[item.plugin_id] = NativeHostEndpoint(
                HostBinding(
                    protocol=1,
                    profile_id="a" * 32,
                    plugin_id=item.plugin_id,
                    runtime_digest=item.runtime_digest,
                    transport_node_id="access-core",
                    service_instance="e" * 32,
                    core_public_key="f" * 64,
                    server_secret="0" * 64,
                ),
                core,
            )
        yield witness
    finally:
        await broker.close()
        for peer in peers:
            await peer.close()
        server.close()
        await server.wait_closed()
        assert all(budget.used == 0 for budget in budgets)


@pytest.mark.parametrize(
    "mode", ["server_streaming", "client_streaming", "bidirectional"]
)
async def test_original_caller_exact_bytes_and_stream_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Core attenuates one actual root and forwards binary frames without rewriting."""
    async with connected(tmp_path, monkeypatch, mode) as witness:
        result = await asyncio.wait_for(
            witness.broker.invoke(
                witness.caller,
                witness.source.plugin_id,
                witness.source.target.capability_id,
                "1.0.0",
                b'{"input":"root"}',
            ),
            5,
        )
        try:
            assert result.payload == b'{"output":"root complete"}'
        finally:
            result.release()
        assert len(witness.observed) == 1
        observed = witness.observed[0]
        assert observed.body == b'{ "input": "exact child", "number": 1.0 }'
        assert observed.context["caller_id"] == witness.caller.principal.identifier
        assert observed.context["caller_kind"] == "client"
        assert observed.context["issuer_node_id"] == "core-profile:" + "a" * 32
        assert observed.context["scope_id"] == "2" * 64
        assert observed.context["parent_scope_id"] == observed.context["root_scope_id"]
        assert (
            witness.ready[0]["remaining_seconds"] is None
            and witness.ready[0]["deadline_unix_ms"] is None
        )
        if mode != "server_streaming":
            projection, media = witness.broker.kernel.decode_transport_v2_packet(
                observed.inputs[1]
            )
            assert (
                media == b"\x00\xff"
                and _OBJECT.validate_json(projection)["sequence"] == 1
            )
        if mode != "client_streaming":
            _projection, media = witness.broker.kernel.decode_transport_v2_packet(
                witness.output[0]
            )
            assert media == b"\xfe\x00"
        assert not witness.broker.active and not witness.broker.delegated


async def test_denied_destination_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only resolution cannot expose or dispatch an ungranted destination."""
    async with connected(tmp_path, monkeypatch, "bidirectional") as witness:
        witness.allowed = False
        write_private(
            tmp_path / "native-inference-policy.json",
            InferenceAccessPolicy(
                revision="f" * 64,
                enabled=True,
                callers=(
                    PrincipalAccess(
                        principal=witness.caller.principal,
                        grants=(witness.source.grant(),),
                    ),
                ),
            )
            .model_dump_json()
            .encode(),
        )
        result = await witness.invoke()
        try:
            assert result.payload == b'{"output":"denied"}'
        finally:
            result.release()
        assert not witness.observed and not witness.broker.delegated


async def test_changed_child_body_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted context cannot authorize different payload bytes on the wire."""
    async with connected(tmp_path, monkeypatch, "bidirectional") as witness:
        witness.tamper_body = True
        with pytest.raises(ValueError, match="outcome unavailable"):
            await witness.invoke()
        assert not witness.observed and not witness.broker.delegated
        assert not witness.broker.active


@pytest.mark.parametrize("withdrawal", ["credential", "policy", "cancel"])
@pytest.mark.parametrize("phase", ["producing", "completion"])
async def test_withdrawal_drains_remote_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    withdrawal: str,
    phase: str,
) -> None:
    """Revocation or cancellation holds the actual root until target cleanup ends."""
    async with connected(tmp_path, monkeypatch, "bidirectional") as witness:
        (witness.produce if phase == "producing" else witness.cleanup).clear()
        witness.retire.clear()
        root = asyncio.create_task(witness.invoke())
        try:
            await asyncio.wait_for(witness.target_started.wait(), 2)
            if withdrawal == "credential":
                witness.broker.callers.revoke(
                    witness.broker.callers.inventory().revision, witness.client_id
                )
            elif withdrawal == "policy":
                write_private(
                    tmp_path / "native-inference-policy.json",
                    InferenceAccessPolicy(revision="f" * 64, enabled=False, callers=())
                    .model_dump_json()
                    .encode(),
                )
            else:
                root.cancel()
            await asyncio.wait_for(witness.retiring.wait(), 2)
            assert not root.done() and not witness.target_cleaned.is_set()
            # Repeated cancellation must not abandon the owned retirement task.
            root.cancel()
            await asyncio.sleep(0)
            assert not root.done()
            witness.retire.set()
            with pytest.raises((ValueError, asyncio.CancelledError)):
                await asyncio.wait_for(root, 3)
            assert witness.target_cleaned.is_set()
            assert not witness.broker.active and not witness.broker.delegated
        finally:
            witness.retire.set()
            if not root.done():
                root.cancel()
            await asyncio.gather(root, return_exceptions=True)


async def test_signed_target_ceiling_narrows_uncapped_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null parent cannot widen a finite destination's actual context or timer."""
    async with connected(tmp_path, monkeypatch, "bidirectional", 0.25) as witness:
        witness.produce.clear()
        root = asyncio.create_task(witness.invoke())
        try:
            await asyncio.wait_for(witness.target_started.wait(), 2)
            deadline = TypeAdapter(int).validate_python(
                witness.observed[0].context["deadline_unix_ms"], strict=True
            )
            assert 0 < deadline - time.time_ns() // 1_000_000 <= 250
            with pytest.raises((ValueError, asyncio.CancelledError, TimeoutError)):
                await asyncio.wait_for(root, 3)
            assert witness.target_cleaned.is_set()
            assert not witness.broker.active and not witness.broker.delegated
            assert not witness.output
        finally:
            if not root.done():
                root.cancel()
            await asyncio.gather(root, return_exceptions=True)
