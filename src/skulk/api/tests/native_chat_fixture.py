"""Independent application over actual authenticated native component sockets.

Only signed installation discovery/owner custody are injected. Caller stores,
compiled declarations/context/frames, broker dispatch, input/output leases and
the HTTP adapters are real. Full signed installation has its separate gate.
"""

import asyncio
import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI, Request, Response
from pydantic import JsonValue, TypeAdapter

from skulk.api.inference_gateway import GatewaySigner
from skulk.api.native_chat import NativeChatGateway
from skulk.api.native_chat_catalog import ApplicationBinding
from skulk.api.native_inference_service import NativeInferenceService
from skulk.api.tests.test_inference_access import application
from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_calls import NativeCalls, NativeHostEndpoint
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    RequestFrame,
    SessionKeys,
)
from skulk.extensions.native_host_rpc import Peer, Reply
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
from skulk.extensions.runtime_files import write_private
from skulk.extensions.tests.test_native_host_protocol import binding
from skulk.shared.types.common import NodeId

_OBJECT = TypeAdapter(dict[str, JsonValue])


def listings() -> tuple[InferenceListing, ...]:
    """Compile an unrelated provider's explicit application bindings."""
    results: list[InferenceListing] = []
    for role, version, revision, mode, hints in (
        (
            "catalog",
            "0.2.0",
            "a",
            "unary",
            {"inference.catalog": "chat.completions@0.2.0"},
        ),
        (
            "chat",
            "0.2.0",
            "b",
            "bidirectional",
            {
                "inference.contract": "chat.completions@0.2.0",
                "inference.authorization": "gateway.ed25519@0.2.0",
            },
        ),
        (
            "continuation",
            "0.1.0",
            "c",
            "bidirectional",
            {
                "inference.contract": "chat.continuation@0.1.0",
                "inference.authorization": "caller.ed25519@0.1.0",
                "inference.chat": "independent.chat@0.2.0",
            },
        ),
    ):
        descriptor = InferenceDescriptor.model_validate_json(
            json.dumps(
                {
                    "contract_profile": "skulk.portable/2",
                    "profile_digest": InferenceProfiles.qualified().portable,
                    "id": "native.fixture." + role,
                    "version": version,
                    "title": role,
                    "description": "Independent native chat fixture",
                    "behavior": "read" if mode == "unary" else "inference",
                    "processing_domain": "local" if mode == "unary" else "external",
                    "io_mode": mode,
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "input_chunk_schema": {"type": "object"}
                    if mode != "unary"
                    else None,
                    "output_chunk_schema": {"type": "object"}
                    if mode != "unary"
                    else None,
                    "annotations": {
                        **hints,
                        "inference.envelope": "skulk.application-envelope@1.0.0",
                        "inference.envelope.contract": "independent.application@1.0.0",
                        "inference.application.id": "independent." + role,
                        "inference.application.version": version,
                        "inference.application.revision": revision * 16,
                    },
                }
            )
        )
        results.append(
            InferenceListing.model_validate_json(
                json.dumps(
                    {
                        "plugin_id": "managed.example",
                        "runtime_digest": "b" * 64,
                        "incarnation": "c" * 32,
                        "target": {
                            "node_id": "native-child",
                            "capability_id": descriptor.id,
                            "version": descriptor.version,
                            "descriptor_revision": descriptor.revision,
                            "behavior": descriptor.behavior,
                            "processing_domain": descriptor.processing_domain,
                        },
                        "descriptor": _OBJECT.validate_json(
                            descriptor.model_dump_json()
                        ),
                        "inference_seconds": None,
                    }
                )
            )
        )
    return tuple(results)


@dataclass
class Invocation:
    """Synthetic application behavior after real native broker admission."""

    identifier: str
    listing: InferenceListing
    proof: dict[str, JsonValue]
    packets: list[bytes]
    input: bytearray = field(default_factory=bytearray)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    end: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Fixture:
    """Actual protected caller state and observable provider-side effects."""

    root: Path
    app: FastAPI
    service: NativeInferenceService
    broker: NativeInferenceCalls
    gateway: NativeChatGateway
    listings: tuple[InferenceListing, ...]
    owner: AuthenticatedInferenceCaller
    client: AuthenticatedInferenceCaller
    token: str
    stranger_token: str
    calls: list[Invocation] = field(default_factory=list)
    metadata_calls: int = 0
    corrupt_end: bool = False
    refuse_chat: bool = False
    reject_input_kind: Literal["started", "chunk", "completed"] | None = None
    receipt_lookups: int = 0
    owner_peer: Peer | None = None
    wait_control: asyncio.Event = field(default_factory=asyncio.Event)

    def policy(
        self, *, owner: bool = True, client: bool = True, continuation: bool = True
    ) -> None:
        """Replace real current authority without changing any bearer or proof."""
        grants = tuple(
            item.grant()
            for item in self.listings
            if continuation or not item.target.capability_id.endswith("continuation")
        )
        write_private(
            self.root / "native-inference-policy.json",
            InferenceAccessPolicy(
                revision=secrets.token_hex(32),
                enabled=True,
                callers=tuple(
                    PrincipalAccess(principal=caller.principal, grants=grants)
                    for caller, enabled in ((self.owner, owner), (self.client, client))
                    if enabled
                ),
            )
            .model_dump_json()
            .encode(),
        )


@asynccontextmanager
async def native_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Fixture]:
    """Run the real broker and HTTP service against authenticated component peers."""
    app, callers = application(root)
    service = NativeInferenceService(
        NodeId("access-core"), connection_path=root / "connection.json"
    )
    broker = await service.host()
    _, token = callers.create(callers.inventory().revision, "Allowed client", None)
    _, stranger = callers.create(callers.inventory().revision, "No grants", None)
    gateway = NativeChatGateway(service, GatewaySigner(root / "gateway.key"))
    fixture = Fixture(
        root,
        app,
        service,
        broker,
        gateway,
        listings(),
        callers.owner(AuthenticatedOwner(NodeId("access-core"))),
        callers.authenticate(token),
        token,
        stranger,
    )
    fixture.policy()
    fixture.wait_control.set()

    async def directory(
        _self: InferenceHosts, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        return fixture.listings

    def check_owner(_self: NativeCalls, plugin: str, runtime: str, peer: Peer) -> None:
        assert plugin == "managed.example" and runtime == "b" * 64
        peer.check()

    monkeypatch.setattr(InferenceHosts, "directory", directory)
    monkeypatch.setattr(NativeCalls, "check_owner", check_owner)
    incoming: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.Queue()
    )
    server = await asyncio.start_server(
        lambda r, w: incoming.put_nowait((r, w)), "127.0.0.1", 0
    )
    _, port = TypeAdapter(tuple[str, int]).validate_python(
        cast(object, server.sockets[0].getsockname()), strict=True
    )
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    owner_reader, owner_writer = await incoming.get()
    core_budget, owner_budget = broker.hosts.budget, NativePayloadBudget()
    invocations: dict[str, Invocation] = {}

    async def refuse(_peer: Peer, _request: RequestFrame, _body: bytes) -> Reply:
        if _request.operation == "delegate_inference":
            handler = broker.hosts.inference_handler
            assert handler is not None
            return await handler(_peer, _request, _body)
        return Reply("pre_dispatch", {})

    async def respond(_peer: Peer, request: RequestFrame, body: bytes) -> Reply:
        kernel = broker.kernel
        if request.operation == "infer":
            fixture.metadata_calls += 1
            metadata = fixture.listings[0]
            assert (
                ApplicationBinding.from_listing(metadata).unwrap(
                    _OBJECT.validate_json(body)
                )
                == {}
            )
            catalog: dict[str, JsonValue] = {
                "contract": "chat.completions@0.2.0",
                "target_node": "application-proof-realm",
                "callable_id": "independent.chat",
                "callable_version": "0.2.0",
                "descriptor_revision": "b" * 16,
                "catalog_revision": "d" * 64,
                "expires_at": time.time() + 300,
                "models": [
                    {
                        "id": "code",
                        "name": "Independent Code",
                        "features": [
                            "image_input",
                            "function_tools",
                            "text_file_input",
                            "pdf_input",
                        ],
                    }
                ],
                "processing_domain": "external",
                "usage": "unknown",
                "usage_label": "Synthetic plan",
                "usage_url": None,
                "maximum_body_bytes": 16 * 1024 * 1024,
            }
            return Reply(
                "complete",
                {},
                json.dumps(
                    ApplicationBinding.from_listing(metadata).wrap(catalog)
                ).encode(),
            )
        identifier = TypeAdapter(str).validate_python(
            request.parameters["call_id"], strict=True
        )
        if request.operation == "infer_stream":
            context = _OBJECT.validate_python(
                request.parameters["context"], strict=True
            )
            executor = _OBJECT.validate_python(context["executor"], strict=True)
            selected = next(
                item
                for item in fixture.listings
                if item.target.capability_id == executor["capability_id"]
            )
            proof = ApplicationBinding.from_listing(selected).unwrap(
                _OBJECT.validate_json(body)
            )
            events: list[dict[str, JsonValue]] = [
                {
                    "kind": "delta",
                    "delta": {
                        "content": "```python\nprint('hello')\n```",
                        "reasoning_content": "Public summary",
                    },
                },
                {
                    "kind": "delta",
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "inspect", "arguments": "{}"},
                            }
                        ]
                    },
                },
                {
                    "kind": "completed",
                    "delta": None,
                    "finish_reason": "tool_calls",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
            ]
            if fixture.refuse_chat:
                events = [{"kind": "failed", "error": "policy_denied"}]
            if selected.target.capability_id.endswith("continuation"):
                application_request = _OBJECT.validate_python(
                    proof["request"], strict=True
                )
                receipt: dict[str, JsonValue] = {
                    "contract": "chat.continuation@0.1.0",
                    "conversation_id": application_request["conversation_id"],
                    "handle": "ct1_" + "e" * 64,
                    "history": {"count": 2, "sha256": "f" * 64},
                    "request_nonce": application_request.get(
                        "request_nonce", application_request["nonce"]
                    ),
                    "expires_at": time.time() + 300,
                }
                if application_request["contract"] == "chat.continuation.receipt@0.1.0":
                    fixture.receipt_lookups += 1
                    events = (
                        [{"receipt": None, "error": "authority_unavailable"}]
                        if fixture.refuse_chat
                        else [{"receipt": receipt, "error": None}]
                    )
                else:
                    events = [
                        {
                            "event": event,
                            "receipt": receipt
                            if event["kind"] == "completed"
                            else None,
                            "continuation_error": None,
                        }
                        for event in events
                    ]
            invocation = Invocation(identifier, selected, proof, [])
            for sequence, event in enumerate(events, 1):
                invocation.packets.append(
                    kernel.encode_transport_v2_packet(
                        json.dumps(
                            {
                                "call_id": identifier,
                                "direction": "provider_to_caller",
                                "sequence": sequence,
                                "kind": "chunk",
                                "payload": ApplicationBinding.from_listing(
                                    selected
                                ).wrap(event),
                            }
                        ),
                        b"",
                    )
                )
            invocation.packets.extend(
                [
                    kernel.encode_transport_v2_packet(
                        json.dumps(
                            {
                                "call_id": identifier,
                                "direction": "provider_to_caller",
                                "sequence": len(events) + 1,
                                "kind": "completed",
                                "payload": {},
                            }
                        ),
                        b"",
                    ),
                    kernel.encode_transport_v2_packet(
                        json.dumps(
                            {
                                "kind": "end",
                                "call_id": "wrong"
                                if fixture.corrupt_end
                                else identifier,
                            }
                        ),
                        b"",
                    ),
                ]
            )
            invocations[identifier] = invocation
            fixture.calls.append(invocation)
            await invocation.end.wait()
            await fixture.wait_control.wait()
            return Reply("complete", {}, b"{}")
        invocation = invocations.get(identifier)
        if request.operation == "stream_ready":
            return Reply("complete", {"ready": invocation is not None})
        assert invocation is not None
        if request.operation == "stream_input":
            source, media = kernel.decode_transport_v2_packet(body)
            fields = _OBJECT.validate_json(source)
            if fields["kind"] == fixture.reject_input_kind:
                invocation.ready.set()
                return Reply("pre_dispatch", {})
            invocation.input.extend(media)
            if fields["kind"] == "completed":
                invocation.ready.set()
            return Reply("complete", {})
        assert request.operation == "stream_next"
        await invocation.ready.wait()
        packet = invocation.packets.pop(0)
        if not invocation.packets:
            invocation.end.set()
        return Reply("complete", {}, packet)

    core = Peer(
        AuthenticatedStream(
            reader,
            writer,
            SessionKeys(outbound=b"a" * 32, inbound=b"b" * 32),
            core_budget,
        ),
        refuse,
        frozenset({"delegate_inference"}),
        lambda: None,
    )
    owner_peer = Peer(
        AuthenticatedStream(
            owner_reader,
            owner_writer,
            SessionKeys(outbound=b"b" * 32, inbound=b"a" * 32),
            owner_budget,
        ),
        respond,
        frozenset(
            {"infer", "infer_stream", "stream_ready", "stream_input", "stream_next"}
        ),
        lambda: None,
    )
    broker.hosts.endpoints["managed.example"] = NativeHostEndpoint(
        binding(Ed25519PrivateKey.generate()), core
    )
    fixture.owner_peer = owner_peer

    async def models(request: Request) -> list[dict[str, JsonValue]]:
        return [
            _OBJECT.validate_json(item.model_dump_json())
            for item in await gateway.models(request)
        ]

    async def chat(request: Request) -> Response:
        return await gateway.chat(
            request, continuation_header=request.headers.get("Skulk-Continuation-Proof")
        )

    app.get("/models")(models)
    app.post("/chat")(chat)
    app.post("/receipt")(gateway.receipt)
    try:
        yield fixture
    finally:
        fixture.wait_control.set()
        await service.close()
        await owner_peer.close()
        server.close()
        await server.wait_closed()
        assert core_budget.used == owner_budget.used == 0
