"""Native HTTP discovery and delivery with real caller state and broker streams."""

import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from starlette.types import Message, Scope

from skulk.api.inference_continuation import ReceiptProof
from skulk.api.main import API
from skulk.api.native_chat import NativeChatGateway
from skulk.api.native_chat_catalog import ApplicationBinding, native_chat_catalogs
from skulk.api.native_chat_stream import (
    NativeChatDeliveryUncertainError,
    NativeChatResponse,
)
from skulk.api.tests.native_chat_fixture import native_fixture
from skulk.api.tests.test_inference_access import client_for
from skulk.api.types import ModelList, ModelListModel
from skulk.extensions.native_host_protocol import ResponseFrame
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import TelemetryView
from skulk.utils.channels import channel

_OBJECT = TypeAdapter(dict[str, JsonValue])
_MODELS = TypeAdapter(list[ModelListModel])
_MAPPINGS = TypeAdapter(list[dict[str, JsonValue]])
_TEXT = TypeAdapter(str)


class NativeRouteAPI(API):
    """Expose only test-selected custody while preserving production HTTP routes."""

    def bind_native_chat(self, gateway: NativeChatGateway) -> None:
        """Bind the authenticated component host for this isolated API fixture."""
        self._native_chat_gateway = gateway


def models(response: httpx.Response) -> list[ModelListModel]:
    """Validate public model entries instead of trusting untyped HTTP JSON."""
    return _MODELS.validate_json(response.content)


async def test_normal_api_routes_compose_native_models_and_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public route wrappers use native discovery without altering local catalogs."""

    async def ordinary(_self: API, status: str | None = None) -> ModelList:
        return ModelList(data=[ModelListModel(id="ordinary/local")])

    monkeypatch.setattr(API, "get_models", ordinary)
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        commands, _ = channel[ForwarderCommand]()
        downloads, _ = channel[ForwarderDownloadCommand]()
        _, events = channel[IndexedEvent]()
        _, elections = channel[ElectionMessage]()
        sender, receiver = channel[ProviderStreamPacket](256)
        api = NativeRouteAPI(
            NodeId("access-core"),
            port=52415,
            event_receiver=events,
            command_sender=commands,
            download_command_sender=downloads,
            election_receiver=elections,
            telemetry_view=TelemetryView(),
            enable_event_log=False,
            mount_dashboard=False,
            provider_stream_sender=sender,
            provider_stream_receiver=receiver,
        )
        api.bind_native_chat(fixture.gateway)
        async with client_for(api.app) as client:
            model: str | None = None
            for path in ("/models", "/v1/models"):
                visible = await client.get(path)
                entries = _MODELS.validate_python(
                    _OBJECT.validate_json(visible.content)["data"]
                )
                assert len(entries) == 2 and entries[0].id == "ordinary/local"
                assert visible.headers["cache-control"] == "no-store"
                model = entries[1].id
                local = await client.get(
                    path, headers={"authorization": "Bearer dummy"}
                )
                assert [
                    entry.id
                    for entry in _MODELS.validate_python(
                        _OBJECT.validate_json(local.content)["data"]
                    )
                ] == ["ordinary/local"]
                downloaded = await client.get(path, params={"status": "downloaded"})
                assert (
                    len(
                        _MODELS.validate_python(
                            _OBJECT.validate_json(downloaded.content)["data"]
                        )
                    )
                    == 1
                )
            assert model is not None
            result = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert result.status_code == 200 and len(fixture.calls) == 1
            parameters = _OBJECT.validate_python(
                _OBJECT.validate_python(
                    _OBJECT.validate_python(api.app.openapi())["paths"]
                )["/v1/chat/continuation/receipt"]
            )
            assert "native_model" in json.dumps(parameters)


async def test_discovery_is_caller_filtered_without_owner_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent owner/client reads cannot share or manufacture native authority."""
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        async with client_for(fixture.app) as owner, client_for(fixture.app) as client:
            permitted, denied = await asyncio.gather(
                owner.get("/models"),
                client.get(
                    "/models",
                    headers={"authorization": "Bearer " + fixture.stranger_token},
                ),
            )
            assert permitted.status_code == denied.status_code == 200
            assert len(models(permitted)) == 1 and denied.json() == []
            visible = await client.get(
                "/models", headers={"authorization": "Bearer " + fixture.token}
            )
            entry = models(visible)[0]
            assert entry.id.startswith(
                "native/managed.example/native.fixture.chat@0.2.0/"
            )
            continuation = _OBJECT.validate_python(
                _OBJECT.validate_python(entry.inference_capability)["continuation"]
            )
            assert continuation["native_model"] == entry.id
            assert (
                _OBJECT.validate_python(continuation["target"])["target_node"]
                == "application-proof-realm"
            )
            assert (
                len(
                    _TEXT.validate_python(
                        _OBJECT.validate_python(continuation["target"])[
                            "descriptor_revision"
                        ]
                    )
                )
                == 16
            )
            for headers in (
                {"authorization": "Bearer dummy"},
                {
                    "x-caller-id": "direct-owner",
                    "authorization": "Bearer " + fixture.stranger_token,
                },
            ):
                assert (await client.get("/models", headers=headers)).json() == []
            fixture.policy(owner=True, client=False)
            assert (
                await client.get(
                    "/models", headers={"authorization": "Bearer " + fixture.token}
                )
            ).json() == []
            assert len(models(await owner.get("/models"))) == 1
        assert fixture.calls == []


@pytest.mark.parametrize(
    "change",
    [
        "runtime",
        "incarnation",
        "child",
        "application_revision",
        "application_contract",
        "processing_domain",
    ],
)
async def test_catalog_cannot_bind_an_unrelated_chat_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Catalog claims do not replace the signed same-child application relationship."""
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        chat = fixture.listings[1]
        if change == "runtime":
            chat = chat.model_copy(update={"runtime_digest": "e" * 64})
        elif change == "incarnation":
            chat = chat.model_copy(update={"incarnation": "e" * 32})
        elif change == "child":
            chat = chat.model_copy(
                update={
                    "target": chat.target.model_copy(
                        update={"node_id": "another-child"}
                    )
                }
            )
        else:
            source = _OBJECT.validate_json(chat.descriptor.model_dump_json())
            if change == "processing_domain":
                source["processing_domain"] = "local"
            else:
                annotations = TypeAdapter(dict[str, str]).validate_python(
                    source["annotations"], strict=True
                )
                annotations[
                    "inference.application.revision"
                    if change == "application_revision"
                    else "inference.envelope.contract"
                ] = (
                    "e" * 16
                    if change == "application_revision"
                    else "another.application"
                )
                source["annotations"] = _OBJECT.validate_python(
                    annotations, strict=True
                )
            descriptor = type(chat.descriptor).model_validate_json(json.dumps(source))
            chat = chat.model_copy(
                update={
                    "descriptor": descriptor,
                    "target": chat.target.model_copy(
                        update={
                            "descriptor_revision": descriptor.revision,
                            "processing_domain": descriptor.processing_domain,
                        }
                    ),
                }
            )
        fixture.listings = (fixture.listings[0], chat, fixture.listings[2])
        fixture.policy()
        assert await native_chat_catalogs(fixture.broker, fixture.owner) == ()
        assert fixture.calls == []


@pytest.mark.parametrize("streamed", [False, True])
async def test_native_chat_preserves_code_tools_summary_and_request_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    """The native broker carries normal application semantics and exact input bytes."""
    async with (
        native_fixture(tmp_path, monkeypatch) as fixture,
        client_for(fixture.app) as client,
    ):
        headers = {"authorization": "Bearer " + fixture.token}
        model = models(await client.get("/models", headers=headers))[0].id
        body = {
            "model": model,
            "stream": streamed,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect code"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": "notes.txt",
                                "file_data": "data:text/plain;base64,Y29kZQ==",
                            },
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
        result = await client.post("/chat", json=body, headers=headers)
        assert result.status_code == 200, result.text
        assert len(fixture.calls) == 1
        sent = _OBJECT.validate_json(bytes(fixture.calls[0].input))
        assert sent["messages"] == body["messages"] and sent["tools"] == body["tools"]
        assert "model" not in sent
        if streamed:
            assert (
                "```python" in result.text
                and '"tool_calls"' in result.text
                and result.text.endswith("data: [DONE]\n\n")
            )
        else:
            message = _OBJECT.validate_python(
                _MAPPINGS.validate_python(
                    _OBJECT.validate_json(result.content)["choices"]
                )[0]["message"]
            )
            assert message["content"] == "```python\nprint('hello')\n```"
            assert message["reasoning_content"] == "Public summary"
            assert _MAPPINGS.validate_python(message["tool_calls"])[0]["function"] == {
                "name": "inspect",
                "arguments": "{}",
            }
        assert fixture.broker.active == {}


@pytest.mark.parametrize("credential", ["invalid", "no_grant"])
async def test_explicit_native_chat_requires_actual_client_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, credential: str
) -> None:
    """A gateway public key or caller-ID header is never a Core bearer permission."""
    async with (
        native_fixture(tmp_path, monkeypatch) as fixture,
        client_for(fixture.app) as client,
    ):
        model = models(await client.get("/models"))[0].id
        result = await client.post(
            "/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={
                "authorization": "Bearer "
                + ("dummy" if credential == "invalid" else fixture.stranger_token),
                "x-caller-id": "direct-owner",
            },
        )
        assert result.status_code == (401 if credential == "invalid" else 403)
        assert fixture.calls == []


async def test_bad_native_end_never_becomes_http_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Application completion does not replace exact native end/control admission."""
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        fixture.corrupt_end = True
        async with client_for(fixture.app) as client:
            model = models(await client.get("/models"))[0].id
            result = await client.post(
                "/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
            assert (
                "[DONE]" not in result.text
                and "native_outcome_unavailable" in result.text
            )
            assert len(fixture.calls) == 1 and fixture.broker.active == {}


async def test_native_continuation_and_receipt_keep_independent_proof_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core selects native transport while the application enforces its own signature.

    This component peer deliberately accepts synthetic signature bytes. Actual
    account/client signature enforcement belongs to the installed app gate.
    """
    async with (
        native_fixture(tmp_path, monkeypatch) as fixture,
        client_for(fixture.app) as client,
    ):
        headers = {"authorization": "Bearer " + fixture.token}
        entry = models(await client.get("/models", headers=headers))[0]
        continuation = _OBJECT.validate_python(
            _OBJECT.validate_python(entry.inference_capability)["continuation"]
        )
        raw = json.dumps(
            {"model": entry.id, "messages": [{"role": "user", "content": "continue"}]}
        ).encode()
        proof: dict[str, JsonValue] = {
            "caller_id": "a" * 32,
            "signature": "b" * 128,
            "request": {
                "contract": "chat.continuation@0.1.0",
                "mode": "start",
                "handle": None,
                "conversation_id": "c" * 64,
                "target": continuation["target"],
                "body": {
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "media_type": "application/vnd.skulk.chat-request+json",
                },
                "nonce": "d" * 64,
                "expires_at": time.time() + 30,
            },
        }
        header = (
            base64.urlsafe_b64encode(json.dumps(proof).encode()).rstrip(b"=").decode()
        )
        result = await client.post(
            "/chat",
            content=raw,
            headers={
                **headers,
                "content-type": "application/json",
                "Skulk-Continuation-Proof": header,
            },
        )
        assert result.status_code == 200, result.text
        receipt = _OBJECT.validate_json(result.content)["skulk_continuation"]
        assert _OBJECT.validate_python(receipt)["request_nonce"] == "d" * 64
        assert bytes(fixture.calls[0].input) == raw
        query = {
            "caller_id": "a" * 32,
            "signature": "b" * 128,
            "request": {
                "contract": "chat.continuation.receipt@0.1.0",
                "conversation_id": "c" * 64,
                "target": continuation["target"],
                "request_nonce": "d" * 64,
                "nonce": "e" * 64,
                "expires_at": time.time() + 30,
            },
        }
        lookup = await client.post(
            "/receipt", params={"native_model": entry.id}, json=query, headers=headers
        )
        assert lookup.status_code == 200, lookup.text
        assert (
            _OBJECT.validate_python(_OBJECT.validate_json(lookup.content)["receipt"])[
                "request_nonce"
            ]
            == "d" * 64
        )
        assert fixture.receipt_lookups == 1 and len(fixture.calls) == 2
        assert not fixture.calls[-1].input
        fixture.policy(continuation=False)
        refused = await client.post(
            "/receipt", params={"native_model": entry.id}, json=query, headers=headers
        )
        assert refused.status_code == 403 and fixture.receipt_lookups == 1


@pytest.mark.parametrize("withdraw", ["policy", "credential", "disconnect"])
async def test_revocation_or_disconnect_cancels_blocked_physical_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, withdraw: str
) -> None:
    """A blocked HTTP send retains its output lease and runs under current authority."""
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        catalog = (await native_chat_catalogs(fixture.broker, fixture.client))[0]
        prepared, body = fixture.gateway.signer.prepare(
            catalog.catalog,
            "code",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        response = NativeChatResponse(
            fixture.broker,
            fixture.client,
            catalog.chat,
            _OBJECT.validate_python(prepared, strict=True),
            model=catalog.model_id(catalog.catalog.models[0]),
            streamed=True,
            body=body,
        )
        entered = asyncio.Event()
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        messages: list[Message] = []
        borrowed = 0

        async def send(message: Message) -> None:
            nonlocal borrowed
            if message["type"] == "http.response.body" and b"print" in message.get(
                "body", b""
            ):
                borrowed = fixture.broker.hosts.budget.used
                entered.set()
                await asyncio.Future[None]()
            messages.append(message)

        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
        }
        task = asyncio.create_task(response(scope, incoming.get, send))
        try:
            async with asyncio.timeout(3):
                await entered.wait()
                assert borrowed > 0
                if withdraw == "policy":
                    fixture.policy(client=False)
                elif withdraw == "credential":
                    fixture.broker.callers.revoke(
                        fixture.broker.callers.inventory().revision,
                        fixture.client.principal.identifier.removeprefix("client:"),
                    )
                else:
                    incoming.put_nowait({"type": "http.disconnect"})
                await task
            assert fixture.broker.active == {} and fixture.broker.hosts.budget.used == 0
            assert not any(
                message["type"] == "http.response.body"
                and b"[DONE]" in message.get("body", b"")
                for message in messages
            )
            assert len(fixture.calls) == 1
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("withdraw", ["policy", "credential", "disconnect", "delegate"])
async def test_final_success_write_retains_revocable_caller_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, withdraw: str
) -> None:
    """Native retirement precedes success without dropping the final write's watcher."""
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        catalog = (await native_chat_catalogs(fixture.broker, fixture.client))[0]
        prepared, body = fixture.gateway.signer.prepare(
            catalog.catalog,
            "code",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        response = NativeChatResponse(
            fixture.broker,
            fixture.client,
            catalog.chat,
            _OBJECT.validate_python(prepared, strict=True),
            model=catalog.model_id(catalog.catalog.models[0]),
            streamed=True,
            body=body,
        )
        entered = asyncio.Event()
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        delivered: list[Message] = []

        async def send(message: Message) -> None:
            if message["type"] == "http.response.body" and b"[DONE]" in message.get(
                "body", b""
            ):
                assert fixture.calls[-1].end.is_set()
                assert len(fixture.broker.active) == 1
                assert all(
                    active.execution_complete
                    for active in fixture.broker.active.values()
                )
                entered.set()
                await asyncio.Future[None]()
            delivered.append(message)

        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
        }
        task = asyncio.create_task(response(scope, incoming.get, send))
        try:
            async with asyncio.timeout(3):
                await entered.wait()
                if withdraw == "policy":
                    fixture.policy(client=False)
                elif withdraw == "credential":
                    fixture.broker.callers.revoke(
                        fixture.broker.callers.inventory().revision,
                        fixture.client.principal.identifier.removeprefix("client:"),
                    )
                elif withdraw == "delegate":
                    (_, parent_call_id), parent = next(
                        iter(fixture.broker.active.items())
                    )
                    context = _OBJECT.validate_json(parent.context.canonical())
                    kernel = fixture.broker.kernel
                    payload = json.dumps(
                        ApplicationBinding.from_listing(fixture.listings[0]).wrap({})
                    ).encode()
                    request = kernel.CompositionV3Contract.from_json(
                        "DelegateRequest",
                        json.dumps(
                            {
                                "protocol": 3,
                                "contract_profile": "skulk.composition/3",
                                "profile_digest": kernel.composition_v3_profile_digest(),
                                "kind": "delegate",
                                "request_id": "0" * 63 + "1",
                                "parent_scope_id": context["scope_id"],
                                "parent_request_sha256": context["request_sha256"],
                                "destination": fixture.listings[0].target.model_dump(
                                    mode="json"
                                ),
                                "request_sha256": hashlib.sha256(payload).hexdigest(),
                                "deadline_unix_ms": None,
                            }
                        ),
                    )
                    policy = kernel.CompositionV3Contract.from_json(
                        "Policy",
                        json.dumps(
                            {
                                **{
                                    key: context[key]
                                    for key in (
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
                                    )
                                },
                                "kind": "policy",
                            }
                        ),
                    )
                    delegated = parent.context.admit(
                        request,
                        policy,
                        parent.listing.target.model_dump_json(),
                        payload,
                        time.time_ns() // 1_000_000,
                        "e" * 64,
                    )
                    owner_peer = fixture.owner_peer
                    assert owner_peer is not None
                    before = fixture.metadata_calls
                    refusal = await owner_peer.request(
                        "delegate_inference",
                        {
                            "parent_call_id": parent_call_id,
                            "request": _OBJECT.validate_json(request.canonical()),
                            "context": _OBJECT.validate_json(delegated.canonical()),
                            "remaining_seconds": None,
                        },
                        payload,
                    )
                    try:
                        assert (
                            isinstance(refusal.frame, ResponseFrame)
                            and refusal.frame.outcome == "unknown"
                        )
                    finally:
                        refusal.release()
                    assert fixture.metadata_calls == before
                else:
                    incoming.put_nowait({"type": "http.disconnect"})
                await task
            assert not any(
                message["type"] == "http.response.body"
                and b"[DONE]" in message.get("body", b"")
                for message in delivered
            )
            assert fixture.broker.active == {} and fixture.broker.hosts.budget.used == 0
            assert len(fixture.calls) == 1
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind", ["chat", "receipt"])
@pytest.mark.parametrize("failure", ["policy", "retirement"])
async def test_collected_payload_is_not_clean_success_after_invocation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, failure: str
) -> None:
    """A started JSON response must abort rather than mask lost native custody."""
    from skulk.extensions import native_inference_calls

    async with native_fixture(tmp_path, monkeypatch) as fixture:
        catalog = (await native_chat_catalogs(fixture.broker, fixture.client))[0]
        model = catalog.model_id(catalog.catalog.models[0])
        if kind == "receipt":
            target = catalog.continuation
            assert target is not None
            proof = ReceiptProof.model_validate_json(
                json.dumps(
                    {
                        "caller_id": "a" * 32,
                        "signature": "b" * 128,
                        "request": {
                            "contract": "chat.continuation.receipt@0.1.0",
                            "conversation_id": "c" * 64,
                            "target": {
                                "target_node": catalog.catalog.target_node,
                                "callable_id": target.application.callable_id,
                                "descriptor_revision": target.application.revision,
                                "model": "code",
                            },
                            "request_nonce": "d" * 64,
                            "nonce": "e" * 64,
                            "expires_at": time.time() + 30,
                        },
                    }
                )
            )
            response = NativeChatResponse(
                fixture.broker,
                fixture.client,
                target,
                _OBJECT.validate_json(proof.model_dump_json()),
                model=model,
                streamed=False,
                receipt=proof,
            )
        else:
            prepared, body = fixture.gateway.signer.prepare(
                catalog.catalog,
                "code",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
            response = NativeChatResponse(
                fixture.broker,
                fixture.client,
                catalog.chat,
                _OBJECT.validate_python(prepared, strict=True),
                model=model,
                streamed=False,
                body=body,
            )
        payload_sent = False
        retirement_failed = False
        delivered: list[Message] = []
        original = native_inference_calls.finish_runtime_work

        async def fail_retirement(task: asyncio.Task[None]) -> None:
            nonlocal retirement_failed
            await original(task)
            if payload_sent and not retirement_failed:
                retirement_failed = True
                raise RuntimeError("synthetic private retirement diagnostic")

        if failure == "retirement":
            monkeypatch.setattr(
                native_inference_calls, "finish_runtime_work", fail_retirement
            )

        async def send(message: Message) -> None:
            nonlocal payload_sent
            delivered.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                assert message.get("more_body") is True
                payload_sent = True
                if failure == "policy":
                    fixture.policy(client=False)

        incoming: asyncio.Queue[Message] = asyncio.Queue()
        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
        }
        with pytest.raises(
            NativeChatDeliveryUncertainError, match="delivery is uncertain"
        ):
            async with asyncio.timeout(3):
                await response(scope, incoming.get, send)
        assert payload_sent
        assert not any(
            message["type"] == "http.response.body"
            and not message.get("more_body", False)
            for message in delivered
        )
        assert fixture.broker.active == {} and fixture.broker.hosts.budget.used == 0
        assert len(fixture.calls) == 1


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("corrupt_end", [False, True])
@pytest.mark.parametrize("reject_input_kind", [None, "started", "chunk", "completed"])
async def test_native_semantic_refusal_waits_for_verified_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streamed: bool,
    corrupt_end: bool,
    reject_input_kind: Literal["started", "chunk", "completed"] | None,
) -> None:
    """Publish an application refusal only after native terminal/end/control admission."""
    async with native_fixture(tmp_path, monkeypatch) as fixture:
        fixture.refuse_chat = True
        fixture.reject_input_kind = reject_input_kind
        fixture.corrupt_end = corrupt_end
        fixture.wait_control.clear()
        catalog = (await native_chat_catalogs(fixture.broker, fixture.client))[0]
        prepared, body = fixture.gateway.signer.prepare(
            catalog.catalog,
            "code",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        response = NativeChatResponse(
            fixture.broker,
            fixture.client,
            catalog.chat,
            _OBJECT.validate_python(prepared, strict=True),
            model=catalog.model_id(catalog.catalog.models[0]),
            streamed=streamed,
            body=body,
        )
        delivered: list[Message] = []
        incoming: asyncio.Queue[Message] = asyncio.Queue()

        async def send(message: Message) -> None:
            if message[
                "type"
            ] == "http.response.body" and b"policy_denied" in message.get("body", b""):
                assert fixture.wait_control.is_set()
                assert fixture.calls[0].end.is_set()
                assert fixture.calls[0].packets == []
                assert all(
                    active.execution_complete
                    for active in fixture.broker.active.values()
                )
            delivered.append(message)

        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
        }
        task = asyncio.create_task(response(scope, incoming.get, send))
        try:
            async with asyncio.timeout(3):
                while not fixture.calls:
                    await asyncio.sleep(0.001)
                await fixture.calls[0].end.wait()
                assert not any(
                    message["type"] == "http.response.body"
                    and b"policy_denied" in message.get("body", b"")
                    for message in delivered
                )
                fixture.wait_control.set()
                await task
            output = b"".join(
                message.get("body", b"")
                for message in delivered
                if message["type"] == "http.response.body"
            )
            assert b"[DONE]" not in output
            if corrupt_end:
                assert b"policy_denied" not in output
                assert b"native_outcome_unavailable" in output
            else:
                assert output.count(b'"code": "policy_denied"') == 1
                assert b"native_outcome_unavailable" not in output
            assert len(fixture.calls) == 1
            assert fixture.broker.active == {} and fixture.broker.hosts.budget.used == 0
        finally:
            fixture.wait_control.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("reject_input_kind", ["started", "chunk", "completed"])
async def test_native_unaccepted_input_never_releases_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streamed: bool,
    reject_input_kind: Literal["started", "chunk", "completed"],
) -> None:
    """Refusal draining cannot publish text/tools or success from unaccepted input."""
    async with (
        native_fixture(tmp_path, monkeypatch) as fixture,
        client_for(fixture.app) as client,
    ):
        fixture.reject_input_kind = reject_input_kind
        fixture.wait_control.set()
        headers = {"authorization": "Bearer " + fixture.token}
        model = models(await client.get("/models", headers=headers))[0].id
        result = await client.post(
            "/chat",
            headers=headers,
            json={
                "model": model,
                "stream": streamed,
                "messages": [{"role": "user", "content": "unaccepted"}],
            },
        )
        assert result.status_code == (200 if streamed else 503)
        assert "native_outcome_unavailable" in result.text
        assert "print(" not in result.text and "tool_calls" not in result.text
        assert "[DONE]" not in result.text and len(fixture.calls) == 1
        assert fixture.broker.active == {} and fixture.broker.hosts.budget.used == 0


@pytest.mark.parametrize("corrupt_end", [False, True])
@pytest.mark.parametrize("reject_input_kind", ["started", "completed"])
async def test_native_early_receipt_refusal_requires_verified_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_end: bool,
    reject_input_kind: Literal["started", "completed"],
) -> None:
    """An early lookup refusal releases no receipt and still verifies end/control."""
    async with (
        native_fixture(tmp_path, monkeypatch) as fixture,
        client_for(fixture.app) as client,
    ):
        fixture.reject_input_kind = reject_input_kind
        fixture.refuse_chat = True
        fixture.corrupt_end = corrupt_end
        headers = {"authorization": "Bearer " + fixture.token}
        model = models(await client.get("/models", headers=headers))[0].id
        query = {
            "caller_id": "a" * 32,
            "signature": "b" * 128,
            "request": {
                "contract": "chat.continuation.receipt@0.1.0",
                "conversation_id": "c" * 64,
                "target": {
                    "target_node": "application-proof-realm",
                    "callable_id": "independent.continuation",
                    "descriptor_revision": "c" * 16,
                    "model": "code",
                },
                "request_nonce": "d" * 64,
                "nonce": "e" * 64,
                "expires_at": time.time() + 30,
            },
        }
        result = await client.post(
            "/receipt", params={"native_model": model}, json=query, headers=headers
        )
        if corrupt_end:
            assert result.status_code == 503
            assert "authority_unavailable" not in result.text
        else:
            assert result.status_code == 200
            assert result.json() == {"receipt": None, "error": "authority_unavailable"}
            assert fixture.calls[0].end.is_set()
        assert len(fixture.calls) == 1 and fixture.receipt_lookups == 1
        assert fixture.broker.active == {} and fixture.broker.hosts.budget.used == 0
