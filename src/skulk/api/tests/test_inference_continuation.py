"""Actual unified HTTP/duplex roundtrips using an independent client-key provider."""

import asyncio
import base64
import hashlib
import json
import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import JsonValue, TypeAdapter

from skulk.api.inference_continuation import (
    ContinuationProof,
    ReceiptProof,
    read_continuation_proof,
)
from skulk.api.inference_gateway import chat_response
from skulk.api.tests.test_inference_gateway import build_api
from skulk.api.tests.test_inference_media import MediaFixture
from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityStreamFrame,
    ExtensionContext,
    descriptor_revision,
)
from skulk.extensions.calls import CapabilityResult
from skulk.extensions.streams import CapabilityStreamSession, InlineMediaAttachment

_OBJECT = TypeAdapter(dict[str, JsonValue])


class ClientProvider(MediaFixture):
    """Unrelated provider enforcing its own registry without a gateway secret."""

    def __init__(self) -> None:
        """Publish the generic explicit relationship alongside unchanged chat metadata."""
        super().__init__("0" * 64)
        self.keys: dict[str, str] = {}
        self.receipts: dict[tuple[str, str], dict[str, JsonValue]] = {}
        self.raw: bytes | None = None
        self.transport_failure = False
        self.bad_receipt = False
        self.use_tools = False
        self.continuation = CapabilityDescriptor(
            id="independent.resume",
            version="0.1.0",
            title="Independent continuation",
            description="Registered client proofs, no gateway credential.",
            io_mode="bidirectional",
            input_schema={"type": "object"},
            input_chunk_schema={"type": "object"},
            output_schema={"type": "object"},
            output_chunk_schema={"type": "object"},
            annotations={
                "inference.contract": "chat.continuation@0.1.0",
                "inference.authorization": "caller.ed25519@0.1.0",
                "inference.chat": self.chat.qualified_id,
            },
        )

    def capabilities(self) -> list[CapabilityDescriptor]:
        """Keep the original chat/catalog descriptors and add one explicit sibling."""
        return [*super().capabilities(), self.continuation]

    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncGenerator[CapabilityStreamFrame]:
        """Resolve a registered key, verify bytes, and emit bounded public outcomes."""
        if call.capability_id != self.continuation.id:
            async for frame in super().handle_input_stream(context, call, input_frames):
                yield frame
            return
        request_value = _OBJECT.validate_python(call.payload).get("request")
        receipt_query = (
            isinstance(request_value, dict)
            and request_value.get("contract") == "chat.continuation.receipt@0.1.0"
        )
        proof = (
            ReceiptProof.model_validate_json(json.dumps(call.payload))
            if receipt_query
            else ContinuationProof.model_validate_json(json.dumps(call.payload))
        )
        canonical = json.dumps(
            {
                "caller_id": proof.caller_id,
                "request": proof.request.model_dump(mode="json"),
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        permitted = False
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.keys.get(proof.caller_id, "0" * 64))
            ).verify(bytes.fromhex(proof.signature), canonical)
            permitted = proof.request.nonce not in self.nonces
        except InvalidSignature:
            pass
        key = proof.caller_id, proof.request.conversation_id
        if isinstance(proof, ContinuationProof) and proof.request.mode == "continue":
            permitted = (
                permitted
                and key in self.receipts
                and self.receipts[key]["handle"] == proof.request.handle
            )
        if not permitted:
            payload: dict[str, object] = (
                {"error": "scope_mismatch"}
                if receipt_query
                else {
                    "event": {"kind": "failed", "error": "policy_denied"},
                    "continuation_error": "scope_mismatch",
                }
            )
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload=payload,
            )
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=2,
                kind="completed",
            )
            return
        self.nonces.add(proof.request.nonce)
        raw = bytearray()
        closed = False
        async for frame in input_frames:
            if frame.kind == "chunk":
                assert isinstance(frame.media, InlineMediaAttachment)
                raw.extend(frame.media.data)
            if frame.kind == "completed":
                closed = True
        assert closed
        sequence = 1
        if isinstance(proof, ContinuationProof):
            assert len(raw) == proof.request.body.size_bytes
            assert hashlib.sha256(raw).hexdigest() == proof.request.body.sha256
            self.raw = bytes(raw)
            self.calls += 1
            receipt: dict[str, JsonValue] = {
                "contract": "chat.continuation@0.1.0",
                "conversation_id": proof.request.conversation_id,
                "request_nonce": "f" * 64 if self.bad_receipt else proof.request.nonce,
                "handle": "ct1_" + secrets.token_hex(32),
                "history": {"count": 2, "sha256": "a" * 64},
                "expires_at": time.time() + 300,
            }
            self.receipts[key] = receipt
            delta: dict[str, JsonValue] = {
                "content": "hello 🦊\n```python\nprint(1)\n```",
                "reasoning_content": "A short summary",
            }
            if self.use_tools:
                delta["tool_calls"] = [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=sequence,
                kind="chunk",
                payload={"event": {"kind": "delta", "delta": delta}},
            )
            sequence += 1
            payload = {
                "event": {
                    "kind": "completed",
                    "finish_reason": "tool_calls" if self.use_tools else "stop",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
                "receipt": receipt,
            }
        else:
            assert not raw
            payload = (
                {"receipt": self.receipts[key]}
                if key in self.receipts
                else {"error": "handle_unavailable"}
            )
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=sequence,
            kind="chunk",
            payload=payload,
        )
        if not self.transport_failure:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=sequence + 1,
                kind="completed",
            )


def sign(
    key: Ed25519PrivateKey, caller: str, request: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Create an independent client signature without invoking a Core signer."""
    value: dict[str, JsonValue] = {"caller_id": caller, "request": request}
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()
    return {**value, "signature": key.sign(raw).hex()}


def prepared(
    provider: ClientProvider, key: Ed25519PrivateKey, *, stream: bool = False
) -> tuple[bytes, dict[str, JsonValue], str]:
    """Bind a noncanonical Unicode HTTP envelope to one approved client key."""
    caller = "b" * 32
    provider.keys[caller] = key.public_key().public_bytes_raw().hex()
    raw = json.dumps(
        {
            "model": f"capability/{provider.chat.id}/code",
            "messages": [{"role": "user", "content": "hello 🦊"}],
            "stream": stream,
            "stream_options": {"include_usage": True},
        },
        ensure_ascii=False,
        indent=3,
    ).encode()
    request: dict[str, JsonValue] = {
        "contract": "chat.continuation@0.1.0",
        "mode": "start",
        "conversation_id": "c" * 64,
        "handle": None,
        "target": {
            "target_node": "fixture-transport",
            "callable_id": provider.continuation.id,
            "descriptor_revision": descriptor_revision(provider.continuation),
            "model": "code",
        },
        "body": {
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "media_type": "application/vnd.skulk.chat-request+json",
        },
        "nonce": secrets.token_hex(32),
        "expires_at": time.time() + 30,
    }
    proof = sign(key, caller, request)
    return (
        raw,
        proof,
        base64.urlsafe_b64encode(json.dumps(proof).encode()).decode().rstrip("="),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tools", [False, True])
async def test_unified_client_proof_exact_bytes_receipt_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, tools: bool
) -> None:
    """Chat, tools, summaries and usage share the existing adapter and recover safely."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    provider, key = ClientProvider(), Ed25519PrivateKey.generate()
    provider.use_tools = tools
    api = build_api(provider)
    raw, proof, header = prepared(provider, key, stream=stream)
    async with (
        api.providers(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app, client=("127.0.0.1", 5000)),
            base_url="http://localhost:52415",
            headers={
                "Origin": "http://localhost:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
        ) as client,
    ):
        listing = await client.get("/v1/models")
        assert '"proof_header":"Skulk-Continuation-Proof"' in listing.text
        response = await client.post(
            "/v1/chat/completions",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "Skulk-Continuation-Proof": header,
            },
        )
        assert response.status_code == 200, response.text
        assert (
            "skulk_continuation" in response.text and "A short summary" in response.text
        )
        assert (
            "total_tokens" in response.text and ("tool_calls" in response.text) == tools
        )
        assert provider.calls == 1 and provider.raw == raw
        request = proof["request"]
        assert isinstance(request, dict)
        lookup = sign(
            key,
            "b" * 32,
            {
                "contract": "chat.continuation.receipt@0.1.0",
                "conversation_id": request["conversation_id"],
                "target": request["target"],
                "request_nonce": request["nonce"],
                "nonce": secrets.token_hex(32),
                "expires_at": time.time() + 30,
            },
        )
        recovered = await client.post("/v1/chat/continuation/receipt", json=lookup)
        assert (
            recovered.status_code == 200
            and recovered.json()["receipt"]["request_nonce"] == request["nonce"]
        )
        assert provider.calls == 1
        replay = await client.post(
            "/v1/chat/completions",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "Skulk-Continuation-Proof": header,
            },
        )
        assert "scope_mismatch" in replay.text and provider.calls == 1
    assert api.active_provider_streams() == 0
    assert not (tmp_path / "inference" / "gateway.key").exists()


@pytest.mark.parametrize(
    "change",
    [
        "bytes",
        "signature",
        "caller",
        "target",
        "revision",
        "model",
        "deadline",
        "privacy",
        "origin",
        "relationship",
        "omitted_handle",
        "omitted_handle_resigned",
    ],
)
async def test_continuation_refuses_before_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Neither HTTP identity nor a changed target/body can grant continuation scope."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    provider, key = ClientProvider(), Ed25519PrivateKey.generate()
    raw, proof, header = prepared(provider, key)
    request = proof["request"]
    assert isinstance(request, dict)
    if change in {"target", "revision", "model"}:
        target = request["target"]
        assert isinstance(target, dict)
        target[
            {
                "target": "target_node",
                "revision": "descriptor_revision",
                "model": "model",
            }[change]
        ] = "d" * 16
    if change == "deadline":
        request["expires_at"] = time.time() - 1
    if change == "caller":
        proof["caller_id"] = "d" * 32
    if change == "signature":
        proof["signature"] = "0" * 128
    if change in {"omitted_handle", "omitted_handle_resigned"}:
        request.pop("handle")
        if change == "omitted_handle_resigned":
            proof = sign(key, "b" * 32, request)
    if change == "bytes":
        raw += b" "
    if change == "privacy":
        body = _OBJECT.validate_json(raw)
        raw = json.dumps({**body, "inference_policy": "local_only"}).encode()
        request["body"] = {
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "media_type": "application/vnd.skulk.chat-request+json",
        }
    if change == "relationship":
        provider.continuation = provider.continuation.model_copy(
            update={
                "annotations": {
                    **(provider.continuation.annotations or {}),
                    "inference.chat": "other.chat@0.2.0",
                }
            }
        )
    header = base64.urlsafe_b64encode(json.dumps(proof).encode()).decode().rstrip("=")
    api = build_api(provider)
    async with (
        api.providers(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app, client=("127.0.0.1", 5000)),
            base_url="http://localhost:52415",
            headers={
                "Origin": "http://untrusted"
                if change == "origin"
                else "http://localhost:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "Skulk-Continuation-Proof": header,
            },
        )
        assert (
            response.status_code in {400, 403, 409} or "scope_mismatch" in response.text
        ), response.text
        if change in {"omitted_handle", "omitted_handle_resigned"}:
            assert response.status_code == 400
        assert provider.calls == 0
    assert api.active_provider_streams() == 0


@pytest.mark.parametrize("failure", ["transport", "receipt"])
async def test_committed_receipt_is_withheld_on_invalid_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A semantic receipt alone cannot become a successful HTTP finish."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    provider, key = ClientProvider(), Ed25519PrivateKey.generate()
    provider.transport_failure = failure == "transport"
    provider.bad_receipt = failure == "receipt"
    raw, _, header = prepared(provider, key, stream=True)
    api = build_api(provider)
    async with (
        api.providers(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app, client=("127.0.0.1", 5000)),
            base_url="http://localhost:52415",
            headers={
                "Origin": "http://localhost:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "Skulk-Continuation-Proof": header,
            },
        )
        assert (
            "skulk_continuation" not in response.text and "[DONE]" not in response.text
        )
        assert '"error"' in response.text and provider.calls == 1


def test_detached_proof_rejects_ambiguous_encoding() -> None:
    """Only bounded canonical base64url and the exact bytes enter dispatch."""
    raw, _, header = prepared(ClientProvider(), Ed25519PrivateKey.generate())
    assert read_continuation_proof(header, raw).request.body.size_bytes == len(raw)
    for malformed in [header + "=", "!", "a" * 4097, "a"]:
        with pytest.raises(ValueError):
            read_continuation_proof(malformed, raw)


async def test_forwarded_stream_has_no_aggregate_generation_cap() -> None:
    """Streaming can forward more text than the collected-response memory guard."""

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        for sequence in range(1, 150):
            yield CapabilityStreamFrame(
                call_id="large",
                direction="provider_to_caller",
                sequence=sequence,
                kind="chunk",
                payload={"kind": "delta", "delta": {"content": "x" * 2048}},
            )
        yield CapabilityStreamFrame(
            call_id="large",
            direction="provider_to_caller",
            sequence=150,
            kind="chunk",
            payload={"kind": "completed", "finish_reason": "stop"},
        )
        yield CapabilityStreamFrame(
            call_id="large",
            direction="provider_to_caller",
            sequence=151,
            kind="completed",
        )

    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="large", ok=True, result={}),
        frames=frames(),
    )
    output = "".join(
        [
            part
            async for part in chat_response(
                session, model="fixture", stream=True, include_usage=False
            )
        ]
    )
    assert (
        output.count("x" * 2048) == 149
        and "[DONE]" in output
        and '"error"' not in output
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_semantic_refusal_drains_transport_before_releasing_the_owner(
    stream: bool,
) -> None:
    """A valid client refusal must not cancel a healthy child's terminal cleanup."""
    raw, _, header = prepared(ClientProvider(), Ed25519PrivateKey.generate())
    proof = read_continuation_proof(header, raw)
    observed, release = asyncio.Event(), asyncio.Event()
    completed = False

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        nonlocal completed
        observed.set()
        yield CapabilityStreamFrame(
            call_id="refused",
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={
                "event": {"kind": "failed", "error": "policy_denied"},
                "continuation_error": "authority_unavailable",
            },
        )
        await release.wait()
        completed = True
        yield CapabilityStreamFrame(
            call_id="refused",
            direction="provider_to_caller",
            sequence=2,
            kind="completed",
        )

    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="refused", ok=True, result={}),
        frames=frames(),
    )
    response = chat_response(
        session, model="fixture", stream=stream, include_usage=False, continuation=proof
    )
    if stream:
        assert '"role": "assistant"' in await anext(response)
    pending = asyncio.create_task(anext(response))
    try:
        await observed.wait()
        await asyncio.sleep(0)
        assert not pending.done()
    finally:
        release.set()
    result = await pending
    assert completed and "authority_unavailable" in result and "[DONE]" not in result
    assert [part async for part in response] == []
