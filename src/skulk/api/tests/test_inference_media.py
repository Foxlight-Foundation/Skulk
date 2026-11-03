"""Provider-neutral media signing and the actual unified API duplex data path."""

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter

from skulk.api.adapters.chat_completions import chat_request_to_text_generation
from skulk.api.inference_gateway import (
    CHAT_BODY_MEDIA_TYPE,
    MAX_CHAT_BODY_BYTES,
    ChatCatalog,
    GatewaySigner,
    send_chat_body,
)
from skulk.api.tests.test_inference_gateway import ChatFixture, build_api
from skulk.api.types import ChatCompletionRequest, ModelList
from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityStreamFrame,
    ExtensionContext,
)
from skulk.extensions.calls import CapabilityResult
from skulk.extensions.streams import (
    CapabilityStreamInput,
    CapabilityStreamSession,
    InlineMediaAttachment,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])


class MediaFixture(ChatFixture):
    """An independent provider that verifies media before producing any output."""

    def __init__(self, key: str, *, reject: bool = False) -> None:
        """Publish a versioned callable without importing a vendor adapter."""
        super().__init__("independent.media", key)
        self.reject = reject
        self.received: dict[str, JsonValue] | None = None
        self.upload_started = asyncio.Event()
        self.release_upload = asyncio.Event()
        self.release_upload.set()
        self.chat = CapabilityDescriptor(
            id=self.namespace + ".chat",
            version="0.2.0",
            title="Independent media chat",
            description="Verify an exact signed body through the public data path.",
            input_schema={"type": "object"},
            input_chunk_schema={"type": "object", "additionalProperties": False},
            io_mode="bidirectional",
            output_schema={"type": "object"},
            output_chunk_schema={"type": "object"},
            annotations={
                "inference.contract": "chat.completions@0.2.0",
                "inference.authorization": "gateway.ed25519@0.2.0",
            },
        )
        self.discovery = CapabilityDescriptor(
            id=self.namespace + ".catalog-v2",
            version="0.2.0",
            title="Independent media catalog",
            description="Cached media-capable choices.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            annotations={"inference.catalog": "chat.completions@0.2.0"},
        )

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        """Reuse independent fixture metadata under the newer input version."""
        value = await super().handle_call(context, call)
        if value:
            value.update(
                contract="chat.completions@0.2.0",
                callable_version="0.2.0",
                maximum_body_bytes=MAX_CHAT_BODY_BYTES,
            )
        return value

    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncGenerator[CapabilityStreamFrame]:
        """Verify media, then use the common signature/text/terminal fixture."""
        proof = _OBJECT.validate_python(call.payload)
        invocation = proof["invocation"]
        assert isinstance(invocation, dict)
        reference = invocation["body"]
        assert isinstance(reference, dict)
        raw = bytearray()
        completed = False
        async for frame in input_frames:
            if frame.kind == "chunk":
                self.upload_started.set()
                await self.release_upload.wait()
                if self.reject:
                    yield CapabilityStreamFrame(
                        call_id=call.call_id,
                        direction="provider_to_caller",
                        sequence=1,
                        kind="chunk",
                        payload={"kind": "failed", "error": "policy_denied"},
                    )
                    yield CapabilityStreamFrame(
                        call_id=call.call_id,
                        direction="provider_to_caller",
                        sequence=2,
                        kind="completed",
                    )
                    return
                assert isinstance(frame.media, InlineMediaAttachment)
                assert frame.media.media_type == CHAT_BODY_MEDIA_TYPE
                raw.extend(frame.media.data)
                assert len(raw) <= MAX_CHAT_BODY_BYTES
            elif frame.kind == "completed":
                completed = True
        assert completed and len(raw) == reference["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
        self.received = _OBJECT.validate_json(raw)
        async for frame in super().handle_stream(context, call):
            yield frame


def request_body(*, stream: bool) -> dict[str, object]:
    """Attach a synthetic document larger than the legacy signing ceiling."""
    return {
        "model": "capability/independent.media.chat/code",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "example.txt",
                            "file_data": "data:text/plain;base64,"
                            + base64.b64encode(b"x" * 1_100_000).decode(),
                        },
                    }
                ],
            }
        ],
        "stream": stream,
    }


@pytest.mark.parametrize("stream", [False, True])
async def test_unified_api_transfers_large_document_and_keeps_metadata_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """The real provider DATA worker delivers exact bytes before generation."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    signer = GatewaySigner(tmp_path / "inference" / "gateway.key")
    fixture = MediaFixture(signer.public_key())
    fixture.release_upload.clear()
    api = build_api(fixture)
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
        task = asyncio.create_task(
            client.post("/v1/chat/completions", json=request_body(stream=stream))
        )
        try:
            await asyncio.wait_for(fixture.upload_started.wait(), 5)
            listing = await client.get("/v1/models")
            assert listing.status_code == 200 and fixture.calls == 0
            fixture.release_upload.set()
            response = await asyncio.wait_for(task, 5)
            assert response.status_code == 200, response.text
            if stream:
                assert "data: [DONE]" in response.text
            else:
                assert (
                    "```python" in response.json()["choices"][0]["message"]["content"]
                )
            assert fixture.calls == 1 and fixture.received is not None
            assert api.active_provider_streams() == 0
        finally:
            fixture.release_upload.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("content", ["single", "list"])
async def test_native_adapter_refuses_files(content: str) -> None:
    """New generic input types cannot silently disappear in native generation."""
    body = request_body(stream=False)
    body["model"] = "native/example"
    messages = body["messages"]
    assert isinstance(messages, list)
    if content == "single":
        messages[0]["content"] = messages[0]["content"][0]
    request = ChatCompletionRequest.model_validate_json(json.dumps(body))
    with pytest.raises(HTTPException) as failure:
        await chat_request_to_text_generation(request)
    assert failure.value.status_code == 400


def test_signer_enforces_advertised_input_bound_and_exact_media_digest(
    tmp_path: Path,
) -> None:
    """A provider cannot expand the gateway's maximum or escape the signature."""
    catalog = ChatCatalog(
        contract="chat.completions@0.2.0",
        callable_version="0.2.0",
        target_node="fixture",
        callable_id="fixture.chat",
        descriptor_revision="a" * 16,
        catalog_revision="b" * 64,
        expires_at=100.0,
        models=(),
        processing_domain="external",
        usage="unknown",
        usage_label="Unknown",
        maximum_body_bytes=100,
    )
    signer = GatewaySigner(tmp_path / "key")
    proof, raw = signer.prepare(catalog, "code", {"messages": []})
    assert raw == b'{"messages":[]}'
    invocation = proof["invocation"]
    assert isinstance(invocation, dict)
    assert invocation["body"]["sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(HTTPException) as failure:
        signer.prepare(catalog, "code", {"messages": "x" * 100})
    assert failure.value.status_code == 413


async def test_disconnected_upload_retires_input_and_provider() -> None:
    """A disconnect before half-close cannot become admitted generation."""
    frames: list[CapabilityStreamFrame] = []
    cancelled = False

    async def send(frame: CapabilityStreamFrame) -> None:
        frames.append(frame)

    sink = CapabilityStreamInput(
        call_id="upload", deadline_at=anyio.current_time() + 10, send_frame=send
    )
    await sink.start()

    async def empty() -> AsyncGenerator[CapabilityStreamFrame]:
        if False:
            yield frames[0]

    async def cancel() -> None:
        nonlocal cancelled
        cancelled = True
        sink.close_locally()

    async def disconnected() -> bool:
        return len(frames) > 1

    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="upload", ok=True, result={}),
        frames=empty(),
        input=sink,
        cancel_output=cancel,
    )
    with pytest.raises(HTTPException) as failure:
        await send_chat_body(session, b"x" * 1_100_000, disconnected)
    assert failure.value.status_code == 499
    assert cancelled and sink.closed
    assert [frame.kind for frame in frames] == ["started", "chunk"]


async def test_dual_catalog_negotiation_keeps_one_stable_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider retains v1 without duplicating models or receiving retries."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    signer = GatewaySigner(tmp_path / "inference" / "gateway.key")
    fixture = MediaFixture(signer.public_key())
    legacy = ChatFixture(fixture.namespace, signer.public_key())
    legacy.name = "independent-legacy"
    api = build_api(legacy, fixture)
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
        listing = ModelList.model_validate_json(
            (await client.get("/v1/models")).content
        ).data
        model = "capability/independent.media.chat/code"
        assert len([item for item in listing if item.id == model]) == 1
        response = await client.post(
            "/v1/chat/completions", json=request_body(stream=False)
        )
        assert response.status_code == 200 and fixture.calls == 1 and legacy.calls == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_provider_input_refusal_has_no_success_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """Early refusal cannot fabricate a successful completion or replay."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    signer = GatewaySigner(tmp_path / "inference" / "gateway.key")
    fixture = MediaFixture(signer.public_key(), reject=True)
    api = build_api(fixture)
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
            "/v1/chat/completions", json=request_body(stream=stream)
        )
        assert "[DONE]" not in response.text and "finish_reason" not in response.text
        assert "error" in response.text or response.status_code == 503
        assert fixture.calls == 0 and api.active_provider_streams() == 0
