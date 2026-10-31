"""The unified chat API routes provider-neutral models under explicit authority."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import JsonValue, TypeAdapter

from skulk.api.inference_gateway import GatewaySigner, chat_response
from skulk.api.main import API
from skulk.api.types import ModelList
from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityStreamFrame,
    ExtensionContext,
    LoadedExtensions,
    descriptor_revision,
)
from skulk.extensions.calls import CapabilityResult
from skulk.extensions.streams import CapabilityStreamSession
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import TelemetryView
from skulk.utils.channels import channel


class ChatFixture:
    """A second provider with no OpenAI dependency or account implementation."""

    name = "independent-chat-fixture"
    skulk_requires = ">=0"

    def __init__(self, namespace: str, key: str, *, delay: float = 0) -> None:
        """Require a separately owner-delegated gateway key."""
        self.namespace, self.key = namespace, key
        self.name = namespace
        self.delay = delay
        self.local = namespace == "different.inference"
        self.calls = 0
        self.closed = False
        self.nonces: set[str] = set()
        self.chat = CapabilityDescriptor(
            id=namespace + ".chat",
            version="0.1.0",
            title="Independent chat",
            description="Credential-free delegated chat fixture.",
            input_schema={"type": "object"},
            io_mode="server_streaming",
            output_schema={"type": "object", "additionalProperties": False},
            output_chunk_schema={"type": "object"},
            annotations={
                "inference.contract": "chat.completions@0.1.0",
                "inference.authorization": "gateway.ed25519@0.1.0",
            },
        )
        self.discovery = CapabilityDescriptor(
            id=namespace + ".catalog",
            version="0.1.0",
            title="Independent catalog",
            description="Cached fixture model choices.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            annotations={"inference.catalog": "chat.completions@0.1.0"},
        )

    def chat_middleware(self) -> None:
        """The fixture does not intercept native chat."""
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        """Publish the same open inference contract using an unrelated namespace."""
        return [self.discovery, self.chat]

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        """Publish fresh fixture facts without making a provider request."""
        await asyncio.sleep(self.delay)
        return {
            "contract": "chat.completions@0.1.0",
            "target_node": "fixture-transport",
            "callable_id": self.chat.id,
            "callable_version": "0.1.0",
            "descriptor_revision": descriptor_revision(self.chat),
            "catalog_revision": "a" * 64,
            "expires_at": time.time() + 60,
            "models": [{"id": "code", "name": "Fixture Code"}],
            "processing_domain": "local" if self.local else "external",
            "usage": "none" if self.local else "unknown",
            "usage_label": "Owner-delegated fixture",
            "usage_url": None,
        }

    async def handle_stream(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> AsyncGenerator[CapabilityStreamFrame]:
        """Validate the actual gateway signature before yielding code or tool output."""
        proof = TypeAdapter(dict[str, JsonValue]).validate_python(call.payload)
        invocation, signature = proof["invocation"], proof["signature"]
        assert isinstance(invocation, dict) and isinstance(signature, str)
        assert (
            invocation["public_key"] == self.key
            and invocation["target_node"] == "fixture-transport"
        )
        assert invocation["callable_id"] == self.chat.id and invocation[
            "descriptor_revision"
        ] == descriptor_revision(self.chat)
        nonce = invocation["nonce"]
        assert isinstance(nonce, str) and nonce not in self.nonces
        self.nonces.add(nonce)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.key)).verify(
            bytes.fromhex(signature),
            json.dumps(
                invocation,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
        )
        self.calls += 1
        try:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={
                    "kind": "delta",
                    "delta": {
                        "content": "Here is code.\n```python\nprint('hello')\n```"
                    },
                },
            )
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=2,
                kind="chunk",
                payload={
                    "kind": "completed",
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
            )
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=3,
                kind="completed",
                payload={},
            )
        finally:
            self.closed = True


class InferenceTestAPI(API):
    """Exercise the ordinary provider workers without starting unrelated services."""

    @asynccontextmanager
    async def providers(self) -> AsyncGenerator[None]:
        """Run the API's real task group and local provider-data subscription."""
        async with self._tg as tasks:
            tasks.start_soon(self._apply_provider_data)
            try:
                yield
            finally:
                tasks.cancel_scope.cancel()


def build_api(provider: ChatFixture, *others: ChatFixture) -> InferenceTestAPI:
    """Build an actual API with its ordinary provider data path and no shared state."""
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    sender, receiver = channel[ProviderStreamPacket](256)
    return InferenceTestAPI(
        NodeId("fixture-api"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        telemetry_view=TelemetryView(),
        provider_stream_sender=sender,
        provider_stream_receiver=receiver,
        extensions=LoadedExtensions([provider, *others]),
    )


@pytest.mark.parametrize("namespace", ["fixture.remote", "different.inference"])
@pytest.mark.parametrize("stream", [False, True])
async def test_public_chat_route_roundtrip_and_direct_owner_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, namespace: str, stream: bool
) -> None:
    """Two unrelated providers use the same catalog, signature and wire adapter."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    signer = GatewaySigner(tmp_path / "inference" / "gateway.key")
    fixture = ChatFixture(namespace, signer.public_key())
    api = build_api(fixture)
    # Start the real API lifespan and its provider receive/dispatch workers.
    async with (
        api.providers(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app, client=("127.0.0.1", 5000)),
            base_url="http://localhost:52415",
            headers={
                "Origin": "http://localhost:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
            timeout=10,
        ) as client,
    ):
        response = await client.get("/v1/inference/gateway")
        assert response.json()["public_key"] == fixture.key

        # Inspect catalog through the actual endpoint while keeping native
        # model-store discovery out of this provider-only fixture.
        listing = await client.get("/v1/models")
        model = f"capability/{namespace}.chat/code"
        assert model in [
            item.id for item in ModelList.model_validate_json(listing.content).data
        ]
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Write code"}],
            "stream": stream,
            "stream_options": {"include_usage": True},
        }
        denied = await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Origin": "http://untrusted"},
        )
        assert denied.status_code == 403 and fixture.calls == 0
        local_only = await client.post(
            "/v1/chat/completions", json={**body, "inference_policy": "local_only"}
        )
        if fixture.local:
            assert local_only.status_code == 200 and fixture.calls == 1
            included = await client.post(
                "/v1/chat/completions",
                json={**body, "inference_policy": "included_only"},
            )
            assert included.status_code == 200 and fixture.calls == 2
        else:
            assert local_only.status_code == 403 and fixture.calls == 0
        before = fixture.calls
        unknown = await client.post(
            "/v1/chat/completions", json={**body, "background": True}
        )
        assert unknown.status_code == 400 and fixture.calls == before
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        if stream:
            assert "print('hello')" in response.text and "[DONE]" in response.text
            assert '"total_tokens": 3' in response.text
        else:
            collected = TypeAdapter(dict[str, JsonValue]).validate_json(
                response.content
            )
            assert "print('hello')" in json.dumps(collected)
            assert response.json()["usage"]["total_tokens"] == 3
        assert fixture.calls == (3 if fixture.local else 1) and fixture.closed


def test_signer_key_stays_private_and_stable(tmp_path: Path) -> None:
    """Restart retains only gateway identity; it never needs account credentials."""
    path = tmp_path / "gateway.key"
    signer = GatewaySigner(path)
    public = signer.public_key()
    assert GatewaySigner(path).public_key() == public
    assert path.stat().st_mode & 0o077 == 0
    path.chmod(0o644)
    with pytest.raises(ValueError):
        signer.public_key()


@pytest.mark.parametrize("healthy_first", [True, False])
async def test_slow_catalogs_do_not_withdraw_a_healthy_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, healthy_first: bool
) -> None:
    """Overall discovery expiry retains healthy models for listing and chat."""
    monkeypatch.setattr("skulk.api.main.SKULK_DATA_HOME", tmp_path)
    signer = GatewaySigner(tmp_path / "inference" / "gateway.key")
    healthy = ChatFixture("healthy.inference", signer.public_key())
    slow = [ChatFixture(f"slow{i}.inference", healthy.key, delay=2) for i in range(4)]
    api = build_api(healthy, *slow) if healthy_first else build_api(*slow, healthy)
    async with (
        api.providers(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app, client=("127.0.0.1", 5000)),
            base_url="http://localhost:52415",
            headers={
                "Origin": "http://localhost:52415",
                "X-Skulk-Dashboard": "pairing-v1",
            },
            timeout=10,
        ) as client,
    ):
        listing = ModelList.model_validate_json(
            (await client.get("/v1/models")).content
        )
        model = "capability/healthy.inference.chat/code"
        assert model in [item.id for item in listing.data]
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Write code"}],
            },
        )
        assert response.status_code == 200 and "print('hello')" in response.text
        assert healthy.calls == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_output_bound_precedes_forwarding_and_closes_the_provider(
    stream: bool,
) -> None:
    """Oversized deltas fail in both wire modes without forwarding their content."""
    closed = False
    cancelled = False

    async def frames() -> AsyncGenerator[CapabilityStreamFrame]:
        nonlocal closed
        try:
            yield CapabilityStreamFrame(
                call_id="bounded",
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"kind": "delta", "delta": {"content": "x" * 300000}},
            )
            raise AssertionError("oversized output must close the provider iterator")
        finally:
            closed = True

    async def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="bounded", ok=True, result={}),
        frames=frames(),
        cancel_output=cancel,
    )
    output = "".join(
        [
            part
            async for part in chat_response(
                session, model="fixture", stream=stream, include_usage=True
            )
        ]
    )
    assert "output_limit" in output and "[DONE]" not in output and len(output) < 1024
    assert closed and cancelled


async def test_close_after_role_cancels_without_reading_a_provider_frame() -> None:
    """Closing the first SSE event retires an already admitted provider call."""
    closed = False
    cancelled = False

    class UnreadFrames:
        """Record receiver closure without allowing the test to consume output."""

        def __aiter__(self) -> "UnreadFrames":
            return self

        async def __anext__(self) -> CapabilityStreamFrame:
            raise AssertionError("the role event must not consume a provider frame")

        async def aclose(self) -> None:
            """Retire the unopened local receiver."""
            nonlocal closed
            closed = True

    async def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    session = CapabilityStreamSession(
        open_result=CapabilityResult(call_id="unread", ok=True, result={}),
        frames=UnreadFrames(),
        cancel_output=cancel,
    )
    response = chat_response(
        session, model="fixture", stream=True, include_usage=False
    )
    assert '"role": "assistant"' in await anext(response)
    await response.aclose()
    assert closed and cancelled
