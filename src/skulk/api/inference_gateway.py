"""Experimental, provider-neutral chat adapter for explicitly delegated capabilities.

No account, provider endpoint, OAuth, or upstream model behavior belongs here.
The managed capability owns those concerns and independently authorizes each
signed request. This adapter initially serves direct owner requests on its host.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import stat
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Literal, cast, final
from uuid import uuid4

import anyio
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
)

from skulk.extensions.streams import CapabilityStreamFrame, CapabilityStreamSession

_OBJECT = TypeAdapter(dict[str, JsonValue])
CONTRACT = "chat.completions@0.1.0"


class ChatModel(BaseModel):
    """One provider-owned opaque model slug and human-readable name."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )
    id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.:@-]+$")
    name: str = Field(min_length=1, max_length=256)
    features: tuple[
        Literal["image_input", "function_tools", "reasoning_toggle"], ...
    ] = Field(default=(), max_length=3)


class ChatCatalog(BaseModel):
    """Cached provider claims, pinned to an exact callable and owner policy."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )
    contract: Literal["chat.completions@0.1.0"]
    target_node: str = Field(min_length=1, max_length=128, pattern=r"^[\w.:@-]+$")
    callable_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.:@-]+$")
    callable_version: Literal["0.1.0"]
    descriptor_revision: str = Field(pattern=r"^[a-f0-9]{16}$")
    catalog_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    expires_at: float = Field(gt=0)
    models: tuple[ChatModel, ...] = Field(max_length=128)
    processing_domain: Literal["local", "private_site", "external"]
    usage: Literal["none", "included_only", "credits_possible", "unknown"]
    usage_label: str = Field(max_length=80)
    usage_url: str | None = Field(default=None, max_length=256)

    @field_validator("usage_url")
    @classmethod
    def safe_usage_url(cls, value: str | None) -> str | None:
        """Allow only HTTPS settings links in the dashboard's provider metadata."""
        from urllib.parse import urlsplit

        if value is not None:
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError("usage URL must be HTTPS without credentials")
        return value

    def model_id(self, model: ChatModel) -> str:
        """Namespace models by capability callable rather than impersonating a card."""
        return f"capability/{self.callable_id}/{model.id}"


@final
class GatewaySigner:
    """Persist a private gateway key locally; providers receive only public proof."""

    def __init__(self, path: Path) -> None:
        """Retain the key path; constructing an API does not create a key."""
        self.path = path

    def _key(self) -> Ed25519PrivateKey:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as target:
                target.write(Ed25519PrivateKey.generate().private_bytes_raw())
                target.flush()
                os.fsync(target.fileno())
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077
                or metadata.st_uid != os.getuid()
            ):
                raise ValueError("gateway signing key is not owner protected")
            raw = source.read(33)
        if len(raw) != 32:
            raise ValueError("invalid gateway key")
        return Ed25519PrivateKey.from_private_bytes(raw)

    def public_key(self) -> str:
        """Read the stable public key without returning private material."""
        return self._key().public_key().public_bytes_raw().hex()

    def sign(
        self, catalog: ChatCatalog, model: str, request: dict[str, JsonValue]
    ) -> dict[str, object]:
        """Bind exact request, target, policy revision, nonce and short expiry."""
        key = self._key()
        invocation: dict[str, JsonValue] = {
            "contract": CONTRACT,
            "public_key": key.public_key().public_bytes_raw().hex(),
            "target_node": catalog.target_node,
            "callable_id": catalog.callable_id,
            "descriptor_revision": catalog.descriptor_revision,
            "catalog_revision": catalog.catalog_revision,
            "model": model,
            "request": request,
            "nonce": secrets.token_hex(32),
            "expires_at": time.time() + 30,
        }
        encoded = json.dumps(
            invocation,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        if len(encoded) > 49152:
            raise HTTPException(
                413, "Capability chat request exceeds the signed input bound"
            )
        return {"invocation": invocation, "signature": key.sign(encoded).hex()}


def _failure(code: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"Capability inference failed: {code}",
            "type": "capability_error",
            "code": code,
        }
    }


def _sse(value: dict[str, JsonValue]) -> str:
    return "data: " + json.dumps(value, allow_nan=False) + "\n\n"


async def _owned_frames(
    session: CapabilityStreamSession,
) -> AsyncGenerator[CapabilityStreamFrame]:
    try:
        async for frame in session.frames:
            yield frame
    finally:
        # ASGI disconnect runs under a cancelled scope. Shield bounded cleanup
        # so provider cancellation and local receiver retirement still happen.
        with anyio.CancelScope(shield=True):
            close = cast(
                Callable[[], Awaitable[None]] | None,
                getattr(session.frames, "aclose", None),
            )
            if close is not None:
                await close()
            if session.cancel_output is not None:
                await asyncio.wait_for(session.cancel_output(), 3)


async def chat_response(
    session: CapabilityStreamSession,
    *,
    model: str,
    stream: bool,
    include_usage: bool,
) -> AsyncIterator[str]:
    """Adapt a cancellable capability stream to the existing chat wire format.

    Success requires both a common completion event and transport completion.
    Mid-stream failures emit a safe error without a success finish or DONE.
    Closing this iterator closes the provider session and cancels its call.
    """
    identifier, created = "chatcmpl-" + uuid4().hex, int(time.time())
    content = ""
    reasoning = ""
    tools: dict[int, dict[str, JsonValue]] = {}
    terminal: dict[str, JsonValue] | None = None
    finished = False

    def chunk(
        delta: dict[str, JsonValue], finish_reason: JsonValue = None
    ) -> dict[str, JsonValue]:
        return {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    async with aclosing(_owned_frames(session)) as frames:
        if stream:
            yield _sse(chunk({"role": "assistant"}))
        async for frame in frames:
            if frame.kind == "failed":
                yield (
                    _sse(_failure("provider_error"))
                    if stream
                    else json.dumps(_failure("provider_error"))
                )
                return
            if frame.kind == "completed":
                finished = True
                break
            if frame.kind != "chunk" or frame.payload is None:
                continue
            event = _OBJECT.validate_python(frame.payload)
            kind = event.get("kind")
            if terminal is not None or kind not in ("delta", "completed", "failed"):
                yield (
                    _sse(_failure("invalid_stream"))
                    if stream
                    else json.dumps(_failure("invalid_stream"))
                )
                return
            if kind == "failed":
                code = event.get("error")
                safe = (
                    code
                    if code
                    in (
                        "policy_denied",
                        "invalid_request",
                        "upstream_failed",
                        "incomplete",
                        "interrupted",
                        "invalid_stream",
                        "output_limit",
                        "usage_limit",
                        "usage_unavailable",
                    )
                    else "provider_error"
                )
                yield (
                    _sse(_failure(str(safe)))
                    if stream
                    else json.dumps(_failure(str(safe)))
                )
                return
            if kind == "completed":
                if event.get("finish_reason") not in ("stop", "tool_calls"):
                    yield (
                        _sse(_failure("invalid_stream"))
                        if stream
                        else json.dumps(_failure("invalid_stream"))
                    )
                    return
                terminal = event
                continue
            delta = event.get("delta")
            if not isinstance(delta, dict):
                yield (
                    _sse(_failure("invalid_stream"))
                    if stream
                    else json.dumps(_failure("invalid_stream"))
                )
                return
            if stream:
                yield _sse(chunk(delta))
            else:
                text, thought = delta.get("content"), delta.get("reasoning_content")
                if isinstance(text, str):
                    content += text
                if isinstance(thought, str):
                    reasoning += thought
                calls = delta.get("tool_calls")
                if isinstance(calls, list):
                    for call in calls:
                        if not isinstance(call, dict) or not isinstance(
                            call.get("index"), int
                        ):
                            continue
                        index = call["index"]
                        assert isinstance(index, int)
                        saved = tools.setdefault(
                            index,
                            {
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if isinstance(call.get("id"), str):
                            saved["id"] = call["id"]
                        function, existing = call.get("function"), saved["function"]
                        if isinstance(function, dict) and isinstance(existing, dict):
                            for field in ("name", "arguments"):
                                addition, before = (
                                    function.get(field),
                                    existing.get(field, ""),
                                )
                                if isinstance(addition, str) and isinstance(
                                    before, str
                                ):
                                    existing[field] = before + addition
            if len(content) + len(reasoning) + len(json.dumps(tools)) > 262144:
                yield json.dumps(_failure("output_limit"))
                return
        if not finished or terminal is None:
            yield (
                _sse(_failure("interrupted"))
                if stream
                else json.dumps(_failure("interrupted"))
            )
            return
        usage = terminal.get("usage")
        if stream:
            yield _sse(chunk({}, terminal["finish_reason"]))
            if include_usage and isinstance(usage, dict):
                yield _sse(
                    {
                        "id": identifier,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
            yield "data: [DONE]\n\n"
        else:
            message: dict[str, JsonValue] = {
                "role": "assistant",
                "content": content or None,
            }
            if reasoning:
                message["reasoning_content"] = reasoning
            if tools:
                message["tool_calls"] = [tools[index] for index in sorted(tools)]
            result: dict[str, JsonValue] = {
                "id": identifier,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": terminal["finish_reason"],
                    }
                ],
            }
            if isinstance(usage, dict):
                result["usage"] = usage
            yield json.dumps(result, allow_nan=False)
