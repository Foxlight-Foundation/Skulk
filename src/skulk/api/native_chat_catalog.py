"""Provider-neutral application bindings over caller-authorized native discovery.

Native compiled identities and application proof identities remain independent.
Annotations describe how to interpret a payload; only current host policy grants
permission, and every dispatch independently pins its observed native listing.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.api.inference_continuation import CONTINUATION_CONTRACT
from skulk.api.inference_gateway import (
    CONTRACT,
    STREAMED_CONTRACT,
    ChatCatalog,
    ChatModel,
)
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_host_protocol import ResponseFrame
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_hosts import InferenceListing

_OBJECT = TypeAdapter(dict[str, JsonValue])
ENVELOPE_CONTRACT = "skulk.application-envelope@1.0.0"


@final
class ApplicationBinding(BaseModel):
    """Explicit application envelope and independent signed proof destination."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    contract: str = Field(
        min_length=1, max_length=256, description="Opaque envelope contract."
    )
    callable_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[\w.:@-]+$",
        description="Application proof callable, not the native address.",
    )
    version: str = Field(
        min_length=1, max_length=64, description="Application proof version."
    )
    revision: str = Field(
        pattern=r"^[a-f0-9]{16}$",
        description="Independent application proof revision, never a truncated native identity.",
    )

    @classmethod
    def from_listing(cls, listing: InferenceListing) -> "ApplicationBinding":
        """Read an explicit signed envelope declaration or refuse the adapter."""
        hints = listing.descriptor.annotations or {}
        if hints.get("inference.envelope") != ENVELOPE_CONTRACT:
            raise ValueError("native application envelope is unsupported")
        return cls.model_validate(
            {
                "contract": hints.get("inference.envelope.contract"),
                "callable_id": hints.get("inference.application.id"),
                "version": hints.get("inference.application.version"),
                "revision": hints.get("inference.application.revision"),
            }
        )

    def wrap(self, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Bind an application payload without rewriting its signed bytes."""
        return {
            "contract": self.contract,
            "revision": self.revision,
            "payload": payload,
        }

    def unwrap(self, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Refuse another envelope or application revision before interpreting data."""
        if (
            set(payload) != {"contract", "revision", "payload"}
            or payload["contract"] != self.contract
            or payload["revision"] != self.revision
        ):
            raise ValueError("native application envelope differs")
        return _OBJECT.validate_python(payload["payload"], strict=True)


@final
@dataclass(frozen=True)
class NativeApplicationTarget:
    """One observed native destination with its independent application binding."""

    listing: InferenceListing
    application: ApplicationBinding


@final
@dataclass(frozen=True)
class NativeChatCatalog:
    """Fresh caller-filtered models and exact same-child stream destinations."""

    catalog: ChatCatalog
    chat: NativeApplicationTarget
    continuation: NativeApplicationTarget | None

    def model_id(self, model: ChatModel) -> str:
        """Expose a native transport address, never a legacy capability alias."""
        listing = self.chat.listing
        return f"native/{listing.plugin_id}/{listing.target.capability_id}@{listing.target.version}/{model.id}"

    def metadata(self, model: ChatModel) -> dict[str, JsonValue]:
        """Describe public model features and independently authorized continuation."""
        catalog = self.catalog
        metadata: dict[str, JsonValue] = {
            "contract": catalog.contract,
            "processing_domain": catalog.processing_domain,
            "usage": catalog.usage,
            "usage_label": catalog.usage_label,
            "usage_url": catalog.usage_url,
            "expires_at": catalog.expires_at,
            "features": list(model.features),
        }
        if self.continuation is not None:
            binding = self.continuation.application
            metadata["continuation"] = {
                "contract": CONTINUATION_CONTRACT,
                "authorization": "caller.ed25519@0.1.0",
                "callable_version": binding.version,
                "proof_header": "Skulk-Continuation-Proof",
                "receipt_path": "/v1/chat/continuation/receipt",
                "native_model": self.model_id(model),
                "target": {
                    "target_node": catalog.target_node,
                    "callable_id": binding.callable_id,
                    "descriptor_revision": binding.revision,
                    "model": model.id,
                },
            }
        return metadata


def _same_child(left: InferenceListing, right: InferenceListing) -> bool:
    return (
        left.plugin_id,
        left.runtime_digest,
        left.incarnation,
        left.target.node_id,
    ) == (
        right.plugin_id,
        right.runtime_digest,
        right.incarnation,
        right.target.node_id,
    )


async def native_chat_catalogs(
    broker: NativeInferenceCalls,
    caller: AuthenticatedInferenceCaller,
    *,
    for_receipt: bool = False,
) -> tuple[NativeChatCatalog, ...]:
    """Read bounded metadata for this actual caller, without an owner-wide cache.

    Metadata and selected inference destinations each require current permission.
    Receipt-only reads require continuation permission instead of permission to
    generate new gateway chat. No provider generation is retried here.
    """
    deadline = asyncio.get_running_loop().time() + 6
    async with asyncio.timeout_at(deadline):
        listings = await broker.directory.directory(streaming=True)
    observed: dict[int, NativeChatCatalog] = {}

    def permitted(listing: InferenceListing) -> bool:
        try:
            broker.callers.authorize(caller, listing.grant())
            return True
        except (OSError, ValueError):
            return False

    async def read(index: int, metadata: InferenceListing) -> None:
        if not permitted(metadata):
            return
        try:
            envelope = ApplicationBinding.from_listing(metadata)
            result = await broker.invoke(
                caller,
                metadata.plugin_id,
                metadata.target.capability_id,
                metadata.target.version,
                json.dumps(envelope.wrap({})).encode(),
                5,
                expected_listing=metadata,
            )
            try:
                if (
                    not isinstance(result.frame, ResponseFrame)
                    or result.frame.outcome != "complete"
                ):
                    return
                catalog = ChatCatalog.model_validate_json(
                    json.dumps(envelope.unwrap(_OBJECT.validate_json(result.payload)))
                )
            finally:
                result.release()
            if catalog.expires_at <= time.time() or catalog.contract != (
                metadata.descriptor.annotations or {}
            ).get("inference.catalog"):
                return
            matches: list[NativeApplicationTarget] = []
            for item in listings:
                if not _same_child(metadata, item):
                    continue
                hints = item.descriptor.annotations or {}
                if (
                    item.descriptor.behavior != "inference"
                    or item.descriptor.processing_domain != catalog.processing_domain
                    or hints.get("inference.contract") != catalog.contract
                    or hints.get("inference.authorization")
                    != (
                        "gateway.ed25519@0.2.0"
                        if catalog.contract == STREAMED_CONTRACT
                        else "gateway.ed25519@0.1.0"
                    )
                    or item.descriptor.io_mode
                    != (
                        "bidirectional"
                        if catalog.contract == STREAMED_CONTRACT
                        else "server_streaming"
                    )
                ):
                    continue
                application = ApplicationBinding.from_listing(item)
                if (
                    application.contract,
                    application.callable_id,
                    application.version,
                    application.revision,
                ) == (
                    envelope.contract,
                    catalog.callable_id,
                    catalog.callable_version,
                    catalog.descriptor_revision,
                ):
                    matches.append(NativeApplicationTarget(item, application))
            if len(matches) != 1:
                return
            chat = matches[0]
            continuation: NativeApplicationTarget | None = None
            continuations: list[NativeApplicationTarget] = []
            if catalog.contract == STREAMED_CONTRACT:
                for item in listings:
                    hints = item.descriptor.annotations or {}
                    if (
                        not _same_child(metadata, item)
                        or item.descriptor.behavior != "inference"
                        or item.descriptor.processing_domain
                        != catalog.processing_domain
                        or item.descriptor.io_mode != "bidirectional"
                        or hints.get("inference.contract") != CONTINUATION_CONTRACT
                        or hints.get("inference.authorization")
                        != "caller.ed25519@0.1.0"
                        or hints.get("inference.chat")
                        != f"{catalog.callable_id}@{catalog.callable_version}"
                    ):
                        continue
                    application = ApplicationBinding.from_listing(item)
                    if (
                        application.contract == envelope.contract
                        and application.version == "0.1.0"
                        and permitted(item)
                    ):
                        continuations.append(NativeApplicationTarget(item, application))
                if len(continuations) == 1:
                    continuation = continuations[0]
            if (
                not permitted(metadata)
                or (for_receipt and continuation is None)
                or (not for_receipt and not permitted(chat.listing))
            ):
                return
            observed[index] = NativeChatCatalog(catalog, chat, continuation)
        except (OSError, ValueError, TimeoutError):
            # A failed metadata read withdraws only this candidate. It cannot
            # authorize a legacy fallback or replay an inference invocation.
            return

    try:
        async with asyncio.timeout_at(deadline):
            async with asyncio.TaskGroup() as tasks:
                for index, item in enumerate(
                    [
                        listing
                        for listing in listings
                        if listing.descriptor.io_mode == "unary"
                        and listing.descriptor.behavior == "read"
                        and (listing.descriptor.annotations or {}).get(
                            "inference.catalog"
                        )
                        in {CONTRACT, STREAMED_CONTRACT}
                    ][:8]
                ):
                    tasks.create_task(read(index, item))
    except TimeoutError:
        pass
    selected: dict[tuple[str, str], NativeChatCatalog] = {}
    for _, value in sorted(observed.items()):
        if value.catalog.expires_at <= time.time():
            continue
        listing = value.chat.listing
        key = (listing.plugin_id, listing.target.capability_id)
        if key not in selected or value.catalog.contract == STREAMED_CONTRACT:
            selected[key] = value
    return tuple(selected.values())
