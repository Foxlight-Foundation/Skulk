"""Unified model/chat/receipt adapters with independent native caller authority."""

import asyncio

from fastapi import HTTPException, Request
from pydantic import JsonValue, TypeAdapter

from skulk.api.inference_access import authenticate_inference_request
from skulk.api.inference_continuation import (
    ContinuationTarget,
    ReceiptProof,
    check_admission_deadline,
    read_continuation_proof,
)
from skulk.api.inference_gateway import MAX_CHAT_BODY_BYTES, GatewaySigner
from skulk.api.native_chat_catalog import NativeChatCatalog, native_chat_catalogs
from skulk.api.native_chat_stream import NativeChatResponse
from skulk.api.native_inference_service import NativeInferenceService
from skulk.api.types import ChatCompletionRequest, ModelListModel
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_inference_calls import NativeInferenceCalls

_OBJECT = TypeAdapter(dict[str, JsonValue])


class NativeChatGateway:
    """Keep generic application adaptation separate from provider implementation."""

    def __init__(self, service: NativeInferenceService, signer: GatewaySigner) -> None:
        """Use the API-owned broker and its existing application gateway signer."""
        self.service, self.signer = service, signer

    async def _caller(
        self, request: Request
    ) -> tuple[NativeInferenceCalls, AuthenticatedInferenceCaller]:
        broker = await self.service.host()
        caller = await authenticate_inference_request(
            request, broker.callers, self.service.node_id
        )
        return broker, caller

    async def models(self, request: Request) -> list[ModelListModel]:
        """Append only current caller-authorized models; ordinary local access survives.

        Invalid/dummy API keys cannot fall back to owner authority. They simply
        contribute no native models to the otherwise public local model list.
        """
        try:
            broker, caller = await self._caller(request)
            catalogs = await native_chat_catalogs(broker, caller)
        except (HTTPException, OSError, ValueError, TimeoutError):
            return []
        return [
            ModelListModel(
                id=catalog.model_id(model),
                name=model.name,
                owned_by="capability",
                description="Caller-authorized native inference capability",
                tasks=["TextGeneration"],
                tags=["capability", "native", catalog.catalog.processing_domain],
                inference_capability=catalog.metadata(model),
            )
            for catalog in catalogs
            for model in catalog.catalog.models
        ]

    async def chat(
        self, request: Request, *, continuation_header: str | None
    ) -> NativeChatResponse:
        """Prepare one exact model call; native and application proofs both apply."""
        broker, caller = await self._caller(request)
        raw = await request.body()
        if len(raw) > MAX_CHAT_BODY_BYTES:
            raise HTTPException(413, "Native chat input exceeds its byte bound")
        try:
            payload = ChatCompletionRequest.model_validate_json(
                raw, strict=True, extra="forbid"
            )
        except ValueError:
            raise HTTPException(
                400, "Unsupported or invalid native chat request"
            ) from None
        if len(request.headers.getlist("Skulk-Continuation-Proof")) > 1:
            raise HTTPException(400, "Ambiguous continuation proof")
        try:
            catalogs = await native_chat_catalogs(broker, caller)
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(503, "Native model discovery is unavailable") from None
        selected = next(
            (
                (catalog, model)
                for catalog in catalogs
                for model in catalog.catalog.models
                if catalog.model_id(model) == str(payload.model)
            ),
            None,
        )
        if selected is None:
            raise HTTPException(403, "Native model is unavailable to this caller")
        catalog, model = selected
        if (
            payload.inference_policy == "local_only"
            and catalog.catalog.processing_domain != "local"
        ) or (
            payload.inference_policy == "included_only"
            and catalog.catalog.usage not in {"none", "included_only"}
        ):
            raise HTTPException(403, "Hard inference policy excludes this capability")
        continuation = None
        target = catalog.chat
        if continuation_header is not None:
            try:
                continuation = read_continuation_proof(continuation_header, raw)
            except ValueError:
                raise HTTPException(400, "Invalid continuation proof or body") from None
            if catalog.continuation is None:
                raise HTTPException(
                    409, "Selected capability has no authorized continuation contract"
                )
            target = catalog.continuation
            expected = ContinuationTarget(
                target_node=catalog.catalog.target_node,
                callable_id=target.application.callable_id,
                descriptor_revision=target.application.revision,
                model=model.id,
            )
            if continuation.request.target != expected:
                raise HTTPException(
                    409, "Continuation destination changed or does not match"
                )
            signed = _OBJECT.validate_json(continuation.model_dump_json())
            body = raw
        else:
            application_body = _OBJECT.validate_json(
                payload.model_dump_json(exclude_none=True, exclude_unset=True)
            )
            application_body.pop("model", None)
            try:
                prepared, body = await asyncio.to_thread(
                    self.signer.prepare, catalog.catalog, model.id, application_body
                )
                signed = _OBJECT.validate_python(prepared, strict=True)
            except (OSError, ValueError):
                raise HTTPException(
                    503, "Inference gateway key is unavailable"
                ) from None
        return NativeChatResponse(
            broker,
            caller,
            target,
            signed,
            model=str(payload.model),
            streamed=payload.stream,
            include_usage=payload.stream_options is not None
            and payload.stream_options.include_usage,
            body=body,
            continuation=continuation,
        )

    async def receipt(
        self, payload: ReceiptProof, request: Request, native_model: str
    ) -> NativeChatResponse:
        """Recover a content-free receipt at an explicitly selected native model.

        The query transport address is independent of the signed application
        proof target. Lookup never generates text or falls back to legacy calls.
        """
        broker, caller = await self._caller(request)
        try:
            check_admission_deadline(payload.request.expires_at)
        except ValueError:
            raise HTTPException(400, "Invalid receipt admission deadline") from None
        try:
            catalogs = await native_chat_catalogs(broker, caller, for_receipt=True)
        except (OSError, ValueError, TimeoutError):
            raise HTTPException(503, "Native receipt destination unavailable") from None
        selected: NativeChatCatalog | None = next(
            (
                catalog
                for catalog in catalogs
                for model in catalog.catalog.models
                if catalog.model_id(model) == native_model
                and model.id == payload.request.target.model
            ),
            None,
        )
        if selected is None or selected.continuation is None:
            raise HTTPException(
                403, "Native receipt destination is unavailable to this caller"
            )
        target = selected.continuation
        expected = ContinuationTarget(
            target_node=selected.catalog.target_node,
            callable_id=target.application.callable_id,
            descriptor_revision=target.application.revision,
            model=payload.request.target.model,
        )
        if payload.request.target != expected:
            raise HTTPException(
                409, "Continuation destination changed or does not match"
            )
        return NativeChatResponse(
            broker,
            caller,
            target,
            _OBJECT.validate_json(payload.model_dump_json()),
            model=native_model,
            streamed=False,
            receipt=payload,
        )
