"""Preserve vLLM request refusals across Skulk's served-engine proxy."""

from typing import cast

import httpx

from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX


class VllmServerRefusalError(RuntimeError):
    """A request was refused by vLLM, with a bounded engine explanation."""


def raise_for_vllm_status(response: httpx.Response) -> None:
    """Raise for an HTTP refusal, preserving its reason and context classification.

    Reads an error response before its streamed connection closes. Context
    overflow uses the shared API sentinel; other refusals remain engine errors.
    Successful responses are left unread so normal SSE streaming stays bounded.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        response.read()
        try:
            body = cast(object, response.json())
        except ValueError:
            body = None
        detail: object = None
        if isinstance(body, dict):
            detail = cast("dict[str, object]", body).get("error")
        kind = ""
        if isinstance(detail, dict):
            fields = cast("dict[str, object]", detail)
            message = str(fields.get("message") or "")
            kind = str(fields.get("type") or "")
        else:
            message = detail if isinstance(detail, str) else response.text
        message = " ".join(message.split())[:300]
        explanation = f"vLLM answered {response.status_code}: " + (
            message or response.reason_phrase
        )
        # vLLM 0.32 labels token/character overflow BadRequestError, so the
        # engine's explicit context phrase is needed alongside typed errors.
        if response.status_code == 400 and (
            kind == "context_length_exceeded"
            or "maximum context length" in message.lower()
        ):
            explanation = f"{CONTEXT_LENGTH_EXCEEDED_PREFIX} {explanation}"
        raise VllmServerRefusalError(explanation) from error
