"""The generic capability call envelope (fabric-citizenship Phase 2b).

A caller invokes a capability a provider serves with a single generic verb:
``call(id@version, payload)``. The fabric moves the payload opaquely; only the
caller and the provider interpret it, validated against the JSON Schemas the
provider published in its :class:`~skulk.extensions.capabilities.CapabilityDescriptor`.

Design constraints (the #510 converged record):

- **Node-addressed and direct**: a call goes to the selected provider node;
  the master is never in the per-call hot path and calls are never
  event-sourced into ``State``.
- **Pinned contract**: a call carries the exact negotiated ``id@version`` AND
  the descriptor revision digest the caller discovered, so discovery and
  invocation cannot silently disagree; a provider whose descriptor drifted
  rejects with ``revision_mismatch`` instead of a confusing payload error.
- **Typed errors**: every failure mode is a :class:`CapabilityError` with a
  machine-readable code; callers never parse prose.
- **Transport-abstract**: the envelope says nothing about the hop. Today it
  rides the direct peer-API path; a Zenoh queryable can carry the same
  envelope later without touching providers or callers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Ceiling on a call or result payload's serialized size. A capability call is
# a control-sized exchange (a memory query, a synthesis request), not a media
# stream; oversized payloads are rejected with `payload_too_large` instead of
# ballooning API-node memory. Streaming media belongs to the Phase 3 surface.
MAX_CALL_PAYLOAD_BYTES = 1_048_576  # 1 MiB

# Default and ceiling for a call's deadline. Every call has a deadline: an
# unbounded provider call could otherwise pin the caller and the provider's
# concurrency slot forever.
DEFAULT_CALL_TIMEOUT_SECONDS = 30.0
MAX_CALL_TIMEOUT_SECONDS = 300.0

CapabilityErrorCode = Literal[
    "not_found",
    "version_mismatch",
    "revision_mismatch",
    "invalid_payload",
    "invalid_result",
    "payload_too_large",
    "overloaded",
    "timeout",
    "provider_error",
    "unreachable",
]
"""Machine-readable failure codes for a capability call.

``not_found``: the target node serves no capability with that id.
``version_mismatch``: the id exists but not at the requested version.
``revision_mismatch``: the provider's descriptor drifted since discovery.
``invalid_payload``: the payload failed the descriptor's input schema.
``invalid_result``: the provider's result failed its own output schema.
``payload_too_large``: payload or result exceeded ``MAX_CALL_PAYLOAD_BYTES``.
``overloaded``: the provider's concurrency bound rejected the call.
``timeout``: the deadline elapsed before the provider finished.
``provider_error``: the handler raised (message carries the summary).
``unreachable``: the target node could not be reached at all.
"""


class CapabilityError(BaseModel):
    """Typed failure for a capability call.

    Attributes:
        code: Machine-readable failure class (see :data:`CapabilityErrorCode`).
        message: Human-readable detail; never required for control flow.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    code: CapabilityErrorCode
    message: str


class CapabilityCall(BaseModel):
    """One capability invocation, addressed to a provider node.

    Attributes:
        call_id: Caller-minted correlation id (opaque string, unique per call).
        capability_id: The capability's ``id`` from its descriptor.
        version: The exact negotiated semantic version.
        descriptor_revision: The revision digest the caller discovered, so the
            provider can reject a drifted contract (``revision_mismatch``).
        caller_node: Node id of the caller (identity, not authorization; the
            fabric is trusted).
        target_node: Node id of the provider being called.
        timeout_seconds: Deadline the provider must honor for this call.
        payload: The opaque call payload; validated against the descriptor's
            ``input_schema`` on the provider before the handler runs.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    call_id: str
    capability_id: str
    version: str
    descriptor_revision: str
    caller_node: str
    target_node: str
    timeout_seconds: float = Field(
        default=DEFAULT_CALL_TIMEOUT_SECONDS,
        gt=0,
        le=MAX_CALL_TIMEOUT_SECONDS,
    )
    payload: dict[str, object]


class CapabilityResult(BaseModel):
    """The outcome of one capability call.

    Exactly one of ``result``/``error`` is set: ``ok=True`` carries the
    handler's result payload (validated against the descriptor's
    ``output_schema`` when one is published); ``ok=False`` carries a typed
    :class:`CapabilityError`.

    Attributes:
        call_id: Echoed correlation id.
        ok: Whether the call succeeded.
        result: The result payload on success, else ``None``.
        error: The typed failure on error, else ``None``.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    call_id: str
    ok: bool
    result: dict[str, object] | None = None
    error: CapabilityError | None = None


def call_failure(
    call_id: str, code: CapabilityErrorCode, message: str
) -> CapabilityResult:
    """Build a failed :class:`CapabilityResult` (the one-liner every guard uses).

    Args:
        call_id: The call being answered.
        code: Machine-readable failure class.
        message: Human-readable detail.

    Returns:
        A ``CapabilityResult`` with ``ok=False`` and the typed error attached.
    """
    return CapabilityResult(
        call_id=call_id,
        ok=False,
        error=CapabilityError(code=code, message=message),
    )
