"""Provider-neutral detached client proofs and content-free continuation receipts.

The capability independently verifies the registered key and current authority.
This module grants no authority and holds no signing keys or provider state.
"""

import base64
import binascii
import hashlib
import re
import time
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

CONTINUATION_CONTRACT = "chat.continuation@0.1.0"
CONTINUATION_PROOF_HEADER = "Skulk-Continuation-Proof"
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[\w.:@-]+$")]
Handle = Annotated[str, Field(pattern=r"^ct1_[a-f0-9]{64}$")]
FailureCode = Literal[
    "authority_unavailable",
    "scope_mismatch",
    "expired",
    "invalid_body",
    "history_mismatch",
    "conversation_exists",
    "handle_unavailable",
    "busy",
    "recovery_required",
    "invalid_transition",
    "storage_unavailable",
    "capacity_exceeded",
]


class ContinuationContract(BaseModel):
    """Strict immutable wire data; unknown fields never expand this contract."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )


class ContinuationTarget(ContinuationContract):
    """Exact destination retained across a conversation and function-tool loop."""

    target_node: Identifier
    callable_id: Identifier
    descriptor_revision: str = Field(pattern=r"^[a-f0-9]{16}$")
    model: Identifier


class ContinuationBody(ContinuationContract):
    """Digest of the original HTTP bytes, including its model envelope."""

    size_bytes: int = Field(ge=2, le=16 * 1024 * 1024)
    sha256: Digest
    media_type: Literal["application/vnd.skulk.chat-request+json"]


class ContinuationRequest(ContinuationContract):
    """One explicit start or continuation, authenticated separately by its owner."""

    contract: Literal["chat.continuation@0.1.0"]
    mode: Literal["start", "continue"]
    conversation_id: Digest
    handle: Handle | None
    target: ContinuationTarget
    body: ContinuationBody
    nonce: Digest
    expires_at: float = Field(gt=0)

    @model_validator(mode="after")
    def explicit_mode(self) -> Self:
        """Refuse a missing handle rather than silently creating a conversation."""
        if (self.mode == "continue") != (self.handle is not None):
            raise ValueError("continuation mode and handle disagree")
        return self


class ContinuationProof(ContinuationContract):
    """A registered client's detached signature, never gateway delegation."""

    caller_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    request: ContinuationRequest
    signature: str = Field(pattern=r"^[a-f0-9]{128}$")


class ReceiptQuery(ContinuationContract):
    """Separate signed lookup domain; it cannot initiate or replay inference."""

    contract: Literal["chat.continuation.receipt@0.1.0"]
    conversation_id: Digest
    target: ContinuationTarget
    request_nonce: Digest
    nonce: Digest
    expires_at: float = Field(gt=0)


class ReceiptProof(ContinuationContract):
    """A fresh client signature for one content-free receipt lookup."""

    caller_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    request: ReceiptQuery
    signature: str = Field(pattern=r"^[a-f0-9]{128}$")


class TranscriptFingerprint(ContinuationContract):
    """Ordered public history identity; contains no messages or canonical state."""

    count: int = Field(ge=0, le=4096)
    sha256: Digest


class ContinuationReceipt(ContinuationContract):
    """Capability commit acknowledgement released only after clean transport end."""

    contract: Literal["chat.continuation@0.1.0"]
    conversation_id: Digest
    handle: Handle
    history: TranscriptFingerprint
    request_nonce: Digest
    expires_at: float = Field(gt=0)

    def matches(self, proof: ContinuationProof | ReceiptProof) -> bool:
        """Bind a returned receipt to the exact requested conversation and attempt."""
        nonce = (
            proof.request.nonce
            if isinstance(proof, ContinuationProof)
            else proof.request.request_nonce
        )
        return (
            self.conversation_id == proof.request.conversation_id
            and self.request_nonce == nonce
        )


class ContinuationEvent(ContinuationContract):
    """Public chat event with an optional receipt or bounded failure category."""

    event: dict[str, JsonValue]
    receipt: ContinuationReceipt | None = None
    continuation_error: FailureCode | None = None

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        """Only a successful semantic terminal carries a committed receipt."""
        kind = self.event.get("kind")
        if kind not in {"delta", "completed", "failed"} or (kind == "completed") != (
            self.receipt is not None
        ):
            raise ValueError("invalid continuation event")
        if self.continuation_error is not None and kind != "failed":
            raise ValueError("invalid continuation failure")
        return self


class ReceiptOutcome(ContinuationContract):
    """One committed receipt or fixed refusal; stored provider content is absent."""

    receipt: ContinuationReceipt | None = None
    error: FailureCode | None = None

    @model_validator(mode="after")
    def exclusive(self) -> Self:
        """Reject ambiguous success or an empty response."""
        if (self.receipt is None) == (self.error is None):
            raise ValueError("invalid receipt outcome")
        return self


def read_continuation_proof(header: str, raw: bytes) -> ContinuationProof:
    """Validate bounded unpadded base64url proof and exact body before dispatch.

    Signature verification and revocation belong to the enforcing capability,
    which resolves the registered public key independently of this API.
    """
    if not 1 <= len(header) <= 4096 or re.fullmatch(r"[A-Za-z0-9_-]+", header) is None:
        raise ValueError("invalid detached continuation proof")
    try:
        encoded = base64.b64decode(
            header + "=" * (-len(header) % 4), altchars=b"-_", validate=True
        )
        proof = ContinuationProof.model_validate_json(encoded)
    except (ValueError, binascii.Error):
        raise ValueError("invalid detached continuation proof") from None
    if base64.urlsafe_b64encode(encoded).decode().rstrip("=") != header:
        raise ValueError("noncanonical proof encoding")
    if (
        proof.request.body.size_bytes != len(raw)
        or proof.request.body.sha256 != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("continuation body mismatch")
    check_admission_deadline(proof.request.expires_at)
    return proof


def check_admission_deadline(expires_at: float) -> None:
    """Reject expired or unbounded admission without truncating an admitted call."""
    now = time.time()
    if not now < expires_at <= now + 60:
        raise ValueError("invalid continuation admission deadline")
