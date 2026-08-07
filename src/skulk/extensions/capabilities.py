"""Self-describing capability contracts for provider extensions.

Fabric citizenship's provider role starts here: a plugin that *serves* a
capability (memory, speech, anything not yet imagined) must be discoverable and
callable by peers that have never seen it. The fabric cannot enumerate future
capabilities, so it standardizes the **description**, never the capabilities
themselves: a :class:`CapabilityDescriptor` is a fixed, self-describing shape
(identifier, semantic version, human/LLM-readable description, JSON Schemas for
input and output, and the call's I/O mode). The open-endedness lives entirely in
the schema payload, which is data, not code.

The shape is aligned with MCP's tool descriptor (name + description +
``inputSchema``/``outputSchema``, list + call semantics) so both programmatic
callers (match id@version + schema) and generative callers (an LLM reads the
description and schema at runtime, the tool-use model) are served by one
artifact. MCP's JSON-RPC session protocol is deliberately NOT part of the
fabric; only the descriptor shape is borrowed.

Discovery layers on the telemetry plane: the descriptor's ``id`` doubles as the
node's advertised capability tag (the light discovery key), and the full
descriptor is fetched with a ``describe`` call so telemetry stays cheap.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CapabilityIoMode = Literal[
    "unary", "server_streaming", "client_streaming", "bidirectional"
]
"""How a capability call moves data.

``unary``: one request payload, one result (memory queries, batch STT).
``server_streaming``: one request, a stream of output chunks (TTS).
``client_streaming``: a stream of input frames, one result (realtime STT to a
batch-backed model). ``bidirectional``: streams both ways (realtime STT with
progressive transcripts). A plain ``streaming`` flag was rejected as too weak
to describe real capabilities (#510).
"""

# Capability identifiers are opaque to the fabric but must be safe as telemetry
# tags and routing-key fragments: lowercase, dot/colon/dash/underscore word
# characters, no whitespace. Examples: "memory", "tts", "embeddings:bge-m3".
_CAPABILITY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")

# Semantic version: MAJOR.MINOR.PATCH. The version is a field of the contract
# (a caller negotiates id@version), distinct from the descriptor revision
# digest, which detects any drift in the published shape.
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class CapabilityDescriptor(BaseModel):
    """One capability a provider offers, described well enough to call it.

    The fabric standardizes this shape and passes call payloads opaquely; only
    the caller and the provider interpret them, validated against the schemas
    published here. Descriptors are immutable: a provider that changes its
    contract publishes a new descriptor (new ``version`` for semantic changes;
    the revision digest changes on any drift).

    Attributes:
        id: Opaque capability identifier (for example ``memory`` or ``tts``).
            Doubles as the telemetry-plane discovery tag.
        version: Semantic version of the capability contract
            (``MAJOR.MINOR.PATCH``). Callers negotiate ``id@version``.
        title: Short human-facing name.
        description: What the capability does, written for both humans and
            generative callers (an LLM reads this plus the schemas at runtime
            to call a capability it has never seen).
        input_schema: JSON Schema (2020-12) for the call payload.
        output_schema: JSON Schema for the unary result, or ``None`` when the
            capability produces no meaningful result body.
        io_mode: How the call moves data (see :data:`CapabilityIoMode`).
        input_chunk_schema: JSON Schema for one inbound stream frame; required
            for ``client_streaming``/``bidirectional``, forbidden otherwise.
        output_chunk_schema: JSON Schema for one outbound stream chunk;
            required for ``server_streaming``/``bidirectional``, forbidden
            otherwise.
        annotations: Optional opaque string hints for callers (for example
            ``{"latency": "interactive"}``); never interpreted by the fabric.
    """

    # Extension-facing contract type: strict, frozen, no silent extras. Not a
    # gossiped wire type (descriptors travel via describe, not telemetry), so
    # no camelCase aliasing.
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: str
    version: str
    title: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object] | None = None
    io_mode: CapabilityIoMode = "unary"
    input_chunk_schema: dict[str, object] | None = None
    output_chunk_schema: dict[str, object] | None = None
    annotations: dict[str, str] | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not _CAPABILITY_ID_PATTERN.match(value):
            raise ValueError(
                f"capability id {value!r} must match "
                f"{_CAPABILITY_ID_PATTERN.pattern} (lowercase, no whitespace)"
            )
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if not _SEMVER_PATTERN.match(value):
            raise ValueError(
                f"capability version {value!r} must be semantic "
                f"(MAJOR.MINOR.PATCH, for example 1.0.0)"
            )
        return value

    @field_validator("title", "description")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_schemas_are_json(self) -> "CapabilityDescriptor":
        # Schema dicts are published verbatim over /v1/capabilities and hashed
        # by descriptor_revision(); a non-JSON value inside (set, bytes, NaN)
        # would surface later as a 500 on the discovery endpoint instead of a
        # clean validation error at descriptor construction. Fail here.
        candidates: dict[str, dict[str, object] | dict[str, str] | None] = {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "input_chunk_schema": self.input_chunk_schema,
            "output_chunk_schema": self.output_chunk_schema,
            "annotations": self.annotations,
        }
        for field_name, value in candidates.items():
            if value is None:
                continue
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} must be JSON-serializable "
                    f"(no sets, bytes, NaN/Infinity, or custom objects): {exc}"
                ) from exc
        return self

    @model_validator(mode="after")
    def _validate_chunk_schemas_match_io_mode(self) -> "CapabilityDescriptor":
        # The chunk schemas ARE the streaming contract; a streaming mode
        # without them is undiscoverable, and chunk schemas on a unary call
        # are dead weight that would drift silently. Enforce the pairing.
        needs_input_chunks = self.io_mode in ("client_streaming", "bidirectional")
        needs_output_chunks = self.io_mode in ("server_streaming", "bidirectional")
        if needs_input_chunks and self.input_chunk_schema is None:
            raise ValueError(
                f"io_mode {self.io_mode!r} requires input_chunk_schema"
            )
        if not needs_input_chunks and self.input_chunk_schema is not None:
            raise ValueError(
                f"io_mode {self.io_mode!r} forbids input_chunk_schema"
            )
        if needs_output_chunks and self.output_chunk_schema is None:
            raise ValueError(
                f"io_mode {self.io_mode!r} requires output_chunk_schema"
            )
        if not needs_output_chunks and self.output_chunk_schema is not None:
            raise ValueError(
                f"io_mode {self.io_mode!r} forbids output_chunk_schema"
            )
        return self

    @property
    def qualified_id(self) -> str:
        """The negotiation key callers use: ``id@version``."""
        return f"{self.id}@{self.version}"


def descriptor_revision(descriptor: CapabilityDescriptor) -> str:
    """Content digest of a descriptor, detecting any drift in its shape.

    Calls carry this revision alongside ``id@version`` so discovery and
    invocation cannot silently disagree: a provider that edits a schema
    without bumping the version still produces a different revision, and the
    mismatch is detectable at call time instead of surfacing as a confusing
    payload validation error.

    Args:
        descriptor: The descriptor to digest.

    Returns:
        A short hex SHA-256 digest of the canonical JSON form.
    """
    canonical = json.dumps(
        descriptor.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
