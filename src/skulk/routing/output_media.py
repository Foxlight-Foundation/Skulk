"""Node-addressed generated-media output packets and binary wire framing.

A finished video is far too large to ride inline in ``DataChunk`` frames, so
the worker that produced it streams the container (and an optional thumbnail)
directly to the owning API node on the ``OUTPUT_MEDIA`` data-plane topic.
Framing mirrors ``VISION_MEDIA`` in the opposite direction: an explicit
``opened`` frame, ordered raw-byte ``chunk`` frames, one ``completed`` frame
carrying the digest, and reverse-routed ``accepted`` or ``transport_failed``
terminals from the API back to the producer.
"""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import ConfigDict, Field, model_validator

from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.pydantic_ext import CamelCaseModel

_HEADER_LENGTH_BYTES = 4
_MAX_HEADER_BYTES = 16_384

OutputMediaPurpose = Literal["video", "thumbnail"]
"""Which artifact of a command an output stream carries."""


class OutputMediaPacket(CamelCaseModel):
    """One bounded generated-media frame or a reverse-routed terminal."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source_node: NodeId = Field(description="Node originating this directed frame.")
    target_node: NodeId = Field(description="Node that must consume this frame.")
    command_id: CommandId = Field(description="Generation command the media belongs to.")
    model: ModelId = Field(description="Model that produced the media.")
    purpose: OutputMediaPurpose = Field(description="Artifact carried by this stream.")
    sequence: int = Field(
        ge=0, description="Lifecycle sequence: open zero, chunks one-based."
    )
    kind: Literal[
        "opened",
        "chunk",
        "completed",
        "accepted",
        "cancelled",
        "transport_failed",
    ] = Field(description="Lifecycle operation carried by this frame.")
    data: bytes = Field(
        default=b"",
        max_length=SKULK_MAX_CHUNK_SIZE,
        description="Raw media bytes carried only by chunk frames.",
    )
    total_chunks: int | None = Field(
        default=None, ge=1, description="Chunk count declared by the producer."
    )
    total_bytes: int | None = Field(
        default=None, ge=1, description="Exact media size declared by the producer."
    )
    content_type: str | None = Field(
        default=None, max_length=128, description="MIME type of the media."
    )
    sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Digest over all chunk bytes in sequence order.",
    )
    error_message: str | None = Field(
        default=None,
        max_length=1024,
        description="Bounded failure detail carried only by transport failures.",
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> OutputMediaPacket:
        if self.kind == "opened":
            if (
                self.sequence != 0
                or self.total_chunks is None
                or self.total_bytes is None
                or self.content_type is None
                or self.sha256 is not None
                or self.data
            ):
                raise ValueError(
                    "output media open requires sequence zero, total_chunks, "
                    "total_bytes, and content_type"
                )
        elif self.kind == "chunk":
            if not self.data:
                raise ValueError("output media chunk must carry bytes")
            if self.total_chunks is None:
                raise ValueError("output media chunk requires total_chunks")
            if self.sequence < 1 or self.sequence > self.total_chunks:
                raise ValueError("output media chunk sequence exceeds total_chunks")
        elif self.data:
            raise ValueError("output media terminal must not carry bytes")

        if self.kind == "completed":
            if self.total_chunks is None or self.total_bytes is None or self.sha256 is None:
                raise ValueError(
                    "output media completion requires total_chunks, total_bytes, "
                    "and sha256"
                )
            if self.sequence != self.total_chunks + 1:
                raise ValueError("output media completion sequence must follow all chunks")
        elif self.kind not in ("opened", "chunk") and any(
            value is not None
            for value in (self.total_chunks, self.total_bytes, self.sha256)
        ):
            raise ValueError("output media terminal has no media metadata")

        if self.kind == "transport_failed" and not self.error_message:
            raise ValueError("transport failure requires an error message")
        if self.kind != "transport_failed" and self.error_message is not None:
            raise ValueError("output media frames cannot carry transport errors")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this packet closes its media stream."""

        return self.kind not in ("opened", "chunk")

    def accepted(self) -> OutputMediaPacket:
        """Build the API's verification acknowledgement for the producer."""

        return OutputMediaPacket(
            source_node=self.target_node,
            target_node=self.source_node,
            command_id=self.command_id,
            model=self.model,
            purpose=self.purpose,
            sequence=self.sequence,
            kind="accepted",
        )

    def transport_failure(self, message: str) -> OutputMediaPacket:
        """Build a terminal failure routed back to the producing worker."""

        reverse_terminal = self.kind in ("accepted", "transport_failed")
        return OutputMediaPacket(
            source_node=self.source_node if reverse_terminal else self.target_node,
            target_node=self.target_node if reverse_terminal else self.source_node,
            command_id=self.command_id,
            model=self.model,
            purpose=self.purpose,
            sequence=self.sequence,
            kind="transport_failed",
            error_message=message,
        )


def encode_output_media_packet(packet: OutputMediaPacket) -> bytes:
    """Encode bounded JSON metadata followed by unmodified media bytes."""

    header = json.dumps(
        {
            "source_node": str(packet.source_node),
            "target_node": str(packet.target_node),
            "command_id": str(packet.command_id),
            "model": str(packet.model),
            "purpose": packet.purpose,
            "sequence": packet.sequence,
            "kind": packet.kind,
            "total_chunks": packet.total_chunks,
            "total_bytes": packet.total_bytes,
            "content_type": packet.content_type,
            "sha256": packet.sha256,
            "error_message": packet.error_message,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("output media packet header is too large")
    return len(header).to_bytes(_HEADER_LENGTH_BYTES, "big") + header + packet.data


def decode_output_media_packet(wire: bytes) -> OutputMediaPacket:
    """Decode and strictly validate one output-media wire packet."""

    if len(wire) < _HEADER_LENGTH_BYTES:
        raise ValueError("output media packet is missing its header length")
    header_length = int.from_bytes(wire[:_HEADER_LENGTH_BYTES], "big")
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError("output media packet header is too large")
    header_end = _HEADER_LENGTH_BYTES + header_length
    if header_end > len(wire):
        raise ValueError("output media packet header is truncated")
    decoded_object = cast(
        object,
        json.loads(wire[_HEADER_LENGTH_BYTES:header_end].decode("utf-8")),
    )
    if not isinstance(decoded_object, dict):
        raise ValueError("output media packet header must be an object")
    return OutputMediaPacket.model_validate(
        {**cast(dict[str, object], decoded_object), "data": wire[header_end:]}
    )
