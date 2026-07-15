"""Node-addressed ephemeral vision input packets and binary wire framing."""

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


class VisionMediaPacket(CamelCaseModel):
    """One bounded image-input frame or source-routed transport failure."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source_node: NodeId = Field(description="Node originating this directed frame.")
    target_node: NodeId = Field(description="Node that must consume this frame.")
    command_id: CommandId = Field(description="Request-scoped transfer identity.")
    model: ModelId = Field(description="Model expected by the admitted task.")
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
        description="Raw base64 text bytes carried only by chunk frames.",
    )
    image_index: int | None = Field(
        default=None, ge=0, description="Original request image index for a chunk."
    )
    total_chunks: int | None = Field(
        default=None, ge=1, description="Total chunk count declared by the source."
    )
    image_count: int | None = Field(
        default=None, ge=1, description="Number of uncached images in the transfer."
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
    def _validate_payload(self) -> "VisionMediaPacket":
        if self.kind == "opened":
            if (
                self.sequence != 0
                or self.total_chunks is None
                or self.image_count is None
                or self.image_index is not None
                or self.sha256 is not None
                or self.data
            ):
                raise ValueError(
                    "vision media open requires only sequence zero, total_chunks, "
                    "and image_count"
                )
        elif self.kind == "chunk":
            if not self.data:
                raise ValueError("vision media chunk must carry bytes")
            if self.image_index is None or self.total_chunks is None:
                raise ValueError(
                    "vision media chunk requires image_index and total_chunks"
                )
            if self.sequence < 1 or self.sequence > self.total_chunks:
                raise ValueError("vision media chunk sequence exceeds total_chunks")
        elif self.data:
            raise ValueError("vision media terminal must not carry bytes")

        if self.kind == "completed":
            if (
                self.total_chunks is None
                or self.image_count is None
                or self.sha256 is None
            ):
                raise ValueError(
                    "vision media completion requires total_chunks, image_count, "
                    "and sha256"
                )
            if self.sequence != self.total_chunks + 1:
                raise ValueError(
                    "vision media completion sequence must follow all chunks"
                )
        elif self.kind not in ("opened", "chunk") and any(
            value is not None
            for value in (self.image_index, self.total_chunks, self.image_count, self.sha256)
        ):
            raise ValueError("vision media terminal has no media metadata")

        if self.kind == "transport_failed" and not self.error_message:
            raise ValueError("transport failure requires an error message")
        if self.kind != "transport_failed" and self.error_message is not None:
            raise ValueError("vision media frames cannot carry transport errors")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this packet closes its media stream."""

        return self.kind not in ("opened", "chunk")

    def accepted(self) -> "VisionMediaPacket":
        """Build a terminal verification acknowledgement for the source API."""

        return VisionMediaPacket(
            source_node=self.target_node,
            target_node=self.source_node,
            command_id=self.command_id,
            model=self.model,
            sequence=self.sequence,
            kind="accepted",
        )

    def transport_failure(self, message: str) -> "VisionMediaPacket":
        """Build a terminal failure routed back to the source API node."""

        return VisionMediaPacket(
            source_node=self.target_node,
            target_node=self.source_node,
            command_id=self.command_id,
            model=self.model,
            sequence=self.sequence,
            kind="transport_failed",
            error_message=message,
        )


def encode_vision_media_packet(packet: VisionMediaPacket) -> bytes:
    """Encode bounded JSON metadata followed by unmodified image bytes."""

    header = json.dumps(
        {
            "source_node": str(packet.source_node),
            "target_node": str(packet.target_node),
            "command_id": str(packet.command_id),
            "model": str(packet.model),
            "sequence": packet.sequence,
            "kind": packet.kind,
            "image_index": packet.image_index,
            "total_chunks": packet.total_chunks,
            "image_count": packet.image_count,
            "sha256": packet.sha256,
            "error_message": packet.error_message,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("vision media packet header is too large")
    return len(header).to_bytes(_HEADER_LENGTH_BYTES, "big") + header + packet.data


def decode_vision_media_packet(wire: bytes) -> VisionMediaPacket:
    """Decode and strictly validate one vision-media wire packet."""

    if len(wire) < _HEADER_LENGTH_BYTES:
        raise ValueError("vision media packet is missing its header length")
    header_length = int.from_bytes(wire[:_HEADER_LENGTH_BYTES], "big")
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError("vision media packet header is too large")
    header_end = _HEADER_LENGTH_BYTES + header_length
    if header_end > len(wire):
        raise ValueError("vision media packet header is truncated")
    decoded_object = cast(
        object,
        json.loads(wire[_HEADER_LENGTH_BYTES:header_end].decode("utf-8")),
    )
    if not isinstance(decoded_object, dict):
        raise ValueError("vision media packet header must be an object")
    return VisionMediaPacket.model_validate(
        {**cast(dict[str, object], decoded_object), "data": wire[header_end:]}
    )
