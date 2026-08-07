"""Node-addressed ephemeral speech media packets and binary wire framing."""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import ConfigDict, Field, model_validator

from skulk.extensions.streams import MAX_INLINE_MEDIA_BYTES
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.pydantic_ext import CamelCaseModel

_HEADER_LENGTH_BYTES = 4
_MAX_HEADER_BYTES = 16_384


class SpeechMediaPacket(CamelCaseModel):
    """One bounded speech-input frame or source-routed transport failure."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source_node: NodeId
    target_node: NodeId
    command_id: CommandId
    sequence: int = Field(ge=0)
    kind: Literal["chunk", "completed", "cancelled", "transport_failed"]
    purpose: Literal["reference_audio", "transcription_audio"] = "reference_audio"
    data: bytes = Field(default=b"", max_length=MAX_INLINE_MEDIA_BYTES)
    filename: str | None = Field(default=None, max_length=255)
    content_type: str | None = Field(default=None, max_length=255)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_message: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _validate_payload(self) -> "SpeechMediaPacket":
        if self.kind == "chunk" and not self.data:
            raise ValueError("speech media chunk must carry bytes")
        if self.kind != "chunk" and self.data:
            raise ValueError("speech media terminal must not carry bytes")
        if self.kind == "completed" and self.sha256 is None:
            raise ValueError("speech media completion requires sha256")
        if self.kind == "transport_failed" and not self.error_message:
            raise ValueError("transport failure requires an error message")
        if self.kind != "transport_failed" and self.error_message is not None:
            raise ValueError("speech media frames cannot carry transport errors")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this packet closes its media stream."""

        return self.kind != "chunk"

    def transport_failure(self, message: str) -> "SpeechMediaPacket":
        """Build a terminal failure routed back to the source API node."""

        return SpeechMediaPacket(
            source_node=self.target_node,
            target_node=self.source_node,
            command_id=self.command_id,
            sequence=self.sequence,
            kind="transport_failed",
            purpose=self.purpose,
            error_message=message,
        )


def encode_speech_media_packet(packet: SpeechMediaPacket) -> bytes:
    """Encode bounded JSON metadata followed by unmodified media bytes."""

    header = json.dumps(
        {
            "source_node": str(packet.source_node),
            "target_node": str(packet.target_node),
            "command_id": str(packet.command_id),
            "sequence": packet.sequence,
            "kind": packet.kind,
            "purpose": packet.purpose,
            "filename": packet.filename,
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
        raise ValueError("speech media packet header is too large")
    return len(header).to_bytes(_HEADER_LENGTH_BYTES, "big") + header + packet.data


def decode_speech_media_packet(wire: bytes) -> SpeechMediaPacket:
    """Decode and strictly validate one speech-media wire packet."""

    if len(wire) < _HEADER_LENGTH_BYTES:
        raise ValueError("speech media packet is missing its header length")
    header_length = int.from_bytes(wire[:_HEADER_LENGTH_BYTES], "big")
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError("speech media packet header is too large")
    header_end = _HEADER_LENGTH_BYTES + header_length
    if header_end > len(wire):
        raise ValueError("speech media packet header is truncated")
    decoded_object = cast(
        object,
        json.loads(wire[_HEADER_LENGTH_BYTES:header_end].decode("utf-8")),
    )
    if not isinstance(decoded_object, dict):
        raise ValueError("speech media packet header must be an object")
    return SpeechMediaPacket.model_validate(
        {**cast(dict[str, object], decoded_object), "data": wire[header_end:]}
    )
