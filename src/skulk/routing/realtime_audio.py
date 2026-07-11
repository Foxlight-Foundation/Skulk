"""Node-addressed realtime audio ingress packets and binary wire framing."""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import ConfigDict, Field, model_validator

from skulk.extensions.streams import MAX_INLINE_MEDIA_BYTES
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.pydantic_ext import CamelCaseModel

_HEADER_LENGTH_BYTES = 4
_MAX_HEADER_BYTES = 16_384


class RealtimeAudioPacket(CamelCaseModel):
    """One binary PCM frame or transport failure addressed to a worker node."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source_node: NodeId
    target_node: NodeId
    command_id: CommandId
    sequence: int = Field(ge=0)
    kind: Literal["chunk", "completed", "cancelled", "transport_failed"]
    data: bytes = Field(default=b"", max_length=MAX_INLINE_MEDIA_BYTES)
    error_message: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _validate_payload(self) -> "RealtimeAudioPacket":
        if self.kind == "chunk" and not self.data:
            raise ValueError("realtime audio chunk must carry PCM bytes")
        if self.kind != "chunk" and self.data:
            raise ValueError("realtime audio terminal must not carry bytes")
        if len(self.data) % 2 != 0:
            raise ValueError("pcm_s16le audio must contain complete 16-bit samples")
        if self.kind == "transport_failed" and not self.error_message:
            raise ValueError("transport failure requires an error message")
        if self.kind != "transport_failed" and self.error_message is not None:
            raise ValueError("audio input frames cannot carry transport errors")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this packet closes its transport stream."""

        return self.kind != "chunk"

    def to_input_frame(self) -> RealtimeAudioInputFrame:
        """Convert a successfully delivered packet into worker-local input."""

        if self.kind == "transport_failed":
            raise ValueError("transport failure is not a worker audio frame")
        return RealtimeAudioInputFrame(
            command_id=self.command_id,
            sequence=self.sequence,
            kind=self.kind,
            data=self.data,
        )

    def transport_failure(self, message: str) -> "RealtimeAudioPacket":
        """Build a terminal failure routed back to the source API node."""

        return RealtimeAudioPacket(
            source_node=self.target_node,
            target_node=self.source_node,
            command_id=self.command_id,
            sequence=self.sequence,
            kind="transport_failed",
            error_message=message,
        )


def encode_realtime_audio_packet(packet: RealtimeAudioPacket) -> bytes:
    """Encode metadata as bounded JSON followed by unmodified PCM bytes."""

    header = json.dumps(
        {
            "source_node": str(packet.source_node),
            "target_node": str(packet.target_node),
            "command_id": str(packet.command_id),
            "sequence": packet.sequence,
            "kind": packet.kind,
            "error_message": packet.error_message,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("realtime audio packet header is too large")
    return len(header).to_bytes(_HEADER_LENGTH_BYTES, "big") + header + packet.data


def decode_realtime_audio_packet(wire: bytes) -> RealtimeAudioPacket:
    """Decode and strictly validate one realtime audio wire packet."""

    if len(wire) < _HEADER_LENGTH_BYTES:
        raise ValueError("realtime audio packet is missing its header length")
    header_length = int.from_bytes(wire[:_HEADER_LENGTH_BYTES], "big")
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError("realtime audio packet header is too large")
    header_end = _HEADER_LENGTH_BYTES + header_length
    if header_end > len(wire):
        raise ValueError("realtime audio packet header is truncated")
    decoded_object = cast(
        object,
        json.loads(wire[_HEADER_LENGTH_BYTES:header_end].decode("utf-8")),
    )
    if not isinstance(decoded_object, dict):
        raise ValueError("realtime audio packet header must be an object")
    decoded = cast(dict[str, object], decoded_object)
    return RealtimeAudioPacket.model_validate(
        {
            **decoded,
            "data": wire[header_end:],
        }
    )
