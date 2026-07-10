"""Provider-stream DATA packets and binary-preserving wire framing."""

from __future__ import annotations

import json
from typing import cast

from pydantic import ConfigDict

from skulk.extensions.streams import (
    CapabilityStreamError,
    CapabilityStreamFrame,
    decode_capability_stream_frame,
    encode_capability_stream_frame,
)
from skulk.shared.types.common import NodeId
from skulk.utils.pydantic_ext import CamelCaseModel

_HEADER_LENGTH_BYTES = 4
_MAX_PROVIDER_HEADER_BYTES = 1_048_576


class ProviderStreamPacket(CamelCaseModel):
    """One provider frame addressed to its receiving fabric node.

    ``owner_node`` is transport routing metadata and is deliberately outside
    :class:`CapabilityStreamFrame`: the public provider contract stays
    transport-independent while the router can unicast the packet to one node.
    """

    model_config = ConfigDict(frozen=True)

    owner_node: NodeId
    frame: CapabilityStreamFrame


def encode_provider_stream_packet(packet: ProviderStreamPacket) -> bytes:
    """Encode a packet as length-prefixed JSON metadata plus optional media."""

    frame_header, media = encode_capability_stream_frame(packet.frame)
    frame_object = cast(object, json.loads(frame_header.decode("utf-8")))
    envelope = json.dumps(
        {"owner_node": str(packet.owner_node), "frame": frame_object},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(envelope) > _MAX_PROVIDER_HEADER_BYTES:
        raise ValueError("provider stream packet header exceeds 1 MiB")
    return len(envelope).to_bytes(_HEADER_LENGTH_BYTES, "big") + envelope + (
        media or b""
    )


def decode_provider_stream_packet(wire: bytes) -> ProviderStreamPacket:
    """Decode and strictly validate one provider DATA wire packet."""

    if len(wire) < _HEADER_LENGTH_BYTES:
        raise ValueError("provider stream packet is missing its header length")
    header_length = int.from_bytes(wire[:_HEADER_LENGTH_BYTES], "big")
    if header_length > _MAX_PROVIDER_HEADER_BYTES:
        raise ValueError("provider stream packet header exceeds 1 MiB")
    header_end = _HEADER_LENGTH_BYTES + header_length
    if header_end > len(wire):
        raise ValueError("provider stream packet header is truncated")
    decoded_object = cast(
        object,
        json.loads(wire[_HEADER_LENGTH_BYTES:header_end].decode("utf-8")),
    )
    if not isinstance(decoded_object, dict):
        raise ValueError("provider stream packet header must be an object")
    decoded = cast(dict[str, object], decoded_object)
    owner_node = decoded.get("owner_node")
    frame_object = decoded.get("frame")
    if not isinstance(owner_node, str) or not owner_node:
        raise ValueError("provider stream packet requires owner_node")
    if not isinstance(frame_object, dict):
        raise ValueError("provider stream packet requires a frame object")

    frame_dict = cast(dict[str, object], frame_object)
    media_header = frame_dict.get("media")
    media_header_dict = (
        cast(dict[str, object], media_header)
        if isinstance(media_header, dict)
        else None
    )
    has_inline_media = (
        media_header_dict is not None
        and media_header_dict.get("kind") == "inline"
    )
    attachment = wire[header_end:] if has_inline_media else None
    if not has_inline_media and header_end != len(wire):
        raise ValueError("raw bytes require an inline provider media header")
    frame_header = json.dumps(
        frame_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return ProviderStreamPacket(
        owner_node=NodeId(owner_node),
        frame=decode_capability_stream_frame(frame_header, attachment),
    )


def provider_stream_rejection_packets(
    packet: ProviderStreamPacket,
    *,
    include_started: bool,
) -> tuple[ProviderStreamPacket, ...]:
    """Build a typed transport failure when router admission cannot proceed."""

    frame = packet.frame
    frames: list[CapabilityStreamFrame] = []
    if include_started:
        frames.append(
            CapabilityStreamFrame(
                call_id=frame.call_id,
                direction=frame.direction,
                sequence=0,
                kind="started",
                synthetic=True,
            )
        )
    frames.append(
        CapabilityStreamFrame(
            call_id=frame.call_id,
            direction=frame.direction,
            sequence=1 if include_started else frame.sequence,
            kind="failed",
            synthetic=True,
            error=CapabilityStreamError(
                code="transport_error",
                message=(
                    "provider DATA transport rejected the stream because remote "
                    "egress capacity is exhausted"
                ),
            ),
        )
    )
    return tuple(
        ProviderStreamPacket(owner_node=packet.owner_node, frame=item)
        for item in frames
    )
