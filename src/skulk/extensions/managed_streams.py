"""Protocol-four local media bridge; no dependency on the private owner SDK.

Each packet has a capped JSON header followed by at most 1 MiB of raw media.
One socket belongs to one call. The cleanup acknowledgment is held separately
from the public terminal so a child cannot release Fabric admission early.
"""

import asyncio
import json
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from skulk.extensions.streams import (
    MAX_INLINE_MEDIA_BYTES,
    CapabilityStreamFrame,
    InlineMediaAttachment,
)

MAX_HEADER_BYTES = 65_536
"""Largest local finite JSON header, excluding its raw media attachment."""


class ManagedStreamEnd(BaseModel):
    """Owner acknowledgment that a call's handler and cleanup have finished."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: Literal["end"] = "end"
    call_id: str = Field(min_length=1, max_length=128, description="Exact public call completed by the owner.")


class _Header(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    protocol: Literal[4]
    frame: dict[str, object]
    media_size: int = Field(ge=0, le=MAX_INLINE_MEDIA_BYTES, description="Number of raw bytes immediately following this header.")


_PACKET: TypeAdapter[CapabilityStreamFrame | ManagedStreamEnd] = TypeAdapter(
    CapabilityStreamFrame | ManagedStreamEnd
)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate stream JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite stream JSON value")


async def write_managed_frame(
    writer: asyncio.StreamWriter, frame: CapabilityStreamFrame
) -> None:
    """Forward one public frame with raw inline bytes and bounded backpressure."""
    data = frame.media.data if isinstance(frame.media, InlineMediaAttachment) else b""
    projection = frame.model_dump(mode="python", exclude={"media": {"data"}})
    header = json.dumps(
        {"protocol": 4, "frame": projection, "media_size": len(data)},
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("managed stream header exceeds bound")
    writer.write(len(header).to_bytes(4, "big") + header + data)
    await writer.drain()


async def read_managed_frame(
    reader: asyncio.StreamReader,
) -> CapabilityStreamFrame | ManagedStreamEnd:
    """Bound untrusted lengths before reading and validate the public frame shape."""
    length = int.from_bytes(await reader.readexactly(4), "big")
    if not 2 <= length <= MAX_HEADER_BYTES:
        raise ValueError("managed stream header exceeds bound")
    raw = await reader.readexactly(length)
    json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    header = _Header.model_validate_json(raw)
    projection = dict(header.frame)
    media_value = projection.get("media")
    media = (
        cast(dict[str, object], media_value) if isinstance(media_value, dict) else None
    )
    if isinstance(media, dict) and media.get("kind") == "inline":
        if "data" in media:
            raise ValueError("managed inline bytes must follow the header")
        projection["media"] = {
            **media,
            "data": await reader.readexactly(header.media_size),
        }
    elif header.media_size:
        raise ValueError("unexpected managed stream attachment")
    return _PACKET.validate_python(projection)
