"""Exercise the public managed boundary without importing the private SDK."""

import asyncio
import json
import socket

import pytest

from skulk.extensions.managed_streams import (
    MAX_HEADER_BYTES,
    ManagedStreamEnd,
    read_managed_frame,
    write_managed_frame,
)
from skulk.extensions.streams import (
    MAX_INLINE_MEDIA_BYTES,
    BlobMediaAttachment,
    CapabilityStreamFrame,
    InlineMediaAttachment,
)


def _packet(header: bytes, media: bytes = b"") -> bytes:
    return len(header).to_bytes(4, "big") + header + media


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.parametrize("inline", [True, False])
async def test_binary_and_blob_round_trip(inline: bool) -> None:
    """Raw media retains arbitrary bytes; blob references never acquire an attachment."""
    left, right = socket.socketpair()
    _, writer = await asyncio.open_connection(sock=left)
    reader, peer = await asyncio.open_connection(sock=right)
    media = (
        InlineMediaAttachment(data=b"\x00\xff\x80\n", media_type="audio/pcm")
        if inline
        else BlobMediaAttachment(
            blob_id="immutable", size_bytes=12, sha256="a" * 64, media_type="audio/wav"
        )
    )
    frame = CapabilityStreamFrame(
        call_id="call",
        direction="provider_to_caller",
        sequence=1,
        kind="chunk",
        payload={"label": "music"},
        media=media,
    )
    try:
        await write_managed_frame(writer, frame)
        assert await read_managed_frame(reader) == frame
    finally:
        writer.close()
        peer.close()
        await asyncio.gather(writer.wait_closed(), peer.wait_closed())


@pytest.mark.parametrize("length", [0, 1, MAX_HEADER_BYTES + 1, 2**32 - 1])
async def test_header_length_rejected_before_payload_read(length: int) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(length.to_bytes(4, "big"))
    # Deliberately omit EOF: a parser that reads first would hang here.
    with pytest.raises(ValueError, match="bound"):
        async with asyncio.timeout(0.2):
            await read_managed_frame(reader)


@pytest.mark.parametrize("data", [b"\x00", (10).to_bytes(4, "big") + b"{}"])
async def test_truncated_packet_rejected(data: bytes) -> None:
    with pytest.raises(asyncio.IncompleteReadError):
        await read_managed_frame(_reader(data))


@pytest.mark.parametrize(
    "header",
    [
        b'{"protocol":4,"protocol":4,"frame":{},"media_size":0}',
        b'{"protocol":4,"frame":{"payload":{"x":1,"x":2}},"media_size":0}',
        b'{"protocol":4,"frame":{"payload":{"x":NaN}},"media_size":0}',
        b'{"protocol":3,"frame":{},"media_size":0}',
        b'{"protocol":4,"frame":{},"media_size":1048577}',
        b'{"protocol":4,"frame":{},"media_size":-1}',
        b'{"protocol":4,"frame":{},"media_size":true}',
        b'{"protocol":4,"frame":{"kind":"end","call_id":"call"},"media_size":1}',
    ],
)
async def test_untrusted_header_rejected_without_attachment_read(header: bytes) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_packet(header))
    with pytest.raises(ValueError):
        async with asyncio.timeout(0.2):
            await read_managed_frame(reader)


@pytest.mark.parametrize("embedded", [False, True])
async def test_inline_attachment_truncation_or_embedded_data_rejected(
    embedded: bool,
) -> None:
    media: dict[str, object] = {"kind": "inline", "media_type": "audio/pcm"}
    if embedded:
        media["data"] = "untrusted base64"
    header = json.dumps(
        {
            "protocol": 4,
            "media_size": 8,
            "frame": {
                "call_id": "call",
                "direction": "provider_to_caller",
                "sequence": 1,
                "kind": "chunk",
                "media": media,
            },
        }
    ).encode()
    with pytest.raises(ValueError if embedded else asyncio.IncompleteReadError):
        await read_managed_frame(_reader(_packet(header, b"short")))


async def test_attachment_limit_and_cleanup_acknowledgment() -> None:
    header = json.dumps(
        {
            "protocol": 4,
            "media_size": MAX_INLINE_MEDIA_BYTES,
            "frame": {
                "call_id": "call",
                "direction": "provider_to_caller",
                "sequence": 1,
                "kind": "chunk",
                "media": {"kind": "inline", "media_type": "application/octet-stream"},
            },
        }
    ).encode()
    frame = await read_managed_frame(
        _reader(_packet(header, b"x" * MAX_INLINE_MEDIA_BYTES))
    )
    assert isinstance(frame, CapabilityStreamFrame)
    assert isinstance(frame.media, InlineMediaAttachment)
    assert len(frame.media.data) == MAX_INLINE_MEDIA_BYTES
    ack = b'{"protocol":4,"frame":{"kind":"end","call_id":"call"},"media_size":0}'
    assert await read_managed_frame(_reader(_packet(ack))) == ManagedStreamEnd(
        call_id="call"
    )


async def test_writer_rejects_oversize_or_nonfinite_headers() -> None:
    left, right = socket.socketpair()
    reader, peer = await asyncio.open_connection(sock=right)
    _, writer = await asyncio.open_connection(sock=left)
    try:
        for value in ("x" * MAX_HEADER_BYTES, float("nan")):
            with pytest.raises(ValueError):
                frame = CapabilityStreamFrame(
                    call_id="call",
                    direction="provider_to_caller",
                    sequence=1,
                    kind="chunk",
                    payload={"value": value},
                )
                await write_managed_frame(writer, frame)
        writer.close()
        await writer.wait_closed()
        assert await reader.read() == b""
    finally:
        writer.close()
        peer.close()
        await asyncio.gather(writer.wait_closed(), peer.wait_closed())
