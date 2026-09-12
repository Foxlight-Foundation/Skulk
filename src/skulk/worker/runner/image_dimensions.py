"""Pixel dimensions of a still image, read from its header alone.

A keyframe decides the canvas of the clip it anchors, so the planner needs
the image's shape before any engine touches it. Only the container headers
are read (PNG, JPEG, WebP: the families the video route accepts), from a
bounded prefix of the file, so a large or hostile file costs nothing more
than that prefix and no image codec runs in the worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

HEADER_BYTES: Final = 256 * 1024
"""Prefix read for the scan; a JPEG's frame header follows its metadata
segments, which stay well inside this for camera and editor output."""

_JPEG_FRAME_MARKERS: Final = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
"""SOFn markers that carry the frame's dimensions (DHT, JPG, and DAC share
the range and are skipped)."""


def image_dimensions(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` of a PNG, JPEG, or WebP file, else ``None``.

    Reads at most ``HEADER_BYTES``; an unreadable file, an unknown format,
    or a header the prefix does not reach yields ``None`` rather than an
    error, since a missing shape only means the canvas falls back to the
    card's default.
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(HEADER_BYTES)
    except OSError:
        return None
    return dimensions_from_header(data)


def dimensions_from_header(data: bytes) -> tuple[int, int] | None:
    """Dimensions parsed from the leading bytes of an image file."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png(data)
    if data.startswith(b"\xff\xd8\xff"):
        return _jpeg(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp(data)
    return None


def _png(data: bytes) -> tuple[int, int] | None:
    # The IHDR chunk is required first: length, type, then width and height.
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return _positive(_be32(data, 16), _be32(data, 20))


def _jpeg(data: bytes) -> tuple[int, int] | None:
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        if marker == 0xFF:
            # Fill bytes precede a marker.
            offset += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            # Standalone markers carry no length.
            offset += 2
            continue
        if marker == 0xD9 or marker == 0xDA:
            # End of image, or scan data before any frame header.
            return None
        length = _be16(data, offset + 2)
        if marker in _JPEG_FRAME_MARKERS:
            if offset + 9 > len(data):
                return None
            return _positive(_be16(data, offset + 7), _be16(data, offset + 5))
        offset += 2 + length
    return None


def _webp(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        # Extended format: 24-bit canvas width and height, each minus one.
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return _positive(width, height)
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        # Lossless: 14-bit width and height minus one, packed after the tag.
        bits = int.from_bytes(data[21:25], "little")
        return _positive((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        # Lossy: the key frame start code, then 14-bit width and height.
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return _positive(width, height)
    return None


def _be16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "big")


def _be32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "big")


def _positive(width: int, height: int) -> tuple[int, int] | None:
    return (width, height) if width > 0 and height > 0 else None
