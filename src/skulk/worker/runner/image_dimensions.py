"""Pixel dimensions of a still image, read from its header alone.

A keyframe decides the canvas of the clip it anchors, so the planner needs
the image's shape before any engine touches it. Only the container headers
are read (PNG, JPEG with its EXIF orientation, WebP, GIF, and the ISO
base-media stills HEIC, HEIF, and AVIF: the families the video route
accepts), from a bounded prefix of the file, so a large or hostile file
costs nothing more than that prefix and no image codec runs in the worker.
The shape returned is the displayed shape: a phone photo stored sideways
with an orientation tag counts as the engine will show it.
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
    """Return the displayed ``(width, height)`` of an image file, else ``None``.

    PNG, JPEG (with its EXIF orientation applied), WebP, GIF, and the ISO
    base-media stills HEIC, HEIF, and AVIF (the primary item, with its
    rotation applied) are read. Reads at most ``HEADER_BYTES``; an
    unreadable file, an unknown format, or a header the prefix does not
    reach yields ``None`` rather than an error, since a missing shape only
    means the canvas falls back to the card's default.
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
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return _gif(data)
    if data[4:8] == b"ftyp":
        return _isobmff(data)
    return None


def _png(data: bytes) -> tuple[int, int] | None:
    # The IHDR chunk is required first: length, type, then width and height.
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return _positive(_be32(data, 16), _be32(data, 20))


def _jpeg(data: bytes) -> tuple[int, int] | None:
    offset = 2
    orientation = 1
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
        if length < 2:
            # A segment shorter than its own length field cannot advance the
            # cursor; a malformed file is no shape, never a stuck runner.
            return None
        if marker == 0xE1 and data[offset + 4 : offset + 10] == b"Exif\x00\x00":
            orientation = _exif_orientation(data[offset + 10 : offset + 2 + length])
        if marker in _JPEG_FRAME_MARKERS:
            if offset + 9 > len(data):
                return None
            width, height = _be16(data, offset + 7), _be16(data, offset + 5)
            if orientation in (5, 6, 7, 8):
                # The decoder transposes these on display; so does the engine.
                width, height = height, width
            return _positive(width, height)
        offset += 2 + length
    return None


def _exif_orientation(tiff: bytes) -> int:
    """The EXIF orientation tag from a TIFF block, 1 when absent or unreadable."""
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return 1
    order = "little" if tiff[:2] == b"II" else "big"

    def read(offset: int, size: int) -> int:
        return int.from_bytes(tiff[offset : offset + size], order)

    if read(2, 2) != 42:
        return 1
    ifd = read(4, 4)
    if ifd + 2 > len(tiff):
        return 1
    count = min(read(ifd, 2), 512)
    for index in range(count):
        entry = ifd + 2 + index * 12
        if entry + 12 > len(tiff):
            return 1
        if read(entry, 2) == 0x0112 and read(entry + 2, 2) == 3:
            value = read(entry + 8, 2)
            return value if 1 <= value <= 8 else 1
    return 1


def _gif(data: bytes) -> tuple[int, int] | None:
    if len(data) < 10:
        return None
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    return _positive(width, height)


_STILL_BRANDS = frozenset(
    {b"heic", b"heix", b"heim", b"heis", b"hevc", b"mif1", b"msf1", b"avif", b"avis"}
)


def _isobmff(data: bytes) -> tuple[int, int] | None:
    """HEIC, HEIF, and AVIF: the primary item's ``ispe``, rotated by its ``irot``.

    A phone still carries several items (the picture, a thumbnail, tiles),
    so the geometry is the one ``pitm`` names, found through the item's
    property associations in ``ipma`` rather than the first ``ispe`` in
    the container. Without ``pitm`` or ``ipma`` the first ``ispe`` stands.
    """
    size = _be32(data, 0)
    if (
        size < 16
        or data[8:12] not in _STILL_BRANDS
        and not any(
            data[16 + 4 * index : 20 + 4 * index] in _STILL_BRANDS
            for index in range((min(size, 64) - 16) // 4)
        )
    ):
        return None
    meta = _box(data, 0, len(data), b"meta")
    if meta is None:
        return None
    # meta is a full box: four bytes of version and flags precede its children.
    iprp = _box(data, meta[0] + 4, meta[1], b"iprp")
    if iprp is None:
        return None
    ipco = _box(data, iprp[0], iprp[1], b"ipco")
    if ipco is None:
        return None
    properties = _children(data, ipco[0], ipco[1])
    wanted = _primary_properties(data, meta, iprp)
    chosen = (
        [properties[index - 1] for index in wanted if 0 < index <= len(properties)]
        if wanted
        else properties
    )
    shape: tuple[int, int] | None = None
    quarter_turns = 0
    for kind, first, last in chosen:
        if kind == b"ispe" and shape is None and first + 12 <= len(data):
            shape = _positive(_be32(data, first + 4), _be32(data, first + 8))
        elif kind == b"irot" and first < min(last, len(data)):
            quarter_turns = data[first] & 0x03
    if shape is None:
        return None
    return (shape[1], shape[0]) if quarter_turns % 2 else shape


def _primary_properties(
    data: bytes, meta: tuple[int, int], iprp: tuple[int, int]
) -> list[int]:
    """Property indices (1-based, in ``ipco`` order) of the primary item."""
    pitm = _box(data, meta[0] + 4, meta[1], b"pitm")
    ipma = _box(data, iprp[0], iprp[1], b"ipma")
    if pitm is None or ipma is None:
        return []
    version = data[pitm[0]] if pitm[0] < len(data) else 0
    primary = _be16(data, pitm[0] + 4) if version == 0 else _be32(data, pitm[0] + 4)
    ipma_version = data[ipma[0]] if ipma[0] < len(data) else 0
    flags = _be32(data, ipma[0]) & 0xFFFFFF
    offset = ipma[0] + 4
    count = _be32(data, offset)
    offset += 4
    for _ in range(min(count, 1024)):
        if ipma_version == 0:
            item = _be16(data, offset)
            offset += 2
        else:
            item = _be32(data, offset)
            offset += 4
        if offset >= len(data):
            return []
        associations = data[offset]
        offset += 1
        indices: list[int] = []
        for _ in range(associations):
            if flags & 1:
                indices.append(_be16(data, offset) & 0x7FFF)
                offset += 2
            else:
                if offset >= len(data):
                    return []
                indices.append(data[offset] & 0x7F)
                offset += 1
        if item == primary:
            return indices
    return []


def _children(data: bytes, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """``(kind, payload start, payload end)`` of every box in a span, in order."""
    found: list[tuple[bytes, int, int]] = []
    offset = start
    while offset + 8 <= min(end, len(data)) and len(found) < 256:
        size = _be32(data, offset)
        header = 8
        if size == 1:
            if offset + 16 > len(data):
                break
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        elif size == 0:
            size = end - offset
        if size < header:
            break
        found.append(
            (data[offset + 4 : offset + 8], offset + header, min(offset + size, end))
        )
        offset += size
    return found


def _box(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int] | None:
    """``(payload start, payload end)`` of the first box of ``kind`` in a span."""
    offset = start
    while offset + 8 <= min(end, len(data)):
        size = _be32(data, offset)
        header = 8
        if size == 1:
            if offset + 16 > len(data):
                return None
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        elif size == 0:
            size = end - offset
        if size < header:
            return None
        if data[offset + 4 : offset + 8] == kind:
            return (offset + header, min(offset + size, end))
        offset += size
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
