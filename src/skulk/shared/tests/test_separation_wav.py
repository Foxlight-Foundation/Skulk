"""Admit complete native WAVs, never format labels or guessed sample geometry."""

import struct

import pytest

from skulk.shared.types.audio_separation import source_from_wav


def _chunk(name: bytes, data: bytes) -> bytes:
    return name + len(data).to_bytes(4, "little") + data + b"\0" * (len(data) % 2)


def _wav(
    *,
    bits: int = 16,
    code: int = 1,
    frames: int = 8000,
    extra_format: bytes = b"",
    samples: bytes | None = None,
) -> bytes:
    alignment = bits // 8
    body = _chunk(
        b"fmt ",
        struct.pack("<HHIIHH", code, 1, 8000, 8000 * alignment, alignment, bits)
        + extra_format,
    )
    body += _chunk(b"JUNK", b"metadata")
    body += _chunk(b"data", bytes(frames * alignment) if samples is None else samples)
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WAVE" + body


@pytest.mark.parametrize(
    "bits,code", [(8, 1), (16, 1), (24, 1), (32, 1), (32, 3), (64, 3)]
)
def test_uncompressed_formats_preserve_native_identity(bits: int, code: int) -> None:
    """PCM/float sources retain real precision rather than a production preset."""
    data = _wav(bits=bits, code=code)
    facts = source_from_wav(data)
    assert (
        facts.frames,
        facts.sample_rate,
        facts.channels,
        facts.bit_depth,
        facts.size_bytes,
    ) == (8000, 8000, 1, bits, len(data))
    assert facts.encoding == ("pcm" if code == 1 else "float")


def test_extensible_pcm_and_odd_padded_chunks_are_accepted() -> None:
    """Common production PCM24 and RIFF alignment require no silent conversion."""
    extra = struct.pack("<HHI", 22, 24, 4) + bytes.fromhex(
        "0100000000001000800000aa00389b71"
    )
    assert (
        source_from_wav(_wav(bits=24, code=0xFFFE, extra_format=extra)).bit_depth == 24
    )
    assert source_from_wav(_wav(bits=8, frames=8001)).frames == 8001


@pytest.mark.parametrize(
    "failure",
    [
        "truncated",
        "size",
        "duplicate_format",
        "duplicate_samples",
        "incomplete_frame",
        "compressed",
        "precision",
        "empty",
        "nan",
        "infinity",
        "float32_overflow",
    ],
)
def test_invalid_container_and_samples_fail_before_placement(failure: str) -> None:
    """Malformed, compressed and nonfinite samples never reach the native loader."""
    data = _wav()
    if failure == "truncated":
        data = data[:-1]
    elif failure == "size":
        data = data[:4] + (len(data) + 1).to_bytes(4, "little") + data[8:]
    elif failure in ("duplicate_format", "duplicate_samples"):
        added = (
            _chunk(b"fmt ", data[20:36])
            if failure == "duplicate_format"
            else _chunk(b"data", b"\0\0")
        )
        body = data[8:] + added
        data = b"RIFF" + len(body).to_bytes(4, "little") + body
    elif failure == "incomplete_frame":
        data = _wav(samples=bytes(16001))
    elif failure == "compressed":
        data = _wav(code=7)
    elif failure == "precision":
        data = _wav(bits=16, code=3)
    elif failure == "empty":
        data = _wav(frames=0)
    else:
        sample = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "float32_overflow": 1e100,
        }[failure]
        data = _wav(
            code=3, bits=64, samples=struct.pack("<d", sample) + bytes(7999 * 8)
        )
    with pytest.raises(ValueError):
        source_from_wav(data)
