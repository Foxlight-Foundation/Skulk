"""Translate bounded text-to-music requests and validate audio.cpp WAV results.

The public API never exposes audio.cpp's arbitrary options or server-local paths.
This module is pure so the sidecar supervisor can use it without coupling model
truth to HTTP, package installation, or the media transport.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import wave
from dataclasses import dataclass
from typing import Final, cast

from skulk.shared.models.model_cards import (
    MusicCardConfig,
    MusicLyricRequirement,
    MusicModelFamily,
)

MAX_MUSIC_WAV_BYTES: Final = 64 * 1024 * 1024
"""Maximum complete WAV delivered from one generation."""
MAX_AUDIO_CPP_RESPONSE_BYTES: Final = 90 * 1024 * 1024
"""Bound the base64 JSON response before allocating a decoded WAV."""


@dataclass(frozen=True, slots=True)
class MusicWav:
    """Validated PCM16 WAV and measured metadata for one completed generation."""

    data: bytes
    sha256: str
    sample_rate: int
    channels: int
    duration_seconds: float
    size_bytes: int


def audio_cpp_music_request(
    *,
    family: MusicCardConfig,
    prompt: str,
    lyrics: str | None,
    seconds: int,
    seed: int | None,
) -> dict[str, object]:
    """Build the fixed audio.cpp request body for one music family.

    Args:
        family: Typed card contract, including lyric and duration rules.
        prompt: Style or musical description.
        lyrics: Optional caller-authored lyrics, only where allowed.
        seconds: Generation target or budget, not exact MiniMax song length.
        seed: Optional nonnegative deterministic seed.

    Returns:
        A JSON-compatible object for POST /v1/tasks/run with no paths.

    Raises:
        ValueError: The request violates the card or global public contract.
    """
    if not prompt.strip() or len(prompt) > 8000:
        raise ValueError("prompt must contain 1 to 8000 characters")
    if not family.min_seconds <= seconds <= family.max_seconds:
        raise ValueError(
            f"seconds must lie between {family.min_seconds} and {family.max_seconds}"
        )
    if lyrics is not None and (not lyrics.strip() or len(lyrics) > 20_000):
        raise ValueError("lyrics must contain 1 to 20000 characters when supplied")
    if family.lyrics == MusicLyricRequirement.Required and lyrics is None:
        raise ValueError("lyrics are required for this model")
    if family.lyrics == MusicLyricRequirement.Unsupported and lyrics is not None:
        raise ValueError("this model does not accept lyrics")
    if seed is not None and not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")

    options: dict[str, str] = {}
    if lyrics is not None:
        options["lyrics"] = lyrics
    if seed is not None:
        options["seed"] = str(seed)
    if family.family == MusicModelFamily.MiniMaxMusic3:
        options["duration_sec"] = str(seconds)
    elif family.family == MusicModelFamily.AceStep15:
        options["route"] = "text2music"
        options["duration_seconds"] = str(seconds)
    else:
        raise ValueError("unsupported music family")
    return {
        "model": "skulk-music",
        "request": {"text": prompt, "options": options},
    }


def decode_audio_cpp_music_response(body: bytes) -> MusicWav:
    """Decode a bounded audio.cpp JSON result into a validated PCM16 WAV.

    The HTTP caller must cap the byte stream while reading; this function
    repeats the bound before JSON parsing so direct callers cannot bypass it.
    """
    if len(body) > MAX_AUDIO_CPP_RESPONSE_BYTES:
        raise ValueError("audio.cpp response exceeds the 90 MiB envelope limit")
    try:
        payload = cast("object", json.loads(body))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("audio.cpp returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("audio.cpp result must be an object")
    audio = cast("dict[str, object]", payload).get("audio")
    if not isinstance(audio, str):
        raise ValueError("audio.cpp result contains no single WAV output")
    if len(audio) > 4 * ((MAX_MUSIC_WAV_BYTES + 2) // 3):
        raise ValueError("music WAV exceeds 64 MiB")
    try:
        wav = base64.b64decode(audio, validate=True)
    except binascii.Error as error:
        raise ValueError("audio.cpp result contains invalid base64") from error
    if not wav or len(wav) > MAX_MUSIC_WAV_BYTES:
        raise ValueError("music WAV is empty or exceeds 64 MiB")
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("audio.cpp returned a non-WAV audio payload")
    if int.from_bytes(wav[4:8], "little") != len(wav) - 8:
        raise ValueError("WAV RIFF size does not match the delivered bytes")
    try:
        with wave.open(io.BytesIO(wav), "rb") as reader:
            if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
                raise ValueError("music result must be uncompressed PCM16 WAV")
            sample_rate = reader.getframerate()
            channels = reader.getnchannels()
            frames = reader.getnframes()
            if sample_rate <= 0 or channels not in (1, 2) or frames <= 0:
                raise ValueError("WAV has invalid audio geometry")
            if len(reader.readframes(frames)) != frames * channels * 2:
                raise ValueError("WAV sample data is truncated")
    except (wave.Error, EOFError) as error:
        raise ValueError("audio.cpp returned a malformed WAV") from error
    declared_rate = cast("dict[str, object]", payload).get("sample_rate")
    declared_channels = cast("dict[str, object]", payload).get("channels")
    if declared_rate != sample_rate or declared_channels != channels:
        raise ValueError("audio.cpp audio metadata disagrees with the WAV")
    return MusicWav(
        data=wav,
        sha256=hashlib.sha256(wav).hexdigest(),
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=frames / sample_rate,
        size_bytes=len(wav),
    )
