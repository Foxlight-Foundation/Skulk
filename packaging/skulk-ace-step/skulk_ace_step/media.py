"""Bounded context preparation and verified native-rate PCM24 music export."""

from __future__ import annotations

import io
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from .audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    SeparationSource,
    source_from_wav,
)
from .bindings import polyphase, soundfile
from .contracts import (
    NativeMusicRequest,
    NativeMusicWav,
    native_music_conditioning_frames,
)


def read_original(path: Path, expected: SeparationSource) -> npt.NDArray[np.float32]:
    """Read one regular bounded original and verify every admitted WAV fact."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        facts = os.fstat(stream.fileno())
        if not stat.S_ISREG(facts.st_mode) or facts.st_size != expected.size_bytes:
            raise ValueError("native music context is not its admitted regular file")
        data = stream.read(MAX_SEPARATION_SOURCE_BYTES + 1)
    if source_from_wav(data) != expected:
        raise ValueError("native music context bytes differ from the admitted source")
    samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if (
        samples.shape != (expected.frames, expected.channels)
        or rate != expected.sample_rate
        or not np.isfinite(samples).all()
    ):
        raise ValueError(
            "native music WAV decoding disagrees with verified source geometry"
        )
    return samples


@dataclass(frozen=True)
class PreparedConditioning:
    """Exact model input samples and their explicit transformation lineage."""

    samples: npt.NDArray[np.float32]
    source: SeparationSource
    gain: float
    padding_frames: int


def prepare_conditioning(request: NativeMusicRequest) -> PreparedConditioning:
    """Create the exact native context; Continue appends silence without changing the prefix.

    Resampling uses a bounded FIR and the same ceiling frame map as the private
    request contract. No source-rate conversion happens when the source is
    already 48 kHz. Explicit zero-tail padding prevents the pinned codec from
    dropping fractional latent frames. Every transformation remains separate
    from the original and the requested edit region.
    """
    source = request.source
    if source is None or request.source_path is None:
        raise ValueError("native music conditioning requires an editing source")
    original = read_original(Path(request.source_path), source)
    frames = (source.frames * 48000 + source.sample_rate - 1) // source.sample_rate
    if source.sample_rate == 48000:
        converted = original
    else:
        converted = polyphase.resample_poly(
            original,
            48000,
            source.sample_rate,
            axis=0,
            window=("kaiser", 5.0),
            padtype="constant",
            cval=0.0,
        )
    if converted.shape != (frames, source.channels) or not np.isfinite(converted).all():
        raise ValueError("native music resampling returned invalid context geometry")
    stereo = np.repeat(converted, 2, axis=1) if source.channels == 1 else converted
    # Pinned process_src_audio clamps to [-1, 1]. Preserve the relative waveform
    # with recorded common attenuation rather than silently clipping overshoot
    # from a valid float input or its resampling filter.
    peak = float(cast(np.float32, np.abs(stereo).max()))
    gain = (1.0 - 2.0**-22) / peak if peak > 1.0 else 1.0
    if gain != 1.0:
        stereo = stereo * gain
    conditioning_frames = native_music_conditioning_frames(request.target_frames)
    padding_frames = conditioning_frames - request.target_frames
    if request.operation == "continue" or padding_frames:
        padded = np.zeros((conditioning_frames, 2), dtype=np.float32)
        padded[:frames] = stereo
        return PreparedConditioning(
            samples=padded, source=source, gain=gain, padding_frames=padding_frames
        )
    return PreparedConditioning(
        samples=stereo, source=source, gain=gain, padding_frames=0
    )


def write_conditioning(
    path: Path, samples: npt.NDArray[np.float32]
) -> SeparationSource:
    """Exclusively write canonical FLOAT WAV and measure the exact model input bytes."""
    if (
        samples.ndim != 2
        or samples.shape[1] != 2
        or not 48000 <= samples.shape[0] <= 120 * 48000
        or not np.isfinite(samples).all()
    ):
        raise ValueError("native music conditioning is not bounded finite stereo audio")
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        soundfile.write(stream, samples, 48000, format="WAV", subtype="FLOAT")
    with path.open("rb") as stream:
        facts = source_from_wav(stream.read(MAX_SEPARATION_SOURCE_BYTES + 1))
    if (
        facts.frames,
        facts.sample_rate,
        facts.channels,
        facts.encoding,
        facts.bit_depth,
    ) != (samples.shape[0], 48000, 2, "float", 32):
        raise ValueError(
            "native music conditioning WAV differs from its complete geometry"
        )
    return facts


def export_result(
    path: Path, samples: npt.NDArray[np.float32], request: NativeMusicRequest
) -> NativeMusicWav:
    """Exclusively export a complete native-rate result and verify its encoded facts.

    Edits retain exact timeline geometry. Generation may report its actual
    bounded length. Only global attenuation required for integer headroom is
    applied; there is no loudness normalization or source mutation.
    """
    if (
        samples.ndim != 2
        or samples.shape[1] != 2
        or not 48000 <= samples.shape[0] <= 120 * 48000
        or not np.isfinite(samples).all()
    ):
        raise ValueError("native music result is not bounded finite stereo audio")
    if request.operation != "generate" and samples.shape[0] != request.target_frames:
        raise ValueError(
            "native music edit result is not aligned with its requested context"
        )
    peak = float(cast(np.float32, np.abs(samples).max()))
    if not math.isfinite(peak):
        raise ValueError("native music result peak is not finite")
    gain = min(1.0, (1.0 - 2.0**-22) / peak) if peak else 1.0
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        soundfile.write(stream, samples * gain, 48000, format="WAV", subtype="PCM_24")
    with path.open("rb") as stream:
        data = stream.read(MAX_SEPARATION_SOURCE_BYTES + 1)
    facts = source_from_wav(data)
    measured, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if (
        (
            facts.frames,
            facts.sample_rate,
            facts.channels,
            facts.encoding,
            facts.bit_depth,
        )
        != (samples.shape[0], 48000, 2, "pcm", 24)
        or measured.shape != samples.shape
        or rate != 48000
        or not np.isfinite(measured).all()
    ):
        raise ValueError("native music exported WAV differs from its complete result")
    return NativeMusicWav(
        sha256=facts.sha256,
        size_bytes=facts.size_bytes,
        frames=facts.frames,
        peak=float(cast(np.float32, np.abs(measured).max())),
        gain=gain,
    )
