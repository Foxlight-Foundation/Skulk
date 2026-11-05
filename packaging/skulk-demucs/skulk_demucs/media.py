"""Bounded original decoding and all-or-nothing aligned PCM24 stem export."""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from .bindings import polyphase
from .bindings import soundfile as sf
from .contracts import (
    MAX_SEPARATION_SOURCE_BYTES,
    SEPARATION_STEMS,
    AudioSeparationOutput,
    SeparatedStem,
    SeparationSource,
    source_from_wav,
)


def read_original(path: Path, expected: SeparationSource) -> npt.NDArray[np.float32]:
    """Read bounded immutable input bytes once and verify decoded native geometry."""
    with path.open("rb") as source:
        data = source.read(MAX_SEPARATION_SOURCE_BYTES + 1)
    actual = source_from_wav(data)
    if actual != expected:
        raise ValueError("separation source bytes differ from admitted metadata")
    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    decoded: npt.NDArray[np.float32] = np.asarray(samples, dtype=np.float32)
    if (
        rate != expected.sample_rate
        or decoded.shape != (expected.frames, expected.channels)
        or not np.isfinite(decoded).all()
    ):
        raise ValueError("WAV decoder disagrees with verified finite source geometry")
    return decoded


def resample_original(
    samples: npt.NDArray[np.float32], source: SeparationSource, target_rate: int
) -> npt.NDArray[np.float32]:
    """Resample verified original frames without an unbounded phase filter bank.

    Keep mono input mono during conversion; the model adapter duplicates it
    afterward. Return the nearest frame count, retaining the original time zero.
    The pinned SciPy polyphase implementation allocates one FIR of length
    ``20 * max(up, down) + 1`` after reducing the rates by their GCD. Bounding
    both rates to 192 kHz therefore bounds its filter workspace even when the
    rates are coprime. Julius instead builds a bank proportional to the product
    of the reduced rates and can exhaust memory on valid 44,101 Hz input.
    """
    if (
        not 8000 <= target_rate <= 192000
        or samples.shape != (source.frames, source.channels)
        or not np.isfinite(samples).all()
    ):
        raise ValueError("resampling requires finite admitted source geometry and rate")
    if source.sample_rate == target_rate:
        return samples
    frames = (2 * source.frames * target_rate + source.sample_rate) // (
        2 * source.sample_rate
    )
    resampled = polyphase.resample_poly(
        samples,
        target_rate,
        source.sample_rate,
        axis=0,
        window=("kaiser", 5.0),
        padtype="constant",
        cval=0.0,
    )
    ceiling = (
        source.frames * target_rate + source.sample_rate - 1
    ) // source.sample_rate
    if (
        resampled.shape != (ceiling, source.channels)
        or not np.isfinite(resampled).all()
    ):
        raise ValueError("polyphase resampling returned invalid finite source geometry")
    # SciPy returns ceil(N * up / down). Our manifest uses nearest-frame
    # duration; discard only that possible final fractional-frame sample, never
    # pad missing audio or trim the beginning to make a malformed output fit.
    return resampled[:frames]


def export_stems(
    directory: Path,
    source: SeparationSource,
    stems: Sequence[npt.NDArray[np.float32]],
    rate: int,
) -> AudioSeparationOutput:
    """Write all four new WAVs with one common attenuation and verify each file.

    Inputs are stereo frame-major views. Never rescale individual stems, replace
    an existing result or publish a manifest for a partial/invalid set. The owner
    removes the command-private directory on failure or cancellation.
    """
    frames = (2 * source.frames * rate + source.sample_rate) // (2 * source.sample_rate)
    if len(stems) != 4 or any(
        stem.shape != (frames, 2) or not np.isfinite(stem).all() for stem in stems
    ):
        raise ValueError(
            "native separation did not return four finite aligned stereo stems"
        )
    pre_gain_peak = max(float(cast(np.float32, np.abs(stem).max())) for stem in stems)
    if not math.isfinite(pre_gain_peak):
        raise ValueError("native separation peak is not finite")
    # Two PCM24 steps leave room for float32 multiplication rounding without
    # clipping positive full scale in libsndfile's integer conversion.
    common_gain = min(1.0, (1.0 - 2.0**-22) / pre_gain_peak) if pre_gain_peak else 1.0
    records: list[SeparatedStem] = []
    for name, samples in zip(SEPARATION_STEMS, stems, strict=True):
        path = directory / f"{name}.wav"
        # Exclusive creation protects originals and previous attempts even when
        # an invalid caller accidentally reuses a command directory.
        with path.open("xb") as destination:
            sf.write(
                destination, samples * common_gain, rate, format="WAV", subtype="PCM_24"
            )
        with path.open("rb") as encoded:
            data = encoded.read(MAX_SEPARATION_SOURCE_BYTES + 1)
        facts = source_from_wav(data)
        if (
            facts.encoding != "pcm"
            or facts.bit_depth != 24
            or (facts.sample_rate, facts.frames, facts.channels) != (rate, frames, 2)
        ):
            raise ValueError(
                "exported WAV differs from the complete native stem geometry"
            )
        measured, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        peak = float(cast(np.float32, np.abs(measured).max()))
        records.append(
            SeparatedStem(
                name=name,
                size_bytes=facts.size_bytes,
                sha256=facts.sha256,
                sample_rate=rate,
                frames=frames,
                peak=peak,
            )
        )
    return AudioSeparationOutput(
        source=source,
        stems=tuple(records),
        common_gain=common_gain,
        pre_gain_peak=pre_gain_peak,
    )
