"""Narrow pinned native audio interfaces where upstream stubs are incomplete."""

from __future__ import annotations

import importlib
from typing import BinaryIO, Literal, Protocol, TypedDict, Unpack, cast

import numpy as np
import numpy.typing as npt


class WavWriteOptions(TypedDict):
    """Exact libsndfile keyword names and the two implemented WAV encodings."""

    format: Literal["WAV"]
    subtype: Literal["PCM_24", "FLOAT"]


class SoundFile(Protocol):
    """Actual libsndfile calls used for verified source and PCM24 result WAVs."""

    def read(
        self, file: BinaryIO, *, dtype: Literal["float32"], always_2d: Literal[True]
    ) -> tuple[npt.NDArray[np.float32], int]:
        """Decode verified uncompressed samples into a bounded frame-major array."""
        ...

    def write(
        self,
        file: BinaryIO,
        data: npt.NDArray[np.float32],
        samplerate: int,
        **kwargs: Unpack[WavWriteOptions],
    ) -> None:
        """Write exact native float conditioning or PCM24 production output."""
        ...


class PolyphaseResampler(Protocol):
    """Pinned bounded SciPy FIR resampler, without a quadratic phase bank."""

    def resample_poly(
        self,
        x: npt.NDArray[np.float32],
        up: int,
        down: int,
        *,
        axis: Literal[0],
        window: tuple[Literal["kaiser"], float],
        padtype: Literal["constant"],
        cval: float,
    ) -> npt.NDArray[np.float32]:
        """Resample verified frame-major samples, retaining time zero."""
        ...


soundfile = cast(SoundFile, cast(object, importlib.import_module("soundfile")))
polyphase = cast(
    PolyphaseResampler, cast(object, importlib.import_module("scipy.signal"))
)
