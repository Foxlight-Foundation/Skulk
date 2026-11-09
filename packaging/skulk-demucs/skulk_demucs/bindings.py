"""Audited typed interfaces for missing/incorrect upstream inference stubs.

These declarations describe the exact pinned runtime APIs, exercised by native
tests. They add no dynamic caller choice, substitute implementation or type-check
suppression. Heavy dependencies remain exclusive to the native process.
"""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager
from typing import BinaryIO, Literal, Protocol, TypedDict, Unpack, cast

import numpy as np
import numpy.typing as npt
import torch


class WavWriteOptions(TypedDict):
    """Exact upstream keyword names for the single implemented export format."""

    format: Literal["WAV"]
    subtype: Literal["PCM_24"]


class SoundFile(Protocol):
    """The two libsndfile operations used for bounded float32 PCM conversion."""

    def read(
        self, file: BinaryIO, *, dtype: Literal["float32"], always_2d: Literal[True]
    ) -> tuple[npt.NDArray[np.float32], int]:
        """Decode complete verified uncompressed sample frames."""
        ...

    def write(
        self,
        file: BinaryIO,
        data: npt.NDArray[np.float32],
        samplerate: int,
        **kwargs: Unpack[WavWriteOptions],
    ) -> None:
        """Write one complete PCM24 WAV without per-source normalization."""
        ...


class TensorArchive(Protocol):
    """Metadata access implemented by the safetensors Rust context manager."""

    def metadata(self) -> dict[str, str] | None:
        """Read constructor metadata without dynamically importing model classes."""
        ...


class SafeTensors(Protocol):
    """Pinned safe_open supports CPU device despite its incomplete wheel stub."""

    def safe_open(
        self, filename: str, *, framework: Literal["pt"], device: Literal["cpu"]
    ) -> AbstractContextManager[TensorArchive]:
        """Open local safetensors metadata without legacy pickle deserialization."""
        ...


class TorchArrayBridge(Protocol):
    """Typed NumPy/seed signatures absent from PyTorch 2.8's generated stubs."""

    def from_numpy(self, ndarray: npt.NDArray[np.float32]) -> torch.Tensor:
        """Share CPU float32 array storage with an inference tensor."""
        ...

    def manual_seed(self, seed: int) -> torch.Generator:
        """Seed the owned process's CPU and selected device generators."""
        ...


class TensorArray(Protocol):
    """CPU float32 Tensor.numpy's actual array result, missing from its stub."""

    def numpy(self) -> npt.NDArray[np.float32]:
        """Share the already verified detached CPU float32 tensor as an array."""
        ...


class PolyphaseResampler(Protocol):
    """Pinned SciPy's NumPy float32 API without its untyped public wrapper."""

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
        """Filter frame-major samples with one finite polyphase coefficient array."""
        ...


soundfile_module: object = importlib.import_module("soundfile")
soundfile = cast(SoundFile, cast(object, soundfile_module))
safetensors_module: object = importlib.import_module("safetensors")
safetensors = cast(SafeTensors, cast(object, safetensors_module))
torch_module: object = importlib.import_module("torch")
torch_arrays = cast(TorchArrayBridge, cast(object, torch_module))
scipy_signal_module: object = importlib.import_module("scipy.signal")
polyphase = cast(PolyphaseResampler, cast(object, scipy_signal_module))


def array_from_tensor(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """Expose finite CPU model samples without an untyped array escaping."""
    value: object = tensor
    return cast(TensorArray, value).numpy()
