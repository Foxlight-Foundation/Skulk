"""Offline exact-artifact HTDemucs execution on an explicitly selected device."""

from __future__ import annotations

import hashlib
import importlib
import os
import random
import resource
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, Self, cast, final

import torch
from torch.version import hip

from .bindings import array_from_tensor, safetensors, torch_arrays
from .contracts import (
    SEPARATION_STEMS,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    SeparationDevice,
    SeparationStemName,
)
from .media import export_stems, read_original, resample_original

WEIGHT_SHA256: Final = (
    "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
)
WEIGHT_SIZE: Final = 84_025_440
MODEL_RATE: Final = 44100


class SeparationModel(Protocol):
    """Narrow interface at the pinned upstream package's untyped boundary."""

    sources: list[str]
    samplerate: int
    audio_channels: int

    def to(self, device: str) -> Self:
        """Move the complete model to the explicit native device before readiness."""
        ...

    def eval(self) -> Self:
        """Select inference behavior without changing the declared model geometry."""
        ...


class ModelLoader(Protocol):
    """Only the verified safetensors loader, never upstream model management."""

    def load_safetensors_model(self, path: Path) -> SeparationModel:
        """Construct the exact checked local artifact without downloads."""
        ...


class ModelApplication(Protocol):
    """Closed apply_model options implemented by the immutable Demucs source."""

    def apply_model(
        self,
        model: SeparationModel,
        mix: torch.Tensor,
        *,
        device: str,
        shifts: int,
        split: bool,
        segment: float,
        overlap: float,
        num_workers: int,
        progress: bool,
        callback: Callable[[Mapping[str, object]], None],
    ) -> torch.Tensor:
        """Separate one batch, keeping long-lived input/output tensors on CPU."""
        ...


def verify_weights(path: Path) -> None:
    """Verify full immutable bytes and a closed constructor before upstream load."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        if os.fstat(source.fileno()).st_size != WEIGHT_SIZE:
            raise ValueError("HTDemucs requires the exact pinned safetensors artifact")
        while block := source.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError(
            "HTDemucs safetensors SHA-256 differs from the selected artifact"
        )
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
    # Even safetensors metadata can name executable Python classes. The full
    # known artifact hash is required before the native loader sees any of it.
    if (
        metadata is None
        or metadata.get("klass") != "demucs.htdemucs.HTDemucs"
        or metadata.get("args") != "[]"
        or "structure" in metadata
    ):
        raise ValueError("HTDemucs constructor metadata is not the admitted model")


def probe_device(device: SeparationDevice) -> str:
    """Execute finite FP32 math on the requested lane, refusing implicit fallback."""
    native = "cpu"
    if device == "mps":
        if (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or not torch.backends.mps.is_available()
        ):
            raise ValueError(
                "MPS separation requires available Metal with fallback disabled"
            )
        native = "mps"
    elif device in ("cuda", "rocm"):
        if not torch.cuda.is_available() or (hip is not None) != (device == "rocm"):
            raise ValueError(
                f"the explicit {device} separation lane is unavailable in this runtime"
            )
        native = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    values = torch.ones((4, 4), dtype=torch.float32, device=native)
    measured = (values @ values).cpu()
    if not bool(torch.equal(measured, torch.full((4, 4), 4.0))):
        raise ValueError("separation device failed its FP32 execution probe")
    return native


@final
@dataclass(frozen=True)
class LoadedSeparator:
    """One verified native model retained until its supervised process exits.

    Only the single native request loop invokes this object. Inference does not
    construct a second model or reconfigure the admitted device/thread policy.
    """

    configuration: NativeSeparationServeConfig
    model: SeparationModel
    native_device: str
    load_seconds: float
    weights_sha256: str = WEIGHT_SHA256
    sample_rate: Literal[44100] = MODEL_RATE
    audio_channels: Literal[2] = 2
    stems: tuple[SeparationStemName, ...] = SEPARATION_STEMS

    def run(
        self,
        job: NativeSeparationJob,
        progress: Callable[[NativeSeparationProgress], None],
    ) -> NativeSeparationResult:
        """Separate one original using the already loaded immutable model policy."""
        return _separate_loaded(self, job, progress)


def load_separator(configuration: NativeSeparationServeConfig) -> LoadedSeparator:
    """Verify exact weights and place their complete model before reporting ready."""
    started = time.monotonic()
    torch.set_num_threads(configuration.threads)
    native = probe_device(configuration.device)
    weights = Path(configuration.weights_path)
    verify_weights(weights)
    loader_module: object = importlib.import_module("demucs.hf")
    loader = cast(ModelLoader, cast(object, loader_module))
    model = loader.load_safetensors_model(weights).eval().to(native)
    if (
        tuple(model.sources) != SEPARATION_STEMS
        or model.samplerate != MODEL_RATE
        or model.audio_channels != 2
    ):
        raise ValueError("loaded HTDemucs geometry differs from the selected artifact")
    return LoadedSeparator(configuration, model, native, time.monotonic() - started)


def separate(
    job: NativeSeparationJob, progress: Callable[[NativeSeparationProgress], None]
) -> NativeSeparationResult:
    """Keep the original single-job validation helper over the resident engine.

    Normal worker integration uses the separately supervised resident lifetime.
    This helper loads once for its one job and retains scoped prototype timing.
    """
    started = time.monotonic()
    progress(NativeSeparationProgress(stage="loading"))
    separator = load_separator(
        NativeSeparationServeConfig(
            weights_path=job.weights_path, device=job.device, threads=job.threads
        )
    )
    result = separator.run(job, progress)
    return result.model_copy(
        update={
            "load_seconds": separator.load_seconds,
            "total_seconds": time.monotonic() - started,
        }
    )


def _separate_loaded(
    separator: LoadedSeparator,
    job: NativeSeparationJob,
    progress: Callable[[NativeSeparationProgress], None],
) -> NativeSeparationResult:
    """Produce a verified complete set from exact local input/model bytes.

    Mirror upstream v4.1 normalization and denormalization, using one seeded
    shift and overlap 0.25. No half-precision policy or automatic CPU fallback is
    applied. The owning supervisor enforces admission, deadline and cancellation.
    """
    started = time.monotonic()
    progress(NativeSeparationProgress(stage="verifying"))
    configuration = separator.configuration
    if (job.weights_path, job.device, job.threads) != (
        configuration.weights_path,
        configuration.device,
        configuration.threads,
    ):
        raise ValueError(
            "job cannot change the resident model, device or thread policy"
        )
    native = separator.native_device
    model = separator.model
    samples = read_original(Path(job.source_path), job.source)
    directory = Path(job.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(path.name != "job.json" for path in directory.iterdir()):
        raise ValueError("separation output directory must be new and command-private")
    resampled = resample_original(samples, job.source, MODEL_RATE)
    waveform = torch_arrays.from_numpy(resampled.T.copy())
    if job.source.channels == 1:
        waveform = waveform.expand(2, -1).contiguous()
    frames = (2 * job.source.frames * MODEL_RATE + job.source.sample_rate) // (
        2 * job.source.sample_rate
    )
    if tuple(waveform.shape) != (2, frames) or not bool(torch.isfinite(waveform).all()):
        raise ValueError(
            "source resampling did not preserve finite stereo frame geometry"
        )
    application_module: object = importlib.import_module("demucs.apply")
    application = cast(ModelApplication, cast(object, application_module))
    # Match the actual upstream tensor API, including epsilon for silence.
    reference = waveform.mean(0)
    mean = reference.mean()
    deviation = reference.std() + 1e-8
    normalized = ((waveform - mean) / deviation)[None]
    random.seed(0)
    torch_arrays.manual_seed(0)
    if native == "cuda":
        torch.cuda.reset_peak_memory_stats()
    completed = 0

    def observed(record: Mapping[str, object]) -> None:
        nonlocal completed
        offset = record.get("segment_offset")
        if record.get("state") == "end" and isinstance(offset, int):
            completed = max(
                completed, min(frames, offset + int(job.segment_seconds * MODEL_RATE))
            )
            progress(
                NativeSeparationProgress(stage="separating", completed_frames=completed)
            )

    progress(NativeSeparationProgress(stage="separating"))
    inference_started = time.monotonic()
    with torch.inference_mode():
        estimates = application.apply_model(
            model,
            normalized,
            device=native,
            shifts=1,
            split=True,
            segment=job.segment_seconds,
            overlap=0.25,
            num_workers=0,
            progress=False,
            callback=observed,
        )
        estimates = (estimates * deviation + mean).cpu()
    if tuple(estimates.shape) != (1, 4, 2, frames) or not bool(
        torch.isfinite(estimates).all()
    ):
        raise ValueError("HTDemucs returned nonfinite or incomplete source estimates")
    inference_seconds = time.monotonic() - inference_started
    progress(NativeSeparationProgress(stage="encoding", completed_frames=frames))
    output = export_stems(
        directory,
        job.source,
        [array_from_tensor(estimates[0, index].T) for index in range(4)],
        MODEL_RATE,
    )
    return NativeSeparationResult(
        output=output,
        device=job.device,
        load_seconds=0.0,
        separation_seconds=inference_seconds,
        total_seconds=time.monotonic() - started,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        device_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if native == "cuda"
        else None,
    )
