"""Closed operation translation and bounded native tensor result validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from .contracts import (
    NATIVE_MUSIC_MAX_FRAMES,
    NATIVE_MUSIC_MIN_DECODE_FRAMES,
    NativeMusicRequest,
    native_music_conditioning_frames,
)
from .upstream_bindings import (
    NativeConstructors,
    NativeHandler,
    NativeResult,
    tensor_samples,
)


@dataclass(frozen=True)
class NativeCall:
    """Opaque native dataclasses owned by the adapter, never a public contract."""

    parameters: object
    configuration: object


@dataclass(frozen=True)
class RenderedAudio:
    """Finite native samples with explicit removal of the known decoder tail."""

    samples: npt.NDArray[np.float32]
    padding_frames_removed: int = 0


def prepare_call(
    request: NativeMusicRequest,
    handler: NativeHandler,
    constructors: NativeConstructors,
    conditioning_path: Path | None,
) -> NativeCall:
    """Translate one operation using the accepted direct base-model policy.

    Continue repaints only the padded tail. Add a part selects the native LEGO
    instruction, rather than pretending a generic repaint isolates an instrument.
    The resolved seed is supplied to both native parameter and batch configuration
    paths. Native defaults cannot silently enable a planner or random seeding.
    """
    if (request.operation == "generate") != (conditioning_path is None):
        raise ValueError("native generation and editing require distinct conditioning")
    if conditioning_path is not None and not conditioning_path.is_absolute():
        raise ValueError("native conditioning path must be private and absolute")
    task = (
        "text2music"
        if request.operation == "generate"
        else ("lego" if request.operation == "add_part" else "repaint")
    )
    instruction = handler.generate_instruction(task, track_name=request.part_class)
    parameters = constructors.parameters(
        task_type=task,
        src_audio=str(conditioning_path) if conditioning_path is not None else None,
        instruction=instruction,
        caption=request.prompt,
        lyrics=request.lyrics,
        instrumental=request.lyrics.strip().casefold() == "[instrumental]",
        duration=request.target_frames / 48000,
        inference_steps=50,
        guidance_scale=7.0,
        seed=request.seed,
        repainting_start=request.region_start_frame / 48000,
        repainting_end=request.region_end_frame / 48000
        if request.region_end_frame is not None
        else -1.0,
        chunk_mask_mode="explicit",
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        dcw_enabled=False,
        enable_normalization=False,
        repaint_strength=0.5,
    )
    configuration = constructors.configuration(
        batch_size=1,
        use_random_seed=False,
        seeds=[request.seed],
        audio_format="wav32",
    )
    return NativeCall(parameters=parameters, configuration=configuration)


def result_samples(result: NativeResult, request: NativeMusicRequest) -> RenderedAudio:
    """Require one complete finite native tensor before any output allocation.

    Read no upstream local output paths. The pinned upstream promises CPU float32
    channel-major audio; a different device, precision, geometry or sample rate
    is an actionable engine mismatch rather than an implicit conversion. Edited
    audio retains the requested timeline geometry. Only the pinned 128-latent
    minimum or explicitly aligned conditioning tail may be removed; every other
    length mismatch fails. Validate the whole tensor before removing padding.
    """
    if not result.success or len(result.audios) != 1:
        raise ValueError("native music inference did not return one successful result")
    audio = result.audios[0]
    tensor = audio.get("tensor")
    if audio.get("sample_rate") != 48000 or not isinstance(tensor, torch.Tensor):
        raise ValueError("native music result is not its expected native-rate tensor")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ValueError("native music result must be CPU float32 audio")
    if (
        tensor.ndim != 2
        or tensor.shape[0] != 2
        or not 48000 <= tensor.shape[1] <= NATIVE_MUSIC_MAX_FRAMES
    ):
        raise ValueError("native music result exceeds bounded stereo geometry")
    samples = tensor_samples(tensor).T
    if not np.isfinite(samples).all():
        raise ValueError("native music result contains nonfinite samples")
    padding = 0
    if request.operation != "generate" and tensor.shape[1] != request.target_frames:
        # Conditioning explicitly ends on a codec boundary; native preparation
        # also has a 128-latent minimum. Neither justifies any other overrun.
        expected_frames = max(
            NATIVE_MUSIC_MIN_DECODE_FRAMES,
            native_music_conditioning_frames(request.target_frames),
        )
        if tensor.shape[1] != expected_frames:
            raise ValueError(
                "native music edit result is not aligned to its requested timeline"
            )
        padding = expected_frames - request.target_frames
        samples = samples[: request.target_frames]
    return RenderedAudio(samples=samples, padding_frames_removed=padding)
