"""Measure complete inpainting inputs before allocating Comfy pixel batches."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from skulk.shared.types.video_inpainting import VideoInpainting
from skulk.worker.runner.video_input import MediaInputRequest, VerifiedMediaInput

if TYPE_CHECKING:
    from skulk.worker.runner.comfy.graph import ReferenceBinding
    from skulk.worker.runner.video_plan import RenderPlan

MAX_INPAINT_PIXEL_BYTES: Final = 16 * 1024**3
"""Combined decoded graph buffers; model weights and sampling need extra memory."""


@dataclass(frozen=True, slots=True)
class InpaintInputFacts:
    """Private decoder facts and the reservation required by this exact graph."""

    source_width: int
    source_height: int
    frames: int
    mask_frames: int
    allocation_bytes: int


def inpaint_allocation_bytes(width: int, height: int, frames: int, *, source_width: int, source_height: int, mask_frames: int, fps: int, opening: bool) -> int:
    """Reserve cached pixel, latent and framewise pyramid buffers together.

    Comfy may retain upstream outputs throughout both stages. Count those
    batches together, including RGB masks, both green contexts, scale/copy
    scratch, both decodes and blends, and an optional opening. Reserve sixteen
    full/half latent buffers and eight stereo PCM buffers independently. Native
    strength-one guides have no spatial attention mask, so they do not allocate
    a dense guide-attenuation matrix. Transformer activations and model weights
    remain the engine's separate responsibility; this is no hardware minimum.

    Args:
        width: Output width on the native 64-pixel grid.
        height: Output height on the same grid.
        frames: Complete requested 8n+1 output frame count.
        source_width: Fully decoded source width before placement.
        source_height: Fully decoded source height before placement.
        mask_frames: One still mask or the entire source timeline.
        fps: Source and output's common native frame clock.
        opening: Whether a full-canvas opening image is retained.

    Returns:
        Conservative combined bytes, without allocating any image tensors.

    Raises:
        ValueError: Geometry or timeline cannot describe this bounded graph.
    """
    if min(width, height) < 128 or max(width, height) > 4096 or width % 64 or height % 64 or not 1 <= frames <= 2881 or frames % 8 != 1 or not 1 <= fps <= 1000:
        raise ValueError("Inpainting needs a bounded complete output canvas and timeline")
    if not 1 <= source_width <= width or not 1 <= source_height <= height or mask_frames not in (1, frames):
        raise ValueError("Inpainting allocation requires measured source and mask geometry")
    full, half = width * height, width // 2 * (height // 2)
    source = source_width * source_height * (frames + mask_frames) * 12
    pixels = (full * 32 + half * 16) * frames * 4
    images = full * 24 if opening else 0
    latents = ((frames - 1) // 8 + 1) * ((width // 32) * (height // 32) + (width // 64) * (height // 64)) * 128 * 4 * 16
    audio = math.ceil(frames * 48000 / fps) * 2 * 4 * 8
    padded_width, padded_height = 1 << (width - 1).bit_length(), 1 << (height - 1).bit_length()
    workspace = padded_width * padded_height * 192 + 64 * 1024**2
    return source + pixels + images + latents + audio + workspace


def prepare_inpaint_media(plan: RenderPlan, control: VideoInpainting, bindings: tuple[ReferenceBinding, ...], input_directory: Path, *, prepare: Callable[[MediaInputRequest], VerifiedMediaInput | None]) -> InpaintInputFacts | None:
    """Decode source, mask and opening through the selected bounded media helper.

    Args:
        plan: Authoritative output geometry and complete trained timeline.
        control: Source offset, without caller-declared source dimensions.
        bindings: Verified worker-local attachments.
        input_directory: Managed Comfy input root.
        prepare: Selected interpreter's cancellable, bounded decoder effect.

    Returns:
        Complete measured facts, or None when the decoder was cancelled.

    Raises:
        ValueError: Source placement, media alignment or combined buffers fail.
    """
    measured: dict[str, VerifiedMediaInput] = {}
    for role in ("source", "mask", "first_frame"):
        binding = next((item for item in bindings if item.spec.role == role), None)
        if binding is None:
            if role != "first_frame":
                raise ValueError("Inpainting requires one complete source clip and its mask")
            continue
        result = prepare(MediaInputRequest(operation="image" if binding.spec.kind == "image" else "inpaint", source=str(input_directory / binding.input_name), duration_seconds=plan.frame_count / plan.fps, width=plan.width, height=plan.height, fps=plan.fps, maximum_frames=plan.frame_count, exact_frames=True))
        if result is None:
            return None
        if result.width is None or result.height is None:
            raise ValueError("Inpainting requires measured decoder geometry")
        if role == "source":
            if result.frames != plan.frame_count:
                raise ValueError("Inpainting source must cover the complete output timeline")
            control.validate_source_canvas(plan.width, plan.height, source_width=result.width, source_height=result.height)
            if result.audio_channels not in (None, 1, 2):
                raise ValueError("Inpainting requires mono or stereo source audio")
        elif role == "mask":
            source = measured["source"]
            expected_frames = 1 if binding.spec.kind == "image" else source.frames
            if (result.width, result.height) != (source.width, source.height) or result.frames != expected_frames:
                raise ValueError("Inpainting mask must match the complete source dimensions and timeline")
            if result.has_audio:
                raise ValueError("Inpainting mask videos cannot contain an audio track")
        elif (result.width, result.height, result.frames) != (plan.width, plan.height, 1):
            raise ValueError("Inpainting opening image must match the complete output canvas")
        measured[role] = result
    source, mask = measured["source"], measured["mask"]
    assert source.width is not None and source.height is not None
    allocation = inpaint_allocation_bytes(plan.width, plan.height, plan.frame_count, source_width=source.width, source_height=source.height, mask_frames=mask.frames, fps=plan.fps, opening="first_frame" in measured)
    if allocation > MAX_INPAINT_PIXEL_BYTES:
        raise ValueError("Inpainting's combined decoded graph buffers exceed the 16-GiB bound; reduce the output size or duration")
    return InpaintInputFacts(source.width, source.height, source.frames, mask.frames, allocation)
