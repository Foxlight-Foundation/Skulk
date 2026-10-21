"""Frame-bounded native Laplacian inpainting blends on the selected device."""

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from kornia.geometry.transform.pyramid import (
    build_laplacian_pyramid,
    build_pyramid,
    pyrup,
)
from torch import Tensor


def inpaint_blend_allocation(width: int, height: int, frames: int) -> tuple[int, int]:
    """Return conservative retained-pixel and one-frame workspace reservations.

    Args:
        width: Actual stage width, between two and 4096 pixels.
        height: Actual stage height, between two and 4096 pixels.
        frames: Complete bounded source timeline, without trimming.

    Returns:
        Distinct input/output tensor bytes and a padded-frame filter reservation.
        Model weights, latent sampling and other graph nodes require their own
        reservations; this is not a universal model or hardware memory minimum.

    Raises:
        ValueError: A dimension is outside the supported pixel-operation bounds.
    """
    if not min(width, height) >= 2 or max(width, height) > 4096 or not 1 <= frames <= 2881:
        raise ValueError("Inpainting blend requires bounded complete pixel geometry")
    padded_width, padded_height = 1 << (width - 1).bit_length(), 1 << (height - 1).bit_length()
    # Three RGB batches (two inputs and output), one mask, and bounded
    # Gaussian/Laplacian/filter/synthesis buffers for one padded frame.
    return frames * width * height * 40, padded_width * padded_height * 192 + 64 * 1024 * 1024


def _pad(image: Tensor, padding: tuple[int, int]) -> Tensor:
    right, down = padding
    return functional.pad(image, (0, right, 0, down), mode="reflect") if right or down else image


def _blend_mask(mask: Tensor, radius: int) -> Tensor:
    if radius == 0:
        return mask
    height, width = mask.shape[-2:]
    scale = 64 / max(height, width)
    low = mask if max(height, width) == 64 else functional.interpolate(mask, size=(max(1, int(round(height * scale))), max(1, int(round(width * scale)))), mode="bilinear", align_corners=False)
    expanded = functional.max_pool2d(low, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return functional.interpolate(expanded, size=(height, width), mode="bilinear", align_corners=False)


def _blend_frame(generated: Tensor, context: Tensor, mask: Tensor, levels: int) -> Tensor:
    generated_levels = build_laplacian_pyramid(generated, max_level=levels)
    context_levels = build_laplacian_pyramid(context, max_level=levels)
    mask_levels = build_pyramid(mask, levels, "reflect", False)
    output = generated_levels[-1] * mask_levels[-1] + context_levels[-1] * (1 - mask_levels[-1])
    for index in range(len(generated_levels) - 2, -1, -1):
        residual = generated_levels[index] * mask_levels[index] + context_levels[index] * (1 - mask_levels[index])
        output = pyrup(output) + residual
    return output


def blend_inpaint_images(generated: Tensor, context: Tensor, mask: Tensor, low_res_dilation: int, *, maximum_bytes: int, device: torch.device, output_device: torch.device, check_cancelled: Callable[[], None]) -> Tensor:
    """Blend a complete decoded take with its green-prepared native context.

    Args:
        generated: Complete native float32 RGB batch; white mask selects it.
        context: Equally shaped green-prepared source frames for this stage.
        mask: One aligned float32 mask per frame; static masks expand upstream.
        low_res_dilation: Native long-side-64 mask radius, zero through 15.
        maximum_bytes: Current runner allocation budget, never a card minimum.
        device: Selected accelerator for the bounded native pyramid operations.
        output_device: Managed intermediate device holding the finished batch.
        check_cancelled: Raise normal cancellation between validation and frames.

    Returns:
        A new float32 RGB take, cropped to the exact original stage canvas.

    Raises:
        ValueError: Pixels, geometry, timeline or budget violate the contract.
        Exception: Cancellation propagates and no input is changed in place.
    """
    if generated.ndim != 4 or generated.shape[-1] != 3 or generated.shape != context.shape or mask.ndim != 3 or mask.shape != generated.shape[:3]:
        raise ValueError("Inpainting blend requires identical full source, output and mask timelines")
    if any(value.dtype != torch.float32 for value in (generated, context, mask)):
        raise ValueError("Inpainting blend requires native float32 pixels")
    if isinstance(low_res_dilation, bool) or not 0 <= low_res_dilation <= 15:
        raise ValueError("Inpainting blend dilation must lie between zero and 15")
    frames, height, width, _ = generated.shape
    retained, workspace = inpaint_blend_allocation(width, height, frames)
    if maximum_bytes < 1 or retained + workspace > maximum_bytes:
        raise ValueError("Inpainting blend exceeds the selected runner's allocation budget")
    for values in (generated, context, mask):
        for frame in values:
            check_cancelled()
            if not bool(torch.isfinite(frame).all()) or bool((frame < 0).any()) or bool((frame > 1).any()):
                raise ValueError("Inpainting blend pixels must be finite and between zero and one")
    padding = ((1 << (width - 1).bit_length()) - width, (1 << (height - 1).bit_length()) - height)
    levels = min(7, min(height + padding[1], width + padding[0]).bit_length() - 1)
    check_cancelled()
    output = torch.empty(generated.shape, dtype=torch.float32, device=output_device)
    # Native math is independent for each frame. Keeping one frame in flight
    # bounds accelerator workspace instead of retaining eight padded pyramids.
    for index in range(frames):
        check_cancelled()
        guide_mask = _blend_mask(mask[index:index + 1].unsqueeze(1), low_res_dilation)
        generated_frame = _pad(generated[index:index + 1].permute(0, 3, 1, 2), padding).to(device)
        context_frame = _pad(context[index:index + 1].permute(0, 3, 1, 2), padding).to(device)
        guide_mask = _pad(guide_mask, padding).to(device)
        blended = _blend_frame(generated_frame, context_frame, guide_mask, levels)
        output[index].copy_(blended[0, :, :height, :width].clamp(0, 1).permute(1, 2, 0))
        # Drop the last frame before allocating the next; an otherwise retained
        # local can double the peak even with a one-frame loop.
        del generated_frame, context_frame, guide_mask, blended
    return output
