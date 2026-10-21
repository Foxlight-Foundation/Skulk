"""Bounded native inpainting mask and trained green-context pixel operations.

The numerical operations follow the pinned LTXVideo inpainting workflow. The
allocation budget and cancellation callback belong to the selected runner;
these helpers neither choose hardware nor silently shorten a source timeline.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor


def _validate_mask(mask: Tensor) -> None:
    if mask.ndim != 3 or not 1 <= mask.shape[0] <= 2881 or min(mask.shape[1:]) < 1 or max(mask.shape[1:]) > 4096:
        raise ValueError("Inpainting mask requires a bounded complete frame batch")
    if mask.dtype != torch.float32:
        raise ValueError("Inpainting mask requires native float32 pixels")


def _validate_values(pixels: Tensor) -> None:
    if not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any()) or bool((pixels > 1).any()):
        raise ValueError("Inpainting pixels must be finite values between zero and one")


def _require_budget(retained_bytes: int, workspace_bytes: int, maximum_bytes: int) -> None:
    if maximum_bytes < 1 or retained_bytes + workspace_bytes > maximum_bytes:
        raise ValueError("Inpainting pixel buffers exceed the selected runner's allocation budget")


def place_inpaint_source(images: Tensor, mask: Tensor, *, width: int, height: int, left: int, top: int, maximum_bytes: int, output_device: torch.device, check_cancelled: Callable[[], None]) -> tuple[Tensor, Tensor]:
    """Place every source frame on an explicit canvas without resize or retiming.

    Args:
        images: Complete float32 RGB source batch, in frame/height/width/channel order.
        mask: Source-sized red-channel mask, static or aligned to every source frame.
        width: Requested output width, on the native 64-pixel grid, up to 4096.
        height: Requested output height, on the native 64-pixel grid, up to 4096.
        left: Nonnegative horizontal source offset, with the full source inside.
        top: Nonnegative vertical source offset, with the full source inside.
        maximum_bytes: Runner budget for both inputs, full outputs and frame workspace.
        output_device: Managed intermediate device receiving both complete batches.
        check_cancelled: Raise normal cancellation between validation and each frame.

    Returns:
        RGB canvas and explicitly expanded mask; padding is white for regeneration.
        The RGB padding is zero and subsequently replaced by the trained green cue.

    Raises:
        ValueError: Geometry, timeline, values or allocation budget are invalid.
        Exception: Cancellation propagates without modifying either input.
    """
    _validate_mask(mask)
    geometry = (width, height, left, top)
    if any(isinstance(value, bool) for value in geometry) or min(width, height) < 128 or max(width, height) > 4096 or width % 64 or height % 64 or min(left, top) < 0:
        raise ValueError("Inpainting requires a bounded 64-pixel output canvas and nonnegative placement")
    if images.ndim != 4 or images.dtype != torch.float32 or images.shape[-1] != 3 or not 1 <= images.shape[0] <= 2881:
        raise ValueError("Inpainting source requires a complete native float32 RGB batch")
    if images.shape[1:3] != mask.shape[1:] or mask.shape[0] not in (1, images.shape[0]) or images.device != mask.device:
        raise ValueError("Inpainting source and mask must have identical geometry and aligned timelines")
    frames, source_height, source_width, _ = images.shape
    if left + source_width > width or top + source_height > height:
        raise ValueError("The complete inpainting source must fit inside the requested canvas")
    # Expanded masks are real retained tensors, even when the caller supplied
    # one still mask. Include them before allocating either full-canvas output.
    retained = (images.numel() + mask.numel() + frames * width * height * 4) * 4
    _require_budget(retained, source_width * source_height * 32 + 4096, maximum_bytes)
    for index in range(frames):
        check_cancelled()
        _validate_values(images[index])
        _validate_values(mask[0 if mask.shape[0] == 1 else index])
    check_cancelled()
    canvas = torch.empty((frames, height, width, 3), dtype=torch.float32, device=output_device)
    masks = torch.empty((frames, height, width), dtype=torch.float32, device=output_device)
    for index in range(frames):
        check_cancelled()
        canvas[index].zero_()
        masks[index].fill_(1)
        canvas[index, top:top + source_height, left:left + source_width].copy_(images[index])
        masks[index, top:top + source_height, left:left + source_width].copy_(mask[0 if mask.shape[0] == 1 else index])
    return canvas, masks


def green_inpaint_context(images: Tensor, mask: Tensor, *, maximum_bytes: int, output_device: torch.device, check_cancelled: Callable[[], None]) -> Tensor:
    """Paint the exact SDR trained key into white mask regions, frame by frame.

    Args:
        images: Complete native float32 RGB frames in batch/height/width/channel order.
        mask: Aligned red-channel mask; one static frame explicitly broadcasts.
        maximum_bytes: Current runner budget for retained inputs, output and workspace.
        output_device: Managed intermediate device receiving the finished context.
        check_cancelled: Raise the normal cancellation exception at bounded intervals.

    Returns:
        A new float32 RGB batch, retaining every source frame and its canvas.

    Raises:
        ValueError: Geometry, pixels or allocation budget violate the native contract.
        Exception: Cancellation is propagated without modifying either input.
    """
    _validate_mask(mask)
    if images.ndim != 4 or not 1 <= images.shape[0] <= 2881 or images.shape[-1] != 3 or images.dtype != torch.float32:
        raise ValueError("Inpainting context requires a complete native float32 RGB batch")
    if images.shape[1:3] != mask.shape[1:] or mask.shape[0] not in (1, images.shape[0]) or images.device != mask.device:
        raise ValueError("Inpainting source and mask must have identical geometry and aligned timelines")
    frame_bytes = images.shape[1] * images.shape[2] * 4
    # Count actual distinct input/output allocations, not overlapping host/GPU
    # telemetry. The frame reserve bounds validation and compositing temporaries.
    retained = (images.numel() + mask.numel() + images.numel()) * 4
    _require_budget(retained, frame_bytes * 16 + 4096, maximum_bytes)
    for index in range(images.shape[0]):
        check_cancelled()
        _validate_values(images[index])
        _validate_values(mask[0 if mask.shape[0] == 1 else index])
    check_cancelled()
    output = torch.empty(images.shape, dtype=torch.float32, device=output_device)
    # The upstream native/HDR workflow deliberately keeps this SDR key. Color
    # conversion here would change the IC-LoRA's trained latent cue.
    green = torch.tensor((102, 255, 0), device=images.device).float() / 255
    for index in range(images.shape[0]):
        check_cancelled()
        selected = mask[0 if mask.shape[0] == 1 else index].unsqueeze(-1)
        prepared = images[index] * (1 - selected) + green.view(1, 1, 3) * selected
        output[index].copy_(prepared)
    return output


def dilate_inpaint_mask(mask: Tensor, spatial_radius: int, *, maximum_bytes: int, check_cancelled: Callable[[], None]) -> Tensor:
    """Apply native spatial max-pooling and the strict greater-than-half mask cut.

    Args:
        mask: Already stage-resized native float32 masks, with no temporal dilation.
        spatial_radius: Half the odd spatial kernel size, from zero through 300.
        maximum_bytes: Current runner budget for retained masks and frame workspace.
        check_cancelled: Raise the normal cancellation exception between frames.

    Returns:
        A new float32 binary mask batch on the input device, without truncation.

    Raises:
        ValueError: The mask, radius or allocation budget is invalid.
        Exception: Cancellation is propagated without changing the input.
    """
    _validate_mask(mask)
    if isinstance(spatial_radius, bool) or not 0 <= spatial_radius <= 300:
        raise ValueError("Inpainting spatial dilation radius must lie between zero and 300")
    frame_bytes = mask.shape[1] * mask.shape[2] * 4
    _require_budget(mask.numel() * 8, frame_bytes * 8 + 4096, maximum_bytes)
    for frame in mask:
        check_cancelled()
        _validate_values(frame)
    check_cancelled()
    output = torch.empty_like(mask)
    for index in range(mask.shape[0]):
        check_cancelled()
        frame = mask[index:index + 1].unsqueeze(1)
        expanded = functional.max_pool2d(frame, kernel_size=spatial_radius * 2 + 1, stride=1, padding=spatial_radius) if spatial_radius else frame
        output[index].copy_((expanded[0, 0] > 0.5).float())
    return output
