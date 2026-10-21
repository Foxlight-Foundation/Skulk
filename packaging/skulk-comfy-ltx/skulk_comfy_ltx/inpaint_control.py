"""Validate the loaded inpainting adapter's full-grid native guide contract."""

from collections.abc import Callable, Mapping
from math import isfinite
from typing import cast

import torch
from torch import Tensor


def validate_inpaint_guide(metadata: object, latent: Mapping[str, object], images: Tensor, expected_frames: int, *, check_cancelled: Callable[[], None]) -> None:
    """Require factor-one metadata and a complete stage-aligned native guide.

    Args:
        metadata: Attachment on the actual sampler's loaded IC-LoRA model.
        latent: Current video latent before appended guide tokens.
        images: Complete green-prepared RGB guide at this stage's canvas.
        expected_frames: Exact complete source timeline on the 8n+1 grid.
        check_cancelled: Raise normal cancellation between frame validations.

    Raises:
        ValueError: Adapter, pixels, geometry or timeline do not match this recipe.
        Exception: Cancellation propagates before native guide VAE allocation.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("Inpainting requires the loaded adapter's reference-grid metadata")
    entries = cast(Mapping[object, object], metadata)
    factors = [value for key, value in entries.items() if isinstance(key, str) and key.endswith("reference_downscale_factor")]
    if len(factors) != 1 or isinstance(factors[0], bool) or not isinstance(factors[0], (str, int, float)):
        raise ValueError("Inpainting adapter must declare exactly one reference downscale factor")
    try:
        factor = float(factors[0])
    except ValueError as error:
        raise ValueError("Inpainting reference downscale factor must be numeric") from error
    if not isfinite(factor) or factor != 1.0:
        raise ValueError("Inpainting requires its trained reference downscale factor of one")
    samples = latent.get("samples")
    if not isinstance(samples, Tensor) or samples.ndim != 5 or samples.shape[0] != 1 or min(samples.shape) < 1:
        raise ValueError("Inpainting requires one native video latent")
    if not 1 <= expected_frames <= 2881 or (expected_frames - 1) % 8 or samples.shape[2] != (expected_frames - 1) // 8 + 1:
        raise ValueError("Inpainting guide must match the complete causal video timeline")
    if images.ndim != 4 or images.shape != (expected_frames, samples.shape[3] * 32, samples.shape[4] * 32, 3) or images.dtype != torch.float32:
        raise ValueError("Inpainting guide must match every frame and the actual stage canvas")
    for frame in images:
        check_cancelled()
        if not bool(torch.isfinite(frame).all()) or bool((frame < 0).any()) or bool((frame > 1).any()):
            raise ValueError("Inpainting guide pixels must be finite and between zero and one")
