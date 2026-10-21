"""Validate the Union adapter's reference grid before native guide encoding."""

from collections.abc import Mapping
from math import isfinite
from typing import cast

import torch
from torch import Tensor


def validate_union_inputs(metadata: object, latent: Mapping[str, object], images: Tensor, expected_frames: int) -> None:
    """Require actual half-grid metadata and an exactly aligned finite video.

    Args:
        metadata: Attachment retained by the actual native LoRA loader.
        latent: Base video latent, before any appended guide tokens.
        images: Guide frames fitted to the base canvas by the graph.
        expected_frames: Complete requested pixel timeline on the 8n+1 grid.

    Raises:
        ValueError: Metadata, frame alignment or reference geometry is unsafe.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("Union requires the loaded adapter's reference-grid metadata")
    entries = cast(Mapping[object, object], metadata)
    factors = [value for key, value in entries.items() if isinstance(key, str) and key.endswith("reference_downscale_factor")]
    if len(factors) != 1 or isinstance(factors[0], bool) or not isinstance(factors[0], (str, int, float)):
        raise ValueError("Union adapter must declare exactly one reference downscale factor")
    try:
        factor = float(factors[0])
    except ValueError as error:
        raise ValueError("Union reference downscale factor must be numeric") from error
    if not isfinite(factor) or factor != 2.0:
        raise ValueError("Union requires its trained reference downscale factor of two")
    samples = latent.get("samples")
    if not isinstance(samples, Tensor) or samples.ndim != 5 or samples.shape[0] != 1:
        raise ValueError("Union requires one video latent with five dimensions")
    if expected_frames < 1 or (expected_frames - 1) % 8 or samples.shape[2] != (expected_frames - 1) // 8 + 1:
        raise ValueError("Union guide must match the complete causal video timeline")
    if samples.shape[3] % 2 or samples.shape[4] % 2:
        raise ValueError("Union base latent must be divisible by its half-grid factor")
    if images.ndim != 4 or images.shape[0] != expected_frames or images.shape[1] != samples.shape[3] * 32 or images.shape[2] != samples.shape[4] * 32 or images.shape[3] < 3:
        raise ValueError("Union guide must contain exactly the requested frames and base canvas")
    if not torch.isfinite(images).all():
        raise ValueError("Union guide pixels must be finite")
