"""Sampling-mask and free-frame boundaries for the pinned DFR keyframe nodes."""

from collections.abc import Mapping, Sequence, Set
from itertools import pairwise

import torch
from torch import Tensor


def prepare_dfr_latent(latent: Mapping[str, object]) -> dict[str, object]:
    """Retain dense video samples and provide an independent 5D sampling mask.

    Upsampling removes the mask. The pinned detailing node otherwise mistakes
    denoised samples for image anchors. An absent mask means all video tokens
    remain sampleable; a present mask retains every frozen image/guide token.
    Samples and unrelated metadata are shared without copying model-sized data.
    """
    samples = latent.get("samples")
    if not isinstance(samples, Tensor) or samples.ndim != 5 or not samples.is_floating_point() or min(samples.shape) < 1:
        raise ValueError("DFR requires one nonempty floating-point 5D video latent")
    mask = latent.get("noise_mask")
    if mask is None:
        # Match the pinned get_noise_mask contract, independently of latent
        # precision. A compact mask broadcasts across channels and pixels.
        mask = torch.ones((samples.shape[0], 1, samples.shape[2], 1, 1), dtype=torch.float32, device=samples.device)
    else:
        if not isinstance(mask, Tensor) or mask.ndim != 5 or not mask.is_floating_point():
            raise ValueError("DFR sampling masks must be floating-point 5D tensors")
        if mask.shape[0] not in (1, samples.shape[0]) or mask.shape[1] != 1 or mask.shape[2] != samples.shape[2] or mask.shape[3] not in (1, samples.shape[3]) or mask.shape[4] not in (1, samples.shape[4]) or mask.device != samples.device:
            raise ValueError("DFR sampling mask does not match the video latent")
        if not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError("DFR sampling masks must contain finite values from zero to one")
        # Downstream image nodes may write masks. Never expose a cached input
        # mask to those writes, even when no initialization was necessary.
        mask = mask.clone()
    return {**latent, "noise_mask": mask}


def detailing_frame_override(num_frames: int, occupied: Set[int], candidates: Sequence[int]) -> str:
    """Keep the stock grid, or place one detail frame inside its largest free gap.

    Short first/last-image clips can occupy every stock stride candidate. The
    explicit fallback avoids replacing anchors or removing detailing entirely.
    Equal gaps choose the earliest; a completely occupied canvas is rejected.
    """
    if not 2 <= num_frames <= 120001:
        raise ValueError("DFR detailing requires a bounded multi-frame canvas")
    if any(index not in occupied for index in candidates):
        return ""
    boundaries = sorted({0, num_frames - 1, *(index for index in occupied if 0 <= index < num_frames)})
    gaps = [(right - left, left, right) for left, right in pairwise(boundaries) if right - left > 1]
    if not gaps:
        raise ValueError("DFR needs an unoccupied pixel frame for detailing")
    _, left, right = max(gaps, key=lambda gap: (gap[0], -gap[1]))
    return str((left + right) // 2)
