"""Bounded native reference-audio tokens for both inpainting sampling stages."""

from collections.abc import Callable

import torch
from torch import Tensor


def inpaint_audio_context(samples: Tensor, *, maximum_bytes: int, check_cancelled: Callable[[], None]) -> tuple[Tensor, Tensor]:
    """Patchify the complete encoded source and freeze every audio time position.

    Args:
        samples: Actual audio VAE output in batch/channel/time/frequency order.
        maximum_bytes: Runner budget covering retained samples, tokens and mask.
        check_cancelled: Raise normal cancellation during bounded validation.

    Returns:
        Native batch/time/channel-frequency tokens and a CPU float32 zero mask.
        The input latent is retained verbatim by the node, without re-encoding.

    Raises:
        ValueError: The latent is malformed, nonfinite or exceeds its budget.
        Exception: Cancellation propagates without changing the encoded input.
    """
    if samples.is_nested or samples.ndim != 4 or samples.shape[0] != 1 or min(samples.shape) < 1 or not samples.is_floating_point():
        raise ValueError("Inpainting requires one complete encoded audio latent")
    batch, channels, frames, frequency = samples.shape
    # Permute+reshape may need a distinct contiguous token buffer. Count that
    # allocation even on a layout where torch could return a view instead.
    retained = samples.numel() * samples.element_size() * 2 + batch * frames * 4
    if maximum_bytes < 1 or retained + channels * frequency * 16 > maximum_bytes:
        raise ValueError("Inpainting reference audio exceeds the runner's allocation budget")
    for index in range(frames):
        check_cancelled()
        if not bool(torch.isfinite(samples[:, :, index]).all()):
            raise ValueError("Inpainting reference audio latent must be finite")
    check_cancelled()
    tokens = samples.permute(0, 2, 1, 3).reshape(batch, frames, channels * frequency)
    return tokens, torch.zeros((batch, 1, frames, 1), dtype=torch.float32, device="cpu")
