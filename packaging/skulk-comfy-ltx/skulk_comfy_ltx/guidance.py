"""Reference per-modality rescaling after CFG, STG and modality guidance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from skulk_comfy_ltx.comfy_sampler import pack_latents, unpack_latents
from skulk_comfy_ltx.layout import patch_modalities, unpatch_modalities


@runtime_checkable
class _PackedGeometry(Protocol):
    latent_shapes: Sequence[Sequence[int]]


class ModelPatcher(Protocol):
    """The pinned model patcher's clone and post-CFG extension boundary."""

    def clone(self) -> ModelPatcher:
        """Clone this node's patches without mutating another graph branch."""
        ...

    def set_model_sampler_post_cfg_function(self, function: Callable[[dict[str, object]], Tensor], disable_cfg1_optimization: bool = False) -> None:
        """Append one operation after the existing guidance calculations."""
        ...


def rescale_modalities(conditioned: Tensor, prediction: Tensor, shapes: Sequence[Sequence[int]], video_scale: float, audio_scale: float) -> Tensor:
    """Match the reference x0 standard deviation separately for video and audio.

    Geometry comes from the actual packed model state, including appended
    guide tokens. Reference float32 arithmetic and output dtype are retained;
    a degenerate prediction fails visibly rather than inventing an epsilon.
    """
    conditioned_parts = patch_modalities(unpack_latents(conditioned, shapes))
    predicted_parts = patch_modalities(unpack_latents(prediction, shapes))
    output: list[Tensor] = []
    for cond, pred, scale in zip(conditioned_parts, predicted_parts, (video_scale, audio_scale), strict=True):
        if not 0 <= scale <= 1:
            raise ValueError("guidance rescaling must lie in [0, 1]")
        if scale == 0:
            output.append(pred)
            continue
        raw = pred.float()
        denominator = raw.std()
        if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
            raise ValueError("HQ guidance prediction has a degenerate standard deviation")
        factor = scale * (cond.float().std() / denominator) + (1 - scale)
        output.append((raw * factor).to(pred.dtype))
    return pack_latents(unpatch_modalities(tuple(output), shapes))


def patch_guidance_rescale(model: ModelPatcher, *, video_scale: float, audio_scale: float) -> ModelPatcher:
    """Append reference rescaling after all guidance on this cloned graph branch."""
    patched = model.clone()

    def post_cfg(arguments: dict[str, object]) -> Tensor:
        conditioned, prediction, geometry = arguments.get("cond_denoised"), arguments.get("denoised"), arguments.get("model")
        if not isinstance(conditioned, Tensor) or not isinstance(prediction, Tensor) or not isinstance(geometry, _PackedGeometry):
            raise ValueError("HQ rescaling requires the pinned packed AV guidance contract")
        return rescale_modalities(conditioned, prediction, geometry.latent_shapes, video_scale, audio_scale)

    patched.set_model_sampler_post_cfg_function(post_cfg, disable_cfg1_optimization=True)
    return patched
