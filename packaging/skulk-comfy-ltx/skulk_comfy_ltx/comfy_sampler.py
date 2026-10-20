"""Translate Comfy's packed AV state without double-applying inpaint masks."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict, cast

import torch
from torch import Tensor

from skulk_comfy_ltx.res2s import Latents, sample_res2s


class LatentModel(Protocol):
    """The pinned base model's exact packed-modality geometry."""

    latent_shapes: Sequence[Sequence[int]]


class Guider(Protocol):
    """The x0 guider underneath Comfy's input/output inpaint wrapper."""

    inner_model: LatentModel

    def __call__(
        self, values: Tensor, sigma: Tensor, *, model_options: dict[str, object], seed: int
    ) -> Tensor:
        """Predict unblended x0 for the packed video and audio states."""
        ...


class SamplerModel(Protocol):
    """The exact KSAMPLER boundary used by the pinned Comfy revision."""

    inner_model: Guider
    latent_image: Tensor


class SamplerProgress(TypedDict):
    """One completed integration interval, in Comfy's callback vocabulary."""

    i: int
    sigma: Tensor
    sigma_hat: Tensor
    denoised: Tensor
    x: Tensor


def unpack_latents(value: Tensor, shapes: Sequence[Sequence[int]]) -> Latents:
    """Unpack actual Comfy modality geometry without coupling their clocks."""
    batch = value.shape[0]
    if len(shapes) != 2 or len(shapes[0]) != 5 or len(shapes[1]) != 4:
        raise ValueError("LTX HQ requires the joint video/audio latent layout")
    if any(shape[0] != batch for shape in shapes):
        raise ValueError("LTX packed modalities must share their batch size")
    counts = [math.prod(shape[1:]) for shape in shapes]
    if value.numel() != batch * sum(counts):
        raise ValueError("LTX packed state differs from its declared modality shapes")
    parts = torch.split(value.reshape(batch, 1, -1), counts, dim=-1)
    return tuple(part.reshape(tuple(shape)) for part, shape in zip(parts, shapes, strict=True))


def pack_latents(values: Latents) -> Tensor:
    """Pack video and audio in the pinned Comfy storage order."""
    return torch.cat([value.reshape(value.shape[0], 1, -1) for value in values], dim=-1)


@torch.no_grad()
def sample_ltx_res2s(
    model: SamplerModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, object] | None = None,
    callback: Callable[[SamplerProgress], None] | None = None,
    disable: bool | None = None,
) -> Tensor:
    """Run the reference midpoint solver inside Comfy's ordinary sampler lifecycle.

    The loop owns x0 blending; calling KSamplerX0Inpaint would apply fractional
    masks twice. Predictions use the underlying guider with the same prepared
    conditions. Callback exceptions propagate to preserve cancellation.
    """
    del disable
    arguments = {} if extra_args is None else extra_args
    seed = arguments.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError("LTX HQ requires an unsigned 64-bit seed")
    options = arguments.get("model_options", {})
    if not isinstance(options, dict):
        raise ValueError("LTX model options must be an object")
    entries = cast("dict[object, object]", options)
    if any(not isinstance(key, str) for key in entries):
        raise ValueError("LTX model options must have string keys")
    model_options = {str(key): value for key, value in entries.items()}
    mask = arguments.get("denoise_mask")
    if mask is not None and not isinstance(mask, Tensor):
        raise ValueError("LTX denoise mask must be a packed tensor")
    shapes = model.inner_model.inner_model.latent_shapes
    clean = unpack_latents(model.latent_image, shapes)
    masks = unpack_latents(torch.ones_like(x) if mask is None else mask, shapes)
    initial = tuple(
        (value * weight + source * (1 - weight)).to(value.dtype)
        for value, weight, source in zip(unpack_latents(x, shapes), masks, clean, strict=True)
    )

    def predict(values: Latents, sigma: Tensor) -> Latents:
        denoised = model.inner_model(
            pack_latents(values), sigma.expand(x.shape[0]), model_options=model_options, seed=seed
        )
        return unpack_latents(denoised, shapes)

    def progress(index: int, denoised: Latents, values: Latents) -> None:
        if callback is not None:
            callback({"i": index, "sigma": sigmas[index], "sigma_hat": sigmas[index], "denoised": pack_latents(denoised), "x": pack_latents(values)})

    return pack_latents(sample_res2s(initial, sigmas, predict, seed=seed, masks=masks, clean=clean, callback=progress))
