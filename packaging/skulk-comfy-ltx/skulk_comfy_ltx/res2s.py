"""Joint-modality Res2s math for a trusted Comfy extension candidate.

Algorithm adapted from the pinned LTX reference; distribution retains its
community license and source attribution. The owning runner installs only an issued wheel and binds its exact build;
this numerical implementation alone is not model qualification.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

Latents = tuple[Tensor, ...]
Predict = Callable[[Latents, Tensor], Latents]


def _coefficients(h: float) -> tuple[float, float, float]:
    def phi(order: int, value: float) -> float:
        if abs(value) < 1e-10:
            return 1.0 / math.factorial(order)
        remainder = sum(value**index / math.factorial(index) for index in range(order))
        return (math.exp(value) - remainder) / value**order
    a21 = 0.5 * phi(1, -h * 0.5)
    b2 = phi(2, -h) / 0.5
    return a21, phi(1, -h) - b2, b2


def _noise(sample: Tensor, generator: torch.Generator) -> Tensor:
    noise = torch.randn(sample.shape, generator=generator, dtype=torch.float64, device=sample.device)
    noise = (noise - noise.mean()) / noise.std()
    return noise.sub_(noise.mean(dim=(-2, -1), keepdim=True)).div_(noise.std(dim=(-2, -1), keepdim=True))


def _preserve(sample: Tensor, mask: Tensor, clean: Tensor) -> Tensor:
    return (sample * mask + clean.float() * (1 - mask)).to(sample.dtype)


def _inject(sample: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor, noise: Tensor) -> Tensor:
    sigma_up = (next_sigma * 0.5).clamp(max=next_sigma * 0.9999)
    residual = (next_sigma**2 - sigma_up**2).clamp(min=0) ** 0.5
    alpha_ratio = torch.nan_to_num(1 - next_sigma + residual, 1.0)
    sigma_down = residual / alpha_ratio
    sigma_down = torch.where(torch.isnan(sigma_down), next_sigma.to(sigma_down.dtype), sigma_down)
    sigma_up = torch.nan_to_num(sigma_up, 0.0)
    epsilon = (sample - denoised) / (sigma - next_sigma)
    clean_prediction = sample - sigma * epsilon
    updated = alpha_ratio * (clean_prediction + sigma_down * epsilon) + sigma_up * noise
    return torch.where(torch.any(sigma_up == 0) | torch.any(next_sigma == 0), denoised, updated.to(denoised.dtype))



def _to_float64(values: Latents) -> Latents:
    return tuple(value.clone().to(torch.float64) for value in values)

def _to_bfloat16(values: Latents) -> Latents:
    return tuple(value.to(torch.bfloat16) for value in values)

def _preserve_all(values: Latents, masks: Latents, clean: Latents) -> Latents:
    return tuple(_preserve(value, mask, source) for value, mask, source in zip(values, masks, clean, strict=True))

def _difference(values: Latents, anchor: Latents) -> Latents:
    return tuple(value.to(torch.float64) - base for value, base in zip(values, anchor, strict=True))

def _advance(anchor: Latents, epsilon: Latents, coefficient: float) -> Latents:
    return tuple(base + coefficient * delta for base, delta in zip(anchor, epsilon, strict=True))

def _combine(anchor: Latents, epsilon: Latents, second: Latents, h: float, b1: float, b2: float) -> Latents:
    return tuple(base + h * (b1 * delta + b2 * (value.to(torch.float64) - base)) for base, delta, value in zip(anchor, epsilon, second, strict=True))

def _noised(anchor: Latents, values: Latents, previous: Latents, masks: Latents, clean: Latents, sigma: Tensor, next_sigma: Tensor, generator: torch.Generator) -> Latents:
    return tuple(_preserve(_inject(base, value, sigma, next_sigma, _noise(old, generator)), mask, source) for base, value, old, mask, source in zip(anchor, values, previous, masks, clean, strict=True))


@torch.no_grad()
def sample_res2s(
    latents: Latents,
    sigmas: Tensor,
    predict: Predict,
    *,
    seed: int,
    masks: Latents,
    clean: Latents,
    callback: Callable[[int, Latents, Latents], None] | None = None,
) -> Latents:
    """Run the reference joint AV midpoint loop with exact per-modality noise.

    Inputs are already noised model latents; ``predict`` returns x0 for every
    modality in matching order. Frozen regions are blended after every model
    evaluation and noise update. Callback and prediction exceptions propagate,
    allowing the owning engine to stop immediately on cancellation.
    """
    if not latents or len(latents) != len(masks) or len(latents) != len(clean):
        raise ValueError('latents, masks and clean modalities must match')
    if latents[0].device.type not in ('cuda', 'cpu'):
        raise ValueError('Res2s candidate requires float64 on its compute device')
    if sigmas.ndim != 1 or len(sigmas) < 2 or not bool(torch.isfinite(sigmas).all()) or not bool((sigmas[:-1] > sigmas[1:]).all()) or bool(sigmas[-1] < 0):
        raise ValueError('Res2s needs a finite strictly decreasing nonnegative schedule')
    main_noise = torch.Generator(device=latents[0].device).manual_seed(seed)
    sub_noise = torch.Generator(device=latents[0].device).manual_seed((seed + 10000) % (2**64))
    terminal = bool(sigmas[-1] == 0)
    intervals = len(sigmas) - 1
    if terminal:
        sigmas = torch.cat((sigmas[:-1], torch.tensor([0.0011, 0.0], device=sigmas.device)))
    step_sizes = -torch.log(sigmas.cpu().to(torch.float64)[1:] / sigmas.cpu().to(torch.float64)[:-1])
    current: Latents = latents
    for index in range(intervals):
        sigma = sigmas[index].to(torch.float64)
        next_sigma = sigmas[index + 1].to(torch.float64)
        anchor = _to_float64(current)
        first = _preserve_all(predict(current, sigmas[index]), masks, clean)
        h = float(step_sizes[index])
        a21, b1, b2 = _coefficients(h)
        middle_sigma = torch.sqrt(sigma * next_sigma)
        epsilon = _difference(first, anchor)
        middle = _advance(anchor, epsilon, h * a21)
        middle = _noised(anchor, middle, current, masks, clean, sigma, middle_sigma, sub_noise)
        if h < 0.5 and bool(sigmas[index] > 0.03):
            for _ in range(100):
                anchor = _advance(middle, epsilon, -h * a21)
                epsilon = _difference(first, anchor)
        second = _preserve_all(predict(_to_bfloat16(middle), middle_sigma), masks, clean)
        combined = _combine(anchor, epsilon, second, h, b1, b2)
        current = _to_bfloat16(_noised(anchor, combined, current, masks, clean, sigmas[index], sigmas[index + 1], main_noise))
        if callback is not None:
            callback(index, first, current)
    if terminal:
        current = _to_bfloat16(_preserve_all(predict(current, sigmas[intervals]), masks, clean))
    return current
