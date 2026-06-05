"""Distribution-preserving speculative sampling (Leviathan & Matias 2023).

At temperature > 0, greedy argmax acceptance biases MTP output toward argmax
tokens (Skulk #180). The correct criterion is probability-ratio rejection
sampling over the *effective* sampling distributions — the ones the
configured sampler actually draws from, after top-p / min-p / top-k
filtering and temperature:

    accept draft x with probability  min(1, p(x) / q(x))
    on reject, resample from         norm(max(0, p - q))

where ``q`` is the drafter's effective distribution and ``p`` the
verifier's. This preserves the target distribution exactly regardless of
draft quality (only the acceptance *rate* varies).

The warp deliberately reuses mlx-lm's own ``apply_top_p`` / ``apply_min_p``
/ ``apply_top_k`` in ``make_sampler``'s order, so the computed
probabilities cannot drift from the sampler's semantics. Note mlx-lm
applies the filters to UNTEMPERED logprobs and temperature only at the
categorical draw — the warp mirrors that exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast, final

import mlx.core as mx
from mlx_lm import sample_utils as _mlx_lm_sample_utils

# mlx-lm ships no type stubs; pin the callables once with explicit types.
_apply_top_p = cast(
    "Callable[[mx.array, float], mx.array]", _mlx_lm_sample_utils.apply_top_p
)
_apply_min_p = cast(
    "Callable[[mx.array, float, int], mx.array]", _mlx_lm_sample_utils.apply_min_p
)
_apply_top_k = cast(
    "Callable[[mx.array, int], mx.array]", _mlx_lm_sample_utils.apply_top_k
)


@final
@dataclass(frozen=True)
class SamplingParams:
    """The sampler-shaping knobs MTP must mirror for ratio acceptance.

    Field semantics and defaults match ``mlx_lm.sample_utils.make_sampler``
    as invoked by ``mlx_generate`` (xtc sampling is not used there).
    """

    temperature: float
    top_p: float = 1.0
    min_p: float = 0.05
    min_tokens_to_keep: int = 1
    top_k: int = 0

    @property
    def is_greedy(self) -> bool:
        """True when the sampler reduces to argmax (temperature 0)."""
        return self.temperature == 0.0


def warp_to_probs(logprobs: mx.array, params: SamplingParams) -> mx.array:
    """Return the effective sampling probabilities for *logprobs*.

    Applies mlx-lm's filter chain (top_p → min_p → top_k, on untempered
    logprobs) then the temperature, exactly as ``make_sampler``'s
    categorical draw does. Input must be normalized log-probabilities
    (min_p thresholds against the top token's probability); output is a
    float32 probability vector summing to 1 over the unfiltered support.
    """
    x = logprobs.astype(mx.float32)
    if 0.0 < params.top_p < 1.0:
        x = _apply_top_p(x, params.top_p)
    if params.min_p != 0.0:
        x = _apply_min_p(x, params.min_p, params.min_tokens_to_keep)
    if params.top_k > 0:
        x = _apply_top_k(x, params.top_k)
    return mx.softmax(x * (1.0 / params.temperature), axis=-1)


def ratio_accept(
    draft_token: int,
    draft_probs: mx.array,
    verify_probs: mx.array,
) -> bool:
    """Leviathan-Chen acceptance test for one drafted token.

    Accepts with probability ``min(1, p(x) / q(x))``. The draft token was
    drawn from ``q`` so ``q(x) > 0``; a token filtered out of ``p`` gives
    ratio 0 and always rejects.
    """
    q_x = float(draft_probs[draft_token].item())
    p_x = float(verify_probs[draft_token].item())
    if q_x <= 0.0:
        # Defensive: a sampler/warp mismatch would put us here. Reject —
        # the residual resample still preserves p.
        return False
    ratio = p_x / q_x
    if ratio >= 1.0:
        return True
    return float(mx.random.uniform().item()) < ratio


def residual_sample(draft_probs: mx.array, verify_probs: mx.array) -> int:
    """Sample the replacement token from ``norm(max(0, p - q))``.

    This is the distribution that, combined with ratio acceptance, makes
    the committed token an exact sample from ``p``. Degenerate case (p <= q
    everywhere up to fp error) falls back to sampling ``p`` directly.
    """
    residual = mx.maximum(verify_probs - draft_probs, 0.0)
    total = float(mx.sum(residual).item())
    if total <= 1e-12:
        residual = verify_probs
        total = float(mx.sum(residual).item())
    token = mx.random.categorical(mx.log(residual / total + 1e-30))
    return int(token.item())
