"""DeepSeek V3/R1 sidecar MTP drafter (projection-only, legacy Phase 1).

Draft path (ported unchanged from the original ``MTPHead``):

    proj   = eh_proj(concat([hnorm(h), enorm(embed(t_next))]))   # hidden_first
    logits = lm_head(shared_norm(proj))

Honesty notes (read before trusting acceptance numbers):

- **Concat order is unverified.** ``hidden_first`` is the inherited
  assumption; no real DeepSeek sidecar has been validated end-to-end. The
  Qwen3.5 equivalent assumption turned out to be wrong (issue #192), so when
  real DeepSeek sidecar weights exist, run the offline matrix experiment
  (``probe_phase2.py``) before believing this head.
- DeepSeek sidecars store RMSNorm weights as **actual scales** (≈1.0), so no
  +1 shift is applied — also worth re-verifying against real weights.
- DeepSeek's MTP module also contains a transformer block; this drafter
  omits it (the Qwen experiment measured the block as decisive). Expect this
  to be promoted to a block-running Phase 2 drafter when validated weights
  are available.

The head is stateless: ``observe`` and ``begin_request`` are no-ops.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence, cast, final

import mlx.core as mx

from skulk.worker.engines.mlx.drafters.introspection import (
    get_embed_fn,
    get_head_fn,
    get_norm_fn,
)

logger = logging.getLogger(__name__)

DEEPSEEK_PREFIXES = ("mtp.0.", "model.mtp.0.")
DEEPSEEK_REQUIRED_KEYS = frozenset({"enorm.weight", "hnorm.weight", "eh_proj.weight"})


def _rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """RMSNorm with an actual-scale weight vector.

    Uses ``mx.fast.rms_norm`` (mlx ships no type stubs for ``mx.fast``,
    hence the cast).
    """
    fast = cast("Callable[[mx.array, mx.array, float], mx.array]", mx.fast.rms_norm)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
    return fast(x, w, eps)


def _dequant_linear(
    x: mx.array,
    w: mx.array,
    scales: mx.array | None,
    biases: mx.array | None,
    bits: int,
    group_size: int,
) -> mx.array:
    """Apply a (possibly quantized) linear layer without bias term."""
    if scales is not None and biases is not None:
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases, transpose=True, bits=bits, group_size=group_size
        )
    return x @ w.T


@final
class DeepseekSidecarDrafter:
    """Stateless projection-only MTP drafter for DeepSeek V3/R1 sidecars.

    Satisfies :class:`~skulk.worker.engines.mlx.drafters.protocol.Drafter`.
    Construct via :func:`build_deepseek_sidecar_drafter`.
    """

    def __init__(
        self,
        *,
        hnorm_w: mx.array,
        enorm_w: mx.array,
        eh_proj_w: mx.array,
        eh_proj_scales: mx.array | None,
        eh_proj_biases: mx.array | None,
        eh_proj_bits: int,
        eh_proj_group_size: int,
        shared_norm_w: mx.array | None,
        embed_fn: Callable[[mx.array], mx.array],
        head_fn: Callable[[mx.array], mx.array],
        norm_fn: Callable[[mx.array], mx.array],
        eps: float,
    ) -> None:
        self._hnorm_w = hnorm_w
        self._enorm_w = enorm_w
        self._eh_proj_w = eh_proj_w
        self._eh_proj_scales = eh_proj_scales
        self._eh_proj_biases = eh_proj_biases
        self._eh_proj_bits = eh_proj_bits
        self._eh_proj_group_size = eh_proj_group_size
        self._shared_norm_w = shared_norm_w
        self._embed_fn = embed_fn
        self._head_fn = head_fn
        self._norm_fn = norm_fn
        self._eps = eps

    def begin_request(self, prompt_cache: Sequence[object]) -> None:
        """No-op: the projection-only head carries no per-request state."""
        del prompt_cache

    def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
        """No-op: the projection-only head has no positional state to advance."""
        del hiddens, next_tokens

    def draft(self, hidden: mx.array, next_token: int, depth: int = 1) -> mx.array:
        """Return (1, vocab) float32 logits for the position after *next_token*.

        The projection-only head cannot chain (it has no block to produce a
        successor hidden), so *depth* is ignored and a single row returns.
        """
        del depth
        e = self._embed_fn(mx.array([next_token]))
        e = _rms_norm(e, self._enorm_w, self._eps)
        h = hidden[None] if hidden.ndim == 1 else hidden
        h = _rms_norm(h, self._hnorm_w, self._eps)

        # hidden_first — inherited DeepSeek assumption, unverified (see module
        # docstring); re-run the matrix experiment on real weights before
        # trusting.
        combined = mx.concatenate([h, e], axis=-1)
        proj = _dequant_linear(
            combined,
            self._eh_proj_w,
            self._eh_proj_scales,
            self._eh_proj_biases,
            self._eh_proj_bits,
            self._eh_proj_group_size,
        )
        if self._shared_norm_w is not None:
            proj = _rms_norm(proj, self._shared_norm_w, self._eps)
        else:
            proj = self._norm_fn(proj)
        logits = self._head_fn(proj)
        return logits[0].astype(mx.float32)[None]


def _detect_quant(
    weights: dict[str, mx.array],
    prefix: str,
    name: str,
) -> tuple[mx.array | None, mx.array | None, int, int]:
    """Return (scales, biases, bits, group_size) for a possibly-quantized weight.

    Defaults to bits=4, group_size=64 (the SWP publishing default) when
    quantization metadata is present.
    """
    scales = weights.get(f"{prefix}{name}_scales")
    biases = weights.get(f"{prefix}{name}_biases")
    if scales is None or biases is None:
        return None, None, 4, 64
    return scales, biases, 4, 64


def detect_deepseek_prefix(weights: dict[str, mx.array]) -> str | None:
    """Return the DeepSeek sidecar key prefix, or ``None`` if not DeepSeek-shaped."""
    for prefix in DEEPSEEK_PREFIXES:
        if all(f"{prefix}{k}" in weights for k in DEEPSEEK_REQUIRED_KEYS):
            return prefix
    return None


def build_deepseek_sidecar_drafter(
    model: object,
    weights: dict[str, mx.array],
) -> DeepseekSidecarDrafter | None:
    """Build the legacy DeepSeek projection-only drafter, or ``None``.

    All failures log a warning and return ``None`` — speculation is an
    optimisation, never a crash.
    """
    prefix = detect_deepseek_prefix(weights)
    if prefix is None:
        logger.warning("MTP: DeepSeek sidecar key layout not recognised — running without MTP")
        return None

    embed_fn = get_embed_fn(model)
    head_fn = get_head_fn(model)
    if embed_fn is None or head_fn is None:
        logger.warning("MTP: could not locate embed_tokens / lm_head on model — running without MTP")
        return None

    norm_fn = get_norm_fn(model)
    effective_norm_fn: Callable[[mx.array], mx.array]
    if norm_fn is None:
        # Fallback to identity — draft quality degrades gracefully.
        logger.warning("MTP: could not locate final norm; logit scale may drift")

        def _identity_norm(x: mx.array) -> mx.array:
            return x

        effective_norm_fn = _identity_norm
    else:
        effective_norm_fn = norm_fn

    scales, biases, bits, group_size = _detect_quant(weights, prefix, "eh_proj.weight")
    drafter = DeepseekSidecarDrafter(
        hnorm_w=weights[f"{prefix}hnorm.weight"],
        enorm_w=weights[f"{prefix}enorm.weight"],
        eh_proj_w=weights[f"{prefix}eh_proj.weight"],
        eh_proj_scales=scales,
        eh_proj_biases=biases,
        eh_proj_bits=bits,
        eh_proj_group_size=group_size,
        shared_norm_w=weights.get(f"{prefix}shared_head.norm.weight"),
        embed_fn=embed_fn,
        head_fn=head_fn,
        norm_fn=effective_norm_fn,
        eps=1e-6,
    )
    logger.info(
        f"MTP drafter initialised (family=deepseek-sidecar, phase=1 projection-only, "
        f"prefix={prefix!r}, quantized={scales is not None})"
    )
    return drafter
