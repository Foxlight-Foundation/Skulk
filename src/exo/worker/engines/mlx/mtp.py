"""MTP (Multi-Token Prediction) speculative decoding head for single-node inference.

Qwen3.5 and DeepSeek V3/R1 bake native MTP prediction heads into their
checkpoint weights.  At inference time Skulk loads those heads from the
published sidecar (`mtp.safetensors`) and uses them to draft one candidate
token cheaply — then verifies the draft with a batched two-token forward pass
through the main model.

Phase 1 implements the projection-only head:
    draft = lm_head(shared_norm(eh_proj(concat(hnorm(h), enorm(embed(t_next))))))

The full transformer-block path (mtp.0.transformer.*) is tracked in issue #152
and will land once we have real sidecar weights to validate against.

Weight format (as saved by SWP / mtp_extractor.py):
  mtp.0.enorm.weight                            float16  (hidden_size,)
  mtp.0.hnorm.weight                            float16  (hidden_size,)
  mtp.0.eh_proj.weight                          int4     (hidden_size, 2*hidden_size//8)
  mtp.0.eh_proj.weight_scales                   float16  (hidden_size, 2*hidden_size//group_size)
  mtp.0.eh_proj.weight_biases                   float16  same as scales
  mtp.0.shared_head.norm.weight                 float16  (hidden_size,)   optional
  mtp.0.shared_head.head.weight                 int4     optional; tied to main lm_head

The model prefix varies by checkpoint family:
  - Top-level keys:   mtp.0.enorm.weight
  - Under model.*:    model.mtp.0.enorm.weight
`build_mtp_head` probes both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, cast

import mlx.core as mx

logger = logging.getLogger(__name__)

_NORM_SHIFT_KEYS = (
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
)

_REQUIRED_KEYS = frozenset({"enorm.weight", "hnorm.weight", "eh_proj.weight"})


def _rms_norm(x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
    """Apply RMSNorm with pre-loaded weight vector."""
    rms_sq: mx.array = mx.mean(x * x, axis=-1, keepdims=True) + eps
    return (x / mx.sqrt(rms_sq)) * (1.0 + w)


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


@dataclass
class MTPHead:
    """Projection-only MTP draft head (Phase 1).

    Produces draft logits for one position ahead from:
    - the main model trunk's hidden state at the current position, and
    - the embedding of the next committed token.

    The full transformer-block forward (Phase 2) will supersede this once
    validated sidecar weights are available.
    """

    # Norm weights (stored as deviations: actual weight = 1.0 + stored_value)
    hnorm_w: mx.array
    enorm_w: mx.array

    # eh_proj: Linear(2*H → H), possibly quantized
    eh_proj_w: mx.array
    eh_proj_scales: mx.array | None
    eh_proj_biases: mx.array | None
    eh_proj_bits: int
    eh_proj_group_size: int

    # Shared final norm (optional — main model norm used as fallback)
    shared_norm_w: mx.array | None

    # Callable references into the main model (set during construction)
    _embed_fn: Callable[[mx.array], mx.array] = field(repr=False)
    _head_fn: Callable[[mx.array], mx.array] = field(repr=False)
    _norm_fn: Callable[[mx.array], mx.array] = field(repr=False)

    eps: float = 1e-6

    def draft(self, hidden: mx.array, next_token_id: int) -> mx.array:
        """Return logits for the position *after* next_token_id.

        Args:
            hidden: (hidden_size,) — trunk output at the current position.
            next_token_id: integer token id of the next committed token.

        Returns:
            (vocab_size,) float32 logits.
        """
        # Embed the next token: (1, hidden_size)
        e = self._embed_fn(mx.array([next_token_id]))
        e = _rms_norm(e, self.enorm_w, self.eps)

        # Normalize hidden state: (1, hidden_size)
        h = hidden[None] if hidden.ndim == 1 else hidden
        h = _rms_norm(h, self.hnorm_w, self.eps)

        # Project combined representation: (1, hidden_size)
        combined = mx.concatenate([h, e], axis=-1)
        proj = _dequant_linear(
            combined,
            self.eh_proj_w,
            self.eh_proj_scales,
            self.eh_proj_biases,
            self.eh_proj_bits,
            self.eh_proj_group_size,
        )

        # Apply final norm (shared with main model if not in sidecar)
        if self.shared_norm_w is not None:
            proj = _rms_norm(proj, self.shared_norm_w, self.eps)
        else:
            proj = self._norm_fn(proj)

        # Output projection via shared lm_head: (1, vocab_size)
        logits = self._head_fn(proj)
        return logits[0].astype(mx.float32)  # (vocab_size,)

    def reset(self) -> None:
        """No-op for Phase 1 (no KV cache to reset)."""


def _extract_prefix(weights: dict[str, mx.array]) -> str | None:
    """Detect the key prefix used for MTP layer 0 in the sidecar."""
    for candidate in ("mtp.0.", "model.mtp.0."):
        if all(f"{candidate}{k}" in weights for k in _REQUIRED_KEYS):
            return candidate
    return None


def _load_weight(
    weights: dict[str, mx.array],
    prefix: str,
    name: str,
) -> mx.array | None:
    return weights.get(f"{prefix}{name}")


def _detect_quant(
    weights: dict[str, mx.array],
    prefix: str,
    name: str,
) -> tuple[mx.array | None, mx.array | None, int, int]:
    """Return (scales, biases, bits, group_size) for a possibly-quantized weight."""
    scales = weights.get(f"{prefix}{name}_scales")
    biases = weights.get(f"{prefix}{name}_biases")
    if scales is None or biases is None:
        return None, None, 4, 64
    # Infer group_size from weight and scales shapes.
    # weight shape for 4-bit: (out, in // (32 // bits))
    # scales shape: (out, in // group_size)
    # We default to bits=4, group_size=64 (SWP default).
    return scales, biases, 4, 64


def _get_embed_fn(model: object) -> Callable[[mx.array], mx.array] | None:
    """Extract embed_tokens callable from the main model."""
    lm: object | None = getattr(model, "language_model", None)
    if lm is not None:
        trunk: object | None = getattr(lm, "model", None)
        if trunk is not None:
            embed: object | None = getattr(trunk, "embed_tokens", None)
            if embed is not None and callable(embed):
                return cast(Callable[[mx.array], mx.array], embed)
    # DeepSeek-style: model.model.embed_tokens
    outer_trunk: object | None = getattr(model, "model", None)
    if outer_trunk is not None:
        outer_embed: object | None = getattr(outer_trunk, "embed_tokens", None)
        if outer_embed is not None and callable(outer_embed):
            return cast(Callable[[mx.array], mx.array], outer_embed)
    return None


def _get_head_fn(model: object) -> Callable[[mx.array], mx.array] | None:
    """Extract lm_head callable from the main model."""
    lm: object | None = getattr(model, "language_model", None)
    if lm is not None:
        head: object | None = getattr(lm, "lm_head", None)
        if head is not None and callable(head):
            return cast(Callable[[mx.array], mx.array], head)
    top_head: object | None = getattr(model, "lm_head", None)
    if top_head is not None and callable(top_head):
        return cast(Callable[[mx.array], mx.array], top_head)
    return None


def _get_norm_fn(model: object) -> Callable[[mx.array], mx.array] | None:
    """Extract final norm callable from the main model trunk."""
    lm: object | None = getattr(model, "language_model", None)
    trunk: object | None = (
        getattr(lm, "model", None) if lm is not None else getattr(model, "model", None)
    )
    if trunk is not None:
        norm: object | None = getattr(trunk, "norm", None)
        if norm is not None and callable(norm):
            return cast(Callable[[mx.array], mx.array], norm)
    return None


def build_mtp_head(
    model: object,
    mtp_weights: dict[str, mx.array],
) -> MTPHead | None:
    """Build an MTPHead from sidecar weights and main model callables.

    Returns ``None`` if:
    - the sidecar does not contain the expected Qwen3.5/DeepSeek key layout, or
    - the main model's embed_tokens or lm_head cannot be located.
    """
    prefix = _extract_prefix(mtp_weights)
    if prefix is None:
        logger.warning(
            "MTP sidecar loaded but key layout not recognised — "
            "expected 'mtp.0.{enorm,hnorm,eh_proj}.weight'; running without MTP"
        )
        return None

    embed_fn = _get_embed_fn(model)
    head_fn = _get_head_fn(model)
    norm_fn = _get_norm_fn(model)

    if embed_fn is None or head_fn is None:
        logger.warning(
            "MTP: could not locate embed_tokens / lm_head on model — "
            "running without MTP"
        )
        return None

    hnorm_w = _load_weight(mtp_weights, prefix, "hnorm.weight")
    enorm_w = _load_weight(mtp_weights, prefix, "enorm.weight")
    eh_proj_w = _load_weight(mtp_weights, prefix, "eh_proj.weight")

    if hnorm_w is None or enorm_w is None or eh_proj_w is None:
        logger.warning("MTP: missing required weight tensors — running without MTP")
        return None

    scales, biases, bits, group_size = _detect_quant(mtp_weights, prefix, "eh_proj.weight")
    shared_norm_w = _load_weight(mtp_weights, prefix, "shared_head.norm.weight")

    eps_candidates = [1e-6, 1e-5]
    eps = eps_candidates[0]

    effective_norm_fn: Callable[[mx.array], mx.array]
    if norm_fn is None:
        # Fallback to identity — draft quality degrades gracefully.
        logger.warning("MTP: could not locate final norm; logit scale may drift")

        def _identity_norm(x: mx.array) -> mx.array:
            return x

        effective_norm_fn = _identity_norm
    else:
        effective_norm_fn = norm_fn

    head = MTPHead(
        hnorm_w=hnorm_w,
        enorm_w=enorm_w,
        eh_proj_w=eh_proj_w,
        eh_proj_scales=scales,
        eh_proj_biases=biases,
        eh_proj_bits=bits,
        eh_proj_group_size=group_size,
        shared_norm_w=shared_norm_w,
        _embed_fn=embed_fn,
        _head_fn=head_fn,
        _norm_fn=effective_norm_fn,
        eps=eps,
    )
    logger.info(f"MTP head initialised (prefix={prefix!r}, quantized={scales is not None})")
    return head
