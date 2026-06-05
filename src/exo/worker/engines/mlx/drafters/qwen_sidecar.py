"""Qwen3.5-family sidecar MTP drafter (Phase 2: full transformer block).

Draft path, validated offline at 72.4% argmax agreement on Qwen3.5-2B
(Skulk issue #192, kite3 matrix experiment 2026-06-04):

    x      = fc(concat([enorm(embed(t_next)), hnorm(h)]))      # embed_first
    out    = decoder_block(x, cache=private_kv)                # mtp.layers.0
    logits = lm_head(final_norm(out))                          # mtp.norm

Three family facts make or break this head — each alone was measured at
0–20% agreement, together 72.4%:

1. **Zero-centered norms.** Qwen3.5 checkpoints store RMSNorm weights as
   deviation-from-1; mlx-lm's ``sanitize()`` adds +1.0 for trunk weights but
   sidecars bypass sanitize, so the shift is applied here at load
   (tell-tale: ``mean(pre_fc_norm_hidden) == -0.28`` raw).
2. **``embed_first`` concat order** — not the DeepSeek-style hidden-first.
3. **The ``mtp.layers.0`` block must run.** It is architecturally one of the
   family's own full-attention decoder layers, so it is instantiated from
   the loaded trunk's layer class and strict-loaded from the sidecar.

The block gives the drafter private KV state keyed to absolute positions via
the pair-stream contract (see :mod:`.protocol`): the loop feeds every
committed position's *(hidden, next-token)* pair exactly once, so RoPE
positions in the private cache always match the target sequence.

Quantized (backbone-matched) sidecars are a planned SWP artifact; today only
the published bf16 sidecars are supported — quantization metadata in the
sidecar is rejected loudly at build time rather than silently mis-loaded.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Literal, Sequence, cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import base as _mlx_lm_base
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.drafters.introspection import (
    build_sibling_attention_layer,
    detect_quantization,
    get_embed_fn,
    get_head_fn,
)

logger = logging.getLogger(__name__)

QWEN35_PREFIX = "mtp."
QWEN35_BLOCK_PREFIX = "mtp.layers.0."
QWEN35_REQUIRED_KEYS = frozenset({
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.fc.weight",
    "mtp.norm.weight",
})

ConcatOrder = Literal["embed_first", "hidden_first"]
NormConvention = Literal["zero_centered", "actual_scale"]


def _rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """RMSNorm with an *actual-scale* weight vector (post-shift if needed).

    Uses ``mx.fast.rms_norm`` to match the kernel the offline validation ran
    on (mlx ships no type stubs for ``mx.fast``, hence the cast).
    """
    fast = cast("Callable[[mx.array, mx.array, float], mx.array]", mx.fast.rms_norm)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
    return fast(x, w, eps)


def _attention_mask(x: mx.array, cache: KVCache) -> mx.array | str | None:
    """Causal mask for the drafter block, offset-aware via the private cache.

    mlx-lm's ``base.create_attention_mask`` is unannotated, hence the cast.
    """
    make_mask = cast(
        "Callable[[mx.array, KVCache], mx.array | str | None]",
        _mlx_lm_base.create_attention_mask,
    )
    return make_mask(x, cache)


@final
class QwenSidecarDrafter:
    """Stateful Phase-2 MTP drafter for Qwen3.5-family sidecars.

    Satisfies :class:`~exo.worker.engines.mlx.drafters.protocol.Drafter`.
    Construct via :func:`build_qwen_sidecar_drafter`, never directly — the
    builder owns weight validation and family-fact resolution.
    """

    def __init__(
        self,
        *,
        hnorm_w: mx.array,
        enorm_w: mx.array,
        fc_w: mx.array,
        final_norm_w: mx.array,
        block: nn.Module,
        embed_fn: Callable[[mx.array], mx.array],
        head_fn: Callable[[mx.array], mx.array],
        concat_order: ConcatOrder,
        eps: float,
    ) -> None:
        self._hnorm_w = hnorm_w
        self._enorm_w = enorm_w
        # fc as a real Linear module so quantize_to_match can swap it for a
        # QuantizedLinear; Linear(x) computes x @ W.T, matching the published
        # raw-matmul semantics exactly.
        fc_linear = nn.Linear(fc_w.shape[1], fc_w.shape[0], bias=False)
        fc_linear.weight = fc_w
        self._fc: nn.Module = fc_linear
        self._final_norm_w = final_norm_w
        self._block = block
        self._embed_fn = embed_fn
        self._head_fn = head_fn
        self._concat_order: ConcatOrder = concat_order
        self._eps = eps
        self._cache = KVCache()

    def quantize_to_match(self, *, group_size: int, bits: int) -> None:
        """Quantize the sidecar block and fc to the target's precision.

        Published sidecars ship bf16; on a quantized target the unquantized
        block dominates draft cost (~350MB of MLP weight reads per draft on
        Qwen3.5-9B — 10.7ms against the 1.2ms K/V-only observe). Drafts are
        always verified by the target, so reduced drafter precision can only
        nudge acceptance, never correctness.
        """
        # mlx ships no annotations for nn.quantize, hence the cast.
        quantize_module = cast("Callable[..., object]", nn.quantize)
        quantize_module(self._block, group_size=group_size, bits=bits)
        fc = self._fc
        if isinstance(fc, nn.Linear):
            self._fc = nn.QuantizedLinear.from_linear(
                fc, group_size=group_size, bits=bits
            )

    def begin_request(self, prompt_cache: Sequence[object]) -> None:
        """Reset the private block KV cache for a new request.

        The target-model *prompt_cache* is unused: sidecar heads draft from
        trunk hiddens, not the target's KV.
        """
        del prompt_cache
        self._cache = KVCache()

    def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
        """Advance the block KV cache over committed pairs without drafting.

        See the pair-stream contract in :mod:`.protocol` — the loop calls
        this for the prompt bulk-ingest and for the single position skipped
        on each verify resolution.
        """
        x = self._project(hiddens, next_tokens)
        mask = _attention_mask(x, self._cache)
        # Output discarded — this call exists to write the block's KV entries.
        self._block(x, mask=mask, cache=self._cache)

    def draft(self, hidden: mx.array, next_token: int, depth: int = 1) -> mx.array:
        """Consume one pair and return (K, vocab) chained greedy draft logits.

        Depth-1 is the trained regime. Deeper rows chain the block on its
        own output hidden — out-of-distribution for the single trained
        block, so conditional acceptance decays fast (measured 86.8% →
        39.2% → 28.2% at depths 1..3 on Qwen3.5-9B); depth 2 is the
        practical ceiling for these heads. Chained cache entries are
        speculative (block-output hiddens, not the canonical trunk-hidden
        pairs) and are trimmed before returning — only the consumed input
        pair persists, preserving the pair-stream contract.
        """
        rows: list[mx.array] = []
        step_hidden = hidden
        step_token = next_token
        chained = 0
        for step in range(max(depth, 1)):
            x = self._project(step_hidden[None], mx.array([step_token]))
            mask = _attention_mask(x, self._cache)
            out = self._block(x, mask=mask, cache=self._cache)
            if step > 0:
                chained += 1
            logits = self._head_fn(_rms_norm(out, self._final_norm_w, self._eps))
            row = logits[0, -1].astype(mx.float32)
            rows.append(row)
            if step + 1 < max(depth, 1):
                # Greedy-select the draft to feed the next chain step (MTP is
                # greedy-only; the loop's sampler is argmax at temp=0).
                step_token = int(mx.argmax(row).item())
                step_hidden = out[0, -1, :]
        if chained:
            self._cache.trim(chained)
        return mx.stack(rows)

    def _project(self, hiddens: mx.array, next_tokens: mx.array) -> mx.array:
        """fc projection of normed (embedding, hidden) pairs: (T,H),(T,) → (1,T,H)."""
        e = self._embed_fn(next_tokens)
        en = _rms_norm(e, self._enorm_w, self._eps)
        hn = _rms_norm(hiddens, self._hnorm_w, self._eps)
        parts = [en, hn] if self._concat_order == "embed_first" else [hn, en]
        combined = mx.concatenate(parts, axis=-1)
        return self._fc(combined)[None]


_PER_EXPERT_KEY = re.compile(
    r"^(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _stack_per_expert_block_weights(
    pairs: list[tuple[str, mx.array]],
) -> list[tuple[str, mx.array]]:
    """Normalize raw per-expert MoE keys to mlx-lm's stacked switch layout.

    SWP sidecars preserve the checkpoint's original per-expert keys
    (``mlp.experts.<n>.{gate,up,down}_proj.weight``), but mlx-lm decoder
    layers hold stacked ``SwitchGLU`` tensors
    (``mlp.switch_mlp.<proj>.weight``, shape ``(num_experts, out, in)``) —
    the conversion normally happens in the family ``sanitize()``, which the
    sidecar strict-load path never runs. Dense sidecars pass through
    untouched.
    """
    stacked_groups: dict[tuple[str, str], dict[int, mx.array]] = {}
    passthrough: list[tuple[str, mx.array]] = []
    for key, value in pairs:
        match = _PER_EXPERT_KEY.match(key)
        if match is None:
            passthrough.append((key, value))
            continue
        group_key = (match.group(1), match.group(3))
        stacked_groups.setdefault(group_key, {})[int(match.group(2))] = value
    for (prefix, projection), by_expert in stacked_groups.items():
        expert_count = len(by_expert)
        if sorted(by_expert) != list(range(expert_count)):
            # A gap means a truncated/corrupt sidecar — let the strict load
            # fail loudly on the missing stacked key rather than stacking
            # a silently wrong tensor.
            passthrough.extend(
                (f"{prefix}.experts.{idx}.{projection}.weight", tensor)
                for idx, tensor in by_expert.items()
            )
            continue
        passthrough.append(
            (
                f"{prefix}.switch_mlp.{projection}.weight",
                mx.stack([by_expert[idx] for idx in range(expert_count)]),
            )
        )
    return passthrough


def build_qwen_sidecar_drafter(
    model: object,
    weights: dict[str, mx.array],
    *,
    norm_convention: NormConvention,
    concat_order: ConcatOrder,
) -> QwenSidecarDrafter | None:
    """Build a Phase-2 Qwen sidecar drafter, or ``None`` to run without MTP.

    Applies the +1.0 shift to every sidecar RMSNorm weight when
    *norm_convention* is ``"zero_centered"``, instantiates the target
    family's own full-attention decoder layer, and strict-loads the
    ``mtp.layers.0.*`` block into it. All failures log a warning and return
    ``None`` — speculation is an optimisation, never a crash.
    """
    if not weights.keys() >= QWEN35_REQUIRED_KEYS:
        missing = sorted(QWEN35_REQUIRED_KEYS - weights.keys())
        logger.warning(f"MTP: Qwen sidecar missing required keys {missing} — running without MTP")
        return None

    if any(key.endswith(("_scales", "_biases")) for key in weights):
        # Backbone-matched quantized sidecars are a planned SWP artifact; the
        # Phase-2 block path has only been validated against bf16 weights.
        logger.warning(
            "MTP: quantized Qwen sidecars are not supported yet — running without MTP"
        )
        return None

    def norm_weight(key: str) -> mx.array:
        v = weights[key]
        # Zero-centered checkpoints store RMSNorm scales as deviation-from-1
        # (mlx-lm sanitize() does this same shift for trunk weights).
        return v + 1.0 if norm_convention == "zero_centered" else v

    embed_fn = get_embed_fn(model)
    head_fn = get_head_fn(model)
    if embed_fn is None or head_fn is None:
        logger.warning("MTP: could not locate embed_tokens / lm_head on model — running without MTP")
        return None

    block = build_sibling_attention_layer(model)
    if block is None:
        logger.warning(
            "MTP: could not instantiate a sibling full-attention decoder layer "
            "for the sidecar transformer block — running without MTP"
        )
        return None

    block_pairs = _stack_per_expert_block_weights(
        [
            (
                key.removeprefix(QWEN35_BLOCK_PREFIX),
                norm_weight(key) if "norm" in key else value,
            )
            for key, value in weights.items()
            if key.startswith(QWEN35_BLOCK_PREFIX)
        ]
    )
    if not block_pairs:
        logger.warning(
            "MTP: sidecar has no mtp.layers.0.* transformer block — the "
            "projection-only path drafts at ~0% acceptance, running without MTP"
        )
        return None
    try:
        block.load_weights(block_pairs, strict=True)
    except ValueError as error:
        # Strict load is the family-drift tripwire: a future family with a
        # different block shape fails loud here instead of drafting garbage.
        logger.warning(f"MTP: sidecar block weights do not match the family decoder layer ({error}) — running without MTP")
        return None

    model_args = getattr(getattr(model, "language_model", None) or model, "args", None)
    eps = float(getattr(model_args, "rms_norm_eps", 1e-6))

    drafter = QwenSidecarDrafter(
        hnorm_w=norm_weight("mtp.pre_fc_norm_hidden.weight"),
        enorm_w=norm_weight("mtp.pre_fc_norm_embedding.weight"),
        fc_w=weights["mtp.fc.weight"],
        final_norm_w=norm_weight("mtp.norm.weight"),
        block=block,
        embed_fn=embed_fn,
        head_fn=head_fn,
        concat_order=concat_order,
        eps=eps,
    )

    quantization = detect_quantization(model)
    if quantization is not None:
        target_group_size, target_bits = quantization
        try:
            drafter.quantize_to_match(group_size=target_group_size, bits=target_bits)
        except ValueError as error:
            # Mismatched dims (e.g. hidden size not divisible by group size)
            # — keep the bf16 sidecar; it is slower, never wrong.
            logger.warning(
                f"MTP: could not quantize sidecar to match target ({error}) — keeping bf16 sidecar"
            )
            quantization = None

    logger.info(
        f"MTP drafter initialised (family=qwen-sidecar, phase=2, "
        f"norm_convention={norm_convention!r}, concat_order={concat_order!r}, "
        f"quantized={quantization!r})"
    )
    return drafter
