"""Drafter construction: layout detection + family-fact resolution.

This is the only place that decides *which* drafter a sidecar gets and
*which family conventions* apply. The decision inputs, in precedence order:

1. **Model-card runtime overrides** (``mtp_norm_convention``,
   ``mtp_concat_order``) — declarative, per-model, for publisher or family
   drift.
2. **Layout-keyed family defaults** — empirically validated facts per
   detected sidecar key layout (the table below).

Adding a new MTP family is intended to be: run the offline matrix experiment
(``probe_phase2.py`` on the test host) against real sidecar weights, read off
the winning row, add a defaults entry and (only if the mechanism is genuinely
new) a drafter module. The generation loop never changes.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from skulk.shared.models.model_cards import RuntimeCapabilityCardConfig
from skulk.worker.engines.mlx.drafters.deepseek_sidecar import (
    build_deepseek_sidecar_drafter,
    detect_deepseek_prefix,
)
from skulk.worker.engines.mlx.drafters.gemma4_assistant import (
    build_gemma4_assistant_drafter,
)
from skulk.worker.engines.mlx.drafters.protocol import Drafter
from skulk.worker.engines.mlx.drafters.qwen_sidecar import (
    QWEN35_REQUIRED_KEYS,
    ConcatOrder,
    NormConvention,
    build_qwen_sidecar_drafter,
)

logger = logging.getLogger(__name__)

# Family defaults, keyed by detected sidecar layout. Validation provenance:
# - qwen-sidecar: kite3 matrix experiment 2026-06-04 (issue #192) —
#   zero_centered + embed_first measured 72.4% argmax agreement; every other
#   combination ≤ 20%.
# - deepseek-sidecar: inherited assumptions, NOT yet validated against real
#   weights (the drafter itself documents this); conventions live in the
#   drafter pending a real-weight matrix run.
_QWEN_DEFAULT_NORM_CONVENTION: NormConvention = "zero_centered"
_QWEN_DEFAULT_CONCAT_ORDER: ConcatOrder = "embed_first"


def build_drafter(
    model: object,
    mtp_weights: dict[str, mx.array] | None,
    *,
    assistant_model: object | None = None,
    runtime: RuntimeCapabilityCardConfig | None = None,
) -> Drafter | None:
    """Detect the sidecar layout and build the matching drafter.

    Args:
        model: The loaded target model (drafters borrow embeddings, the
            output head, and block structure from it).
        mtp_weights: Raw sidecar tensors as published (no shifts applied).
        runtime: Optional model-card runtime section carrying per-model
            convention overrides.

    Returns:
        A constructed :class:`Drafter`, or ``None`` to run without
        speculation (unrecognised layout or any construction failure —
        always logged, never raised).
    """
    if assistant_model is not None:
        # Gemma 4 assistant pattern: a separate chain-trained draft model
        # cross-attending over the target's KV (gemma4-mtp Phase C).
        return build_gemma4_assistant_drafter(model, assistant_model)

    if mtp_weights is None:
        return None

    if mtp_weights.keys() >= QWEN35_REQUIRED_KEYS:
        norm_convention: NormConvention = (
            runtime.mtp_norm_convention
            if runtime is not None and runtime.mtp_norm_convention is not None
            else _QWEN_DEFAULT_NORM_CONVENTION
        )
        concat_order: ConcatOrder = (
            runtime.mtp_concat_order
            if runtime is not None and runtime.mtp_concat_order is not None
            else _QWEN_DEFAULT_CONCAT_ORDER
        )
        return build_qwen_sidecar_drafter(
            model,
            mtp_weights,
            norm_convention=norm_convention,
            concat_order=concat_order,
        )

    if detect_deepseek_prefix(mtp_weights) is not None:
        return build_deepseek_sidecar_drafter(model, mtp_weights)

    logger.warning(
        "MTP sidecar loaded but key layout not recognised — expected the "
        "Qwen3.5 'mtp.{pre_fc_norm_*,fc,norm}' + 'mtp.layers.0.*' layout or "
        "the DeepSeek 'mtp.0.{enorm,hnorm,eh_proj}' layout; running without MTP"
    )
    return None
