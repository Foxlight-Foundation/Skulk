"""Model-introspection helpers shared by drafter implementations.

Drafters borrow callables (embeddings, output head, final norm) and structure
(the family's own decoder-layer class) from the already-loaded target model
instead of re-implementing family math. Every helper degrades to ``None`` so
builders can fall back to running without speculation rather than crashing.
"""

from __future__ import annotations

from typing import Callable, Protocol, cast

import mlx.core as mx
import mlx.nn as nn


def get_text_model(model: object) -> object | None:
    """Return the text-model wrapper (``TextModel``-like) for *model*.

    Handles both multimodal wrappers (``model.language_model``) and plain
    causal LMs (the model itself).
    """
    lm: object | None = getattr(model, "language_model", None)
    return lm if lm is not None else model


def get_trunk(model: object) -> object | None:
    """Return the inner trunk (the module holding ``embed_tokens``/``layers``)."""
    text_model = get_text_model(model)
    return getattr(text_model, "model", None)


def get_model_args(model: object) -> object | None:
    """Return the family ``ModelArgs``-style dataclass carried by the text model."""
    text_model = get_text_model(model)
    return getattr(text_model, "args", None)


def get_embed_fn(model: object) -> Callable[[mx.array], mx.array] | None:
    """Extract the ``embed_tokens`` callable from the main model."""
    trunk = get_trunk(model)
    if trunk is not None:
        embed: object | None = getattr(trunk, "embed_tokens", None)
        if embed is not None and callable(embed):
            return cast(Callable[[mx.array], mx.array], embed)
    return None


def _tied_head_fn(trunk: object | None) -> Callable[[mx.array], mx.array] | None:
    """Output head for tied-embedding models: ``embed_tokens.as_linear``.

    mlx-lm >= 0.31.3 qwen3_5 TextModel has no ``lm_head`` attribute when
    ``tie_word_embeddings`` is set — its ``__call__`` projects through
    ``self.model.embed_tokens.as_linear`` instead.
    """
    if trunk is None:
        return None
    embed: object | None = getattr(trunk, "embed_tokens", None)
    as_linear: object | None = getattr(embed, "as_linear", None)
    if as_linear is not None and callable(as_linear):
        return cast(Callable[[mx.array], mx.array], as_linear)
    return None


def get_head_fn(model: object) -> Callable[[mx.array], mx.array] | None:
    """Extract the lm_head callable (or tied-embedding equivalent)."""
    text_model = get_text_model(model)
    if text_model is not None:
        head: object | None = getattr(text_model, "lm_head", None)
        if head is not None and callable(head):
            return cast(Callable[[mx.array], mx.array], head)
    return _tied_head_fn(get_trunk(model))


def get_norm_fn(model: object) -> Callable[[mx.array], mx.array] | None:
    """Extract the final-norm callable from the main model trunk."""
    trunk = get_trunk(model)
    if trunk is not None:
        norm: object | None = getattr(trunk, "norm", None)
        if norm is not None and callable(norm):
            return cast(Callable[[mx.array], mx.array], norm)
    return None


class _QuantizedLinearFacts(Protocol):
    """The two quantization parameters mlx's untyped QuantizedLinear carries."""

    group_size: int
    bits: int


def detect_quantization(model: object) -> tuple[int, int] | None:
    """Return the target trunk's ``(group_size, bits)``, or ``None`` if bf16.

    Drafters that borrow structure from the target should match its
    quantization: an unquantized bf16 sidecar block on a 4-bit target makes
    the draft forward memory-bound on weights several times larger than the
    verifier's own layers read.
    """
    trunk = get_trunk(model)
    if not isinstance(trunk, nn.Module):
        return None
    modules = cast(
        "list[tuple[str, nn.Module]]",
        trunk.named_modules(),  # pyright: ignore[reportUnknownMemberType]
    )
    for _name, module in modules:
        if isinstance(module, nn.QuantizedLinear):
            facts = cast(_QuantizedLinearFacts, cast(object, module))
            return int(facts.group_size), int(facts.bits)
    return None


def build_sibling_attention_layer(model: object) -> nn.Module | None:
    """Construct an uninitialised sibling of the trunk's full-attention layer.

    MTP transformer blocks in Qwen3Next-descended families are architecturally
    one of the family's own decoder layers, so the most future-proof way to
    build one is to find a full-attention layer in the loaded trunk and
    instantiate ``type(layer)(args, layer_idx=<same index>)``. New families
    then work without new block code — and if a future family changes its
    constructor signature, this returns ``None`` and the caller falls back to
    running without MTP (loud log, no crash).
    """
    trunk = get_trunk(model)
    args = get_model_args(model)
    layers: object | None = getattr(trunk, "layers", None)
    if trunk is None or args is None or not isinstance(layers, list):
        return None
    for layer_idx, layer in enumerate(cast("list[object]", layers)):
        # Full-attention layers carry `self_attn`; linear/SSM layers do not.
        if getattr(layer, "self_attn", None) is None:
            continue
        try:
            sibling = type(layer)(args, layer_idx=layer_idx)  # type: ignore[call-arg]
        except TypeError:
            return None
        return cast(nn.Module, sibling)
    return None
