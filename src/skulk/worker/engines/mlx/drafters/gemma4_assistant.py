"""Gemma 4 assistant-model drafter (Phase C of the gemma4-mtp initiative).

Gemma 4 ships speculative decoding as a separate 4-layer *assistant* model
(published per target, e.g. ``mlx-community/gemma-4-26B-A4B-it-assistant-bf16``)
rather than embedded MTP heads. The assistant:

- computes **no K/V of its own** — every layer cross-attends over the
  *target's* KV cache (the K/V of the target's last full-attention and last
  sliding-attention layers), which is why
  :meth:`~exo.worker.engines.mlx.drafters.protocol.Drafter.begin_request`
  receives the target cache handles;
- is **trained for multi-step drafting**, unlike the one-step Qwen sidecar
  block whose chained acceptance craters past depth 1 (86.8% → 39.2%
  conditional, Skulk #192 findings) — depth 2–4 is where this drafter is
  expected to pay (upstream reports 3.94× at block size 4);
- consumes the target's **final (post-norm) hidden state** — the gemma4
  trunk path returns exactly that (the pre-norm wrapper is gated to
  qwen-shaped trunks).

The adapter drives the upstream model's ``__call__`` (which returns logits)
rather than ``draft_block`` (which returns sampled tokens), so it satisfies
Skulk's :class:`Drafter` protocol with real ``(K, vocab)`` rows — greedy
chains at depth, and at depth 1 the row is the assistant's true
distribution, so sampled (ratio-acceptance) decoding works unchanged.

Shared-KV extraction is a faithful port of mlx-vlm's
``_mtp_shared_kv_from_prompt_cache`` (zip layers with caches, ``state[:2]``
as (k, v), un-ring-buffer ``RotatingKVCache`` via ``_temporal_order``,
last layer of each ``layer_type`` wins).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast, final

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import RotatingKVCache

from skulk.worker.engines.mlx.drafters.introspection import get_trunk

logger = logging.getLogger(__name__)

SharedKV = dict[str, tuple[mx.array, mx.array]]


class AssistantModel(Protocol):
    """Structural type for mlx-vlm's ``Gemma4AssistantDraftModel`` surface.

    mlx-vlm ships no type stubs; this pins the exact members the adapter
    touches so the rest stays strictly typed.
    """

    def bind(self, target_model: object) -> "AssistantModel": ...

    def reset(self, target_model: object) -> list[object]: ...

    def set_shared_kv(
        self,
        shared_kv_states: SharedKV,
        kv_offset: int,
        position: int | None = None,
        kv_valid_len: int | None = None,
    ) -> None: ...

    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv_states: SharedKV,
        position_ids: mx.array,
    ) -> tuple[mx.array, mx.array]: ...

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]: ...

    def load_weights(
        self, weights: list[tuple[str, mx.array]], strict: bool = True
    ) -> object: ...

    def parameters(self) -> dict[str, object]: ...


def _extract_shared_kv(
    layers: Sequence[object], prompt_cache: Sequence[object]
) -> SharedKV:
    """Port of mlx-vlm's ``_mtp_shared_kv_from_prompt_cache``.

    Walks target layers and their caches in lockstep and keeps the K/V of
    the LAST layer of each ``layer_type`` (full_attention /
    sliding_attention). Rotating caches are restored to temporal order so
    the assistant's masks index real positions.
    """
    shared: SharedKV = {}
    for layer, layer_cache in zip(layers, prompt_cache, strict=False):
        layer_type = cast("str | None", getattr(layer, "layer_type", None))
        if layer_type is None or layer_cache is None:
            continue
        state = cast(
            "tuple[mx.array | None, ...] | None", getattr(layer_cache, "state", None)
        )
        if state is None or len(state) < 2:
            continue
        keys, values = state[0], state[1]
        if keys is None or values is None:
            continue
        if isinstance(layer_cache, RotatingKVCache):
            temporal = cast(
                "Callable[[mx.array], mx.array] | None",
                getattr(layer_cache, "_temporal_order", None),
            )
            if callable(temporal):
                keys = temporal(keys)
                values = temporal(values)
        shared[str(layer_type)] = (keys, values)
    return shared


@final
class Gemma4AssistantDrafter:
    """Stateful drafter wrapping mlx-vlm's ``Gemma4AssistantDraftModel``.

    Satisfies :class:`~exo.worker.engines.mlx.drafters.protocol.Drafter`.
    Construct via :func:`build_gemma4_assistant_drafter`.
    """

    # The assistant cross-attends the TARGET's KV cache: the loop must keep
    # that cache fully committed before every draft (no deferred replay).
    reads_target_cache = True

    def __init__(self, *, assistant: AssistantModel, target_model: object) -> None:
        self._assistant = assistant
        self._target = target_model
        self._prompt_cache: Sequence[object] = []
        trunk = get_trunk(target_model)
        layers = getattr(trunk, "layers", None)
        self._layers: list[object] = (
            cast("list[object]", layers) if isinstance(layers, list) else []
        )

    def begin_request(self, prompt_cache: Sequence[object]) -> None:
        """Rebind to the target and keep the per-request cache handles.

        The assistant has no private cache (``make_cache() -> []``); its
        only per-round state arrives via ``set_shared_kv`` inside
        :meth:`draft`.
        """
        self._assistant.reset(self._target)
        # Hold the LIVE cache sequence, never a copy: the loop's
        # reject-restore REPLACES rotating entries in its cache list
        # (trim_cache assigns fresh objects), so a copied list freezes this
        # drafter's cross-attention view at the first reject — measured as
        # progressive acceptance decay (56% -> 26% over 150 tokens) on
        # snapshot-path targets. Native-rollback targets mutate in place
        # and masked the bug.
        self._prompt_cache = prompt_cache

    def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
        """No-op: the assistant carries no positional history of its own."""
        del hiddens, next_tokens

    def draft(self, hidden: mx.array, next_token: int, depth: int = 1) -> mx.array:
        """Chain up to *depth* drafts from the target's post-norm hidden.

        Mirrors the upstream ``draft_block`` loop but collects logits per
        step: input = ``concat([target_embed(tok) * embed_scale, h_prev])``,
        ``position_ids`` held constant at the bonus offset, ``h_prev``
        advanced with the assistant's own ``post_projection`` output.
        """
        if len(self._prompt_cache) > len(self._layers):
            raise RuntimeError(
                "Gemma 4 assistant drafter: more caches than target layers "
                f"({len(self._layers)} layers vs {len(self._prompt_cache)} caches)"
            )
        # KV-shared models (E2B/E4B: num_kv_shared_layers > 0) own caches only
        # for the first N layers (gemma4 make_cache builds
        # layers[:first_kv_shared_layer_idx]); the deeper layers reuse that
        # K/V — and the last cache-owning layer of each type is exactly what
        # the assistant cross-attends over.
        shared = _extract_shared_kv(
            self._layers[: len(self._prompt_cache)], self._prompt_cache
        )
        # The bonus position == the target cache offset at draft time (the
        # position next_token is about to occupy).
        offset = max(
            (
                int(cast(int, getattr(layer_cache, "offset", 0)))
                for layer_cache in self._prompt_cache
            ),
            default=0,
        )
        # Upstream anchors the drafter's constant query position at the LAST
        # CACHED position (kv_offset - 1), not at kv_offset — set_shared_kv's
        # default (position = kv_offset) is one off, and that RoPE off-by-one
        # measured a ~21pp acceptance loss (48% vs upstream's 69.3% on
        # identical E4B-8bit artifacts) before this was matched to
        # mlx-vlm's _mtp_draft_position.
        draft_position = max(offset - 1, 0)
        self._assistant.set_shared_kv(
            shared, kv_offset=offset, position=draft_position, kv_valid_len=offset
        )
        position_ids = mx.array([[draft_position]])

        embed_fn = cast(
            "Callable[[mx.array], mx.array] | None",
            getattr(self._assistant, "_input_embed", None),
        )
        if embed_fn is None:
            raise RuntimeError(
                "Gemma 4 assistant drafter: bind(target) did not resolve the "
                "target's input embeddings"
            )
        embed_scale = float(
            cast(float, getattr(self._assistant, "_input_embed_scale", 1.0))
        )

        # (H,) post-norm target hidden -> (1, 1, backbone_hidden_size)
        h_prev = hidden[None, None, :]
        token = next_token
        rows: list[mx.array] = []
        for _ in range(max(depth, 1)):
            tok_embed = embed_fn(mx.array([[token]])) * embed_scale
            inputs_embeds = mx.concatenate(
                [tok_embed.astype(h_prev.dtype), h_prev], axis=-1
            )
            h_prev, logits = self._assistant(inputs_embeds, shared, position_ids)
            row = logits[0, -1].astype(mx.float32)
            rows.append(row)
            # Greedy-internal chain (matches the Qwen drafter and the loop's
            # depth-1 rule under sampling).
            token = int(mx.argmax(row).item())
        return mx.stack(rows)


def build_gemma4_assistant_drafter(
    model: object,
    assistant_model: object,
) -> Gemma4AssistantDrafter | None:
    """Wrap a loaded assistant model as a Skulk drafter, or ``None``.

    All failures log a warning and return ``None`` — speculation is an
    optimisation, never a crash.
    """
    trunk = get_trunk(model)
    layers = getattr(trunk, "layers", None)
    if not isinstance(layers, list) or not layers:
        logger.warning(
            "Gemma 4 assistant: target trunk layers not introspectable — "
            "running without speculation"
        )
        return None
    typed_layers = cast("list[object]", layers)
    if not any(getattr(layer, "layer_type", None) for layer in typed_layers):
        logger.warning(
            "Gemma 4 assistant: target layers carry no layer_type metadata "
            "(not a gemma4-family trunk?) — running without speculation"
        )
        return None
    assistant = cast(AssistantModel, assistant_model)
    try:
        assistant = assistant.bind(model)
    except Exception as error:  # noqa: BLE001 — bind probes model structure
        logger.warning(
            f"Gemma 4 assistant: bind(target) failed ({error}) — "
            "running without speculation"
        )
        return None
    if not callable(getattr(assistant, "_input_embed", None)):
        logger.warning(
            "Gemma 4 assistant: bind(target) did not resolve the target's "
            "input embeddings — assistant drafting disabled"
        )
        return None
    drafter = Gemma4AssistantDrafter(assistant=assistant, target_model=model)
    logger.info("Gemma 4 assistant drafter initialised (family=gemma4-assistant)")
    return drafter


def load_assistant_model(model_dir: Path) -> object | None:
    """Load a Gemma 4 assistant checkpoint as an mlx-vlm draft model.

    Enforces bf16 (fp16 assistants degenerate after ~50 tokens — unscaled
    QK per the phase-c spec). Returns ``None`` on any failure, logged.
    """
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        logger.warning(
            f"Gemma 4 assistant at {model_dir} is incomplete "
            "(needs config.json + model.safetensors) — assistant drafting disabled"
        )
        return None
    try:
        # Deferred import: mlx-vlm is darwin-only and heavy; importing here
        # keeps non-darwin environments (and the missing-file fast path)
        # from ever touching it.
        from mlx_vlm.speculative.drafters.gemma4_assistant import (  # pyright: ignore[reportMissingTypeStubs]
            Gemma4AssistantDraftModel,
            ModelConfig,
        )

        with open(config_path) as fh:
            config_dict = cast("dict[str, object]", json.load(fh))
        config = ModelConfig.from_dict(config_dict)  # pyright: ignore[reportUnknownMemberType]
        assistant = cast(
            AssistantModel, cast(object, Gemma4AssistantDraftModel(config))
        )
        raw = cast("dict[str, mx.array]", mx.load(str(weights_path)))
        weights = assistant.sanitize(raw)
        # bf16 enforcement: cast any fp16 tensors up-front.
        weights = {
            k: (v.astype(mx.bfloat16) if v.dtype == mx.float16 else v)
            for k, v in weights.items()
        }
        assistant.load_weights(list(weights.items()), strict=True)
        flattened = cast(
            "list[tuple[str, mx.array]]", tree_flatten(assistant.parameters())
        )
        mx.eval([value for _, value in flattened])
        logger.info(
            f"Gemma 4 assistant loaded from {model_dir} "
            f"({len(flattened)} tensors, bf16)"
        )
        return assistant
    except Exception as error:  # noqa: BLE001 — never crash the runner for speculation
        logger.warning(
            f"Gemma 4 assistant load failed ({error}) — running without speculation"
        )
        return None
