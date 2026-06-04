# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# NOTE: this module is written against the PRE-0.31.3 BatchGenerator, whose
# shape the .mlx_typings stubs no longer describe (they track 0.31.3). It is
# runtime-gated off on >= 0.31.3 via _BATCHGEN_COMPATIBLE and quarantined from
# strict typing until the re-port (#187), at which point these pragmas go away.
"""Optimised ``BatchGenerator.next()`` for decode-only steps.

Replaces the default mlx-lm ``BatchGenerator.next`` with a faster path that
precomputes top-k logprobs asynchronously during decode. The precomputed
values are attached to the ``Response`` object and consumed by
``extract_top_logprobs`` one step later, overlapping GPU work with token
emission.

Written against the pre-0.31.3 ``BatchGenerator`` internals. mlx-lm 0.31.3
split that class into ``PromptProcessingBatch`` / ``GenerationBatch`` plus an
orchestrating ``BatchGenerator`` with no nested ``Response``, so this patch
cannot apply there. ``_BATCHGEN_COMPATIBLE`` gates application: on
incompatible versions the patch is skipped with a warning and
``extract_top_logprobs`` uses its synchronous fallback path — a per-token
perf regression (sync argpartition), not a correctness change. Re-port
against the split architecture is tracked in
https://github.com/Foxlight-Foundation/Skulk/issues/187.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
from loguru import logger
from mlx_lm.generate import BatchGenerator, generation_stream

_PRECOMPUTE_TOP_K = 20

# The optimisation rewrites BatchGenerator.next against pre-0.31.3 internals;
# the nested Response class is the cheapest reliable fingerprint of that era.
_BATCHGEN_COMPATIBLE = hasattr(BatchGenerator, "Response")

_original_public_next: Callable[[BatchGenerator], list[BatchGenerator.Response]] | None = (
    getattr(BatchGenerator, "next", None)
)

_pending_topk_idx: mx.array | None = None
_pending_topk_val: mx.array | None = None
_pending_selected_lps: mx.array | None = None


def _fast_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    tic = time.perf_counter()
    batch = self.active_batch
    assert batch is not None
    batch_size = len(batch)

    prev_tokens = batch.y
    prev_logprobs = batch.logprobs

    has_processors = any(p for ps in batch.logits_processors for p in ps)
    if has_processors:
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate([toks, prev_tokens[i : i + 1]])

    logits = self.model(prev_tokens[:, None], cache=batch.cache)
    logits = logits[:, -1, :]

    if has_processors:
        processed_logits: list[mx.array] = []
        for e in range(batch_size):
            sample_logits: mx.array = logits[e : e + 1]
            for processor in batch.logits_processors[e]:
                sample_logits = processor(batch.tokens[e], sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if (
        batch_size == 1
        or any(batch.samplers)
        and all(s is batch.samplers[0] for s in batch.samplers)
    ):
        sampler = batch.samplers[0] or self.sampler
        batch.y = sampler(logprobs)
    elif any(batch.samplers):
        all_samples: list[mx.array] = []
        for e in range(batch_size):
            s = batch.samplers[e] or self.sampler
            all_samples.append(s(logprobs[e : e + 1]))
        batch.y = mx.concatenate(all_samples, axis=0)
    else:
        batch.y = self.sampler(logprobs)
    batch.logprobs = list(logprobs)

    global _pending_topk_idx, _pending_topk_val, _pending_selected_lps

    emit_topk_indices: list[list[int]] = (
        cast(list[list[int]], _pending_topk_idx.tolist())
        if _pending_topk_idx is not None
        else []
    )
    emit_topk_values: list[list[float]] = (
        cast(list[list[float]], _pending_topk_val.tolist())
        if _pending_topk_val is not None
        else []
    )
    emit_selected_lps: list[float] = (
        cast(list[float], _pending_selected_lps.tolist())
        if _pending_selected_lps is not None
        else []
    )

    needs_topk: bool = getattr(self, "_needs_topk", False)
    if needs_topk:
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        _pending_topk_idx = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        _pending_topk_val = mx.take_along_axis(logprobs, _pending_topk_idx, axis=1)
        sort_order = mx.argsort(-_pending_topk_val, axis=1)
        _pending_topk_idx = mx.take_along_axis(_pending_topk_idx, sort_order, axis=1)
        _pending_topk_val = mx.take_along_axis(_pending_topk_val, sort_order, axis=1)
        _pending_selected_lps = logprobs[mx.arange(batch_size), batch.y]
        mx.async_eval(
            batch.y,
            *batch.logprobs,
            *batch.tokens,
            _pending_topk_idx,
            _pending_topk_val,
            _pending_selected_lps,
        )
    else:
        _pending_topk_idx = None
        _pending_topk_val = None
        _pending_selected_lps = None
        mx.async_eval(batch.y, *batch.logprobs, *batch.tokens)

    prev_token_list: list[int] = cast(list[int], prev_tokens.tolist())

    toc = time.perf_counter()
    self._stats.generation_time += toc - tic

    keep_idx: list[int] = []
    end_idx: list[int] = []
    responses: list[Any] = []
    stop_tokens = self.stop_tokens

    for e in range(batch_size):
        t = prev_token_list[e]
        uid = batch.uids[e]
        num_tok = batch.num_tokens[e] + 1
        batch.num_tokens[e] = num_tok

        if t in stop_tokens:
            finish_reason = "stop"
            end_idx.append(e)
        elif num_tok >= batch.max_tokens[e]:
            finish_reason = "length"
            end_idx.append(e)
        else:
            finish_reason = None
            keep_idx.append(e)

        cache = None
        if finish_reason is not None:
            cache = batch.extract_cache(e)
        response = self.Response(uid, t, prev_logprobs[e], finish_reason, cache)
        if emit_topk_indices and e < len(emit_topk_indices):
            response._topk_indices = emit_topk_indices[e]
            response._topk_values = emit_topk_values[e]
            response._selected_logprob = emit_selected_lps[e]
        responses.append(response)

    if end_idx:
        if keep_idx:
            batch.filter(keep_idx)
            if (
                _pending_topk_idx is not None
                and _pending_topk_val is not None
                and _pending_selected_lps is not None
            ):
                ki = mx.array(keep_idx)
                _pending_topk_idx = _pending_topk_idx[ki]
                _pending_topk_val = _pending_topk_val[ki]
                _pending_selected_lps = _pending_selected_lps[ki]
        else:
            self.active_batch = None
            _pending_topk_idx = None
            _pending_topk_val = None
            _pending_selected_lps = None

    self._next_count += 1
    if self._next_count % 512 == 0:
        mx.clear_cache()
    self._stats.generation_tokens += len(responses)
    return responses


def _patched_public_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    batch = self.active_batch
    # Only do decode with fast_next
    if batch is not None and not self.unprocessed_prompts:
        with mx.stream(generation_stream):
            return _fast_next(self)
    # Only installed via apply_batch_gen_patch when _BATCHGEN_COMPATIBLE,
    # in which case the original ``next`` was captured above.
    assert _original_public_next is not None
    return _original_public_next(self)


def apply_batch_gen_patch() -> None:
    """Monkey-patch ``BatchGenerator.next`` with the optimised decode path.

    No-op (with a warning) on mlx-lm versions whose BatchGenerator internals
    moved (>= 0.31.3 split) — see module docstring.
    """
    if not _BATCHGEN_COMPATIBLE:
        logger.warning(
            "opt_batch_gen patch skipped: mlx-lm BatchGenerator internals "
            "changed (0.31.3 class split); top-logprobs precompute disabled, "
            "extract_top_logprobs will use its synchronous fallback."
        )
        return
    BatchGenerator.next = _patched_public_next
