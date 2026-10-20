"""Bounded soft scene routing for the pinned LTX video attention contract.

The fixed profile matches the native LTX relay's Gaussian plateau, including
its integer midpoint and constant sigma. Global and connector-register columns
stay free; this never selects a single scene's keys or allocates a full-video
attention bias. Integration supplies actual, finite query coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

RELAY_WORKSPACE_BYTES = 32 * 1024 * 1024
"""Maximum combined temporary bias storage, excluding existing attention tensors."""


@dataclass(frozen=True, slots=True)
class RelaySegment:
    """A local text-column interval and its free window on the latent timeline."""

    token_start: int
    token_end: int
    midpoint: int
    half_window: int


def relay_lengths(weights: tuple[int, ...], latent_frames: int) -> tuple[int, ...]:
    """Allocate the actual grid proportionally, retaining one frame per scene.

    Largest remainders receive spare frames, with earlier scenes breaking ties.
    This preserves the existing Foxlight native request's relative-length rule.
    """
    if not 2 <= len(weights) <= 8 or type(latent_frames) is not int or not len(weights) <= latent_frames <= 15001:
        raise ValueError("relay requires two to eight scenes with at least one latent frame each")
    if any(type(weight) is not int or not 1 <= weight <= 120 for weight in weights):
        raise ValueError("relay scene weights must be integers in [1, 120]")
    remaining = latent_frames - len(weights)
    total = sum(weights)
    lengths = [1 + remaining * weight // total for weight in weights]
    remainder_order = sorted(range(len(weights)), key=lambda index: (-(remaining * weights[index] % total), index))
    for index in remainder_order[:latent_frames - sum(lengths)]:
        lengths[index] += 1
    return tuple(lengths)


def relay_segments(token_ranges: tuple[tuple[int, int], ...], weights: tuple[int, ...], latent_frames: int) -> tuple[RelaySegment, ...]:
    """Bind measured text ranges to the fixed native Gaussian plateau profile."""
    lengths = relay_lengths(weights, latent_frames)
    if len(token_ranges) != len(lengths):
        raise ValueError("relay token ranges must describe every scene")
    cursor = 0
    previous_end = 0
    segments: list[RelaySegment] = []
    for (start, end), length in zip(token_ranges, lengths, strict=True):
        if type(start) is not int or type(end) is not int or not previous_end <= start < end:
            raise ValueError("relay text ranges must be ordered, nonempty and nonoverlapping")
        segments.append(RelaySegment(start, end, (2 * cursor + length) // 2, max(length // 2 - 2, 0)))
        previous_end = end
        cursor += length
    return tuple(segments)


def relay_bias(query_frames: Tensor, segments: tuple[RelaySegment, ...], context_tokens: int, real_queries: Tensor) -> Tensor:
    """Build one FP32 query chunk; actual surviving guide rows remain unpenalized.

    ``query_frames`` and ``real_queries`` have shape (batch, queries). The
    integration validates finite coordinates once per forward, avoiding a device
    synchronization at every layer and chunk. Context bounds always fail closed.
    """
    if query_frames.ndim != 2 or not torch.is_floating_point(query_frames) or query_frames.shape != real_queries.shape or real_queries.dtype != torch.bool or query_frames.device != real_queries.device:
        raise ValueError("relay requires aligned floating coordinates and boolean real-query rows")
    if context_tokens < 1 or not segments or any(segment.token_start < 0 or segment.token_end > context_tokens or segment.token_start >= segment.token_end for segment in segments):
        raise ValueError("relay scene ranges exceed the actual text context")
    batch, queries = query_frames.shape
    bias = torch.zeros((batch, 1, queries, context_tokens), device=query_frames.device, dtype=torch.float32)
    positions = query_frames.float()
    inverse_variance = math.log(1000.0) ** 2 / 2.0
    for segment in segments:
        distance = (positions - segment.midpoint).abs().sub(segment.half_window).clamp_min(0)
        penalty = distance.square().mul(-inverse_variance)
        penalty = torch.where(real_queries, penalty, 0.0)
        bias[:, 0, :, segment.token_start:segment.token_end] = penalty.unsqueeze(-1)
    return bias


def relay_attention(query: Tensor, key: Tensor, value: Tensor, *, query_frames: Tensor, real_queries: Tensor, segments: tuple[RelaySegment, ...], attention: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor], mask: Tensor | None = None, workspace_bytes: int = RELAY_WORKSPACE_BYTES) -> Tensor:
    """Apply soft routing through the caller's real optimized attention in chunks.

    Projections, all-head normalization, rotary embeddings and output gating
    remain the integration's responsibility and execute once, outside this loop.
    Existing floating masks are sliced before conversion. The conservative
    32-byte-per-cell budget covers FP32 bias, combination, conversion and the
    Gaussian's row vectors without retaining a dense whole-video bias or
    duplicating output.
    """
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("relay requires the pinned batch/token/channel attention geometry")
    batch, queries, _ = query.shape
    context_tokens = key.shape[1]
    if query_frames.shape != (batch, queries) or real_queries.shape != query_frames.shape:
        raise ValueError("relay coordinates do not match the actual attention queries")
    if query.device != key.device or key.device != value.device or query.device != query_frames.device:
        raise ValueError("relay attention and coordinates must share the placed device")
    if not torch.is_floating_point(query) or query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("relay requires aligned floating attention projections")
    mask_heads = 1
    if mask is not None:
        if mask.ndim != 4 or not torch.is_floating_point(mask) or mask.device != query.device or mask.shape[0] not in (1, batch) or mask.shape[2] not in (1, queries) or mask.shape[3] != context_tokens:
            raise ValueError("relay requires a broadcastable pinned floating text-attention mask")
        mask_heads = mask.shape[1]
    if batch < 1 or queries < 1 or context_tokens < 1 or mask_heads < 1 or type(workspace_bytes) is not int:
        raise ValueError("relay attention dimensions and workspace must be positive")
    rows_per_chunk = min(1024, workspace_bytes // (32 * batch * mask_heads * context_tokens))
    if rows_per_chunk < 1:
        raise ValueError("relay text context exceeds the bounded attention workspace")
    output = torch.empty_like(query)
    for start in range(0, queries, rows_per_chunk):
        end = min(start + rows_per_chunk, queries)
        bias = relay_bias(query_frames[:, start:end], segments, context_tokens, real_queries[:, start:end])
        if mask is not None:
            original = mask if mask.shape[2] == 1 else mask[:, :, start:end, :]
            bias = bias + original.float()
        result = attention(query[:, start:end], key, value, bias.to(dtype=query.dtype))
        if result.shape != query[:, start:end].shape:
            raise ValueError("relay optimized attention returned incompatible output geometry")
        output[:, start:end] = result
    return output
