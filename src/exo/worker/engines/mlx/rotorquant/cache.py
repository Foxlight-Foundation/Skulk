"""Experimental pure-MLX IsoQuant KV cache with optional deferred prefill.

This module intentionally does not implement the fused RotorQuant+QJL
attention path from the RotorQuant paper. It stores compressed K/V indices
and norms, then materializes dequantized fp16 K/V for normal MLX attention.
That makes it useful for isolated cache experiments, but not a production
substitute for the upstream fused llama.cpp path.

The idea: while the engine is processing a prompt (``update_and_fetch``
called with ``num_steps > 1``), keep K and V in fp16. Quantization only
happens once, on the first decode-shaped call (``num_steps == 1``).
This avoids compounding centroid-roundtrip errors through every prefill
attention chain.
"""

from collections.abc import Sequence
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    RotatingKVCache,
    create_attention_mask,
)

from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.engines.mlx.rotorquant.quantizer import IsoQuantizer
from exo.worker.engines.mlx.rotorquant.tables import ISO3_BLOCK_SIZE


class RotorQuantKVCache:
    """Per-layer KV cache backed by IsoQuant 3-bit compression.

    Storage layout (one slot per token per kv-head):
      - ``key_indices``:   ``(B, kv_heads, T, head_dim)`` uint8
      - ``key_norms``:     ``(B, kv_heads, T, head_dim // 128)`` fp16
      - ``value_indices``: same shape as keys
      - ``value_norms``:   same shape as key_norms

    During the deferred-prefill window the class additionally holds
    ``_pending_keys`` and ``_pending_values`` as fp16 buffers, and
    ``update_and_fetch`` returns the concatenation of those buffers
    directly without ever quantizing. The first call with
    ``num_steps == 1`` flushes the pending buffers through the
    quantizer in a single batched operation.
    """

    step = 256

    def __init__(self, *, defer_prefill: bool = True, seed: int = 42) -> None:
        # Deferred prefill is on by default because it is both an
        # accuracy and a peak-memory-amortization win; tests can flip
        # it off to exercise the always-quantized path.
        self.defer_prefill = defer_prefill
        self.seed = seed
        self.offset = 0

        self._quantizer: IsoQuantizer | None = None
        self.head_dim: int | None = None

        self._key_indices: mx.array | None = None
        self._key_norms: mx.array | None = None
        self._value_indices: mx.array | None = None
        self._value_norms: mx.array | None = None

        # Deferred prefill scratch buffers — kept in fp16 until the
        # first decode-shaped call triggers ``_flush_deferred``.
        self._pending_keys: mx.array | None = None
        self._pending_values: mx.array | None = None
        self._pending_active: bool = defer_prefill

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _ensure_quantizer(self, head_dim: int) -> None:
        if head_dim % ISO3_BLOCK_SIZE != 0:
            raise ValueError(
                f"RotorQuantKVCache requires head_dim divisible by "
                f"{ISO3_BLOCK_SIZE}, got {head_dim}"
            )
        if self._quantizer is None:
            self._quantizer = IsoQuantizer()
            self.head_dim = head_dim

    def _expand_storage(
        self,
        batch_size: int,
        kv_heads: int,
        steps: int,
        head_dim: int,
    ) -> None:
        n_blocks = head_dim // ISO3_BLOCK_SIZE
        needed = self.offset + steps
        if self._key_indices is not None and needed <= self._key_indices.shape[2]:
            return

        new_steps = ((steps + self.step - 1) // self.step) * self.step
        if self._key_indices is None:
            shape_idx = (batch_size, kv_heads, new_steps, head_dim)
            shape_norm = (batch_size, kv_heads, new_steps, n_blocks)
            self._key_indices = mx.zeros(shape_idx, dtype=mx.uint8)
            self._key_norms = mx.zeros(shape_norm, dtype=mx.float16)
            self._value_indices = mx.zeros(shape_idx, dtype=mx.uint8)
            self._value_norms = mx.zeros(shape_norm, dtype=mx.float16)
            return

        # Concatenate a fresh slab of zeros to the right of the live
        # region. The live region ends at ``self.offset``; everything
        # past that is stale capacity from a previous expansion.
        assert self._key_norms is not None
        assert self._value_indices is not None
        assert self._value_norms is not None
        zeros_idx = mx.zeros(
            (batch_size, kv_heads, new_steps, head_dim),
            dtype=mx.uint8,
        )
        zeros_norm = mx.zeros(
            (batch_size, kv_heads, new_steps, n_blocks),
            dtype=mx.float16,
        )
        self._key_indices = mx.concatenate(
            [self._key_indices[..., : self.offset, :], zeros_idx],
            axis=2,
        )
        self._key_norms = mx.concatenate(
            [self._key_norms[..., : self.offset, :], zeros_norm],
            axis=2,
        )
        self._value_indices = mx.concatenate(
            [self._value_indices[..., : self.offset, :], zeros_idx],
            axis=2,
        )
        self._value_norms = mx.concatenate(
            [self._value_norms[..., : self.offset, :], zeros_norm],
            axis=2,
        )

    # ------------------------------------------------------------------
    # Update / fetch
    # ------------------------------------------------------------------

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Insert ``keys``/``values`` and return the full live KV cache.

        While deferred prefill is active, the call appends to the fp16
        scratch buffers and returns the concatenated fp16 view directly.
        Once the first decode-shaped call (``num_steps == 1``) arrives,
        the scratch buffers are flushed through the quantizer in one
        batched operation, after which all subsequent calls quantize on
        insert and return a freshly dequantized view.
        """
        _, _, num_steps, head_dim = keys.shape
        self._ensure_quantizer(head_dim)

        # Deferred-prefill path: keep raw fp16 until the first decode token.
        # On a multi-token call we append to the scratch and bail out
        # immediately. On a single-token call we flush whatever we have
        # accumulated, then fall through to the normal quantized insert
        # path so the new decode token is stored alongside the rest.
        if self._pending_active:
            if num_steps > 1:
                self._append_pending(keys, values)
                assert self._pending_keys is not None
                assert self._pending_values is not None
                # Advance the logical offset so downstream attention
                # (RoPE position via ``cache.offset``) sees the correct
                # token count when prefill is split into multiple
                # chunks. Without this, every chunk after the first is
                # positioned as if it started at token 0.
                self.offset += num_steps
                return self._pending_keys, self._pending_values
            self._flush_deferred(keys.dtype)

        if num_steps > 0:
            self._quantize_and_store(keys, values)

        return self._dequantize_live(keys.dtype)

    def _append_pending(self, keys: mx.array, values: mx.array) -> None:
        """Append new K/V to the deferred-prefill scratch buffers."""
        if self._pending_keys is None:
            self._pending_keys = keys.astype(mx.float16)
            self._pending_values = values.astype(mx.float16)
            return
        assert self._pending_values is not None
        self._pending_keys = mx.concatenate(
            [self._pending_keys, keys.astype(mx.float16)],
            axis=2,
        )
        self._pending_values = mx.concatenate(
            [self._pending_values, values.astype(mx.float16)],
            axis=2,
        )

    def _flush_deferred(self, dtype: mx.Dtype) -> None:
        """Quantize the deferred prefill buffer and free the fp16 scratch.

        If the first call into ``update_and_fetch`` is already decode-shaped
        (e.g., a 1-token prompt) the deferred buffers will be empty. In that
        case there is nothing to flush — we just exit the deferred phase and
        let the caller's normal quantize-on-insert path handle the new
        token.
        """
        del dtype  # the live cache stores indices regardless of caller dtype
        assert self._quantizer is not None

        if self._pending_keys is None or self._pending_values is None:
            self._pending_keys = None
            self._pending_values = None
            self._pending_active = False
            return

        pending_keys = self._pending_keys
        pending_values = self._pending_values
        batch_size, kv_heads, num_pending, head_dim = pending_keys.shape

        # ``offset`` was advanced as tokens were appended to the pending
        # scratch, so the live region we are about to populate is
        # ``[offset - num_pending, offset)``. ``_expand_storage`` sizes
        # capacity from ``offset``, so temporarily rewind it to the
        # pre-pending position before expanding so we don't allocate
        # ``num_pending`` extra slots.
        live_end = self.offset
        self.offset = live_end - num_pending
        self._expand_storage(batch_size, kv_heads, num_pending, head_dim)
        assert self._key_indices is not None
        assert self._key_norms is not None
        assert self._value_indices is not None
        assert self._value_norms is not None

        key_indices, key_norms = self._quantizer.quantize(pending_keys)
        value_indices, value_norms = self._quantizer.quantize(pending_values)

        self._key_indices[..., self.offset : live_end, :] = key_indices
        self._key_norms[..., self.offset : live_end, :] = key_norms
        self._value_indices[..., self.offset : live_end, :] = value_indices
        self._value_norms[..., self.offset : live_end, :] = value_norms
        self.offset = live_end

        # Free the scratch — from now on we are in the steady-state
        # decode path and quantize on insert.
        self._pending_keys = None
        self._pending_values = None
        self._pending_active = False

    def _quantize_and_store(self, keys: mx.array, values: mx.array) -> None:
        """Quantize freshly arrived tokens and append to the live cache."""
        assert self._quantizer is not None
        batch_size, kv_heads, num_steps, head_dim = keys.shape

        self._expand_storage(batch_size, kv_heads, num_steps, head_dim)
        assert self._key_indices is not None
        assert self._key_norms is not None
        assert self._value_indices is not None
        assert self._value_norms is not None

        key_indices, key_norms = self._quantizer.quantize(keys)
        value_indices, value_norms = self._quantizer.quantize(values)

        end = self.offset + num_steps
        self._key_indices[..., self.offset : end, :] = key_indices
        self._key_norms[..., self.offset : end, :] = key_norms
        self._value_indices[..., self.offset : end, :] = value_indices
        self._value_norms[..., self.offset : end, :] = value_norms
        self.offset = end

    def _dequantize_live(
        self, output_dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        """Materialize the full live cache as fp16 K/V for SDPA."""
        assert self._quantizer is not None
        assert self._key_indices is not None
        assert self._key_norms is not None
        assert self._value_indices is not None
        assert self._value_norms is not None

        keys_out = self._quantizer.dequantize(
            self._key_indices[..., : self.offset, :],
            self._key_norms[..., : self.offset, :],
            output_dtype,
        )
        values_out = self._quantizer.dequantize(
            self._value_indices[..., : self.offset, :],
            self._value_norms[..., : self.offset, :],
            output_dtype,
        )
        return keys_out, values_out

    # ------------------------------------------------------------------
    # mlx-lm cache protocol
    # ------------------------------------------------------------------

    def size(self) -> int:
        # ``offset`` is the logical token count in both the deferred and
        # steady-state phases (it is advanced on append in either path).
        return self.offset

    @property
    def state(self) -> object:
        """Pytree of array leaves for snapshot/restore and ``mx.eval``.

        The shape of the tuple disambiguates phases for the setter:
          - 2 elements → deferred-prefill scratch ``(pending_keys, pending_values)``
          - 4 elements → live quantized cache ``(key_indices, key_norms, value_indices, value_norms)``
          - empty list → uninitialized cache

        We deliberately do **not** include a Python ``str`` tag here:
        ``mx.eval`` and ``mx.save_safetensors`` expect every leaf in the
        cache pytree to be an ``mx.array``, and a ``str`` leaf raises at
        runtime. The phase is also tracked in ``meta_state`` via
        ``_pending_active`` for human inspection.
        """
        if self._pending_active:
            if self._pending_keys is None:
                return []
            assert self._pending_values is not None
            return (self._pending_keys, self._pending_values)
        if self._key_indices is None:
            return []
        assert self._key_norms is not None
        assert self._value_indices is not None
        assert self._value_norms is not None
        return (
            self._key_indices[..., : self.offset, :],
            self._key_norms[..., : self.offset, :],
            self._value_indices[..., : self.offset, :],
            self._value_norms[..., : self.offset, :],
        )

    @state.setter
    def state(self, value: object) -> None:
        seq = cast(Sequence[mx.array], value)
        if len(seq) == 0:
            return
        if len(seq) == 2:
            self._pending_keys = seq[0]
            self._pending_values = seq[1]
            self._pending_active = True
            # Keep ``offset`` consistent with the logical token count
            # restored from the pending scratch.
            self.offset = int(self._pending_keys.shape[2])
            return
        if len(seq) == 4:
            self._key_indices = seq[0]
            self._key_norms = seq[1]
            self._value_indices = seq[2]
            self._value_norms = seq[3]
            self.offset = int(self._key_indices.shape[2])
            self._pending_active = False
            return
        raise ValueError(
            f"RotorQuantKVCache.state expects 0, 2, or 4 array leaves, got {len(seq)}"
        )

    @property
    def meta_state(self) -> object:
        head_dim = -1 if self.head_dim is None else self.head_dim
        return (
            str(self.offset),
            str(self.seed),
            str(int(self.defer_prefill)),
            str(int(self._pending_active)),
            str(head_dim),
        )

    @meta_state.setter
    def meta_state(self, value: object) -> None:
        offset, seed, defer, pending, head_dim = (
            int(part) for part in cast(Sequence[str], value)
        )
        self.offset = offset
        self.seed = seed
        self.defer_prefill = bool(defer)
        self._pending_active = bool(pending)
        self.head_dim = None if head_dim < 0 else head_dim
        if self.head_dim is not None and self._quantizer is None:
            self._quantizer = IsoQuantizer()

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        if self._pending_active and self._pending_keys is not None:
            assert self._pending_values is not None
            available = int(self._pending_keys.shape[2])
            trimmed = min(available, n)
            new_len = available - trimmed
            self._pending_keys = self._pending_keys[..., :new_len, :]
            self._pending_values = self._pending_values[..., :new_len, :]
            self.offset -= trimmed
            return trimmed
        trimmed = min(self.offset, n)
        self.offset -= trimmed
        return trimmed

    def make_mask(self, *args: object, **kwargs: object) -> object:
        if len(args) == 0 or not isinstance(args[0], int):
            return None
        window_size = kwargs.get("window_size")
        return_array = kwargs.get("return_array", False)
        if not isinstance(window_size, (int, type(None))):
            return None
        if not isinstance(return_array, bool):
            return None
        return create_attention_mask(
            args[0],
            offset=self.size(),
            window_size=window_size,
            return_array=return_array,
        )

    @property
    def is_deferred(self) -> bool:
        """Whether the cache is still in the deferred-prefill phase.

        Public so tests and observability code can ask without poking
        at private attributes.
        """
        return self._pending_active

    @property
    def has_live_storage(self) -> bool:
        """Whether the quantized live cache has been allocated yet."""
        return self._key_indices is not None

    @property
    def has_pending_storage(self) -> bool:
        """Whether the deferred-prefill scratch buffer is currently live."""
        return self._pending_keys is not None

    def empty(self) -> bool:
        if self._pending_active:
            return self._pending_keys is None
        return self._key_indices is None

    @property
    def nbytes(self) -> int:
        total = 0
        if self._key_indices is not None:
            assert self._key_norms is not None
            assert self._value_indices is not None
            assert self._value_norms is not None
            total += int(self._key_indices.nbytes)
            total += int(self._key_norms.nbytes)
            total += int(self._value_indices.nbytes)
            total += int(self._value_norms.nbytes)
        if self._pending_keys is not None:
            assert self._pending_values is not None
            total += int(self._pending_keys.nbytes)
            total += int(self._pending_values.nbytes)
        return total


# ----------------------------------------------------------------------
# Compatibility check + factory functions (mirrors turboquant/cache.py)
# ----------------------------------------------------------------------


def ensure_rotorquant_compatible(model: Model) -> None:
    """Validate that the model uses cache types RotorQuant can wrap.

    RotorQuant can only replace plain ``KVCache`` entries; ``ArraysCache``
    (SSM) and ``RotatingKVCache`` (sliding window) entries are passed
    through to mlx-lm unchanged. Anything else is rejected so we never
    silently corrupt a model with custom cache layouts.
    """
    if not hasattr(model, "make_cache"):
        return

    sample_cache = cast(
        list[object],
        model.make_cache(),  # type: ignore[reportUnknownMemberType]
    )
    if len(sample_cache) == 0:
        return
    if not all(
        isinstance(entry, (KVCache, ArraysCache, RotatingKVCache))
        for entry in sample_cache
    ):
        cache_types = ", ".join(
            sorted({type(entry).__name__ for entry in sample_cache})
        )
        raise ValueError(
            "RotorQuant backend currently supports KVCache entries plus optional "
            "ArraysCache and RotatingKVCache passthrough; "
            f"found cache type(s): {cache_types}"
        )


def make_rotorquant_cache_from_template(
    model: Model,
    *,
    defer_prefill: bool = True,
    seed: int = 42,
) -> KVCacheType:
    """Build one RotorQuantKVCache per attention layer.

    Mirrors :func:`make_turboquant_cache_from_template`. Non-KV entries
    such as Mamba SSM caches are preserved unchanged.
    """
    ensure_rotorquant_compatible(model)
    if not hasattr(model, "make_cache"):
        return [
            RotorQuantKVCache(defer_prefill=defer_prefill, seed=seed + index)
            for index, _layer in enumerate(model.layers)
        ]

    template_cache = cast(
        list[object],
        model.make_cache(),  # type: ignore[reportUnknownMemberType]
    )
    caches: list[object] = []
    kv_index = 0
    for entry in template_cache:
        if isinstance(entry, KVCache):
            caches.append(
                RotorQuantKVCache(defer_prefill=defer_prefill, seed=seed + kv_index)
            )
            kv_index += 1
        else:
            caches.append(entry)
    return caches


def make_rotorquant_adaptive_cache(
    model: Model,
    *,
    fp16_layers: int,
    defer_prefill: bool = True,
    seed: int = 42,
) -> KVCacheType:
    """Build a RotorQuant cache with FP16 protection on edge layers.

    The first ``fp16_layers`` and last ``fp16_layers`` attention layers
    use plain mlx-lm ``KVCache`` instances; everything in between uses
    :class:`RotorQuantKVCache`. Mirrors
    :func:`make_turboquant_adaptive_cache` exactly.
    """
    ensure_rotorquant_compatible(model)
    if not hasattr(model, "make_cache"):
        caches: list[object] = []
        for index, _layer in enumerate(model.layers):
            if index < fp16_layers or index >= len(model.layers) - fp16_layers:
                caches.append(KVCache())
            else:
                caches.append(
                    RotorQuantKVCache(
                        defer_prefill=defer_prefill,
                        seed=seed + index,
                    )
                )
        return caches

    template_cache = cast(
        list[object],
        model.make_cache(),  # type: ignore[reportUnknownMemberType]
    )
    kv_positions = [
        index
        for index, entry in enumerate(template_cache)
        if isinstance(entry, KVCache)
    ]
    caches = []
    for index, entry in enumerate(template_cache):
        if not isinstance(entry, KVCache):
            caches.append(entry)
            continue

        kv_order = kv_positions.index(index)
        if kv_order < fp16_layers or kv_order >= len(kv_positions) - fp16_layers:
            caches.append(KVCache())
        else:
            caches.append(
                RotorQuantKVCache(
                    defer_prefill=defer_prefill,
                    seed=seed + kv_order,
                )
            )
    return caches
