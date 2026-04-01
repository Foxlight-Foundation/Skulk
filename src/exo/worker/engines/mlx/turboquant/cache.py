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
from exo.worker.engines.mlx.turboquant.quantizer import TurboQuantizer


class TurboQuantKVCache:
    step = 256

    def __init__(
        self,
        *,
        key_bits: int,
        value_bits: int,
        seed: int = 42,
    ) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.seed = seed
        self.offset = 0

        self.key_indices: mx.array | None = None
        self.key_norms: mx.array | None = None
        self.value_indices: mx.array | None = None
        self.value_norms: mx.array | None = None

        self.key_dim: int | None = None
        self.value_dim: int | None = None
        self._key_quantizer: TurboQuantizer | None = None
        self._value_quantizer: TurboQuantizer | None = None

    def _ensure_quantizers(self, key_dim: int, value_dim: int) -> None:
        if self._key_quantizer is None:
            self._key_quantizer = TurboQuantizer(key_dim, self.key_bits, self.seed)
            self.key_dim = key_dim
        if self._value_quantizer is None:
            self._value_quantizer = TurboQuantizer(
                value_dim,
                self.value_bits,
                self.seed + 1,
            )
            self.value_dim = value_dim

    def _expand_storage(
        self,
        batch_size: int,
        kv_heads: int,
        steps: int,
        key_dim: int,
        value_dim: int,
    ) -> None:
        needed = self.offset + steps
        if self.key_indices is not None and needed <= self.key_indices.shape[2]:
            return

        new_steps = ((steps + self.step - 1) // self.step) * self.step
        base_shape = (batch_size, kv_heads, new_steps)
        new_key_indices = mx.zeros((*base_shape, key_dim), dtype=mx.uint8)
        new_key_norms = mx.zeros(base_shape, dtype=mx.float32)
        new_value_indices = mx.zeros((*base_shape, value_dim), dtype=mx.uint8)
        new_value_norms = mx.zeros(base_shape, dtype=mx.float32)

        old_key_indices = self.key_indices
        old_key_norms = self.key_norms
        old_value_indices = self.value_indices
        old_value_norms = self.value_norms
        if old_key_indices is None:
            self.key_indices = new_key_indices
            self.key_norms = new_key_norms
            self.value_indices = new_value_indices
            self.value_norms = new_value_norms
            return

        assert old_key_norms is not None
        assert old_value_indices is not None
        assert old_value_norms is not None
        self.key_indices = mx.concatenate(
            [old_key_indices[..., : self.offset, :], new_key_indices],
            axis=2,
        )
        self.key_norms = mx.concatenate(
            [old_key_norms[..., : self.offset], new_key_norms],
            axis=2,
        )
        self.value_indices = mx.concatenate(
            [old_value_indices[..., : self.offset, :], new_value_indices],
            axis=2,
        )
        self.value_norms = mx.concatenate(
            [old_value_norms[..., : self.offset], new_value_norms],
            axis=2,
        )

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch_size, kv_heads, steps, key_dim = keys.shape
        value_dim = int(values.shape[3])
        prev = self.offset

        self._ensure_quantizers(key_dim, value_dim)
        self._expand_storage(batch_size, kv_heads, steps, key_dim, value_dim)

        assert self.key_indices is not None
        assert self.key_norms is not None
        assert self.value_indices is not None
        assert self.value_norms is not None
        assert self._key_quantizer is not None
        assert self._value_quantizer is not None

        key_indices, key_norms = self._key_quantizer.quantize(keys.reshape(-1, key_dim))
        value_indices, value_norms = self._value_quantizer.quantize(
            values.reshape(-1, value_dim)
        )

        self.key_indices[..., prev : prev + steps, :] = key_indices.reshape(
            batch_size,
            kv_heads,
            steps,
            key_dim,
        )
        self.key_norms[..., prev : prev + steps] = key_norms.reshape(
            batch_size,
            kv_heads,
            steps,
        )
        self.value_indices[..., prev : prev + steps, :] = value_indices.reshape(
            batch_size,
            kv_heads,
            steps,
            value_dim,
        )
        self.value_norms[..., prev : prev + steps] = value_norms.reshape(
            batch_size,
            kv_heads,
            steps,
        )
        self.offset += steps

        dequant_keys = self._key_quantizer.dequantize(
            self.key_indices[..., : self.offset, :],
            self.key_norms[..., : self.offset],
            keys.dtype,
        )
        dequant_values = self._value_quantizer.dequantize(
            self.value_indices[..., : self.offset, :],
            self.value_norms[..., : self.offset],
            values.dtype,
        )
        return dequant_keys, dequant_values

    def size(self) -> int:
        return self.offset

    @property
    def state(self) -> object:
        if self.key_indices is None:
            return []
        assert self.key_norms is not None
        assert self.value_indices is not None
        assert self.value_norms is not None
        return (
            self.key_indices[..., : self.offset, :],
            self.key_norms[..., : self.offset],
            self.value_indices[..., : self.offset, :],
            self.value_norms[..., : self.offset],
        )

    @state.setter
    def state(self, v: object) -> None:
        (
            self.key_indices,
            self.key_norms,
            self.value_indices,
            self.value_norms,
        ) = cast(Sequence[mx.array], v)
        self.offset = int(self.key_indices.shape[2])

    @property
    def meta_state(self) -> object:
        key_dim = -1 if self.key_dim is None else self.key_dim
        value_dim = -1 if self.value_dim is None else self.value_dim
        return (
            str(self.offset),
            str(self.key_bits),
            str(self.value_bits),
            str(self.seed),
            str(key_dim),
            str(value_dim),
        )

    @meta_state.setter
    def meta_state(self, v: object) -> None:
        (
            offset,
            key_bits,
            value_bits,
            seed,
            key_dim,
            value_dim,
        ) = map(int, cast(Sequence[str], v))
        self.offset = offset
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.seed = seed
        self.key_dim = None if key_dim < 0 else key_dim
        self.value_dim = None if value_dim < 0 else value_dim
        self._key_quantizer = (
            None
            if self.key_dim is None
            else TurboQuantizer(self.key_dim, self.key_bits, self.seed)
        )
        self._value_quantizer = (
            None
            if self.value_dim is None
            else TurboQuantizer(self.value_dim, self.value_bits, self.seed + 1)
        )

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
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
            offset=self.offset,
            window_size=window_size,
            return_array=return_array,
        )

    def empty(self) -> bool:
        return self.key_indices is None

    @property
    def nbytes(self) -> int:
        if self.key_indices is None:
            return 0
        assert self.key_norms is not None
        assert self.value_indices is not None
        assert self.value_norms is not None
        return (
            self.key_indices.nbytes
            + self.key_norms.nbytes
            + self.value_indices.nbytes
            + self.value_norms.nbytes
        )


def ensure_standard_attention(model: Model) -> None:
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
            "TurboQuant backend currently supports KVCache entries plus optional ArraysCache "
            "and RotatingKVCache entries; "
            f"found cache type(s): {cache_types}"
        )


def make_turboquant_cache_from_template(
    model: Model,
    *,
    key_bits: int,
    value_bits: int,
    seed: int = 42,
) -> KVCacheType:
    ensure_standard_attention(model)
    if not hasattr(model, "make_cache"):
        return [
            TurboQuantKVCache(
                key_bits=key_bits,
                value_bits=value_bits,
                seed=seed + index,
            )
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
                TurboQuantKVCache(
                    key_bits=key_bits,
                    value_bits=value_bits,
                    seed=seed + kv_index,
                )
            )
            kv_index += 1
        else:
            caches.append(entry)
    return caches


def make_turboquant_adaptive_cache(
    model: Model,
    *,
    key_bits: int,
    value_bits: int,
    fp16_layers: int,
    seed: int = 42,
) -> KVCacheType:
    ensure_standard_attention(model)
    if not hasattr(model, "make_cache"):
        caches: list[object] = []
        for index, _layer in enumerate(model.layers):
            if index < fp16_layers or index >= len(model.layers) - fp16_layers:
                caches.append(KVCache())
            else:
                caches.append(
                    TurboQuantKVCache(
                        key_bits=key_bits,
                        value_bits=value_bits,
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
                TurboQuantKVCache(
                    key_bits=key_bits,
                    value_bits=value_bits,
                    seed=seed + kv_order,
                )
            )
    return caches
