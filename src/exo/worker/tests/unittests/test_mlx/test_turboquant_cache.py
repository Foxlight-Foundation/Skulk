from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.shared.types.mlx import Model
from exo.worker.engines.mlx.turboquant.cache import (
    TurboQuantKVCache,
    ensure_standard_attention,
    make_turboquant_adaptive_cache,
    make_turboquant_cache_from_template,
)


class FakeStandardModel:
    def __init__(self, num_layers: int) -> None:
        self.layers = [object() for _ in range(num_layers)]

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


class FakeUnsupportedCache:
    pass


class FakeUnsupportedModel:
    def __init__(self) -> None:
        self.layers = [object()]

    def make_cache(self) -> list[FakeUnsupportedCache]:
        return [FakeUnsupportedCache()]


class FakeMixedCacheModel:
    def __init__(self) -> None:
        self.layers = [object(), object(), object()]

    def make_cache(self) -> list[object]:
        return [ArraysCache(1), KVCache(), KVCache()]


class FakeRotatingMixedCacheModel:
    def __init__(self) -> None:
        self.layers = [object(), object(), object()]

    def make_cache(self) -> list[object]:
        return [RotatingKVCache(max_size=32), KVCache(), KVCache()]


def test_turboquant_cache_roundtrip_shape_and_offset() -> None:
    cache = TurboQuantKVCache(key_bits=3, value_bits=4)
    keys = mx.random.normal(shape=(1, 2, 3, 128)).astype(mx.float16)
    values = mx.random.normal(shape=(1, 2, 3, 128)).astype(mx.float16)

    fetched_keys, fetched_values = cache.update_and_fetch(keys, values)

    assert fetched_keys.shape == keys.shape
    assert fetched_values.shape == values.shape
    assert cache.offset == 3


def test_turboquant_cache_state_and_meta_state_roundtrip() -> None:
    cache = TurboQuantKVCache(key_bits=3, value_bits=4, seed=99)
    keys = mx.random.normal(shape=(1, 1, 2, 128)).astype(mx.float16)
    values = mx.random.normal(shape=(1, 1, 2, 128)).astype(mx.float16)
    _ = cache.update_and_fetch(keys, values)

    restored = TurboQuantKVCache(key_bits=1, value_bits=1)
    restored.state = cache.state
    restored.meta_state = cache.meta_state

    assert restored.offset == cache.offset
    assert restored.meta_state == cache.meta_state


def test_turboquant_cache_trim() -> None:
    cache = TurboQuantKVCache(key_bits=3, value_bits=4)
    keys = mx.random.normal(shape=(1, 1, 4, 128)).astype(mx.float16)
    values = mx.random.normal(shape=(1, 1, 4, 128)).astype(mx.float16)
    _ = cache.update_and_fetch(keys, values)

    trimmed = cache.trim(2)

    assert trimmed == 2
    assert cache.offset == 2


def test_adaptive_cache_uses_fp16_outer_layers() -> None:
    caches = make_turboquant_adaptive_cache(
        cast(Model, cast(object, FakeStandardModel(6))),
        key_bits=3,
        value_bits=4,
        fp16_layers=1,
    )

    assert isinstance(caches[0], KVCache)
    assert isinstance(caches[-1], KVCache)
    assert isinstance(caches[1], TurboQuantKVCache)
    assert isinstance(caches[-2], TurboQuantKVCache)


def test_ensure_standard_attention_rejects_unsupported_cache() -> None:
    with pytest.raises(ValueError, match="RotatingKVCache entries"):
        ensure_standard_attention(cast(Model, cast(object, FakeUnsupportedModel())))


def test_make_turboquant_cache_from_template_preserves_arrays_cache() -> None:
    caches = make_turboquant_cache_from_template(
        cast(Model, cast(object, FakeMixedCacheModel())),
        key_bits=3,
        value_bits=4,
    )

    assert isinstance(caches[0], ArraysCache)
    assert isinstance(caches[1], TurboQuantKVCache)
    assert isinstance(caches[2], TurboQuantKVCache)


def test_make_turboquant_cache_from_template_preserves_rotating_cache() -> None:
    caches = make_turboquant_cache_from_template(
        cast(Model, cast(object, FakeRotatingMixedCacheModel())),
        key_bits=3,
        value_bits=4,
    )

    assert isinstance(caches[0], RotatingKVCache)
    assert isinstance(caches[1], TurboQuantKVCache)
    assert isinstance(caches[2], TurboQuantKVCache)
