"""Check planner cache costs against actual allocator buffers, not ideal ratios."""

from collections.abc import Callable
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import QuantizedKVCache

from skulk.shared.models.engine_memory_settings import EngineMemorySettings
from skulk.shared.models.memory_estimate import estimate_kv_cache_bytes
from skulk.shared.models.tests.test_engine_memory_settings import memory_card
from skulk.worker.engines.mlx.turboquant.cache import TurboQuantKVCache


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_affine_cache_budget_covers_real_packed_buffers(bits: int) -> None:
    card = memory_card().model_copy(update={"n_layers": 1, "num_key_value_heads": 2})
    cache = QuantizedKVCache(group_size=64, bits=bits)
    data = mx.ones((1, 2, 257, 128), dtype=mx.float16)
    update = cast(Callable[[mx.array, mx.array], object], cache.update_and_fetch)
    update(data, data)
    mx.eval(cache.keys, cache.values)
    assert cache.keys is not None and cache.values is not None
    actual = sum(item.nbytes for item in (*cache.keys, *cache.values))
    estimate = estimate_kv_cache_bytes(
        card,
        1,
        257,
        resolved_backend="mlx-metal",
        memory_settings=EngineMemorySettings(
            cache_backend="mlx_quantized", key_bits=bits, value_bits=bits
        ),
    )
    assert actual <= estimate.in_bytes


def test_turboquant_budget_covers_real_byte_codes_and_norms() -> None:
    card = memory_card().model_copy(update={"n_layers": 1, "num_key_value_heads": 2})
    cache = TurboQuantKVCache(key_bits=3, value_bits=4)
    data = mx.ones((1, 2, 17, 128), dtype=mx.float16)
    keys, values = cache.update_and_fetch(data, data)
    assert cache.key_indices is not None and cache.key_norms is not None
    assert cache.value_indices is not None and cache.value_norms is not None
    mx.eval(
        keys,
        values,
        cache.key_indices,
        cache.key_norms,
        cache.value_indices,
        cache.value_norms,
    )
    actual = sum(
        item.nbytes
        for item in (
            cache.key_indices,
            cache.key_norms,
            cache.value_indices,
            cache.value_norms,
            keys,
            values,
        )
    )
    estimate = estimate_kv_cache_bytes(
        card,
        1,
        17,
        resolved_backend="mlx-metal",
        memory_settings=EngineMemorySettings(
            cache_backend="turboquant", key_bits=3, value_bits=4
        ),
    )
    assert actual <= estimate.in_bytes
