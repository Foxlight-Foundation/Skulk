"""Tests for ``RotorQuantKVCache`` including the deferred-prefill path."""

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.rotorquant.cache import RotorQuantKVCache
from exo.worker.engines.mlx.rotorquant.tables import ISO3_BLOCK_SIZE


def _kv_chunk(num_steps: int, *, seed: int = 0) -> tuple[mx.array, mx.array]:
    """Build a (B, H, T, head_dim) prefill chunk for tests."""
    keys = mx.random.normal(
        shape=(1, 4, num_steps, ISO3_BLOCK_SIZE),
        key=mx.random.key(seed),
    )
    values = mx.random.normal(
        shape=(1, 4, num_steps, ISO3_BLOCK_SIZE),
        key=mx.random.key(seed + 1),
    )
    return keys, values


def test_deferred_prefill_keeps_storage_unallocated_during_prefill() -> None:
    """Multi-token calls should not allocate the quantized live cache."""
    cache = RotorQuantKVCache(defer_prefill=True)
    keys, values = _kv_chunk(64)
    cache.update_and_fetch(keys, values)

    assert cache.is_deferred is True
    assert cache.has_live_storage is False
    assert cache.offset == 0
    assert cache.size() == 64


def test_deferred_prefill_flushes_on_first_decode_call() -> None:
    """The first ``num_steps == 1`` call should flush the deferred buffer."""
    cache = RotorQuantKVCache(defer_prefill=True)
    pre_k, pre_v = _kv_chunk(48, seed=10)
    cache.update_and_fetch(pre_k, pre_v)
    assert cache.is_deferred is True

    dec_k, dec_v = _kv_chunk(1, seed=11)
    out_k, out_v = cache.update_and_fetch(dec_k, dec_v)

    assert cache.is_deferred is False
    assert cache.has_pending_storage is False
    assert cache.has_live_storage is True
    assert cache.offset == 49  # 48 prefill + 1 decode
    assert out_k.shape == (1, 4, 49, ISO3_BLOCK_SIZE)
    assert out_v.shape == (1, 4, 49, ISO3_BLOCK_SIZE)
    assert out_k.dtype == dec_k.dtype


def test_deferred_prefill_returns_raw_fp16_during_prefill() -> None:
    """The fp16 view returned during prefill should be lossless."""
    cache = RotorQuantKVCache(defer_prefill=True)
    keys, values = _kv_chunk(32, seed=20)
    out_k, out_v = cache.update_and_fetch(keys, values)

    # Equality with the input within fp16 round-tripping tolerance.
    diff_k = mx.max(mx.abs(out_k.astype(mx.float32) - keys.astype(mx.float32))).item()
    diff_v = mx.max(mx.abs(out_v.astype(mx.float32) - values.astype(mx.float32))).item()
    assert diff_k < 1e-3
    assert diff_v < 1e-3


def test_non_deferred_path_quantizes_immediately() -> None:
    """With ``defer_prefill=False`` even prefill chunks hit the quantizer."""
    cache = RotorQuantKVCache(defer_prefill=False)
    keys, values = _kv_chunk(16, seed=30)
    out_k, _ = cache.update_and_fetch(keys, values)

    assert cache.is_deferred is False
    assert cache.has_live_storage is True
    assert cache.offset == 16
    # Reconstruction is lossy through the 3-bit codebook — verify the
    # output isn't bit-identical to the input but is still close in MSE.
    err = (out_k.astype(mx.float32) - keys.astype(mx.float32)) ** 2
    rel_mse = (mx.sum(err) / mx.sum(keys.astype(mx.float32) ** 2)).item()
    assert 1e-6 < rel_mse < 0.2


def test_decode_after_prefill_appends_correctly() -> None:
    """After flush, additional decode tokens should append to the live cache."""
    cache = RotorQuantKVCache(defer_prefill=True)
    pre_k, pre_v = _kv_chunk(8, seed=40)
    cache.update_and_fetch(pre_k, pre_v)

    for step in range(3):
        dk, dv = _kv_chunk(1, seed=50 + step)
        out_k, _ = cache.update_and_fetch(dk, dv)
        assert out_k.shape[2] == 9 + step

    assert cache.offset == 11
    assert cache.size() == 11


def test_trim_works_in_both_phases() -> None:
    """``trim`` should remove tokens from whichever buffer is live."""
    pending_cache = RotorQuantKVCache(defer_prefill=True)
    keys, values = _kv_chunk(20, seed=60)
    pending_cache.update_and_fetch(keys, values)
    assert pending_cache.size() == 20
    pending_cache.trim(5)
    assert pending_cache.size() == 15

    flushed_cache = RotorQuantKVCache(defer_prefill=True)
    flushed_cache.update_and_fetch(*_kv_chunk(10, seed=70))
    flushed_cache.update_and_fetch(*_kv_chunk(1, seed=71))  # triggers flush
    assert flushed_cache.size() == 11
    flushed_cache.trim(3)
    assert flushed_cache.size() == 8


def test_head_dim_must_be_block_aligned() -> None:
    cache = RotorQuantKVCache(defer_prefill=False)
    bad_keys = mx.zeros((1, 1, 2, 100), dtype=mx.float16)
    bad_values = mx.zeros((1, 1, 2, 100), dtype=mx.float16)
    with pytest.raises(ValueError):
        cache.update_and_fetch(bad_keys, bad_values)
