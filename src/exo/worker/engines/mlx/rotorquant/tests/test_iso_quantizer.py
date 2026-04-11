"""Correctness tests for the IsoQuant 3-bit quantizer.

A bit-exact comparison against a precomputed llama.cpp CPU reference
fixture is desirable but blocked on producing the fixture (which
requires building the llama.cpp fork). For v1 we validate via
self-consistency properties that any correct implementation must
satisfy: rotation orthogonality, round-trip reconstruction error,
norm-correction unbiasedness, and shape contracts.
"""

import math

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.rotorquant.quantizer import IsoQuantizer
from exo.worker.engines.mlx.rotorquant.rotation import (
    iso3_rotate_forward,
    iso3_rotate_inverse,
)
from exo.worker.engines.mlx.rotorquant.tables import ISO3_BLOCK_SIZE


def _random_kv(shape: tuple[int, ...], seed: int = 0) -> mx.array:
    """Sample a Gaussian tensor matching a typical attention KV layout."""
    return mx.random.normal(shape=shape, key=mx.random.key(seed))


def test_quaternion_rotation_is_orthogonal() -> None:
    """``||R x|| == ||x||`` for the per-block rotation."""
    x = _random_kv((2, 4, 8, ISO3_BLOCK_SIZE))
    rotated = iso3_rotate_forward(x.astype(mx.float32))
    norm_in = mx.sqrt(mx.sum(x * x, axis=-1))
    norm_out = mx.sqrt(mx.sum(rotated * rotated, axis=-1))
    diff = mx.max(mx.abs(norm_in - norm_out)).item()
    assert diff < 1e-4, f"rotation not orthogonal: max ||x|| - ||Rx|| = {diff}"


def test_rotation_inverse_round_trips() -> None:
    """``R^{-1} R x == x`` to floating-point precision."""
    x = _random_kv((1, 2, 3, ISO3_BLOCK_SIZE), seed=1).astype(mx.float32)
    restored = iso3_rotate_inverse(iso3_rotate_forward(x))
    diff = mx.max(mx.abs(x - restored)).item()
    assert diff < 1e-4, f"R^-1 R x diverged by {diff}"


def test_quantize_dequantize_shapes() -> None:
    """Indices and norms have the documented shapes for a typical KV layer."""
    quantizer = IsoQuantizer()
    keys = _random_kv((2, 8, 16, 256))  # head_dim 256 = 2 iso3 blocks
    indices, norms = quantizer.quantize(keys)
    assert indices.shape == (2, 8, 16, 256)
    assert indices.dtype == mx.uint8
    assert norms.shape == (2, 8, 16, 2)
    assert norms.dtype == mx.float16
    restored = quantizer.dequantize(indices, norms, mx.float16)
    assert restored.shape == keys.shape
    assert restored.dtype == mx.float16


def test_quantize_indices_in_valid_range() -> None:
    """Index values must lie in ``[0, 7]`` for the 3-bit codebook."""
    quantizer = IsoQuantizer()
    x = _random_kv((1, 1, 4, ISO3_BLOCK_SIZE), seed=2)
    indices, _ = quantizer.quantize(x)
    max_idx = int(mx.max(indices).item())
    min_idx = int(mx.min(indices).item())
    assert min_idx >= 0
    assert max_idx <= 7


def test_round_trip_reconstruction_is_within_3bit_budget() -> None:
    """End-to-end reconstruction error stays within the 3-bit centroid budget.

    With 8 Lloyd-Max centroids on a Gaussian-ish post-rotation
    distribution, the per-element MSE should be around 0.04 of the
    input variance. We use a generous threshold (0.15) so the test
    catches outright bugs without flaking on RNG variance.
    """
    quantizer = IsoQuantizer()
    x = _random_kv((1, 4, 32, ISO3_BLOCK_SIZE), seed=3).astype(mx.float32)
    indices, norms = quantizer.quantize(x)
    x_hat = quantizer.dequantize(indices, norms, mx.float32)

    err = x - x_hat
    rel_mse = (mx.sum(err * err) / mx.sum(x * x)).item()
    assert rel_mse < 0.15, f"3-bit round-trip relative MSE {rel_mse} exceeds budget"


def test_norm_correction_keeps_magnitudes_unbiased() -> None:
    """The norm-correction trick should leave per-block ``||x||`` unbiased.

    Without correction, centroid quantization shrinks the unit-vector
    norm and the dequantized output systematically underestimates the
    input magnitude. With correction the average magnitude ratio
    should sit at ~1.0 across many random blocks.
    """
    quantizer = IsoQuantizer()
    x = _random_kv((4, 8, 64, ISO3_BLOCK_SIZE), seed=4).astype(mx.float32)
    indices, norms = quantizer.quantize(x)
    x_hat = quantizer.dequantize(indices, norms, mx.float32)

    norm_in = mx.sqrt(mx.sum(x * x, axis=-1))
    norm_out = mx.sqrt(mx.sum(x_hat * x_hat, axis=-1))
    ratio = (norm_out / mx.maximum(norm_in, 1e-8)).reshape(-1)
    mean_ratio = float(mx.mean(ratio).item())
    # Slack covers fp16 round-tripping plus the variance of a finite
    # sample. The unbiased target is exactly 1.0.
    assert math.isclose(mean_ratio, 1.0, abs_tol=0.05), (
        f"norm-correction biased mean ratio = {mean_ratio}"
    )


def test_quantize_rejects_non_multiple_head_dim() -> None:
    quantizer = IsoQuantizer()
    bad = _random_kv((1, 1, 1, 100), seed=5)  # 100 is not a multiple of 128
    with pytest.raises(ValueError):
        quantizer.quantize(bad)
