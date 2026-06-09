from skulk.shared.models.memory_estimate import (
    GPU_WORKING_SET_FRACTION,
    MEMORY_OVERHEAD_FACTOR,
    MEMORY_OVERHEAD_FLOOR,
    estimate_kv_cache_bytes,
    estimate_shard_footprint,
    gpu_working_set_ceiling,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory


def _card(
    storage_gb: float, *, kv_heads: int | None = None, n_layers: int = 32
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(storage_gb),
        n_layers=n_layers,
        hidden_size=1000,
        supports_tensor=True,
        num_key_value_heads=kv_heads,
        tasks=[ModelTask.TextGeneration],
    )


def test_kv_is_zero_without_kv_heads():
    assert estimate_kv_cache_bytes(_card(8), 32, 8192).in_bytes == 0


def test_kv_zero_for_nonpositive_args():
    card = _card(8, kv_heads=8)
    assert estimate_kv_cache_bytes(card, 0, 8192).in_bytes == 0
    assert estimate_kv_cache_bytes(card, 16, 0).in_bytes == 0


def test_kv_scales_linearly_with_layers_and_context():
    card = _card(8, kv_heads=8)
    base = estimate_kv_cache_bytes(card, 16, 8192).in_bytes
    assert base > 0
    assert estimate_kv_cache_bytes(card, 32, 8192).in_bytes == 2 * base
    assert estimate_kv_cache_bytes(card, 16, 16384).in_bytes == 2 * base


def test_shard_footprint_whole_model_is_weights_times_factor_plus_floor():
    card = _card(8)  # no KV
    expected = Memory.from_gb(8) * MEMORY_OVERHEAD_FACTOR + MEMORY_OVERHEAD_FLOOR
    assert estimate_shard_footprint(card, 1.0).in_bytes == expected.in_bytes


def test_shard_footprint_zero_fraction_is_zero():
    assert estimate_shard_footprint(_card(8), 0.0).in_bytes == 0


def test_shard_footprint_fraction_scales_weights_and_kv_not_floor():
    card = _card(8, kv_heads=8)
    half = estimate_shard_footprint(card, 0.5)
    whole = estimate_shard_footprint(card, 1.0)
    assert half.in_bytes < whole.in_bytes
    # Weights and KV both scale with the fraction; only the flat floor does not,
    # so (footprint - floor) is linear in the fraction.
    floor = MEMORY_OVERHEAD_FLOOR.in_bytes
    assert abs((half.in_bytes - floor) * 2 - (whole.in_bytes - floor)) <= 4


def test_gpu_working_set_ceiling_is_fraction_of_total():
    expected = Memory.from_gb(16) * GPU_WORKING_SET_FRACTION
    assert gpu_working_set_ceiling(Memory.from_gb(16)).in_bytes == expected.in_bytes
