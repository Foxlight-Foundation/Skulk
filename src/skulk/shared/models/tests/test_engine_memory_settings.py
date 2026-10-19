"""Actual runtime policy must drive configuration-specific admission."""

import pytest

from skulk.shared.models.engine_memory_settings import (
    EngineMemorySettings,
    resolve_engine_memory_settings,
)
from skulk.shared.models.memory_estimate import (
    estimate_shard_footprint,
    per_token_kv_bytes,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory


def memory_card() -> ModelCard:
    """Return one selected quantized artifact with published standard geometry."""
    return ModelCard(
        model_id=ModelId("test/cache-policy"),
        storage_size=Memory.from_gb(8),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        num_key_value_heads=8,
        attention_head_dim=128,
        family="qwen3",
        quantization="Q4",
        tasks=[ModelTask.TextGeneration],
    )


@pytest.mark.parametrize("raw,expected", [("",32),("8",8),("+8",8),("0",32),("-3",32),("nonsense",32)])
def test_vllm_capacity_is_part_of_the_actual_memory_policy(raw: str, expected: int) -> None:
    """Placement and worker cache policy include the native sequence limit."""
    policy = resolve_engine_memory_settings({"SKULK_VLLM_MAX_CONCURRENT_REQUESTS":raw}, engine="vllm")
    assert policy.vllm_max_concurrent_requests == expected
    assert policy.key_bits == policy.value_bits == 16


def test_weight_quantization_does_not_select_cache_quantization() -> None:
    card = memory_card()
    default = resolve_engine_memory_settings({})
    compressed = resolve_engine_memory_settings(
        {"SKULK_KV_CACHE_BACKEND": "mlx_quantized", "SKULK_KV_CACHE_BITS": "4"}
    )
    assert default.cache_backend == "default"
    assert (
        per_token_kv_bytes(card, resolved_backend="mlx-metal", memory_settings=default)
        == 131072
    )
    assert estimate_shard_footprint(
        card, 1.0, resolved_backend="mlx-metal", memory_settings=compressed
    ) < estimate_shard_footprint(
        card, 1.0, resolved_backend="mlx-metal", memory_settings=default
    )
    assert card.quantization == "Q4"


def test_turboquant_accounts_for_byte_codes_and_temporary_full_precision() -> None:
    card = memory_card()
    policy = resolve_engine_memory_settings(
        {
            "SKULK_KV_CACHE_BACKEND": "turboquant",
            "SKULK_TQ_K_BITS": "3",
            "SKULK_TQ_V_BITS": "4",
        }
    )
    # Byte codes plus FP32 norms for all layers, and one FP16 attention layer.
    expected = 32 * 8 * (128 + 4) * 2 + 8 * 128 * 2 * 2
    assert (
        per_token_kv_bytes(card, resolved_backend="mlx-metal", memory_settings=policy)
        == expected
    )
    assert expected > 32 * 8 * 128 * (3 + 4) // 8


def test_adaptive_pipeline_charges_full_precision_edges_on_every_rank() -> None:
    card = memory_card()
    policy = EngineMemorySettings(
        cache_backend="turboquant_adaptive",
        key_bits=3,
        value_bits=4,
        fp16_edge_layers=4,
    )
    # Four eight-layer ranks each retain all eight local layers in FP16.
    rank = estimate_shard_footprint(
        card,
        0.25,
        resolved_backend="mlx-metal",
        memory_settings=policy,
        cache_layer_count=8,
    )
    baseline = estimate_shard_footprint(
        card,
        0.25,
        resolved_backend="mlx-metal",
        memory_settings=EngineMemorySettings(
            cache_backend="mlx_quantized", key_bits=4, value_bits=4
        ),
        cache_layer_count=8,
    )
    assert rank > baseline
    assert (
        per_token_kv_bytes(
            card,
            resolved_backend="mlx-metal",
            memory_settings=policy,
            cache_layer_count=8,
        )
        == 131072
    )


def test_unknown_cache_layout_and_optiq_fallback_do_not_invent_savings() -> None:
    card = memory_card().model_copy(update={"family": "unknown"})
    policy = EngineMemorySettings(
        cache_backend="mlx_quantized", key_bits=4, value_bits=4
    )
    assert (
        per_token_kv_bytes(card, resolved_backend="mlx-metal", memory_settings=policy)
        == 131072
    )
    assert (
        per_token_kv_bytes(
            memory_card(),
            resolved_backend="mlx-metal",
            memory_settings=EngineMemorySettings(
                cache_backend="optiq", key_bits=4, value_bits=4
            ),
        )
        == 131072
    )


def test_engine_selection_controls_overhead_and_cache_independently_of_file_format() -> (
    None
):
    card = memory_card().model_copy(update={"gguf_file": "model.gguf"})
    quant = EngineMemorySettings(
        cache_backend="mlx_quantized", key_bits=4, value_bits=4
    )
    assert estimate_shard_footprint(
        card, 1.0, resolved_backend="vllm-cuda", memory_settings=quant
    ) > estimate_shard_footprint(card, 1.0, resolved_backend="llama_server-cuda")
    assert (
        per_token_kv_bytes(card, resolved_backend="vllm-cuda", memory_settings=quant)
        == 131072
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"SKULK_KV_CACHE_BACKEND": "mlx_quantized"},
        {"SKULK_KV_CACHE_BACKEND": "mlx_quantized", "SKULK_KV_CACHE_BITS": "1"},
        {"SKULK_KV_CACHE_BACKEND": "turboquant", "SKULK_TQ_K_BITS": "9"},
    ],
)
def test_invalid_active_cache_controls_fail_before_optimistic_estimation(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        resolve_engine_memory_settings(environment)


def test_invalid_cache_controls_do_not_break_unrelated_engines() -> None:
    """Bad MLX controls remain visible without disabling valid native engines."""
    from skulk.shared.models.engine_memory_settings import (
        observe_engine_memory_settings,
    )

    env = {"SKULK_KV_CACHE_BACKEND": "mlx_quantized"}
    assert observe_engine_memory_settings(env).configuration_error is not None
    assert (
        observe_engine_memory_settings(env, engine="audio_cpp").configuration_error
        is None
    )
