"""Effective runner cache controls shared by node facts and memory admission."""

from collections.abc import Mapping
from typing import Literal, final

from pydantic import Field

from skulk.utils.pydantic_ext import FrozenModel

CacheBackend = Literal[
    "default", "mlx_quantized", "turboquant", "turboquant_adaptive", "optiq"
]
CACHE_BACKENDS: tuple[CacheBackend, ...] = (
    "default",
    "mlx_quantized",
    "turboquant",
    "turboquant_adaptive",
    "optiq",
)


@final
class EngineMemorySettings(FrozenModel):
    """Actual cache configuration, independent of stored weight quantization.

    Non-MLX runners currently use fixed two-byte attention elements. MLX controls
    are resolved from the same inherited environment used by the child runner.
    Bit widths describe quantizer precision; they do not imply packed storage.
    """

    cache_backend: CacheBackend = Field(
        default="default",
        description="Effective cache implementation, after runner fallback for unknown selectors.",
    )
    key_bits: int = Field(
        default=16,
        ge=1,
        le=16,
        description="Key quantizer precision; default cache stores full precision.",
    )
    value_bits: int = Field(
        default=16,
        ge=1,
        le=16,
        description="Value quantizer precision, independent of stored model weights.",
    )
    group_size: int = Field(
        default=64,
        gt=0,
        description="Elements per group for MLX affine cache scales and biases.",
    )
    fp16_edge_layers: int = Field(
        default=0,
        ge=0,
        description="Number of full-precision attention layers retained at each edge.",
    )
    configuration_error: str | None = Field(
        default=None,
        max_length=1000,
        description="Invalid active cache controls; the engine must not allocate with this policy. Other node engines remain usable.",
    )


def resolve_engine_memory_settings(
    environment: Mapping[str, str],
    *,
    engine: str = "mlx",
) -> EngineMemorySettings:
    """Resolve effective settings using runner defaults and environment precedence.

    The node applies configuration to its environment before this resolver runs.
    Invalid bit widths fail explicitly rather than creating an optimistic budget.
    Non-MLX engines use their explicitly pinned full-precision attention policy.
    """
    if engine != "mlx":
        return EngineMemorySettings()
    requested = environment.get("SKULK_KV_CACHE_BACKEND", "default") or "default"
    backend = next((item for item in CACHE_BACKENDS if item == requested), "default")
    if backend == "default":
        return EngineMemorySettings()
    if backend == "mlx_quantized":
        raw = environment.get("SKULK_KV_CACHE_BITS")
        if raw is None:
            raise ValueError("mlx_quantized requires SKULK_KV_CACHE_BITS")
        bits = int(raw)
        if bits not in (2, 3, 4, 6, 8):
            raise ValueError("MLX cache quantization supports 2, 3, 4, 6 or 8 bits")
        return EngineMemorySettings(
            cache_backend=backend, key_bits=bits, value_bits=bits
        )
    if backend in ("turboquant", "turboquant_adaptive"):
        key = int(environment.get("SKULK_TQ_K_BITS", "3") or "3") or 3
        value = int(environment.get("SKULK_TQ_V_BITS", "4") or "4") or 4
        if key not in (2, 3, 4) or value not in (2, 3, 4):
            raise ValueError("TurboQuant cache supports 2, 3 or 4 bits")
        return EngineMemorySettings(
            cache_backend=backend,
            key_bits=key,
            value_bits=value,
            fp16_edge_layers=int(environment.get("SKULK_TQ_FP16_LAYERS", "4"))
            if backend == "turboquant_adaptive"
            else 0,
        )
    bits = int(environment.get("SKULK_OPTIQ_BITS", "4"))
    return EngineMemorySettings(
        cache_backend="optiq",
        key_bits=bits,
        value_bits=bits,
        fp16_edge_layers=int(environment.get("SKULK_OPTIQ_FP16_LAYERS", "4")),
    )


def observe_engine_memory_settings(
    environment: Mapping[str, str],
    *,
    engine: str = "mlx",
) -> EngineMemorySettings:
    """Retain an invalid engine setting as a diagnostic without breaking the node.

    No compression savings are claimed for invalid controls. Placement and the
    worker refuse that engine configuration; unrelated engines remain available.
    """
    try:
        return resolve_engine_memory_settings(environment, engine=engine)
    except ValueError as error:
        return EngineMemorySettings(configuration_error=str(error)[:1000])
