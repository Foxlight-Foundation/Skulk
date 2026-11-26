"""Effective runner cache controls shared by node facts and memory admission."""

from collections.abc import Iterable, Mapping
from typing import Final, Literal, final

from pydantic import Field

from skulk.shared.backends import memory_settings_key, package_engine_of
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

VLLM_DEFAULT_MAX_CONCURRENT_REQUESTS: Final = 32


def _is_none(value: object) -> bool:
    return value is None
VLLM_MAX_CONCURRENT_REQUESTS_ENV: Final = "SKULK_VLLM_MAX_CONCURRENT_REQUESTS"


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

    vllm_max_concurrent_requests: int = Field(
        default=VLLM_DEFAULT_MAX_CONCURRENT_REQUESTS,
        ge=1,
        description="Effective vLLM sequence/admission limit, used for native MTP workspace sizing; ignored by other engines.",
    )
    package_overhead_bytes: int | None = Field(
        default=None,
        ge=0,
        le=64 * 1024**3,
        exclude_if=_is_none,
        description="Memory an engine package's signed manifest adds to a model's weight bytes for admission; absent for built-in engines, so their settings read and serialize as before.",
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
    if engine == "vllm":
        try:
            capacity = int(environment.get(VLLM_MAX_CONCURRENT_REQUESTS_ENV, "") or VLLM_DEFAULT_MAX_CONCURRENT_REQUESTS)
        except ValueError:
            capacity = VLLM_DEFAULT_MAX_CONCURRENT_REQUESTS
        return EngineMemorySettings(
            vllm_max_concurrent_requests=capacity if capacity > 0 else VLLM_DEFAULT_MAX_CONCURRENT_REQUESTS
        )
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


def engine_package_memory_settings(overhead_bytes: int) -> EngineMemorySettings:
    """The memory settings of one engine package build on a node.

    A package states its own memory need in its signed manifest. Core's cache
    controls for built-in engines do not apply to it, so nothing is read from
    the environment.
    """
    return EngineMemorySettings(package_overhead_bytes=overhead_bytes)


def backend_engine_memory_settings(
    backend: str,
    package_overhead_bytes: Mapping[str, int],
    environment: Mapping[str, str],
) -> EngineMemorySettings:
    """The settings this node applies to one backend tag.

    The node's advertisement and its own load guard both come through here,
    so the figure placement stamped is the figure the guard compares.
    """
    overhead = package_overhead_bytes.get(backend)
    if overhead is not None and package_engine_of(backend) is not None:
        return engine_package_memory_settings(overhead)
    return observe_engine_memory_settings(
        environment, engine=backend.split("-", 1)[0]
    )


def node_engine_memory_settings(
    backends: Iterable[str],
    package_overhead_bytes: Mapping[str, int],
    environment: Mapping[str, str],
) -> dict[str, EngineMemorySettings]:
    """Every engine's settings on this node, keyed as ``memory_settings_key`` keys them."""
    return {
        memory_settings_key(backend): backend_engine_memory_settings(
            backend, package_overhead_bytes, environment
        )
        for backend in backends
    }


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
