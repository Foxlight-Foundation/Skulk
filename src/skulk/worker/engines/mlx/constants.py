import os
from typing import Literal, cast

from skulk.shared.constants import (
    DEFAULT_MAX_OUTPUT_TOKENS as SHARED_DEFAULT_MAX_OUTPUT_TOKENS,
)
from skulk.shared.constants import (
    MAX_OUTPUT_TOKENS,
    preferred_env_value,
)
from skulk.shared.models.engine_memory_settings import observe_engine_memory_settings

ENGINE_MEMORY_SETTINGS = observe_engine_memory_settings(os.environ)

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
# Backwards-compatible engine-local alias. The default is shared with served
# engines because it is an API contract, not an MLX implementation detail.
DEFAULT_MAX_OUTPUT_TOKENS: int = SHARED_DEFAULT_MAX_OUTPUT_TOKENS
MAX_TOKENS: int = MAX_OUTPUT_TOKENS
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = (
    ENGINE_MEMORY_SETTINGS.key_bits
    if ENGINE_MEMORY_SETTINGS.cache_backend == "mlx_quantized"
    else None
)
KVCacheBackend = Literal[
    "default",
    "mlx_quantized",
    "turboquant",
    "turboquant_adaptive",
    "optiq",
]
DEFAULT_KV_CACHE_BACKEND: KVCacheBackend = "default"
VALID_KV_CACHE_BACKENDS: tuple[KVCacheBackend, ...] = (
    "default",
    "mlx_quantized",
    "turboquant",
    "turboquant_adaptive",
    "optiq",
)
_kv_cache_backend_value = preferred_env_value(
    "SKULK_KV_CACHE_BACKEND",
    default=DEFAULT_KV_CACHE_BACKEND,
)
KV_CACHE_BACKEND: KVCacheBackend = cast(
    KVCacheBackend,
    _kv_cache_backend_value if _kv_cache_backend_value else DEFAULT_KV_CACHE_BACKEND,
)
TURBOQUANT_K_BITS: int | None = (
    ENGINE_MEMORY_SETTINGS.key_bits
    if ENGINE_MEMORY_SETTINGS.cache_backend in ("turboquant", "turboquant_adaptive")
    else None
)
TURBOQUANT_V_BITS: int | None = (
    ENGINE_MEMORY_SETTINGS.value_bits
    if ENGINE_MEMORY_SETTINGS.cache_backend in ("turboquant", "turboquant_adaptive")
    else None
)
TURBOQUANT_FP16_LAYERS: int = (
    ENGINE_MEMORY_SETTINGS.fp16_edge_layers
    if ENGINE_MEMORY_SETTINGS.cache_backend == "turboquant_adaptive"
    else 4
)
DEFAULT_TURBOQUANT_K_BITS: int = 3
DEFAULT_TURBOQUANT_V_BITS: int = 4
OPTIQ_BITS: int = (
    ENGINE_MEMORY_SETTINGS.key_bits
    if ENGINE_MEMORY_SETTINGS.cache_backend == "optiq"
    else 4
)
OPTIQ_FP16_LAYERS: int = (
    ENGINE_MEMORY_SETTINGS.fp16_edge_layers
    if ENGINE_MEMORY_SETTINGS.cache_backend == "optiq"
    else 4
)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
