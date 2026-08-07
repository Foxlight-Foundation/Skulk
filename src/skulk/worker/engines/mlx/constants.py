import os
from typing import Literal, cast

from skulk.shared.constants import (
    DEFAULT_MAX_OUTPUT_TOKENS as SHARED_DEFAULT_MAX_OUTPUT_TOKENS,
)
from skulk.shared.constants import (
    MAX_OUTPUT_TOKENS,
    preferred_env_value,
)

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
    int(os.environ["SKULK_KV_CACHE_BITS"]) if "SKULK_KV_CACHE_BITS" in os.environ else None
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
    int(os.environ.get("SKULK_TQ_K_BITS", os.environ.get("SKULK_TQ_K_BITS", "")))
    if os.environ.get("SKULK_TQ_K_BITS", os.environ.get("SKULK_TQ_K_BITS"))
    else None
)
TURBOQUANT_V_BITS: int | None = (
    int(os.environ.get("SKULK_TQ_V_BITS", os.environ.get("SKULK_TQ_V_BITS", "")))
    if os.environ.get("SKULK_TQ_V_BITS", os.environ.get("SKULK_TQ_V_BITS"))
    else None
)
TURBOQUANT_FP16_LAYERS: int = int(os.environ.get("SKULK_TQ_FP16_LAYERS", "4"))
DEFAULT_TURBOQUANT_K_BITS: int = 3
DEFAULT_TURBOQUANT_V_BITS: int = 4
OPTIQ_BITS: int = int(os.environ.get("SKULK_OPTIQ_BITS", "4"))
OPTIQ_FP16_LAYERS: int = int(os.environ.get("SKULK_OPTIQ_FP16_LAYERS", "4"))

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
