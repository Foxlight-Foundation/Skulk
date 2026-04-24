import os
from typing import Literal, cast

from exo.shared.constants import preferred_env_value

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
# Default generated-token budget when an API client omits max_tokens. This must
# be an output cap, not a proxy for model context length: very large defaults can
# push distributed decode into long-running native MLX/Metal waits before the
# user has asked for that much output.
DEFAULT_MAX_OUTPUT_TOKENS: int = 4096
_max_tokens_value = preferred_env_value(
    "SKULK_MAX_OUTPUT_TOKENS",
    "EXO_MAX_TOKENS",
    str(DEFAULT_MAX_OUTPUT_TOKENS),
)
MAX_TOKENS: int = int(_max_tokens_value) if _max_tokens_value else DEFAULT_MAX_OUTPUT_TOKENS
if MAX_TOKENS < 1:
    raise ValueError("SKULK_MAX_OUTPUT_TOKENS must be greater than 0")
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = (
    int(os.environ["EXO_KV_CACHE_BITS"]) if "EXO_KV_CACHE_BITS" in os.environ else None
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
    "EXO_KV_CACHE_BACKEND",
    DEFAULT_KV_CACHE_BACKEND,
)
KV_CACHE_BACKEND: KVCacheBackend = cast(
    KVCacheBackend,
    _kv_cache_backend_value if _kv_cache_backend_value else DEFAULT_KV_CACHE_BACKEND,
)
TURBOQUANT_K_BITS: int | None = (
    int(os.environ.get("SKULK_TQ_K_BITS", os.environ.get("EXO_TQ_K_BITS", "")))
    if os.environ.get("SKULK_TQ_K_BITS", os.environ.get("EXO_TQ_K_BITS"))
    else None
)
TURBOQUANT_V_BITS: int | None = (
    int(os.environ.get("SKULK_TQ_V_BITS", os.environ.get("EXO_TQ_V_BITS", "")))
    if os.environ.get("SKULK_TQ_V_BITS", os.environ.get("EXO_TQ_V_BITS"))
    else None
)
TURBOQUANT_FP16_LAYERS: int = int(os.environ.get("EXO_TQ_FP16_LAYERS", "4"))
DEFAULT_TURBOQUANT_K_BITS: int = 3
DEFAULT_TURBOQUANT_V_BITS: int = 4
OPTIQ_BITS: int = int(os.environ.get("EXO_OPTIQ_BITS", "4"))
OPTIQ_FP16_LAYERS: int = int(os.environ.get("EXO_OPTIQ_FP16_LAYERS", "4"))

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
