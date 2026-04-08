import os
from typing import Literal, cast

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
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
    "rotorquant",
    "rotorquant_adaptive",
]
DEFAULT_KV_CACHE_BACKEND: KVCacheBackend = "default"
KV_CACHE_BACKEND: KVCacheBackend = cast(
    KVCacheBackend,
    os.environ.get(
        "SKULK_KV_CACHE_BACKEND",
        os.environ.get("EXO_KV_CACHE_BACKEND", DEFAULT_KV_CACHE_BACKEND),
    ),
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
ROTORQUANT_FP16_LAYERS: int = int(
    os.environ.get("SKULK_ROTORQUANT_FP16_LAYERS", os.environ.get("EXO_ROTORQUANT_FP16_LAYERS", "4"))
)
# Deferred prefill keeps K/V in fp16 during prompt processing and flushes
# the buffer to compressed storage on the first decode token. It is the
# load-bearing accuracy improvement of this backend; only disable for
# debugging.
ROTORQUANT_DEFER_PREFILL: bool = (
    os.environ.get(
        "SKULK_ROTORQUANT_DEFER_PREFILL",
        os.environ.get("EXO_ROTORQUANT_DEFER_PREFILL", "1"),
    )
    not in ("0", "false", "False", "")
)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
