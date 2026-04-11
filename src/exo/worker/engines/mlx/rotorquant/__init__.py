"""Experimental RotorQuant-named KV cache backend (IsoQuant variant).

Pure-MLX port of the IsoQuant 3-bit KV cache compression from
johndpope/llama-cpp-turboquant (MIT) and scrya-com/rotorquant (MIT).
This is not the fused RotorQuant+QJL attention implementation described in
the RotorQuant paper; it is a storage/dequant cache used for isolated MLX
experiments.

Key properties vs the older TurboQuant native backend:
- Block-diagonal 4D quaternion rotations instead of randomized Hadamard
- Norm-correction trick for unbiased magnitudes after centroid quantization
- Optional deferred prefill: K/V stay in fp16 during prompt processing,
  flushed to compressed storage on the first decode-shaped call
- GQA-native (compression is per-(kv_head, token), Q heads fan out at SDPA)

The backend stores indices and norms; ``update_and_fetch`` returns fully
dequantized fp16 K/V to standard ``mx.fast.scaled_dot_product_attention``.
The centroid-space attention optimization from OptiQ is intentionally
deferred to a follow-up; v1 prioritizes correctness and the deferred-prefill
quality win over the rotated-space SDPA perf win.
"""

from exo.worker.engines.mlx.rotorquant.cache import (
    RotorQuantKVCache,
    make_rotorquant_adaptive_cache,
    make_rotorquant_cache_from_template,
)

__all__ = [
    "RotorQuantKVCache",
    "make_rotorquant_adaptive_cache",
    "make_rotorquant_cache_from_template",
]
