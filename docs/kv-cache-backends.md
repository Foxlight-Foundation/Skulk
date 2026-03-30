<!-- Copyright 2025 Foxlight Foundation -->

# KV Cache Backends

Skulk includes several opt-in KV cache backends for MLX text generation. These backends are intended for long-context and memory-pressure experiments, while preserving existing behavior unless explicitly enabled.

## Current Status

- `default`: existing behavior
- `mlx_quantized`: MLX LM built-in `QuantizedKVCache`
- `turboquant`: correctness-first TurboQuant-inspired KV cache for standard `KVCache` layers
- `turboquant_adaptive`: keeps outer KV layers in FP16 and applies TurboQuant to middle KV layers

If `EXO_KV_CACHE_BACKEND` is unset, or is set to `default`, Skulk behaves as before.

## Recommended Setting

The current best experimental setting during local testing has been:

```bash
EXO_KV_CACHE_BACKEND=turboquant_adaptive \
EXO_TQ_K_BITS=3 \
EXO_TQ_V_BITS=4 \
EXO_TQ_FP16_LAYERS=4 \
uv run exo
```

This mode keeps the first and last 4 KV layers in normal FP16-style cache and applies TurboQuant only to the middle KV layers.

## Available Environment Variables

- `EXO_KV_CACHE_BACKEND`
  - `default`
  - `mlx_quantized`
  - `turboquant`
  - `turboquant_adaptive`
- `EXO_KV_CACHE_BITS`
  - Required when `EXO_KV_CACHE_BACKEND=mlx_quantized`
- `EXO_TQ_K_BITS`
  - Optional for `turboquant` and `turboquant_adaptive`
  - Defaults to `3`
- `EXO_TQ_V_BITS`
  - Optional for `turboquant` and `turboquant_adaptive`
  - Defaults to `4`
- `EXO_TQ_FP16_LAYERS`
  - Used by `turboquant_adaptive`
  - Defaults to `4`

## Invocation Examples

Default behavior:

```bash
EXO_KV_CACHE_BACKEND=default uv run exo
```

MLX quantized KV cache:

```bash
EXO_KV_CACHE_BACKEND=mlx_quantized EXO_KV_CACHE_BITS=4 uv run exo
```

TurboQuant adaptive:

```bash
EXO_KV_CACHE_BACKEND=turboquant_adaptive EXO_TQ_K_BITS=3 EXO_TQ_V_BITS=4 EXO_TQ_FP16_LAYERS=4 uv run exo
```

Full TurboQuant:

```bash
EXO_KV_CACHE_BACKEND=turboquant EXO_TQ_K_BITS=3 EXO_TQ_V_BITS=4 uv run exo
```

## Practical Expectations

These backends should currently be thought of as memory-oriented, not throughput-oriented.

- `default`
  - Best baseline for quality
  - Highest KV memory use
- `mlx_quantized`
  - Lower KV memory than `default`
  - Easier comparison point against built-in MLX quantization
- `turboquant_adaptive`
  - Lower KV memory with safer quality than full TurboQuant
  - Best current experimental candidate
- `turboquant`
  - Most aggressive compression path currently available
  - Higher quality risk than adaptive mode

In the current implementation, these backends may be slower than baseline decode in exchange for reduced KV memory use and better long-context headroom.

## Supported Cache Layouts

The current TurboQuant implementation compresses only standard `KVCache` entries and preserves these cache types unchanged:

- `ArraysCache`
- `RotatingKVCache`

Mixed cache layouts such as these are supported:

- `KVCache` + `ArraysCache`
- `KVCache` + `RotatingKVCache`
- `KVCache` + `ArraysCache` + `RotatingKVCache`

If a model uses other cache types, Skulk will fail with an explicit error showing the discovered cache types.

## Current Limitations

These backends are still experimental.

- Quantized KV cache backends do not yet support batch/history mode
- When `mlx_quantized`, `turboquant`, or `turboquant_adaptive` is enabled, Skulk currently forces sequential generation as a temporary safety fallback
- The current TurboQuant path is correctness-first and does not yet include:
  - QJL residual correction
  - bit-packed storage
  - fused kernels
  - optimized incremental dequant paths
  - full paper-faithful TurboQuant estimator behavior

The forced sequential fallback should be removed once quantized KV cache backends support batch/history semantics properly.

## Manual Testing Guidance

The most useful current comparisons are:

1. `default`
2. `mlx_quantized`
3. `turboquant_adaptive`

Recommended prompt sequence:

1. Short prompt sanity check
2. Multi-turn factual recall
3. Long-context retrieval
4. Long-context summarization

Things to compare:

- completion success vs failure
- obvious quality degradation
- follow-up turn correctness
- long-context recall
- subjective latency
- memory headroom and stability

## Roadmap

The next major steps are:

1. add observability and backend-specific runtime logging
2. build a repeatable evaluation harness
3. add TurboQuantProd-style QJL residual correction so inner product estimates are unbiased
4. replace the current Gaussian codebooks with precomputed Beta codebooks
5. retune adaptive defaults across more models after the estimator path is improved
6. add real batch/history support so the forced sequential fallback can be removed
7. implement bit-packing and better decode efficiency
8. evaluate whether deeper kernel-level optimization, including fused kernels, is justified
