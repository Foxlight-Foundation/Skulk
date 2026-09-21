---
id: kv-cache-backends
title: KV Cache Backends
sidebar_position: 4
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk includes several opt-in KV cache backends for MLX text generation. These backends are intended for long-context and memory-pressure experiments, while preserving existing behavior unless explicitly enabled.

## Current Status

- `default`: existing behavior (no cache quantization)
- `mlx_quantized`: MLX LM built-in `QuantizedKVCache`
- `turboquant`: correctness-first TurboQuant-inspired KV cache for standard `KVCache` layers
- `turboquant_adaptive`: keeps outer KV layers in FP16 and applies TurboQuant to middle KV layers
- `optiq`: rotation-based KV cache via [mlx-optiq](https://mlx-optiq.pages.dev/); uses randomized orthogonal rotations with Lloyd-Max quantization and rotated-space attention for compatible attention layouts

If `SKULK_KV_CACHE_BACKEND` is unset, or is set to `default`, Skulk behaves as before.

## Configuration examples

### mlx-optiq

```bash
SKULK_KV_CACHE_BACKEND=optiq \
SKULK_OPTIQ_BITS=4 \
SKULK_OPTIQ_FP16_LAYERS=4 \
uv run skulk
```

The optiq backend uses mlx-optiq's rotation-based vector quantization, which eliminates per-key rotation overhead at inference time via rotated-space attention. It keeps the first and last N KV layers in FP16 for adaptive quality.

### TurboQuant Adaptive

```bash
SKULK_KV_CACHE_BACKEND=turboquant_adaptive \
SKULK_TQ_K_BITS=3 \
SKULK_TQ_V_BITS=4 \
SKULK_TQ_FP16_LAYERS=4 \
uv run skulk
```

This mode keeps the first and last 4 KV layers in normal FP16-style cache and applies TurboQuant only to the middle KV layers. Validate output quality and memory use with the exact model and context length you intend to serve.

## Available Environment Variables

| Variable | Backends | Default | Description |
|----------|----------|---------|-------------|
| `SKULK_KV_CACHE_BACKEND` | all | `default` | Backend selection |
| `SKULK_KV_CACHE_BITS` | `mlx_quantized` | *(required)* | Bit width for MLX quantized cache |
| `SKULK_OPTIQ_BITS` | `optiq` | `4` | Bit width for mlx-optiq cache |
| `SKULK_OPTIQ_FP16_LAYERS` | `optiq` | `4` | Edge layers kept in FP16 |
| `SKULK_TQ_K_BITS` | `turboquant`, `turboquant_adaptive` | `3` | Key quantization bits |
| `SKULK_TQ_V_BITS` | `turboquant`, `turboquant_adaptive` | `4` | Value quantization bits |
| `SKULK_TQ_FP16_LAYERS` | `turboquant_adaptive` | `4` | Edge layers kept in FP16 |

## Invocation Examples

Default behavior:

```bash
SKULK_KV_CACHE_BACKEND=default uv run skulk
```

mlx-optiq (rotation-based):

```bash
SKULK_KV_CACHE_BACKEND=optiq SKULK_OPTIQ_BITS=4 SKULK_OPTIQ_FP16_LAYERS=4 uv run skulk
```

MLX quantized KV cache:

```bash
SKULK_KV_CACHE_BACKEND=mlx_quantized SKULK_KV_CACHE_BITS=4 uv run skulk
```

TurboQuant adaptive:

```bash
SKULK_KV_CACHE_BACKEND=turboquant_adaptive SKULK_TQ_K_BITS=3 SKULK_TQ_V_BITS=4 SKULK_TQ_FP16_LAYERS=4 uv run skulk
```

## Practical Expectations

Quantization can reduce the memory used by standard KV layers, with additional
compute and possible output-quality changes. Bit width, retained edge layers,
attention layout and context length determine the trade-off; there is no universal
quality or speed ranking. Compare a representative workload against `default`
before choosing a setting. Model weights and unchanged recurrent or rotating
caches are not compressed by these switches.

## Supported Cache Layouts

All quantized backends (optiq, turboquant, mlx_quantized) compress only standard `KVCache` entries and preserve these cache types unchanged:

- `ArraysCache`
- `RotatingKVCache`

Mixed cache layouts are supported:

- `KVCache` + `ArraysCache`
- `KVCache` + `RotatingKVCache`
- `KVCache` + `ArraysCache` + `RotatingKVCache`

## Current Limitations

- All quantized KV cache backends force sequential generation (no batch/history mode)
- Optiq checks the observed attention geometry before patching: non-power-of-two
  head dimensions or detected grouped-query attention (different query/KV head
  counts) log a warning and fall back to the default cache. Check logs to confirm
  whether quantization actually engaged.
- The optiq backend requires `mlx-optiq` to be installed (`pip install mlx-optiq`)
- The optiq backend's `patch_attention()` monkey-patches MLX's SDPA, so avoid switching between optiq and other backends within the same process lifetime without a restart

## Implementation reference

The accepted backend names and defaults live in
`src/skulk/worker/engines/mlx/constants.py`. Cache conversion, compatibility
checks and Optiq attention patching live in
`src/skulk/worker/engines/mlx/cache.py`. These settings affect MLX runners;
they do not configure llama.cpp, vLLM, speech or video engines.
