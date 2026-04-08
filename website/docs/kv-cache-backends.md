---
id: kv-cache-backends
title: KV Cache Backends
sidebar_position: 4
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk includes several opt-in KV cache backends for MLX text generation. These backends are intended for long-context and memory-pressure experiments, while preserving existing behavior unless explicitly enabled.

## Current Status

- `default`: existing behavior — no cache quantization
- `mlx_quantized`: MLX LM built-in `QuantizedKVCache`
- `turboquant`: correctness-first TurboQuant-inspired KV cache for standard `KVCache` layers
- `turboquant_adaptive`: keeps outer KV layers in FP16 and applies TurboQuant to middle KV layers
- `optiq`: rotation-based KV cache via [mlx-optiq](https://mlx-optiq.pages.dev/) — uses randomized orthogonal rotations with Lloyd-Max quantization and rotated-space attention. Falls back to default for GQA models.
- `rotorquant`: **[NEW]** pure-MLX port of IsoQuant 3-bit from [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) and [johndpope/llama-cpp-turboquant](https://github.com/johndpope/llama-cpp-turboquant). Block-diagonal quaternion rotations + deferred prefill. GQA-native.
- `rotorquant_adaptive`: **[NEW]** as above with FP16 protection on the first/last N attention layers. Recommended starting point.

If `SKULK_KV_CACHE_BACKEND` is unset, or is set to `default`, Skulk behaves as before.

## Recommended Settings

### RotorQuant Adaptive (recommended starting point)

```bash
SKULK_KV_CACHE_BACKEND=rotorquant_adaptive \
SKULK_ROTORQUANT_FP16_LAYERS=4 \
uv run skulk
```

The rotorquant backend implements IsoQuant 3-bit compression with two key properties:

- **Block-diagonal quaternion rotations**: each 4D group of a head dimension is rotated by a fixed unit quaternion, costing `O(d)` per token instead of the `O(d log d)` randomized Hadamard used by the legacy turboquant backend or the `O(d²)` random orthogonal matrix used by mlx-optiq. The quaternion table and 3-bit Lloyd-Max centroids are vendored verbatim from the llama.cpp fork so the math agrees with the upstream C reference.
- **Deferred prefill**: K and V are kept in fp16 throughout prompt processing and quantized once on the first decode token. This eliminates the compounding centroid-roundtrip error that quantizing-on-insert introduces during prefill, and matches the published 5.3× prefill / PPL 6.91 numbers from the upstream llama.cpp fork. The deferred-prefill flush is the unique contribution of this MLX port — the upstream fork only ships it on CUDA.

GQA models work natively (no fallback) because compression is per-(kv_head, token) and Q heads fan out at SDPA. The head dimension must be a multiple of 128 (the IsoQuant block size); standard Llama and Qwen heads at 64 / 128 / 256 are all multiples and work directly.

### mlx-optiq (best quality)

```bash
SKULK_KV_CACHE_BACKEND=optiq \
SKULK_OPTIQ_BITS=4 \
SKULK_OPTIQ_FP16_LAYERS=4 \
uv run skulk
```

The optiq backend uses mlx-optiq's rotation-based vector quantization, which eliminates per-key rotation overhead at inference time via rotated-space attention. It keeps the first and last N KV layers in FP16 for adaptive quality.

### TurboQuant Adaptive (proven stable)

```bash
SKULK_KV_CACHE_BACKEND=turboquant_adaptive \
SKULK_TQ_K_BITS=3 \
SKULK_TQ_V_BITS=4 \
SKULK_TQ_FP16_LAYERS=4 \
uv run skulk
```

This mode keeps the first and last 4 KV layers in normal FP16-style cache and applies TurboQuant only to the middle KV layers. Proven stable across most models.

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
| `SKULK_ROTORQUANT_FP16_LAYERS` | `rotorquant_adaptive` | `4` | Edge layers kept in FP16 |
| `SKULK_ROTORQUANT_DEFER_PREFILL` | `rotorquant`, `rotorquant_adaptive` | `1` | Set to `0` to disable deferred prefill (debugging only) |

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

| Backend | Memory | Quality | Speed | Notes |
|---------|--------|---------|-------|-------|
| `default` | Highest | Baseline | Fastest | No quantization |
| `rotorquant_adaptive` | Low | Best quantized | Near-baseline | IsoQuant 3-bit + deferred prefill, GQA-native |
| `rotorquant` | Lowest | Good | Near-baseline | All layers IsoQuant 3-bit + deferred prefill |
| `optiq` | Low | Good | Near-baseline | Rotation-based, no GQA support |
| `turboquant_adaptive` | Low | Good | Moderate | Proven stable, Hadamard-based |
| `turboquant` | Low | Variable | Moderate | All layers, Hadamard-based |
| `mlx_quantized` | Low | Good | Moderate | MLX built-in quantization |

## Supported Cache Layouts

All quantized backends (rotorquant, optiq, turboquant, mlx_quantized) compress only standard `KVCache` entries and preserve these cache types unchanged:

- `ArraysCache`
- `RotatingKVCache`

Mixed cache layouts are supported:

- `KVCache` + `ArraysCache`
- `KVCache` + `RotatingKVCache`
- `KVCache` + `ArraysCache` + `RotatingKVCache`

## Current Limitations

- All quantized KV cache backends force sequential generation (no batch/history mode)
- The optiq backend requires `mlx-optiq` to be installed (`pip install mlx-optiq`)
- The optiq backend's `patch_attention()` monkey-patches MLX's SDPA — avoid switching between optiq and other backends within the same process lifetime without a restart

## About mlx-optiq

The `optiq` backend is powered by [mlx-optiq](https://mlx-optiq.pages.dev/), which provides:

- **Rotation-based vector quantization**: Random orthogonal rotations + Lloyd-Max centroids
- **Rotated-space attention**: Eliminates per-key rotation overhead (O(d²) fixed cost vs O(seq_len × d²))
- **Superior long-context quality**: Claims 100% needle retrieval at 4-bit vs 73% FP16

mlx-optiq also provides mixed-precision weight quantization (per-layer sensitivity analysis via KL divergence), which Skulk plans to integrate as a model store feature in a future release.
