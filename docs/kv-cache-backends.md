<!-- Copyright 2025 Foxlight Foundation -->

# KV Cache Backends

Skulk includes several opt-in KV cache backends for MLX text generation. These backends are intended for long-context and memory-pressure experiments, while preserving existing behavior unless explicitly enabled.

## Current Status

- `default`: existing behavior — no cache quantization
- `mlx_quantized`: MLX LM built-in `QuantizedKVCache`
- `turboquant`: correctness-first TurboQuant-inspired KV cache for standard `KVCache` layers
- `turboquant_adaptive`: keeps outer KV layers in FP16 and applies TurboQuant to middle KV layers
- `optiq`: **[NEW]** rotation-based KV cache via [mlx-optiq](https://mlx-optiq.pages.dev/) — uses randomized orthogonal rotations with Lloyd-Max quantization and rotated-space attention for superior long-context quality
- `rotorquant`: **experimental, gated** pure-MLX IsoQuant-style storage/dequant cache. This is not the fused RotorQuant+QJL implementation from the RotorQuant paper.
- `rotorquant_adaptive`: **experimental, gated** as above with FP16 protection on the first/last N attention layers.

If `SKULK_KV_CACHE_BACKEND` is unset, or is set to `default`, Skulk behaves as before.

## Recommended Settings

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

### RotorQuant / IsoQuant (experimental only)

The `rotorquant` names are intentionally gated behind an explicit opt-in:

```bash
SKULK_ENABLE_EXPERIMENTAL_ROTORQUANT=1 \
SKULK_KV_CACHE_BACKEND=rotorquant_adaptive \
SKULK_ROTORQUANT_FP16_LAYERS=4 \
uv run skulk
```

This backend is a pure-MLX IsoQuant-style cache that compresses storage and
then returns fully dequantized fp16 K/V to normal MLX attention. It does not
implement RotorQuant's fused Metal/CUDA attention path or QJL residual
correction, so it should not be used as the default distributed inference
baseline.

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
| `SKULK_ENABLE_EXPERIMENTAL_ROTORQUANT` | `rotorquant`, `rotorquant_adaptive` | `0` | Required opt-in for experimental RotorQuant/IsoQuant backends |
| `SKULK_ROTORQUANT_FP16_LAYERS` | `rotorquant_adaptive` | `4` | Edge layers kept in FP16 |
| `SKULK_ROTORQUANT_DEFER_PREFILL` | `rotorquant`, `rotorquant_adaptive` | `1` | Set to `0` to disable deferred prefill while debugging |

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
| `optiq` | Low | Best quantized | Near-baseline | Rotation-based, best long-context |
| `turboquant_adaptive` | Low | Good | Moderate | Proven stable, Hadamard-based |
| `turboquant` | Lowest | Variable | Moderate | Most aggressive compression |
| `mlx_quantized` | Low | Good | Moderate | MLX built-in quantization |
| `rotorquant_adaptive` | Low | Experimental | Experimental | Gated pure-MLX IsoQuant storage/dequant cache |
| `rotorquant` | Lowest | Experimental | Experimental | Gated pure-MLX IsoQuant storage/dequant cache |

## Supported Cache Layouts

All quantized backends (optiq, turboquant, mlx_quantized, and the gated rotorquant variants) compress only standard `KVCache` entries and preserve these cache types unchanged:

- `ArraysCache`
- `RotatingKVCache`

Mixed cache layouts are supported:

- `KVCache` + `ArraysCache`
- `KVCache` + `RotatingKVCache`
- `KVCache` + `ArraysCache` + `RotatingKVCache`

## Current Limitations

- All quantized KV cache backends force sequential generation (no batch/history mode)
- Gemma 4 text generation also forces sequential generation for now because distributed BatchGenerator mode can produce degenerate repetition with its sliding-window cache layout
- The optiq backend requires `mlx-optiq` to be installed (`pip install mlx-optiq`)
- The optiq backend's `patch_attention()` monkey-patches MLX's SDPA — avoid switching between optiq and other backends within the same process lifetime without a restart
- The rotorquant backends are disabled unless `SKULK_ENABLE_EXPERIMENTAL_ROTORQUANT=1` is set. If selected without the gate, Skulk falls back to `default`.

## About mlx-optiq

The `optiq` backend is powered by [mlx-optiq](https://mlx-optiq.pages.dev/), which provides:

- **Rotation-based vector quantization**: Random orthogonal rotations + Lloyd-Max centroids
- **Rotated-space attention**: Eliminates per-key rotation overhead (O(d²) fixed cost vs O(seq_len × d²))
- **Superior long-context quality**: Claims 100% needle retrieval at 4-bit vs 73% FP16

mlx-optiq also provides mixed-precision weight quantization (per-layer sensitivity analysis via KL divergence), which Skulk plans to integrate as a model store feature in a future release.
