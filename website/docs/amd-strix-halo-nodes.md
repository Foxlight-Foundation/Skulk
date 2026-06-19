# AMD Strix Halo nodes (Linux / ROCm)

Skulk clusters are not Mac-only. An AMD Ryzen AI Max (Strix Halo, `gfx1151`) box
running Linux can join a cluster as a worker that serves **GGUF models through
llama.cpp on its integrated Radeon GPU**, alongside Apple Silicon nodes serving
MLX models. The cluster is heterogeneous: each model is placed on the nodes that
can actually run it.

This page covers what such a node needs, how to bring one up, and how the
cluster decides what to run where.

## What runs where

A Skulk node advertises the compute **backends** it can serve, written as
`<engine>-<compute>` tags:

- A macOS node advertises `mlx` and `mlx-metal`.
- A Linux node with llama.cpp built for the Radeon GPU advertises `llama_cpp`
  and `llama_cpp-vulkan`.

Each model card declares which backends can run it. A GGUF model card lists
llama.cpp backends as compatible; an MLX/safetensors model lists MLX. When you
launch a model, the master places it only on nodes whose advertised backends
include one the card accepts, and prefers the card's higher-ranked backend when
several nodes qualify. So a GGUF model lands on the AMD node and an MLX model
lands on the Macs, automatically.

## The GPU path: Vulkan, not HIP

On `gfx1151` the reliable, well-supported way to run llama.cpp on the GPU today
is the **Vulkan backend** (Mesa's RADV driver), not the ROCm/HIP backend. ROCm
is still installed for its runtime and driver stack, but Skulk's llama.cpp runner
offloads through Vulkan. On a Ryzen AI Max+ 395 (Radeon 8060S) this fully
offloads a 7B Q4_K_M model to the iGPU and decodes at interactive speed, which is
what makes the box useful as a cluster node rather than a CPU-only fallback.

## Validated configuration

| Component        | Version                                          |
| ---------------- | ------------------------------------------------ |
| Hardware         | AMD Ryzen AI Max+ 395 w/ Radeon 8060S (gfx1151)  |
| OS               | Ubuntu 26.04 LTS (kernel 7.0)                    |
| ROCm             | 6.4                                              |
| GPU compute      | Vulkan via Mesa RADV (`STRIX_HALO`)             |
| Python / uv      | 3.13 / 0.11                                       |
| llama-cpp-python | 0.3.30, built with the Vulkan backend            |

## Bring-up

The full operational steps and a launcher template live in
[`deployment/rocm/`](https://github.com/Foxlight-Foundation/Skulk/tree/main/deployment/rocm)
in the repo. In outline:

1. **Install the GPU stack**: ROCm plus a working Vulkan driver (RADV). Confirm
   with `rocminfo | grep gfx` (expect `gfx1151`) and `vulkaninfo | grep deviceName`
   (expect the Radeon device via RADV).
2. **Build the Skulk environment**: `git clone` the repo and run `uv sync`. The
   Rust networking bindings compile here. No MLX is needed on a non-Mac node.
3. **Build llama-cpp-python with Vulkan**: the default install is a CPU wheel,
   so reinstall it with the Vulkan backend:
   ```bash
   CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --force-reinstall \
     --no-cache-dir --python .venv/bin/python llama-cpp-python
   ```
   Re-run this after any `uv sync`, which restores the CPU wheel.
4. **Launch the node**: point it at the rest of the cluster and declare its
   backend, using the `launch-skulk.sh.example` template (sets
   `SKULK_LLAMA_CPP_BACKENDS=vulkan`). On Linux there is no launchd, so start it
   detached so it survives an SSH disconnect:
   ```bash
   setsid bash -c 'exec ~/launch-skulk.sh > ~/skulk.log 2>&1' </dev/null >/dev/null 2>&1 &
   ```

A headless node needs no dashboard build: the API serves without the UI, and you
reach the dashboard from any node that has it.

## Serving a model on the AMD node

Launch a GGUF model (its card lists llama.cpp backends as compatible) and the
master places it on the AMD node. From any node's API:

```bash
curl -s -X POST http://<any-node>:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"<org>/<gguf-repo>","instance_meta":"MlxRing","min_nodes":1}'
```

Skulk downloads only the preferred quantization (not every quant in a multi-quant
repo), loads it through llama.cpp on the Radeon GPU, and serves it through the
same OpenAI-compatible endpoints as any other model.

## Scope

A single AMD node serves GGUF models on its own GPU. Sharding one model across an
AMD node and a Mac is not supported (the two engines do not share a runtime);
multi-node GGUF inference across several llama.cpp nodes is tracked separately.
The interconnect doctrine still applies: the cluster fabric is trusted, so put
untrusted segments behind your own network controls.
