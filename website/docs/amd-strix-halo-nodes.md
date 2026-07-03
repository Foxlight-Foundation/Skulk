# AMD Strix Halo nodes (Linux / Vulkan)

Skulk clusters are not Mac-only. An AMD Ryzen AI Max (Strix Halo, `gfx1151`) box
running Linux can join a cluster as a worker that serves **GGUF models on its
integrated Radeon GPU**, alongside Apple Silicon nodes serving MLX models. The
cluster is heterogeneous: each model is placed on the nodes that can actually run
it.

An AMD node can run GGUF models through **two engines**:

- **`llama_cpp`** (in-process): the default GGUF path. Skulk loads the model with
  `llama-cpp-python` and decodes on the Radeon GPU via Vulkan. Single-node.
- **`llama_server`** (served): Skulk launches an external `llama-server` process
  and proxies its OpenAI API. This is the only path to llama.cpp's **native
  multi-token prediction** (`--spec-type draft-mtp`), so it is how you get
  speculative-decoding speedups on an AMD node. Single-node; enabled per node by
  pointing `SKULK_LLAMA_SERVER_BIN` at a `llama-server` binary.

This page covers what such a node needs, how to bring one up, both engines, and
how the cluster decides what to run where.

## What runs where

A Skulk node advertises the compute **backends** it can serve, written as
`<engine>-<compute>` tags:

- A macOS node advertises `mlx` and `mlx-metal`.
- A Linux node with llama.cpp built for the Radeon GPU advertises `llama_cpp`
  and `llama_cpp-vulkan`.
- If that node also has a `llama-server` binary (`SKULK_LLAMA_SERVER_BIN`), it
  additionally advertises the served backend for its GPU, e.g.
  `llama_server-vulkan` (and `llama_server-cpu` as a floor).

Each model card declares which backends can run it. A GGUF model card lists
llama.cpp backends as compatible; an MLX/safetensors model lists MLX. When you
launch a model, the master places it only on nodes whose advertised backends
include one the card accepts, and prefers the card's higher-ranked backend when
several nodes qualify. So a GGUF model lands on the AMD node and an MLX model
lands on the Macs, automatically.

## The GPU path: Vulkan, not HIP

On `gfx1151` the reliable, well-supported way to run llama.cpp on the GPU today
is the **Vulkan backend** (Mesa's RADV driver), not the ROCm/HIP backend. ROCm is
not required for inference. It is optional and used only for the `rocminfo`
diagnostic. Skulk's llama.cpp runner offloads through Vulkan (Mesa RADV). On a
Ryzen AI Max+ 395 (Radeon 8060S) this fully
offloads a 7B Q4_K_M model to the iGPU and decodes at interactive speed, which is
what makes the box useful as a cluster node rather than a CPU-only fallback.

## Validated configuration

| Component        | Version                                          |
| ---------------- | ------------------------------------------------ |
| Hardware         | AMD Ryzen AI Max+ 395 w/ Radeon 8060S (gfx1151)  |
| OS               | Ubuntu 26.04 LTS (kernel 7.x)                     |
| GPU compute      | Vulkan via Mesa RADV 26.x (Vulkan 1.4, `STRIX_HALO`) |
| Build toolchain  | cmake 4.x, gcc/g++ 15.x, glslc (shaderc)         |
| Python / uv      | uv-managed Python, uv 0.11+                       |
| llama-cpp-python | Built from source with the Vulkan backend        |
| llama.cpp        | Vulkan build (b9820+), for native MTP (`llama-server`) |

On Ubuntu 26.04 the whole Vulkan path is distro-native (Mesa RADV and
`vulkan-tools` from `main`, `glslc` and the optional `rocminfo` diagnostic from
`universe`, GPU firmware in `linux-firmware`), so a Skulk node needs **no
third-party AMD/ROCm apt repository**. Inference runs on Vulkan, not HIP: the
`llama-server` binary links `libvulkan`, not `libamdhip`, so ROCm is optional and
used here only for `rocminfo`.

## Unified memory (kernel parameters)

The Ryzen AI Max has a single pool of unified LPDDR5X. The BIOS carves part of it
out as dedicated GPU VRAM (for example 64 GiB on a 128 GiB box) and leaves the
rest as system RAM. Two kernel parameters let the GPU address the whole pool, so
a model larger than the VRAM carve-out still runs on the GPU (its weights map
through the GTT aperture into system RAM):

- `amdgpu.gttsize=126976` caps the GPU's unified (GTT) aperture at 124 GiB
  (126976 MiB / 1024).
- `ttm.pages_limit=32505856` raises the pinned-memory limit to the same 124 GiB
  (32505856 x 4 KiB pages = 126976 MiB).

Set both to fit your box (scale down proportionally on a smaller machine, for
example a 64 GiB box). Without them the GPU is capped near the VRAM carve-out and
large models fail to load. Skulk already understands this: it shows the machine's
full unified memory in the dashboard (VRAM carve-out plus system RAM) and its
planner admits models against the whole pool, so a 128 GiB box serves ~128 GiB of
model rather than only the non-carve-out slice.

A third parameter is optional and trades IOMMU isolation for throughput and
stability on a trusted node:

- `amd_iommu=off` disables the AMD IOMMU entirely, which removes translation
  overhead and avoids IOMMU edge cases. Prefer it over `iommu=pt` **only where
  the node is trusted** (the cluster-fabric assumption), since it removes device
  isolation.

Apply through GRUB and reboot (the parameters live in `GRUB_CMDLINE_LINUX`):

```bash
# swap iommu=pt for amd_iommu=off, keeping the memory caps:
sudo sed -i 's/iommu=pt/amd_iommu=off/' /etc/default/grub
# the line should now read, e.g.:
#   GRUB_CMDLINE_LINUX="amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856"
sudo update-grub && sudo reboot
# after reboot, confirm:
cat /proc/cmdline
```

## Bring-up

The full operational steps, a one-shot dependency installer, and a launcher
template live in
[`deployment/rocm/`](https://github.com/Foxlight-Foundation/Skulk/tree/main/deployment/rocm)
in the repo.

### Install the dependencies (one script)

On a fresh Ubuntu box, `deployment/rocm/install-deps.sh` installs everything the
node needs below the Skulk repo (the Vulkan GPU stack, the build toolchain, `uv`,
and GPU device-group membership) and, with flags, builds the MTP `llama-server`
binary and the Skulk `uv` env in the same pass. It is idempotent:

```bash
git clone https://github.com/Foxlight-Foundation/Skulk.git && cd Skulk

# System deps only:
deployment/rocm/install-deps.sh
# Everything, including the MTP llama-server and the Vulkan llama-cpp-python:
deployment/rocm/install-deps.sh --with-llama-server --with-skulk-env
# Verify a box without installing:
deployment/rocm/install-deps.sh --check
```

If it adds you to the `render` / `video` GPU groups for the first time, log out
and back in before starting Skulk. The manual steps below are what the script
automates.

### Manual steps

1. **Install the GPU stack**: a working Vulkan driver (Mesa RADV); ROCm is
   optional (only `rocminfo`). Confirm with `vulkaninfo | grep deviceName`
   (expect the Radeon device via RADV) and, if installed, `rocminfo | grep gfx`
   (expect `gfx1151`).
2. **Build the Skulk environment**: `git clone` the repo and run `uv sync`. The
   Rust networking bindings compile here. No MLX is needed on a non-Mac node.
3. **Build llama-cpp-python with Vulkan**: the default install is a CPU wheel,
   so reinstall it built from source with the Vulkan backend. `--no-binary`
   forces the source build, otherwise uv installs the prebuilt CPU wheel and
   `CMAKE_ARGS` is ignored:
   ```bash
   CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --force-reinstall \
     --no-cache-dir --no-binary llama-cpp-python \
     --python .venv/bin/python llama-cpp-python
   ```
   You build this once. The service entrypoint runs `uv sync --inexact` on a node
   that declares a GPU backend, so a routine sync no longer prunes this
   source-built wheel; rebuild it by hand only when bumping the llama.cpp version.
   As a safety net, if the wheel is ever replaced by a CPU-only one (no GPU
   offload compiled in), the node detects that and advertises `llama_cpp-cpu`
   instead of its GPU backend, so GPU work is never routed to a degraded build.
4. **Build `llama-server` for native MTP (optional)**: speculative decoding on
   the AMD node runs through an external `llama-server` process (the in-process
   binding does not expose native MTP). Build it once with Vulkan and point the
   node at it (see the MTP section below). `install-deps.sh --with-llama-server`
   does this for you.
5. **Launch the node**: declare its backend and point it at the rest of the
   cluster, using the `launch-skulk.sh.example` template (sets
   `SKULK_LLAMA_CPP_BACKENDS=vulkan`). Nodes on the same LAN segment find each
   other automatically (mDNS); if this node is on a different segment, set
   `SKULK_BOOTSTRAP_PEERS` to dial the existing nodes. On Linux there is no
   launchd, so start it detached so it survives an SSH disconnect:
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

## What the AMD node can serve

A GGUF model on the AMD node uses the same OpenAI-compatible endpoints as any
other Skulk model, with the same streaming behavior. It matches the MLX nodes on
the generation capabilities llama.cpp supports:

- **Text generation**, streamed token by token.
- **Vision / image input** for a vision GGUF (with an `mmproj` projector): send
  images as OpenAI `image_url` content and the model describes or reasons over
  them. The projector loads through llama.cpp's multimodal chat handler.
- **Logprobs** (opt-in): per-token logprobs and ranked alternatives (`logprobs`
  / `top_logprobs`). This needs the model loaded so it retains per-token logits,
  which pre-allocates a large buffer (context length x vocab), so it is off by
  default. Enable it per node with `SKULK_LLAMA_CPP_LOGITS_ALL=1`; doing so
  further caps the served context to a bounded window
  (`SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX`, default 8192) so the buffer stays small.
  With it off, a logprobs request returns a clear error rather than silently
  omitting them. Either way the served context window is sized to the memory the
  cluster admitted for the instance, never the model's full trained context (a
  128k-context model loaded at full context would otherwise allocate a KV cache
  far larger than placement reserved and could exhaust the node on load).
- **Tool calling**: a request's `tools` are passed to the model. Whether the
  model returns a *structured* tool call or describes the call in prose depends
  on the model and its embedded chat template, which Skulk uses as-is; either way
  the request completes through the normal streaming path.

**Vision / multimodal GGUF** models are served. A vision GGUF that ships a
separate `mmproj` projector (LLaVA / Qwen-VL style) runs on an AMD node: the
projector downloads alongside the weights and the runner loads it through
llama.cpp's multimodal chat handler, so image chat requests (OpenAI `image_url`
content) work. A repo is recognized as a vision model from its `config.json`
vision section, or, when it ships none, from the presence of the `mmproj` file.

## Native MTP on the AMD node (served engine)

Speculative decoding (multi-token prediction) **does** run on an AMD node,
through the served `llama_server` engine. llama.cpp's native MTP lives in the
`llama-server` application, not in the in-process `llama-cpp-python` binding, so
Skulk reaches it by launching `llama-server` with `--spec-type draft-mtp` and
proxying its OpenAI API. A model whose card declares `served_spec_type =
draft_mtp` and lists `llama_server-*` backends is placed on a node that has a
`llama-server` binary and served with MTP on.

There are two MTP shapes, both served this way:

- **Baked-in heads**: a GGUF that ships MTP prediction tensors (for example the
  Qwen3.5 / Qwen3.6 MTP GGUFs). No separate draft model is needed.
- **Draft-model MTP**: a base GGUF plus a small separate draft GGUF passed as
  `--model-draft` (for example Gemma 4 31B with a published draft). The card
  co-fetches both through the store.

To enable it on a node, build a recent `llama-server` with the Vulkan backend and
point `SKULK_LLAMA_SERVER_BIN` at it before launching Skulk. Native MTP
(`--spec-type draft-mtp`) landed in llama.cpp build b9196, so use a newer tag:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j"$(nproc)"
# then in ~/.skulk/skulk.env (or launch-skulk.sh):
#   SKULK_LLAMA_SERVER_BIN=$HOME/llama.cpp/build/bin/llama-server
```

`deployment/rocm/install-deps.sh --with-llama-server` runs exactly this and prints
the path to set. On a Ryzen AI Max the
Vulkan `llama-server` serves an MTP-capable GGUF on the Radeon GPU and generates
with speculation active, streaming reasoning and tool calls through the same
`/v1` endpoints as any other model. Acceptance is per-model: a pairing that does
not pay is turned off in that model's card, the same discipline MLX MTP uses. See
[Speculative Decoding](speculative-decoding.md) for how MTP works and what
speedups to expect.

## What is not on the AMD path today

- **MTP across the in-process `llama_cpp` engine.** The in-process binding does
  not expose native MTP; use the served `llama_server` engine (above) for MTP.
  A GGUF served through `llama_cpp` runs plain autoregressive decoding.
- **Sharding one model across an AMD node and a Mac** (the two engines do not
  share a runtime), and **multi-node GGUF inference** across several llama.cpp
  nodes (tracked separately). A single AMD node serves each GGUF model on its own
  GPU.

The interconnect doctrine still applies: the cluster fabric is trusted, so put
untrusted segments behind your own network controls.
