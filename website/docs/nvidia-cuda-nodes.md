# NVIDIA CUDA nodes (Linux)

An NVIDIA GPU box running Linux can join a Skulk cluster as a worker, serving
**GGUF models through a CUDA `llama-server`** and, optionally, high-concurrency
workloads through **vLLM**, alongside Apple Silicon nodes serving MLX models.
As on AMD nodes, the cluster is heterogeneous: each model is placed on the
nodes that can actually run it.

An NVIDIA node can serve through three engines:

- **`llama_server`** (served, CUDA): Skulk launches an external `llama-server`
  process built with CUDA and proxies its OpenAI API. This is the GGUF path and
  the only path to llama.cpp's **native multi-token prediction**
  (`--spec-type draft-mtp`); see
  [Speculative Decoding](speculative-decoding.md). Single-node per model, with
  multi-node GGUF pooling available through the same RPC mechanism the
  [AMD guide](amd-strix-halo-nodes.md) describes.
- **`vllm`** (served): Skulk launches an external `vllm serve` process and
  proxies its OpenAI API. This is the high-concurrency serving path on CUDA;
  see [The vLLM engine](vllm-engine.md).
- **`llama_cpp`** (in-process): possible via a CUDA source build of
  `llama-cpp-python` (the `deployment/cuda/install-deps.sh` script performs
  it), but the served engines are the recommended CUDA paths.

## Prerequisites: the driver, nothing else

The only thing an NVIDIA node needs preinstalled is the **NVIDIA driver**:
any machine where `nvidia-smi` works is ready. In particular:

- **No CUDA toolkit install is required.** The managed CUDA engine wheel does
  not rehost the CUDA runtime; it resolves it from NVIDIA's official PyPI
  wheels (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`), which install as
  ordinary Python dependencies.
- **No NVML setup is required.** GPU detection and VRAM telemetry use
  `nvidia-ml-py`, which is an ordinary Linux dependency of Skulk itself and
  installs automatically with the environment.

## One-command install

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash
```

On a Linux machine where `nvidia-smi` reports a GPU, the installer provisions,
beyond the usual toolchain (uv, Rust, the repo checkout, the Python
environment, and the dashboard built by Skulk's bundled Node.js runtime):

- **The `skulk-llama-server-cuda` engine wheel**, fetched from the Foxlight
  wheel index at `wheels.foxlight.ai` (the CUDA wheel exceeds PyPI's per-file
  size limit; PyPI stays the source for the NVIDIA runtime dependencies). The
  wheel carries Foxlight-built `llama-server` and `ggml-rpc-server` binaries
  compiled from the pinned upstream llama.cpp release, behind a shim that
  wires the CUDA runtime libraries onto the loader path. Skulk's engine
  provisioning discovers the installed wheel and wires it as the node's
  served engine with no configuration.
- **Build provenance** you can verify: the wheels carry sigstore attestations,
  checked with

  ```bash
  gh attestation verify <wheel-file> --owner Foxlight-Foundation
  ```

- **vLLM, if requested** with the `--with-vllm` flag (see below).

The installer finishes by running `skulk doctor --fix`, so any remaining gap is
printed with its consequence and remediation rather than discovered at serving
time.

An operator's own build always wins: point `SKULK_LLAMA_SERVER_BIN` at a custom
`llama-server` and provisioning never runs. An invalid override is reported as
a loud `invalid_engine_binary` conflict rather than silently replaced. Set
`SKULK_NO_ENGINE_AUTOPROVISION=1` to disable engine auto-provisioning entirely.

## What the node advertises

Skulk detects the GPU and derives the node's backends automatically: with the
CUDA engine wheel installed the node advertises the served
`llama_server-cuda` backend, and with `SKULK_VLLM_BIN` set it additionally
advertises `vllm-cuda`. Model cards declare which backends can run them, so
GGUF and vLLM-carded models land on the NVIDIA node while MLX models land on
the Macs, automatically. Run `uv run skulk doctor` to audit exactly what the
node will advertise.

## The CUDA wheel's compute-capability floor

The CUDA engine wheel is compiled for **compute capability 8.0 and newer**
(Ampere onward: SM 80/86/89/90, plus forward-compatible PTX). On an older GPU
(a T4 is 7.5, a V100 is 7.0) Skulk deliberately skips the CUDA wheel, because
its kernels would fail only at model load, and falls back to the **Vulkan**
engine build instead, which drives NVIDIA GPUs through their Vulkan ICD. That
fallback works on **bare metal** with a full driver install.

## Container GPU clouds (rented pods)

Rented-GPU containers (RunPod-style pods) inject a **compute-only driver
stack**: the CUDA driver interface is present but no working Vulkan ICD is, so
the Vulkan fallback is unavailable there. Serving in a container therefore
runs through the CUDA paths: the CUDA engine wheel for GGUF, and **vLLM as the
served path for high-concurrency workloads** (neither needs Vulkan; both need
only the driver the pod image already carries).

**Install on the container disk, not the network volume.** Pod network
volumes (RunPod's `/workspace`) break the Python installer mid-sync with
stale-file-handle errors and make a slow home for the environment. The
installer detects a network-filesystem target and refuses with the fix;
pass `--dir "$HOME/skulk"` (or any container-local path) instead. Network
mounts you know to behave can be forced with
`SKULK_INSTALL_ALLOW_NETWORK_FS=1`.

For repeated rented-GPU sessions there is a **prebaked pod image** on GHCR:

```
ghcr.io/foxlight-foundation/skulk-cuda-pod:latest
```

It bakes in the two slow CUDA compiles (the CUDA `llama-server` +
`ggml-rpc-server` build and the CUDA `llama-cpp-python` wheel, compiled for
SM 80/86/89) plus the toolchain, cutting a session's setup from about an hour
to minutes: inside the pod, `/opt/skulk/pod-bootstrap.sh [git-ref]` clones the
requested Skulk ref, syncs its environment, and swaps in the prebaked wheel.
Per-commit tags are published alongside `:latest`.

## vLLM for concurrent serving

Install it in the same pass as everything else:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash -s -- --with-vllm
```

vLLM lives in its **own virtual environment** (`~/.skulk/vllm-env`) with
Skulk's validated dependency matrix, because Skulk's environment and vLLM
currently require conflicting dependency versions; Skulk drives the `vllm` CLI
as an external served engine through `SKULK_VLLM_BIN`, which the installer
records in `~/.skulk/skulk.env`. The download is several GB. See
[The vLLM engine](vllm-engine.md) for scope, knobs, and when it wins.

One extra requirement applies to models carded with a DFlash speculator
(the Laguna cards): the speculator JIT-compiles its kernels through NVRTC at
engine start, which needs a **CUDA 12.8 or newer toolchain** on the node
(older headers predate the FP8 types it uses). Nodes running a CUDA 12.8+
driver stack normally have this already; container images pinned to an older
CUDA toolkit need the newer `cuda-nvcc`/`cudart-dev` packages installed even
though the GPU driver itself is fine. Models without a DFlash card section
are unaffected.

## Troubleshooting

`uv run skulk doctor` is the audit: it inspects the same facts snapshot Skulk's
capability pipeline uses (visible GPUs, usable engines, declared-versus-observed
configuration, storage headroom) and prints every non-OK verdict with its
consequence and fix; `skulk doctor --fix` applies the safe remediations first.
A misconfigured node also shows up cluster-wide in `/state`'s `nodeHealth` with
capability-conflict codes such as `gpu_serving_disabled` or
`gpu_detection_degraded`. See [Node doctor](node-doctor.md) and the
[Operator Runbook](operations.md) for the full code list.

The interconnect doctrine is the same as everywhere in Skulk: the cluster
fabric is trusted, so put untrusted segments behind your own network controls.
