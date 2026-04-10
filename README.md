# **Skulk**

<!-- Copyright 2025 Foxlight Foundation -->

<div align="center">
  <img src="docs/imgs/skulk-logo.svg" width="200" height="200" alt="Skulk logo">
</div>

Skulk is a fork of EXO for running AI models across one or more machines as a cluster.
It keeps EXO's distributed inference foundation, then extends it with a central model store,
a more modern dashboard, richer API workflows, sophisticated cache quantization, support for more model families such as embeddings and TTS, and cluster-friendly configuration management.

> Skulk is maintained by [Foxlight Foundation](https://github.com/foxlight-foundation) and forked from [exo](https://github.com/exo-explore/exo).

**[Documentation](https://foxlight-foundation.github.io/Skulk/)** · **[API Guide](https://foxlight-foundation.github.io/Skulk/api/)** · **[Architecture](https://foxlight-foundation.github.io/Skulk/architecture/)**

## What Skulk Is Good At

- Run a model on a single machine through the dashboard or API.
- Form a small cluster of Macs and split larger models across them.
- Use a central model store so the cluster downloads once and stages locally.
- Talk to the cluster through OpenAI Chat Completions, OpenAI Responses, Claude Messages, or Ollama-compatible APIs.
- Push KV cache memory down hard with rotation-based 3-bit quantization (RotorQuant, OptiQ, TurboQuant) and pick a backend per workload from the dashboard.
- Use a model-aware reasoning contract that handles toggleable and non-toggleable thinking models without baking assumptions into client code.
- Experiment with advanced placement modes, RDMA, and KV cache backends when you are ready.
- Run non-chat workloads such as embeddings and other specialized model flows.
- Build TTS-oriented and other API-driven workflows on top of the cluster.
- Actually use your cluster for real inference workloads instead of treating it as a demo.

## Everything Different About Skulk

This is the running list of where Skulk diverges from upstream [exo](https://github.com/exo-explore/exo). It is a living section — every meaningful change should land here when it ships, so anyone evaluating Skulk can see the surface area at a glance.

### Inference and KV cache

- **RotorQuant KV cache backend** — pure-MLX port of IsoQuant 3-bit (block-diagonal quaternion rotations + Lloyd-Max centroids) with **deferred prefill on Metal**, a contribution that does not exist in any upstream project (the llama.cpp fork ships it CUDA-only). GQA-native; no fallback for grouped-query models. See [docs/kv-cache-backends.md](docs/kv-cache-backends.md).
- **TurboQuant native and adaptive backends** — randomized Hadamard rotation + Lloyd-Max centroids, with an adaptive variant that keeps edge attention layers in fp16 for accuracy.
- **OptiQ KV cache integration** — wraps `mlx-optiq`'s rotated-space attention path so the rotation cost stays out of the per-token loop on supported (non-GQA) models.
- **OptiQ mixed-precision weight quantization pipeline** — async wrapper around `mlx-optiq`'s sensitivity analysis and KL-divergence per-layer bit allocation, exposed as a model-store optimization job.
- **KV prefix cache with snapshot/restore** — LRU-evicted prompt-prefix cache that snapshots SSM and rotating-window cache states so prefix matches are reusable across conversation turns even for hybrid Mamba/Transformer architectures.
- **Pipeline-parallel prefill for short prompts** — pipelined models now route every prefill through the pipeline path, fixing prior warmup hangs on Gemma-class models.
- **Force-sequential fallback** — quantized backends transparently fall back to a sequential generator when batch/history mode is incompatible with their cache layout.

### Model capability system

- **Two-layer capability model** — declarative `ModelCard` (with optional `reasoning`, `modalities`, `tooling`, and `runtime` sections) plus a normalized `ResolvedCapabilityProfile` derived from the card and conservative family defaults. This is the source of truth for prompt rendering, output parsing, tool-call handling, and the `/v1/models` `resolved_capabilities` field.
- **Phase 2 thinking contract** — `enable_thinking`, `reasoning_effort`, and the dashboard thinking toggle are all driven by `supports_thinking_toggle`, so non-toggleable reasoning models behave correctly without leaking model-specific quirks into client code.
- **Output parser selection** — model cards declare `output_parser` (`generic`, `gemma4`, `gpt_oss`, `deepseek_v32`, etc.), so reasoning markers are normalized into structured `reasoning_content` per family.
- **Model store metadata pipeline** — capability resolution feeds `/v1/models` so dashboards and clients can discover thinking, multimodal, and tool support without hardcoding model lists.

### API surface

- **Claude Messages API** — `/v1/messages` adapter, including streaming, tool use, image inputs, and capability-aware thinking controls.
- **Ollama compatibility** — both `/api/chat` and `/api/generate`, with adapter-side reasoning normalization.
- **OpenAI Responses API** — `/v1/responses` adapter alongside chat completions.
- **Embeddings endpoint** for non-chat workloads.
- **Model store endpoints** — search, add, download, capability resolution, optimization jobs, and registry management, all exposed under stable URLs and documented in the OpenAPI spec.
- **Cluster-wide config endpoints** — `GET`/`POST` config that gossipsubs to every node and writes back to `skulk.yaml`.
- **Tracing, downloads, instance previews, and placement endpoints** — distributed-system observability and pre-launch placement inspection that upstream does not expose.

### Dashboard

- **React dashboard (default)** — replaces upstream's Svelte UI with a typed React + styled-components app that ships with the binary. The legacy Svelte dashboard is kept only as a fallback in the repo.
- **Cluster topology view** with live device icons, GPU stats, network mesh visualization, and connection status banners.
- **Placement preview / placement manager** for inspecting and choosing valid placements before launching.
- **Model store browser** with HuggingFace search, family sidebar, model filters, capability badges, recent models, and per-model launch controls.
- **Reasoning-aware chat UI** that splits inline `<think>` and Gemma `<|channel>` markers into a dedicated thinking panel and merges them with `reasoning_content` deltas from the API.
- **Image attachments and multimodal chat affordances** for vision models.
- **Cluster-wide settings panel** that writes to `skulk.yaml` and syncs across nodes via gossipsub.
- **Light and dark themes** with first-class theme tokens, screenshots in both modes for documentation work.

### Centralized logging and observability

- **Structured JSON stdout** when `logging.enabled` is set, configurable from the dashboard Settings panel and synced cluster-wide.
- **Vector + VictoriaLogs + Grafana stack** — local Vector log shipper on each node, central VictoriaLogs storage, ready-made Grafana dashboards. Stack definition lives in `deployment/logging/`.
- **Distributed tracing** opt-in via `EXO_TRACING_ENABLED`.

### Model store

- **Centralized model store host** — one node downloads, the rest of the cluster stages over the LAN.
- **Persistent registry** with capability resolution and download tracking.
- **Custom model card support** — add your own model with `POST /models/add`.
- **Image and embedding model cards** behind feature flags.
- **Optimization job pipeline** for mlx-optiq mixed-precision weight quantization.

### Cluster operation

- **Cluster-wide settings sync** for KV cache backend, logging, model store host, HF token, and other inference toggles.
- **Bootstrap peer config from `skulk.yaml`, env, or CLI** for fixed-topology clusters.
- **Election (bully algorithm) + master/worker split** for indexing events and broadcasting state.
- **`SKULK_*` environment variables** alongside the legacy `EXO_*` set, so new options can land without colliding with upstream.
- **`skulk.yaml`** as the canonical config file, with `exo.yaml` kept for backwards compatibility.

### Build, type system, and dev workflow

- **Strict basedpyright** type checking — zero-error policy for new code.
- **Ruff** linting and **`nix fmt`** formatting in CI.
- **Nix flake** for reproducible toolchain setup.
- **Docusaurus docs site** with auto-generated OpenAPI per-endpoint pages and TypeDoc HTML reference for the dashboard, both built from source.
- **Pre-commit checklist** documented in `CLAUDE.md` and enforced in CI.

### Hardware and platform

- **Apple Silicon as the primary target**, including RDMA over Thunderbolt 5 on supported hardware and matched macOS versions.
- **Linux supported** (CPU-oriented in this fork; GPU work happens on Apple Silicon).

## Prerequisites

### macOS

- [Xcode](https://developer.apple.com/xcode/)
- [uv](https://github.com/astral-sh/uv)
- [node](https://github.com/nodejs/node)
- [rustup](https://rustup.rs/)
- `macmon` for Apple Silicon monitoring

```bash
brew install uv macmon node
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

### Linux

- [uv](https://github.com/astral-sh/uv)
- Node 18+
- [rustup](https://rustup.rs/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

## Getting Started

If you are brand new to Skulk, follow this order:

1. Install the prerequisites for your platform.
2. Clone the repo.
3. Build the dashboard.
4. Run `uv sync`.
5. Start Skulk with `uv run exo`.
6. Open the dashboard at `http://localhost:52415`.
7. Confirm your node or cluster appears in the topology view.
8. Launch a model from the Model Store view, or place one through the API.
9. Wait until the model is placed and ready.
10. Then chat in the dashboard or send API requests.

Skulk's core runtime flow is:

1. start one or more nodes
2. confirm topology
3. place a model
4. wait for it to become ready
5. then use the dashboard or API

Important behavior:

- The dashboard will not let you chat unless a model is already placed and ready.
- The API behaves the same way in practice. If you send a chat request too early, you will usually get `404 No instance found for model ...`.

## Choose Your Path

- **I want the fastest first success**: follow [Single-Node Quick Start](#single-node-quick-start).
- **I want a multi-node cluster**: follow [Cluster Quick Start](#cluster-quick-start).
- **I want shared storage and fewer duplicate downloads**: read [Model Store](#model-store) after the cluster quick start.
- **I want to integrate with code**: jump to [API Guide](#api-guide) and then [docs/api.md](docs/api.md).

## Platform Support

| Platform | Current state |
|----------|---------------|
| macOS on Apple Silicon | Primary target. Best experience today. |
| Multi-Mac clusters | Supported. Best results on matched macOS versions and fast networking. |
| RDMA over Thunderbolt 5 | Supported on eligible macOS 26.2+ hardware after OS-level setup. |
| Linux | Supported, but currently CPU-oriented in this fork. |

## Core Features

- **Distributed inference**: split work across devices instead of treating each machine as an island.
- **Skulk Dashboard**: React dashboard for topology, model store, chat, settings, and placement workflows.
- **Model Store**: centralize model files on one node and stage them to the rest of the cluster over the LAN.
- **Cluster-wide config sync**: update config from the dashboard and sync it across nodes.
- **Placement previews**: inspect valid placements before launching a model.
- **Thinking-aware chat UI**: chat with compatible models and surface reasoning content.
- **Alternative API compatibility**: OpenAI Chat Completions, OpenAI Responses, Claude Messages, and Ollama.
- **Rotation-based KV cache backends**: RotorQuant (IsoQuant 3-bit + deferred prefill), OptiQ, TurboQuant, and TurboQuant Adaptive — pick per workload from the dashboard.
- **Capability-driven thinking contract**: model cards declare reasoning support; the API and dashboard route accordingly.
- **Experimental inference tuning**: long-context and memory experiments via the KV cache backends above.

## Dashboard

Skulk serves a built-in dashboard at `http://localhost:52415`.
The React dashboard is the default UI. The legacy Svelte dashboard is kept only as a fallback in the repo.
The normal dashboard flow is: confirm topology, launch a model, wait for it to become ready, then open chat.

<p align="center">
  <img src="docs/imgs/dash-1.png" alt="Skulk dashboard showing cluster topology and currently running models" width="80%" />
</p>
<p align="center"><em>Start here: confirm the node or cluster looks healthy in the cluster view.</em></p>

<p align="center">
  <img src="docs/imgs/dash-2.png" alt="Skulk dashboard model store" width="80%" />
</p>
<p align="center"><em>Next: launch or download a model from the Model Store view.</em></p>

<p align="center">
  <img src="docs/imgs/dash-3.png" alt="Skulk dashboard chat view" width="80%" />
</p>
<p align="center"><em>Then: chat once a model is placed and ready.</em></p>

## Single-Node Quick Start

This path is for getting one machine working end-to-end from zero.

### 1. Install Prerequisites

Use the instructions in [Prerequisites](#prerequisites).

### 2. Clone the Repo, Build the Dashboard, and Start Skulk

```bash
git clone https://github.com/foxlight-foundation/Skulk.git
cd Skulk
npm --prefix dashboard-react install
npm --prefix dashboard-react run build
uv sync
uv run exo
```

This starts the dashboard and API at `http://localhost:52415`.

### 3. Open the Dashboard

Go to `http://localhost:52415`.

From there:

1. Confirm your node appears in the topology view.
2. Open the Model Store view.
3. Launch a model.
4. Wait for the model to become ready.
5. Open chat and start using it.

### 4. Launch a Model with the API Instead

If you would rather use the API directly, this is the simplest flow.

1. Preview placements:

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Llama-3.2-1B-Instruct-4bit"
```

2. Quick-launch a placement:

```bash
curl -X POST http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1
  }'
```

3. Send a chat request:

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello from Skulk"}]
  }'
```

If you get `404 No instance found for model ...`, the model has not been placed yet or is not running.

## Cluster Quick Start

Use this path when you want more than one machine in the cluster.

1. Install Skulk on each node.
2. Build the dashboard on each node if you are running from source.
3. Start `uv run exo` on each machine.
4. Open the dashboard on one node and confirm the cluster topology looks correct.
5. Use placement preview or the placement manager to launch a model.
6. Send chat requests through the dashboard or API.

Skulk can discover peers automatically in many local setups. If you want a fixed cluster topology, use `--bootstrap-peers` or the `EXO_BOOTSTRAP_PEERS` environment variable.

Example:

```bash
uv run exo --bootstrap-peers /ip4/192.168.1.20/tcp/5678/p2p/12D3KooW...
```

## Model Store

The model store is one of Skulk's biggest additions over upstream EXO.

Without it, each node may download model data independently.
With it, one node acts as the store host and the rest of the cluster stages from that machine over the LAN.

Use the model store when:

- your models are large
- you have multiple nodes
- you want cleaner offline behavior after the first download
- you want model files to live on a large local or network-attached volume

Recommended path:

1. Start Skulk on all nodes.
2. Open the dashboard on the node that should hold the model store.
3. Go to **Settings**.
4. Toggle **This node is the store host**.
5. Choose the store path.
6. Save.
7. Restart Skulk on all nodes if the UI tells you the change requires restart.

For the full guide, see [docs/model-store.md](docs/model-store.md).

## API Guide

Skulk exposes several API surfaces:

- **OpenAI Chat Completions**: `/v1/chat/completions`
- **OpenAI Responses**: `/v1/responses`
- **Claude Messages**: `/v1/messages`
- **Ollama-compatible endpoints**: `/ollama/api/...`
- **Skulk control endpoints**: placement, model store, config, tracing, downloads, cluster state

The most important API doc lives here:

- [docs/api.md](docs/api.md)

That guide is written to be both newcomer-friendly and integration-friendly. It includes:

- a first-success launch flow
- exact endpoint behavior
- copy-paste examples
- common failure cases
- store and config endpoints

## Common Workflows

### List Known Models

```bash
curl http://localhost:52415/v1/models
```

### List Downloaded Models Only

```bash
curl "http://localhost:52415/v1/models?status=downloaded"
```

### Search Hugging Face

```bash
curl "http://localhost:52415/models/search?query=qwen3&limit=5"
```

### Add a Custom Model Card

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mlx-community/my-custom-model"}'
```

### Use the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Remember: that model must already be placed and running.

## Configuration

Skulk supports both environment variables and `exo.yaml`.

`exo.yaml` is especially useful for:

- `model_store`
- `inference.kv_cache_backend`
- `hf_token`

The dashboard Settings UI can write and sync config for you.

See:

- [exo.yaml.example](exo.yaml.example)
- [docs/model-store.md](docs/model-store.md)
- [docs/kv-cache-backends.md](docs/kv-cache-backends.md)

## Useful CLI Options

Current common options:

- `--no-api`
- `--api-port`
- `--no-worker`
- `--no-downloads`
- `--offline`
- `--no-batch`
- `--bootstrap-peers`
- `--libp2p-port`
- `--fast-synch`
- `--no-fast-synch`

Examples:

```bash
uv run exo --offline
uv run exo --no-worker
uv run exo --api-port 52416
uv run exo --bootstrap-peers /ip4/192.168.1.20/tcp/5678/p2p/12D3KooW...
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXO_MODELS_PATH` | Extra colon-separated search paths for local or shared models | None |
| `EXO_MODELS_DIR` | Primary downloaded-model directory | platform-specific |
| `EXO_OFFLINE` | Use only local or pre-staged models | `false` |
| `EXO_ENABLE_IMAGE_MODELS` | Enable image model cards and image workflows | `false` |
| `EXO_LIBP2P_NAMESPACE` | Custom namespace for cluster isolation | None |
| `EXO_FAST_SYNCH` | Control MLX fast synch behavior | Auto |
| `EXO_TRACING_ENABLED` | Enable distributed tracing | `false` |
| `EXO_KV_CACHE_BACKEND` | KV cache backend selection | `default` |
| `EXO_KV_CACHE_BITS` | Bit width for `mlx_quantized` | None |
| `EXO_TQ_K_BITS` | Key-cache bits for TurboQuant backends | `3` |
| `EXO_TQ_V_BITS` | Value-cache bits for TurboQuant backends | `4` |
| `EXO_TQ_FP16_LAYERS` | Edge FP16 layers for `turboquant_adaptive` | `4` |
| `EXO_NO_BATCH` | Force sequential generation | `false` |
| `EXO_OPTIQ_BITS` | Bit width for `optiq` | `4` |
| `EXO_OPTIQ_FP16_LAYERS` | Edge FP16 layers for `optiq` | `4` |
| `SKULK_ENABLE_EXPERIMENTAL_ROTORQUANT` | Enable experimental pure-MLX RotorQuant/IsoQuant cache backends | `false` |
| `SKULK_ROTORQUANT_FP16_LAYERS` | Edge FP16 layers for `rotorquant_adaptive` | `4` |
| `SKULK_ROTORQUANT_DEFER_PREFILL` | Set to `0` to disable deferred prefill (debugging only) | `1` |
| `EXO_BOOTSTRAP_PEERS` | Comma-separated static peers to dial on startup | None |
| `HF_TOKEN` | Hugging Face token | None |

Examples:

```bash
EXO_OFFLINE=true uv run skulk
EXO_ENABLE_IMAGE_MODELS=true uv run skulk
EXO_KV_CACHE_BACKEND=optiq EXO_OPTIQ_BITS=4 EXO_OPTIQ_FP16_LAYERS=4 uv run skulk
SKULK_KV_CACHE_BACKEND=default SKULK_MLX_HANG_DEBUG=1 uv run skulk -vv
```

The `rotorquant` and `rotorquant_adaptive` cache backends are experimental
pure-MLX IsoQuant storage/dequant backends, not the fused RotorQuant+QJL
implementation from the paper. They fall back to `default` unless
`SKULK_ENABLE_EXPERIMENTAL_ROTORQUANT=1` is also set, and should be used only
for isolated cache experiments.

## RDMA on macOS

RDMA is relevant only if you are building a multi-node Mac cluster on supported Thunderbolt 5 hardware.

High-level process:

1. Boot into Recovery.
2. Run `rdma_ctl enable`.
3. Reboot.
4. Make sure your cabling and macOS versions are appropriate.

Important caveats:

- RDMA clusters need the right hardware and cabling.
- Matching macOS versions matter.
- On Mac Studio, avoid the Thunderbolt 5 port next to Ethernet for this setup.
- If running from source, the repo contains `tmp/set_rdma_network_config.sh` for network setup help.

## Benchmarks

<details>
  <summary>Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-1-qwen3-235b.jpeg" alt="Benchmark - Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
</details>

<details>
  <summary>DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-2-deepseek-3.1-671b.jpeg" alt="Benchmark - DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
</details>

<details>
  <summary>Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-3-kimi-k2-thinking.jpeg" alt="Benchmark - Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
</details>

## More Documentation

- [docs/api.md](docs/api.md)
- [docs/model-store.md](docs/model-store.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/kv-cache-backends.md](docs/kv-cache-backends.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute code, docs, testing help, or design feedback.

## About EXO

EXO is the upstream distributed inference project that Skulk builds on top of.
Skulk keeps that foundation, then pushes further on model-store workflows, dashboard UX, and newcomer-friendly cluster operation.
