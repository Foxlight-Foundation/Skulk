# **Skulk**

<!-- Copyright 2025 Foxlight Foundation -->

<div align="center">
  <img src="docs/imgs/skulk-logo-2.png" width="200" height="200" alt="Skulk logo">
</div>

Skulk is a fork of EXO for running AI models across one or more machines as a cluster.
It keeps EXO's distributed inference foundation, then extends it with a central model store,
a more modern dashboard, richer API workflows, sophisticated cache quantization, support for more model families such as embeddings and TTS, and cluster-friendly configuration management.

> Skulk is maintained by [Foxlight Foundation](https://github.com/foxlight-foundation) and forked from [exo](https://github.com/exo-explore/exo).

**[Documentation](https://foxlight-foundation.github.io/Skulk/)** · **[Build And Runtime Paths](https://foxlight-foundation.github.io/Skulk/build-and-runtime/)** · **[Release Notes](https://foxlight-foundation.github.io/Skulk/release-notes/1.0.2/)** · **[Architecture](https://foxlight-foundation.github.io/Skulk/architecture/)**

## What Skulk Is Good At

- Run a model on a single machine through the dashboard or API.
- Form a small cluster of Macs and split larger models across them.
- Use a central model store so the cluster downloads once and stages locally.
- Talk to the cluster through OpenAI Chat Completions, OpenAI Responses, Claude Messages, or Ollama-compatible APIs.
- Experiment with advanced placement modes, RDMA, and KV cache backends when you are ready.
- Run non-chat workloads such as embeddings and other specialized model flows.
- Build TTS-oriented and other API-driven workflows on top of the cluster.
- Actually use your cluster for real inference workloads instead of treating it as a demo.

## Prerequisites

### macOS

- [Xcode](https://developer.apple.com/xcode/)
- [uv](https://github.com/astral-sh/uv)
- [node](https://github.com/nodejs/node)
- [rustup](https://rustup.rs/)
- `macmon` for Apple Silicon monitoring
- [Nix](https://nixos.org/download/) for `nix fmt`, `nix flake check`, and the repo dev shell

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

Build/runtime note:

- `uv` is the canonical source and runtime path for Skulk on macOS, including the official `mlx` + `mlx-metal` wheel stack.
- Nix is kept for reproducible development tooling, formatting, and `flake`-based validation. It should match the `uv` runtime contract instead of silently substituting a different MLX build.

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
- **Experimental inference tuning**: OptiQ and other KV cache backends for long-context and memory experiments.

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
| `EXO_BOOTSTRAP_PEERS` | Comma-separated static peers to dial on startup | None |
| `HF_TOKEN` | Hugging Face token | None |

Examples:

```bash
EXO_OFFLINE=true uv run exo
EXO_ENABLE_IMAGE_MODELS=true uv run exo
EXO_KV_CACHE_BACKEND=optiq EXO_OPTIQ_BITS=4 EXO_OPTIQ_FP16_LAYERS=4 uv run exo
```

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
