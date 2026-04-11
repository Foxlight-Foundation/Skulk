---
id: everything-different
title: Everything Different About Skulk
sidebar_position: 2
---

<!-- Copyright 2025 Foxlight Foundation -->

This is the running list of where Skulk diverges from upstream [exo](https://github.com/exo-explore/exo).

It is intentionally a **living document** — every meaningful change should land here when it ships, so anyone evaluating Skulk can see the surface area at a glance and the team has a single place to point at when describing what makes this fork distinct.

If you are about to land a feature or fix that adds, removes, or materially changes a behavior relative to upstream, update this page in the same PR.

## Inference and KV cache

- **RotorQuant KV cache backend** — pure-MLX port of IsoQuant 3-bit (block-diagonal quaternion rotations + Lloyd-Max centroids) with **deferred prefill on Metal**, a contribution that does not exist in any upstream project (the llama.cpp fork ships it CUDA-only). GQA-native; no fallback for grouped-query models. See [KV cache backends](kv-cache-backends).
- **TurboQuant native and adaptive backends** — randomized Hadamard rotation + Lloyd-Max centroids, with an adaptive variant that keeps edge attention layers in fp16 for accuracy.
- **OptiQ KV cache integration** — wraps `mlx-optiq`'s rotated-space attention path so the rotation cost stays out of the per-token loop on supported (non-GQA) models.
- **OptiQ mixed-precision weight quantization pipeline** — async wrapper around `mlx-optiq`'s sensitivity analysis and KL-divergence per-layer bit allocation, exposed as a model-store optimization job.
- **KV prefix cache with snapshot/restore** — LRU-evicted prompt-prefix cache that snapshots SSM and rotating-window cache states so prefix matches are reusable across conversation turns even for hybrid Mamba/Transformer architectures.
- **Pipeline-parallel prefill for short prompts** — pipelined models now route every prefill through the pipeline path, fixing prior warmup hangs on Gemma-class models.
- **Force-sequential fallback** — quantized backends transparently fall back to a sequential generator when batch/history mode is incompatible with their cache layout.

## Model capability system

- **Two-layer capability model** — declarative `ModelCard` (with optional `reasoning`, `modalities`, `tooling`, and `runtime` sections) plus a normalized `ResolvedCapabilityProfile` derived from the card and conservative family defaults. This is the source of truth for prompt rendering, output parsing, tool-call handling, and the `/v1/models` `resolved_capabilities` field.
- **Phase 2 thinking contract** — `enable_thinking`, `reasoning_effort`, and the dashboard thinking toggle are all driven by `supports_thinking_toggle`, so non-toggleable reasoning models behave correctly without leaking model-specific quirks into client code. See [model capabilities](model-capabilities).
- **Output parser selection** — model cards declare `output_parser` (`generic`, `gemma4`, `gpt_oss`, `deepseek_v32`, etc.), so reasoning markers are normalized into structured `reasoning_content` per family.
- **Model store metadata pipeline** — capability resolution feeds `/v1/models` so dashboards and clients can discover thinking, multimodal, and tool support without hardcoding model lists.

## API surface

- **Claude Messages API** — `/v1/messages` adapter, including streaming, tool use, image inputs, and capability-aware thinking controls.
- **Ollama compatibility** — both `/api/chat` and `/api/generate`, with adapter-side reasoning normalization.
- **OpenAI Responses API** — `/v1/responses` adapter alongside chat completions.
- **Embeddings endpoint** for non-chat workloads.
- **Model store endpoints** — search, add, download, capability resolution, optimization jobs, and registry management, all exposed under stable URLs and documented in the OpenAPI spec.
- **Cluster-wide config endpoints** — `GET`/`POST` config that gossipsubs to every node and writes back to `skulk.yaml`. Stable KV cache backend selection goes through this same endpoint, so the dashboard Settings panel can switch supported backends cluster-wide without env vars. Experimental `rotorquant` values remain env-gated and fall back to `default` unless explicitly enabled.
- **Tracing, downloads, instance previews, and placement endpoints** — distributed-system observability and pre-launch placement inspection that upstream does not expose.

## Dashboard

- **React dashboard (default)** — replaces upstream's Svelte UI with a typed React + styled-components app that ships with the binary. The legacy Svelte dashboard is kept only as a fallback in the repo.
- **Cluster topology view** with live device icons, GPU stats, network mesh visualization, and connection status banners.
- **Placement preview / placement manager** for inspecting and choosing valid placements before launching.
- **Model store browser** with HuggingFace search, family sidebar, model filters, capability badges, recent models, and per-model launch controls.
- **Reasoning-aware chat UI** that splits inline `<think>` and Gemma `<|channel>` markers into a dedicated thinking panel and merges them with `reasoning_content` deltas from the API.
- **Image attachments and multimodal chat affordances** for vision models.
- **Cluster-wide settings panel** that writes to `skulk.yaml` and syncs across nodes via gossipsub.
- **Light and dark themes** with first-class theme tokens.

## Centralized logging and observability

- **Structured JSON stdout** when `logging.enabled` is set, configurable from the dashboard Settings panel and synced cluster-wide.
- **Vector + VictoriaLogs + Grafana stack** — local Vector log shipper on each node, central VictoriaLogs storage, ready-made Grafana dashboards. Stack definition lives in `deployment/logging/`.
- **Distributed tracing** opt-in via `EXO_TRACING_ENABLED`.

## Model store

- **Centralized model store host** — one node downloads, the rest of the cluster stages over the LAN.
- **Persistent registry** with capability resolution and download tracking.
- **Custom model card support** — add your own model with `POST /models/add`.
- **Image and embedding model cards** behind feature flags.
- **Optimization job pipeline** for mlx-optiq mixed-precision weight quantization.

## Cluster operation

- **Cluster-wide settings sync** for KV cache backend, logging, model store host, HF token, and other inference toggles.
- **Bootstrap peer config from `skulk.yaml`, env, or CLI** for fixed-topology clusters.
- **Election (bully algorithm) + master/worker split** for indexing events and broadcasting state.
- **`SKULK_*` environment variables** alongside the legacy `EXO_*` set, so new options can land without colliding with upstream.
- **`skulk.yaml`** as the canonical config file, with `exo.yaml` kept for backwards compatibility.

## Build, type system, and dev workflow

- **Strict basedpyright** type checking — zero-error policy for new code.
- **Ruff** linting and **`nix fmt`** formatting in CI.
- **Nix flake** for reproducible toolchain setup.
- **Docusaurus docs site** with auto-generated OpenAPI per-endpoint pages and TypeDoc HTML reference for the dashboard, both built from source.
- **Pre-commit checklist** documented in `CLAUDE.md` and enforced in CI.

## Hardware and platform

- **Apple Silicon as the primary target**, including RDMA over Thunderbolt 5 on supported hardware and matched macOS versions.
- **Linux supported** (CPU-oriented in this fork; GPU work happens on Apple Silicon).

## How to update this page

When you ship something that distinguishes Skulk from upstream exo, add it here in the same PR. A few ground rules:

- Group your bullet under the right category. If none fit, add a new category — but only when there's clearly a new area of divergence.
- Lead with what the change *is*, then a sentence on *why* it matters compared to upstream.
- Link to the relevant guide, code path, or issue when one exists.
- Remove or rewrite bullets when they become inaccurate. This page is canonical, so stale bullets are worse than missing ones.
