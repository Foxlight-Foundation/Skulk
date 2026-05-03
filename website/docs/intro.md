---
id: intro
title: Skulk Developer Docs
sidebar_position: 1
slug: /
---

<!-- Copyright 2025 Foxlight Foundation -->

## Why Skulk

Most inference tools assume a single machine. Skulk assumes a cluster.

**Run models that don't fit on one machine.** Skulk splits a model across every device in your cluster and routes inference through the pipeline automatically. A 70B model that won't fit in your Mac's unified memory might fit across two.

**Every device counts.** MacBooks, Mac Studios, Mac Pros, Linux boxes — if it runs MLX, it joins the cluster. Skulk elects a master, places model instances across available nodes, and rebalances when a node goes down or comes back.

**Always on, self-healing.** Skulk runs as a supervised service on macOS and Linux. It starts at boot, restarts on crash, and recovers cluster state from its event log — no babysitting required. A startup preflight detects port conflicts and tells the supervisor to back off and retry rather than failing silently.

**Manage it from anywhere.** Add Tailscale to your nodes and your phone and you have a private overlay network to your cluster. The dashboard works over plain HTTP on the Tailscale address. The built-in operator panel is mobile-optimized — it shows memory, GPU, and temperature for every node and lets you restart any node with a two-tap confirmation. You don't need SSH.

**OpenAI-compatible API.** Any client that speaks the OpenAI chat completions format works with Skulk out of the box — no SDK changes, no custom clients.

**Observable by default.** Runtime tracing, a cross-cluster flight recorder, per-node diagnostics, and structured JSON logging that ships to VictoriaLogs via Vector. You can see exactly what's happening across every rank during an inference request.

---

## Start Here

If you are getting oriented, start with these pages:

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md) — installation, first run, quick-start
- [Build and runtime paths](build-and-runtime) — how `uv` and Nix fit together
- [Run as a service](run-skulk-as-a-service) — autostart and crash recovery for macOS and Linux
- [Remote access](tailscale) — reach your dashboard and operator panel from anywhere via Tailscale
- [API guide](api-guide) — place a model, then call the API
- [Model store guide](model-store) — shared model storage and download workflows
- [Tracing and debugging](tracing) — runtime tracing, cluster browsing, and operator workflow
- [Architecture overview](architecture) — how the node, cluster, and event model fit together

## Common Jobs

- I want to browse the backend API: [API Reference](/api/skulk-api)
- I want frontend and TypeScript symbols: [TypeScript API](https://foxlight-foundation.github.io/Skulk/typedoc/)
- I want implementation context before I integrate: [Architecture overview](architecture)
- I want to debug live inference: [Tracing and debugging](tracing)
- I want to manage my cluster remotely: [Remote access via Tailscale](tailscale)
- I want nodes in different locations in one cluster: [Multi-network clustering](tailscale-clustering)

## What Lives Here

- Hand-written guides for setup, architecture, and operational workflows
- Generated OpenAPI output for the FastAPI backend, browsable per-endpoint with a try-it-out console
- Generated TypeDoc output for selected TypeScript modules in `dashboard-react`

## Keep In Mind

For text generation, Skulk is not just a stateless HTTP API. A model generally needs to be placed and running before chat-style requests succeed. The dashboard enforces this, and the compatibility APIs reflect the same runtime reality.
