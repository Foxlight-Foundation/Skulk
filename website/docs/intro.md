---
id: intro
title: Skulk
sidebar_position: 1
slug: /
---

<!-- Copyright 2025 Foxlight Foundation -->

**Skulk is an interconnect fabric for multi-node AI compute.** It joins several
machines into one cluster and moves work across them as if they were a single
device.

Its headline use today is **distributed inference**: point Skulk at a few
machines and it pools their memory and GPUs behind one OpenAI-compatible
endpoint, so you can run models far larger than any single machine could hold.

## Get started

1. Install and start Skulk on each machine. The
   [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
   covers installation and the first run.
2. Open the dashboard (default `http://localhost:52415`), pick a model, and
   launch it. Skulk places it across the cluster and starts serving when it is
   ready.
3. Call the OpenAI-compatible endpoint at `/v1/chat/completions` with any client
   that speaks that format.

For the runtime details, see [build and runtime paths](build-and-runtime) and
[run as a service](run-skulk-as-a-service).

## Why Skulk

**Run models that don't fit on one machine.** Skulk splits a model across every
device in the cluster and routes the work through the pipeline automatically. A
70B model that won't fit in one Mac's unified memory can run across two.

**Every device counts.** MacBooks, Mac Studios, Mac Pros, and Linux boxes all
join the same cluster. Skulk elects a master, places models across the available
nodes, and rebalances when a node leaves or rejoins.

**Always on, self-healing.** Skulk runs as a supervised service on macOS and
Linux: it starts at boot, restarts on crash, and rebuilds cluster state on
recovery. If the master node dies mid-request, a new one is elected and serving
continues.

**Manage it from anywhere.** Put your nodes on a Tailscale network and the
mobile-friendly operator panel gives you live memory, GPU, and temperature for
every node, plus one-tap node restarts, over plain HTTP. No SSH required.

**OpenAI-compatible.** Any client that speaks the OpenAI chat-completions format
works out of the box. No SDK changes, no custom client.

**Observable by default.** Runtime tracing, a cross-cluster flight recorder,
per-node diagnostics, and structured logs you can ship to VictoriaLogs let you
see exactly what each node is doing during a request.

## Common tasks

- **Use the API** to run inference: [API guide](api-guide), and the browsable
  [API reference](/api/skulk-api).
- **Manage the cluster** (place models, watch nodes, recover): the
  [dashboard and operations guide](operations), and
  [remote access via Tailscale](tailscale).
- **Debug the cluster** during a request: [tracing and debugging](tracing).
- **Add models to the model store**: [model store guide](model-store).
- **Span locations or networks** with one cluster:
  [multi-network clustering](tailscale-clustering).

## What Skulk is, and where it's going

Skulk separates cluster traffic into three planes: a **compute** plane (the
high-speed interconnect that exchanges model activations between nodes), a
**control** plane (cluster decisions, task lifecycle, and node health), and a
**data** plane (generated output streamed back to the requesting node). Keeping
these separate is what makes Skulk a general fabric rather than a single-purpose
inference server: inference is the first workload to ride it, not the limit of
what it can carry.

That foundation opens up more than running one model across machines. The same
interconnect is built to support disaggregating a model so different nodes handle
different parts of it, treating memory as its own kind of node, mixing inference
backends, and composing clusters out of smaller ones. The
[architecture overview](architecture) explains how the pieces fit together
today.
