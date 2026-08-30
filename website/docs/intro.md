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

1. Install Skulk on each machine. The packaged apps are the recommended path:

   **Apple Silicon macOS 15 or newer**

   ```bash
   brew install --cask Foxlight-Foundation/skulk/skulk
   ```

   Open **Skulk** from Applications and choose **Start Skulk**.

   **Ubuntu or Debian desktop (`amd64` or `arm64`)**

   ```bash
   curl -fLO https://apt.foxlight.ai/foxlight-archive-keyring.deb
   sudo apt install ./foxlight-archive-keyring.deb
   sudo apt update
   sudo apt install skulk
   ```

   Open **Skulk** from the application menu and choose **Start Skulk**. See the
   complete [installation guide](install) for updates, headless Linux, cluster
   namespaces, other Linux distributions, and development builds.
2. Open the dashboard (default `http://localhost:52415`), pick a model, and
   launch it. The supported catalog comes from Skulk's TUF-verified external
   model-card registry; each card identifies one exact selectable artifact.
   Skulk places it across the cluster and starts serving when it is ready.
3. Call the OpenAI-compatible endpoint at `/v1/chat/completions` with any client
   that speaks that format. The [API guide](api-guide) walks through a first
   request step by step, from placement to first token.

Speech works the same way: launch a speech model and the dashboard chat gains a
hands-free voice loop, while the cluster serves OpenAI-compatible
`/v1/audio/speech` and `/v1/audio/transcriptions` endpoints plus a realtime
transcription WebSocket at `/v1/realtime`
([speech guide](speech-fabric-realtime)).

For source and runtime internals, see
[source builds and runtime paths](build-and-runtime). For advanced service
configuration around a source checkout, see
[run as a service](run-skulk-as-a-service).

## Why Skulk

**Run models that don't fit on one machine.** Skulk splits a model across as many
machines as it needs and routes the work through the pipeline automatically. A
70B model that won't fit in one Mac's unified memory can run across two.

**Every device counts.** MacBooks, Mac Studios, Mac Pros, and Linux boxes all
join the same cluster. Skulk elects a master, places models across the available
nodes, and rebalances when a node leaves or rejoins.

**Always on, self-healing.** Skulk runs as a supervised service on macOS and
Linux: it starts at boot, restarts on crash, and rebuilds cluster state on
recovery. If the master node dies, a new one is elected and the models already
placed keep running, so the cluster stays available (an in-flight request at the
moment of failover may need to be retried).

**Manage it from anywhere.** Put your nodes on a Tailscale network and the
mobile-friendly operator panel gives you live memory, GPU, and temperature for
every node, plus one-tap node restarts, over plain HTTP. No SSH required.

**OpenAI-compatible.** Any client that speaks the OpenAI chat-completions format
works out of the box. No SDK changes, no custom client.

**Observable by default.** Runtime tracing, a cross-cluster flight recorder,
per-node diagnostics, and structured logs you can ship to VictoriaLogs let you
see exactly what each node is doing during a request.

**Air-gap durable.** Every complete model-store or staged artifact retains its
full effective card and hashed manifest beside the bytes. Installed models load
from that local truth before registry access, and node caches can reconcile into
the central store without downloading again from Hugging Face.

## Common tasks

- **Use the API** to run inference: [API guide](api-guide), and the browsable
  [API reference](/api/skulk-api).
- **Manage the cluster** (place models, watch nodes, recover): the
  [dashboard and operations guide](operations), and
  [remote access via Tailscale](tailscale).
- **Debug the cluster** during a request: [tracing and debugging](tracing).
- **Add models to the model store**: [model store guide](model-store).
- **Understand model identity and catalog precedence**:
  [model cards](model-cards).
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
