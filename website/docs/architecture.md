---
id: architecture
title: Skulk Architecture Overview
sidebar_position: 5
---

<!-- Copyright 2025 Foxlight Foundation -->

This page is the quick mental model for how Skulk fits together.

You do not need to understand every subsystem to use Skulk, but it helps to know what is happening when you start a node, join a cluster, place a model, and send a request.

## The Short Version

A single Skulk node runs several cooperating systems:

- networking and peer discovery
- election and master coordination
- worker execution and model loading
- the API server
- the dashboard served by that API server

When you add more nodes, Skulk forms a cluster. The master coordinates placement and state, workers do execution, and the API exposes both compatibility endpoints and Skulk-specific control endpoints.

## The Main User Flow

Most real usage follows this shape:

1. start one or more Skulk nodes
2. let the cluster discover itself or connect through bootstrap peers
3. inspect topology in the dashboard or through `/state`
4. preview or choose a placement for a model
5. place the model
6. download or stage the required model files
7. load the model on the chosen nodes
8. send chat or other generation requests

That is why placement is such an important idea in Skulk docs: generation depends on runtime state, not just API calls.

## The Main Systems

### Master

The master coordinates cluster state and placement decisions.

Responsibilities include:

- ordering events
- coordinating placement
- maintaining the shared view of the cluster

### Worker

Each node runs a worker.

The worker is responsible for:

- gathering node information
- managing downloads and staging
- loading and unloading model runners
- executing inference-related tasks

### Runner

Runners execute model work in an isolated process.

This is where Skulk selects inference behavior such as:

- model loading strategy
- MLX execution path
- KV cache backend choice

The runtime behavior for a specific model family is increasingly driven by
model-card-backed capability resolution rather than scattered family checks. See
[Model Cards](model-cards), [Model Capabilities](model-capabilities), and
[Gemma 4 behavior notes](model-behaviors/gemma4) for the current shape of that
system.

### API

The API server exposes:

- OpenAI-compatible endpoints
- Claude-compatible endpoints
- Ollama-compatible endpoints
- Skulk-specific control endpoints for placement, config, store, state, and tracing

The API server also serves the dashboard.

### Election

Election handles who becomes master in a distributed cluster.

That lets Skulk keep operating even when connectivity changes or nodes come and go.

## Message Flow

Skulk uses explicit message passing between systems.

At a high level:

- commands ask for something to happen
- events record what already happened
- state is rebuilt by applying events in order

This is why the system often feels more like a distributed application platform than a single local inference process.

## Event Sourcing

Skulk uses an event-sourced state model.

In practice, that means:

- cluster changes are represented as events
- those events are ordered and applied into a shared state object
- commands and current state together drive future work

Follower recovery does not have to mean replaying an entire session forever.
Newer Skulk builds can bootstrap from a master-published snapshot and then
replay only the retained tail after that snapshot. This keeps restart and
rejoin time bounded on long-lived clusters while preserving the same
authoritative model: master state plus indexed events.

This does introduce an operational rollout rule: once a master starts
compacting old replay history after writing snapshots, older nodes that only
understand "replay from event `0`" should be considered temporary guests during
the rollout window, not indefinitely supported members of the cluster. Upgrade
all nodes before relying on bounded retention as the normal steady state.

A simple rule of thumb:

- events are past tense
- commands are imperative

Examples:

- "place this model" is a command
- "this instance was created" is an event

## API Adapters

Skulk supports multiple external API styles by adapting them into one internal execution path.

At a high level:

```text
OpenAI Chat Completions -> adapter -> internal text generation task
Claude Messages         -> adapter -> internal text generation task
OpenAI Responses        -> adapter -> internal text generation task
Ollama APIs             -> adapter -> internal text generation task
```

This is why one placed model can be accessed through several compatibility formats.

## Topics and Communication

The major communication patterns include:

- command topics for explicit requests
- local events from workers and nodes
- global events broadcast by the master
- election messages for leader coordination
- connection messages for networking updates

You do not usually need to work with these directly as a user, but they explain why state, placement, and trace behavior look the way they do.

## Where the Model Store Fits

The model store does not replace the cluster architecture.

Instead, it changes how model artifacts are sourced:

- without a store, nodes download independently
- with a store, one host keeps shared model files and other nodes stage from it

The rest of the system still uses the same master, worker, API, and placement model.

## Where the Dashboard Fits

The dashboard is not a separate product or service.

It is the main operator interface for the same Skulk runtime:

- topology view
- model store workflows
- settings and config
- chat
- placement workflows

That is why the docs often describe dashboard and API flows as parallel ways of driving the same underlying system.

## Where Logging Fits

Skulk supports centralized log aggregation for the cluster.

Each node can emit structured JSON on stdout alongside the human-readable stderr output. A local Vector agent reads stdout and ships logs to a central VictoriaLogs instance, where they can be queried via Grafana or VictoriaLogs' built-in VMUI.

The key pieces:

- `src/exo/shared/logging.py` — loguru setup with a JSON stdout sink
- `deployment/logging/vector.yaml` — Vector config (stdin → VictoriaLogs)
- `deployment/logging/docker-compose.yml` — VictoriaLogs + Grafana stack
- `skulk.yaml` `logging.enabled` + `logging.ingest_url` — enables the JSON sink (configurable via dashboard Settings, synced to all nodes)

This is opt-in. Without the logging config, skulk behaves identically to before.

## Where Tracing Fits

Skulk also has a separate tracing surface for debugging live inference work.

The important user-facing model is:

- tracing is off by default
- you turn it on at runtime from the dashboard traces view
- the toggle applies cluster-wide for new requests
- traces can be browsed from any reachable node through cluster trace endpoints
- local trace deletion remains local-only in v1

Tracing is meant for targeted debugging sessions, not as a permanently enabled
always-on telemetry pipeline. When you need it, use the dashboard bug icon or
the `/v1/tracing` API to enable it, reproduce the workload, then inspect the
result through the traces UI or the `/v1/traces*` endpoints.

For operator workflow and endpoint details, read [Tracing and debugging](tracing)
and the [API guide](api-guide).

## Debugging MLX Hangs

When a model appears to stall during warmup, prefill, or distributed generation,
Skulk can emit phase-specific hang diagnostics from the runner process.

Set these environment variables before starting `skulk`:

- `SKULK_MLX_HANG_DEBUG=1`
- `SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS=10`

When enabled, the runner logs:

- entry and exit for warmup and prefill phases
- which prefill path was selected (`stream_generate` or `pipeline_parallel_prefill`)
- whether prefill yielded its first token
- periodic Python stack traces while the active phase remains stuck

`SKULK_...` is the preferred prefix. `EXO_MLX_HANG_DEBUG` and
`EXO_MLX_HANG_DEBUG_INTERVAL_SECONDS` remain accepted as compatibility
fallbacks while older scripts are updated.

## Pipeline Warmup Policy

Distributed pipeline models now use an intentionally minimal warmup request.

The goal of warmup is only to validate that MLX prefill and first-token
generation are functional while matching the current `generate.prefill()`
routing behavior: short single-chunk prompts are sent through
`stream_generate` because `pipeline_parallel_prefill` has been observed to
wedge for that case on multi-node pipeline models.

For distributed pipeline warmup, Skulk always uses:

- no synthetic instructions
- a single user message with content `hello`
- `enable_thinking=False`
- the normal sampler defaults used by the warmup path

The following debug-only environment variables are available:

- `SKULK_DEBUG_WARMUP_REPEAT_COUNT`
- `SKULK_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS`

These are only honored for single-node debugging. For distributed pipeline
warmup, Skulk intentionally ignores them and stays on the minimal sanity-check
prompt.

Legacy compatibility aliases are also accepted:

- `EXO_DEBUG_WARMUP_REPEAT_COUNT`
- `EXO_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS`

There is also an emergency bypass for debugging:

- `SKULK_SKIP_LLM_WARMUP=1`

That bypass marks the runner ready without issuing the synthetic warmup request
and should only be used for diagnosis, not normal operation. Distributed
groups ignore it so warmup coordination cannot diverge across ranks.

## When to Read More

If you are:

- trying to get started, go back to the [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- integrating against the API, read the [API guide](api-guide)
- setting up shared storage, read the [model store guide](model-store)
- setting up cluster logging, read `deployment/logging/` and the [CONTRIBUTING guide](https://github.com/Foxlight-Foundation/Skulk/blob/main/CONTRIBUTING.md#centralized-logging)
