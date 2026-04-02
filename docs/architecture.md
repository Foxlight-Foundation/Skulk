<!-- Copyright 2025 Foxlight Foundation -->

# Skulk Architecture Overview

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
- `exo.yaml` `logging.structured_stdout` — enables the JSON sink

This is opt-in. Without the logging config, exo behaves identically to before.

## When to Read More

If you are:

- trying to get started, go back to the [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- integrating against the API, read the [API guide](api.md)
- setting up shared storage, read the [model store guide](model-store.md)
- setting up cluster logging, read `deployment/logging/` and the [CONTRIBUTING guide](https://github.com/Foxlight-Foundation/Skulk/blob/main/CONTRIBUTING.md#centralized-logging)
