---
id: tracing
title: Tracing and Debugging
sidebar_position: 5
---

<!-- Copyright 2025 Foxlight Foundation -->

This guide explains how to use Skulk's runtime tracing feature.

The short version:

- tracing is off by default
- you turn it on from the dashboard traces view or via `PUT /v1/tracing`
- the toggle applies to new requests across the current cluster session
- image, text-generation, and embedding requests can all produce traces
- traces can be browsed from any reachable node through the cluster trace view

## How To Turn It On

In the dashboard:

1. click the bug icon in the toolbar
2. open the **Traces** page
3. use the tracing toggle card at the top of the page

The UI copy intentionally says:

- **Applies to new requests on all nodes**

That is the correct mental model. Tracing is a runtime cluster toggle, but it
does not retroactively attach to work that was already running before you
enabled it.

You can also control it through the API:

```bash
curl http://localhost:52415/v1/tracing

curl -X PUT http://localhost:52415/v1/tracing \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true}'
```

## What Gets Traced

Tracing is no longer image-only.

Skulk can now emit traces for:

- image generation and image edits
- text generation
- text embeddings

The trace metadata includes:

- task kind
- model ID
- source node information
- tags
- arbitrary attributes for richer debugging views

Tool-related text-generation activity is also marked so the traces page can
filter for tool activity.

## Local vs Cluster Browsing

The traces page supports two browsing scopes:

- **Local**: show trace artifacts saved on the current node
- **Cluster**: fan out to reachable peer APIs, deduplicate by `task_id`, and
  browse traces from any reachable node

This distinction matters:

- browsing is available from any reachable node in cluster scope
- deletion is still local-only in v1
- if some peers are offline or unreachable, cluster results may be partial

## API Surface

Runtime tracing control:

- `GET /v1/tracing`
- `PUT /v1/tracing`

Local trace browsing:

- `GET /v1/traces`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`
- `POST /v1/traces/delete`

Cluster trace browsing:

- `GET /v1/traces/cluster`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`

For request and response details, see the [API guide](api-guide) and the
interactive [API Reference](/api/skulk-api).

## Common Workflow

When you want to debug a live problem:

1. open the traces page
2. enable tracing
3. reproduce the workload
4. refresh the traces list
5. inspect local or cluster scope as needed
6. use filters for task kind, model, source node, category, or tool activity
7. download the raw trace JSON or open it in Perfetto when you need timeline detail

## Operator Notes

- tracing is meant for debugging sessions, not as a permanent always-on mode
- enabling tracing affects new requests only
- the old env-var path still exists as a hidden developer boot override, but it
  is no longer the normal user workflow
- cluster browsing is read-only in v1
- local deletion remains explicit and local-only in v1
