# Skulk Developer Docs

This site combines practical guides with generated reference material for Skulk.

Skulk is a distributed inference platform, so these docs try to serve both of the audiences that show up most often:

- people who are new and want a clear path to a first working setup
- developers who want exact API, type, and integration reference material

## Start Here

If you are getting oriented, start with these pages:

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md) for installation, first run, and quick-start paths
- [API guide](api.md) for the happy path: place a model, then call the API
- [Model store guide](model-store.md) for shared model storage and download workflows
- [KV cache backends](kv-cache-backends.md) for backend and runtime tuning
- [Architecture overview](architecture.md) for how the node, cluster, and event model fit together

## Common Jobs

- I want to browse the backend API: [API Reference (ReDoc)](reference/api-reference.md)
- I want the raw machine-readable schema: [OpenAPI schema](reference/openapi.md)
- I want frontend and TypeScript symbols: [TypeScript API](reference/typedoc/README.md)
- I want implementation context before I integrate: [Architecture overview](architecture.md)

## What Lives Here

- Hand-written guides for setup, architecture, and operational workflows
- Generated OpenAPI output for the FastAPI backend
- Generated TypeDoc output for selected TypeScript modules in `dashboard-react`

## Keep In Mind

For text generation, Skulk is not just a stateless HTTP API. A model generally needs to be placed and running before chat-style requests succeed. The dashboard enforces this, and the compatibility APIs reflect the same runtime reality.
