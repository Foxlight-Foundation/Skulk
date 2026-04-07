---
id: intro
title: Skulk Developer Docs
sidebar_position: 1
slug: /
---

<!-- Copyright 2025 Foxlight Foundation -->

This site combines practical guides with generated reference material for Skulk.

Skulk is a distributed inference platform, so these docs try to serve both of the audiences that show up most often:

- people who are new and want a clear path to a first working setup
- developers who want exact API, type, and integration reference material

## Start Here

If you are getting oriented, start with these pages:

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md) for installation, first run, and quick-start paths
- [API guide](api-guide) for the happy path: place a model, then call the API
- [Model store guide](model-store) for shared model storage and download workflows
- [KV cache backends](kv-cache-backends) for backend and runtime tuning
- [Model cards](model-cards) for the metadata and capability declarations Skulk uses to describe models
- [Model capabilities](model-capabilities) for how declarative card fields become runtime behavior
- [Architecture overview](architecture) for how the node, cluster, and event model fit together

## Common Jobs

- I want to browse the backend API: [API Reference](/api/skulk-api)
- I want frontend and TypeScript symbols: [TypeScript API](https://foxlight-foundation.github.io/Skulk/typedoc/)
- I want implementation context before I integrate: [Architecture overview](architecture)

## What Lives Here

- Hand-written guides for setup, architecture, and operational workflows
- Generated OpenAPI output for the FastAPI backend, browsable per-endpoint with a try-it-out console
- Generated TypeDoc output for selected TypeScript modules in `dashboard-react`

## Keep In Mind

For text generation, Skulk is not just a stateless HTTP API. A model generally needs to be placed and running before chat-style requests succeed. The dashboard enforces this, and the compatibility APIs reflect the same runtime reality.
