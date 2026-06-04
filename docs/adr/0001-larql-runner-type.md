# ADR 0001: LARQL Runner Type

Status: Accepted for planning
Date: 2026-05-12
Parent roadmap: https://github.com/Foxlight-Foundation/Skulk/issues/173
Tracking issue: https://github.com/Foxlight-Foundation/Skulk/issues/153
Source plan: https://github.com/Foxlight-Foundation/Skulk/blob/claude/understand-larql-repo-sJqA1/docs/slice-placement-and-vindex-publisher-plan.md

## Context

Skulk's current execution model is centered on worker-managed runner
subprocesses. A worker observes event-sourced placement state, downloads or
stages the assigned model artifacts, starts a runner subprocess, and reports
lifecycle transitions through the same event stream every node applies.

LARQL introduces a different execution role: a process that serves vindex-backed
FFN or expert slices over HTTP. That process still needs the same operational
properties Skulk expects from model runners: deterministic startup, supervised
shutdown, crash visibility, logging, readiness state, and eventual placement
metadata.

## Decision

Skulk will model LARQL as a first-class `LarqlRunner` runner type managed by
the worker. The worker will supervise a child `larql-server` process alongside
the existing MLX runner subprocesses.

The `LarqlRunner` will use Skulk's runner lifecycle conventions:

- worker-owned process supervision
- stdout/stderr forwarding into Skulk logging
- readiness and failure reporting through event-sourced state
- shutdown driven by instance and runner lifecycle events

The initial implementation will treat `larql-server` as an upstream binary
dependency, not as in-tree Rust or Python code.

## Consequences

`LarqlRunner` becomes part of Skulk's runner taxonomy. Future implementation
work must add explicit runner metadata rather than overloading MLX shard
metadata or treating LARQL as an external sidecar.

Operators should be able to reason about LARQL-backed slices through the same
dashboard, state, diagnostics, and logging surfaces used for MLX runners.

Skulk remains insulated from LARQL internals. The integration boundary is the
LARQL server process and its HTTP contract.

## Rejected Alternatives

### Sidecar

A sidecar would be quick to prototype, but it moves process lifecycle,
readiness, logs, and crash recovery outside Skulk. That creates a second
operator workflow and makes slice placement harder to explain and diagnose.

### In-tree Port

Reimplementing LARQL's slice protocol inside Skulk would be a large, slow fork
of upstream LARQL. It would also make it harder to pick up future LARQL
improvements in vindex format, server behavior, and FFN/expert endpoints.
