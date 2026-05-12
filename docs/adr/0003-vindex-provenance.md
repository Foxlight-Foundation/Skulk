# ADR 0003: Vindex Provenance

Status: Accepted for planning
Date: 2026-05-12
Parent roadmap: https://github.com/Foxlight-Foundation/Skulk/issues/173
Tracking issue: https://github.com/Foxlight-Foundation/Skulk/issues/155
Publisher issue: https://github.com/Foxlight-Foundation/Skulk/issues/156
Source plan: https://github.com/Foxlight-Foundation/Skulk/blob/claude/understand-larql-repo-sJqA1/docs/slice-placement-and-vindex-publisher-plan.md

## Context

LARQL vindexes are directory-shaped artifacts derived from source model
weights. Extracting them can require substantial scratch disk, time, and
toolchain setup. Running that extraction on every Skulk user's machine would
make first use slow and operationally fragile.

Skulk already has model-store and download concepts for consuming artifacts.
The clean boundary is to make Skulk a vindex consumer and move extraction into
a dedicated publisher workflow.

## Decision

Skulk will consume vindexes from HuggingFace URIs such as `hf://...`. Skulk will
not extract vindexes in-tree.

Extraction, publication, manifest curation, and catalogue governance live in a
separate sibling repository: `skulk-vindex-publisher`.

The publisher repository owns scheduled LARQL extraction and publication jobs.
Skulk owns runtime consumption, local caching/staging, and placement metadata.

## Consequences

Skulk does not add a Rust toolchain or LARQL extraction dependency to its normal
runtime setup.

Future model-store work must support directory-shaped vindex artifacts, but it
does not need to know how to produce them.

The published vindex URI convention becomes part of the contract between the
publisher repo and Skulk's placement/runtime code.

## Rejected Alternatives

### Extract Inside Skulk

This would push heavyweight extraction work onto every operator machine,
including laptops that only need to run inference. It also expands Skulk's
runtime dependency surface for a build-time artifact-production task.

### No Curated Catalogue

Without a curated catalogue, users would need to find community vindexes or
produce their own before slice mode is useful. That weakens the operator
experience and makes supported-model behavior harder to reproduce.
