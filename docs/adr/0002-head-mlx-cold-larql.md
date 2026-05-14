# ADR 0002: MLX Head With LARQL Cold Tier

Status: Accepted for planning; gated by Phase 3 feasibility
Date: 2026-05-12
Parent roadmap: https://github.com/Foxlight-Foundation/Skulk/issues/173
Tracking issue: https://github.com/Foxlight-Foundation/Skulk/issues/154
Gate issues: https://github.com/Foxlight-Foundation/Skulk/issues/161 and https://github.com/Foxlight-Foundation/Skulk/issues/162
Source plan: https://github.com/Foxlight-Foundation/Skulk/blob/claude/understand-larql-repo-sJqA1/docs/slice-placement-and-vindex-publisher-plan.md

## Context

Skulk's strongest runtime path is MLX on Apple Silicon. Slice placement should
extend that path instead of replacing it. The desired architecture lets a Mac
head node keep the hot attention/router path local while RAM-rich commodity
peers serve cold FFN or expert weights from LARQL vindexes.

This decision is expensive to reverse once placement state, runner assignment,
and API surfaces begin to encode slice responsibilities.

## Decision

The MLX runner remains the head runtime for slice mode. It owns the standard
MLX weights needed for the head role: embeddings, attention, norms, router, and
any locally assigned layers. It delegates selected per-layer FFN or expert work
to `LarqlRunner` peers over HTTP.

The MLX head never loads a vindex. Vindexes are cold-tier artifacts consumed by
LARQL peers.

The default wire format for delegated tensors is f16. i8 remains an explicit
future opt-in only where the LARQL contract supports it and Skulk can preserve
correctness.

## Feasibility Gate

This ADR is accepted for planning, not yet accepted for irreversible runtime
implementation. Phase 3 must prove that Skulk's MLX path can delegate a
per-layer FFN or expert step and continue generation with acceptable overhead.

If MLX does not expose usable hooks and a manual forward-pass split is too
fragile, too invasive, or too slow, this ADR must be superseded before Phase 4
slice-placement work starts.

## Consequences

Existing MLX single-node and MLX pipeline placement remain the default path for
models that fit on the selected head node.

Slice placement is additive. It is only considered when the normal MLX path
cannot fit the model or when the operator explicitly chooses a slice-mode flow
in future UI/API work.

The slice plan must identify which LARQL peer serves which preset, layer range,
expert range, and vindex URI so the MLX head can dispatch remote FFN/expert
calls deterministically.

## Rejected Alternatives

### Replace the Head Runtime With LARQL

Replacing MLX would discard Skulk's current strongest execution path and make
Apple Silicon performance dependent on a new serving stack.

### Load Vindexes on the Head

Loading vindexes on the head duplicates cold-tier storage and undermines the
purpose of using commodity RAM-rich peers for dormant weights.

### General Remote-Compute Abstraction

The v1 design targets LARQL's concrete FFN/expert server contract. A generic
remote execution abstraction would add surface area before Skulk has proven the
basic slice-mode value proposition.
