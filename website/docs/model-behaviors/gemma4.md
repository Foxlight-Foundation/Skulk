---
id: gemma4
title: Gemma 4
sidebar_position: 1
---

<!-- Copyright 2025 Foxlight Foundation -->

Gemma 4 is one of the first model families in Skulk that required explicit,
model-specific runtime handling.

That makes it a useful reference point for how the capability system is meant to
work.

## Why Gemma 4 Is Special

Gemma 4 is not just “a text model with vision.”

It brings together several distinct behaviors:

- custom multimodal prompt structure
- channel-delimited reasoning blocks
- native multimodal execution paths
- model-family-specific tool formatting considerations
- variant-specific modality support, including audio on some smaller variants

Generic least-common-denominator handling was enough to get partial behavior, but not enough to get reliable, correct behavior.

## Prompt Handling

Plain Gemma 4 requests use a dedicated renderer instead of the generic tokenizer chat template when Skulk needs exact control over the prompt shape.

That is especially important for:

- multimodal user messages
- assistant generation prefix handling
- reasoning channel initialization

## Reasoning Format

Gemma 4 reasoning uses a channel-delimited format rather than the simpler token-delimited approach used by some other models.

In practice, that means Skulk needs to:

- render the correct thought-channel structure
- parse the channel markers correctly
- route reasoning text away from visible assistant content

This is exactly the kind of behavior the capability system is meant to describe explicitly.

## Vision and Native Multimodality

Gemma 4 can use a native multimodal execution path.

That means model support is not just a matter of accepting image content parts. The runtime also needs to know:

- whether native multimodal execution is expected
- what processor/model type is used
- how to interpret media token regions

## Current Capability Declarations

The built-in Gemma 4 cards now declare advanced capability sections so the runtime does not have to infer everything from scattered family checks.

Today that includes declarations for:

- reasoning toggle support
- reasoning format
- prompt renderer
- output parser
- native multimodal support
- tool-call format family

## Current Gaps

Phase 1 does **not** mean Gemma 4 is fully feature-complete yet.

Some follow-up work is intentionally tracked separately, including:

- reasoning budget support
- audio input support for variants that expose it upstream
- fuller Gemma 4 tool grammar support

## Debugging Gemma 4 Stalls

If Gemma 4 appears to hang during warmup or prefill, enable the MLX
hang-debug instrumentation before starting Skulk:

```bash
export SKULK_MLX_HANG_DEBUG=1
export SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS=10
uv run skulk
```

This causes the runner to log:

- warmup and prefill phase boundaries
- the selected prefill path
- whether the first prefill token was ever produced
- repeated Python stack traces while the stuck phase is still active

The older `EXO_MLX_HANG_DEBUG` names still work as compatibility fallbacks, but
new scripts and docs should prefer the `SKULK_` prefix.

For distributed Gemma 4 pipeline warmup specifically, Skulk now forces a
minimal synthetic prompt by design:

- no synthetic instructions
- a single user message with content `hello`

This is intentional. During debugging, richer synthetic warmup prompts were
observed to trigger the distributed warmup hang path on multi-node pipeline
setups. Pipeline models now stay on Skulk's explicit pipeline-prefill path even
for short one-chunk prompts, while distributed warmup keeps the synthetic prompt
minimal and suppresses unnecessary distributed progress polling for those tiny
prefills.

## Current Clustered Runtime Envelope

Gemma 4 currently has a narrower trusted clustered path than more generic text
models.

Today, the boring path is:

- Gemma 4-specific prompt rendering
- Gemma 4-specific thinking-channel parsing
- `SequentialGenerator` for distributed inference
- explicit pipeline prefill during warmup
- default KV cache as the baseline

This is conservative by design. It reflects the current support envelope that
has proven stable in cluster debugging.

Two debug-only warmup shaping env vars exist:

- `SKULK_DEBUG_WARMUP_REPEAT_COUNT`
- `SKULK_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS`

However, distributed pipeline warmup intentionally ignores them and stays on the
minimal sanity-check prompt. They remain useful for single-node investigation.

If you need to bypass synthetic warmup entirely during diagnosis, you can use:

```bash
export SKULK_SKIP_LLM_WARMUP=1
```

That bypass should be treated as a temporary debugging escape hatch only.
Distributed pipeline groups ignore it so one rank cannot skip warmup while
peers are still participating in warmup coordination.

## Why This Matters

Gemma 4 is the proof point for the model capability system.

If Skulk can express Gemma 4 behavior through model-card-backed capability declarations plus a resolved runtime profile, then future model-family support gets much cleaner:

- less hidden coupling
- fewer one-off patches
- more accurate dashboard and API behavior

## Related

- [Model Cards](../model-cards)
- [Model Capabilities](../model-capabilities)
