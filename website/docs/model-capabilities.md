---
id: model-capabilities
title: Model Capabilities
sidebar_position: 5
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk supports a wide range of models, but not every model behaves the same way.

Some models need:

- custom prompt rendering
- non-generic reasoning delimiters
- specialized tool-call formats
- native multimodal execution
- model-specific API controls

The model capability system exists so Skulk can support those differences without turning the runtime into a pile of hidden one-off checks.

## The Two Layers

Skulk now treats capability handling as two related layers:

### 1. Declarative model card

The model card stores broad static metadata plus optional advanced capability sections.

This is the durable, syncable, editable source of truth.

### 2. Resolved runtime profile

At runtime, Skulk resolves the card plus tokenizer/model-family facts into a normalized capability profile.

That resolved profile answers questions like:

- should this request use a custom prompt renderer?
- what reasoning format should be expected?
- which output parser should run?
- what defaults should be used when thinking is toggled on or off?

## Why Not Only One Layer?

If Skulk used only model cards directly at runtime:

- execution code would be full of `None` checks and partial fallbacks
- every hot path would need to re-interpret optional metadata
- backward compatibility would be harder to preserve cleanly

If Skulk used only hard-coded runtime profiles:

- custom cards would not be expressive enough
- API and dashboard metadata would drift away from runtime behavior
- model support would become scattered again

The combined approach gives us:

- one declarative source of truth
- one normalized execution contract

## Current Phase 1 Goal

Phase 1 is about building the capability spine that UI and API work can depend on.

That means:

- cards can declare advanced capability sections
- old cards still work
- runtime behavior for key decisions is capability-driven
- model metadata exposed by the API can begin surfacing refined behavior to clients

Phase 1 specifically focuses on:

- reasoning/thinking defaults
- prompt renderer selection
- output parser selection

## Phase 2 Thinking Contract

Phase 2 keeps the existing public controls:

- `enable_thinking`
- `reasoning_effort`

But their behavior is now explicitly model-aware through `resolved_capabilities`.

### Toggleable reasoning models

If `resolved_capabilities.supports_thinking_toggle` is `true`:

- `enable_thinking=true` enables thinking using the model profile's default effort unless an explicit non-disabled effort is provided
- `enable_thinking=false` disables thinking using the profile's disabled effort
- `reasoning_effort="none"` also disables thinking

### Non-toggleable reasoning models

If a model supports reasoning but does not support thinking toggle:

- clients should not offer a toggle
- explicit toggle overrides are normalized away
- requests fall back to the model's supported default behavior

This keeps the public API stable without pretending every reasoning-capable model can switch on and off cleanly.

## Fallback Behavior

If a model card does not define advanced sections, Skulk should still work.

The runtime resolves that model to a conservative generic profile:

- generic prompt rendering
- generic parser behavior
- no assumptions about special reasoning controls
- no assumptions about special modalities or tool grammars

This is critical for compatibility with existing built-in and custom cards.

## Precedence Rules

The resolved runtime profile follows a simple precedence model:

1. explicit advanced fields from the model card win
2. model-family defaults fill in known behavior for important families
3. generic fallback preserves compatibility for everything else

Phase 1 intentionally keeps those heuristics conservative. The goal is not to
guess every possible advanced feature, but to preserve current behavior while
letting extended cards make support more precise.

## What This Enables

Once the capability spine exists, Skulk can evolve cleanly toward:

- model-aware thinking controls
- reasoning budget support
- audio modality support
- richer tool grammars
- safer dashboard controls based on real support instead of guesswork

## Related

- [Model Cards](model-cards)
- [Architecture Overview](architecture)
- [Gemma 4 behavior notes](model-behaviors/gemma4)
