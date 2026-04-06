---
id: deepseek-v32
title: DeepSeek V3.2 / DSML
sidebar_position: 3
---

<!-- Copyright 2025 Foxlight Foundation -->

DeepSeek V3.2 is the Phase 1 reference point for DSML-based prompt and parsing
behavior.

## Why DeepSeek V3.2 Is Special

DeepSeek V3.2 uses a DSML-oriented flow instead of the generic tokenizer chat
template and generic output parsing path.

That means the runtime needs to know:

- when to use DSML prompt encoding
- when to parse DSML-style reasoning and tool-call output
- when a model should be treated as a DeepSeek V3.2 family member even if the
  tokenizer metadata alone is not enough

## Current Capability Handling

Phase 1 supports DeepSeek V3.2 through resolved family defaults:

- DSML prompt renderer
- DeepSeek V3.2 output parser
- DSML tool-call format
- thinking-toggle aware family behavior

## Built-In Card Status

At the moment, Phase 1 documents DeepSeek V3.2 as a first-class resolved family,
but built-in declarative card coverage may still lag behind Gemma 4 and GPT-OSS.

That is acceptable for Phase 1 because:

- the runtime behavior is now covered by the resolved capability layer
- the docs make the gap explicit instead of leaving it hidden

## Why This Matters

DeepSeek V3.2 demonstrates that the capability system can represent a model
family whose key differentiation is not vision, but a custom prompt/parser
contract.

## Related

- [Model Cards](../model-cards)
- [Model Capabilities](../model-capabilities)
