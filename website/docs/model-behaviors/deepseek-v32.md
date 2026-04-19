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

The validated built-in `mlx-community/DeepSeek-V3.2-{4bit,8bit}` cards now
declare the same DSML contract that the runtime already used through resolved
family defaults:

- toggleable reasoning metadata
- DSML tool-call format
- DSML prompt rendering
- DeepSeek V3.2 output parsing

The resolved capability layer still matters for compatibility and future aliases,
but the trusted V3.2 quantizations no longer rely on hidden family inference
alone.

## Why This Matters

DeepSeek V3.2 demonstrates that the capability system can represent a model
family whose key differentiation is not vision, but a custom prompt/parser
contract.

## Related

- [Model Cards](../model-cards)
- [Model Capabilities](../model-capabilities)
