---
id: gpt-oss
title: GPT-OSS
sidebar_position: 2
---

<!-- Copyright 2025 Foxlight Foundation -->

GPT-OSS is one of the first non-Gemma families that Phase 1 treats as a
first-class capability-system consumer.

## Why GPT-OSS Is Special

GPT-OSS needs more than coarse `thinking` metadata.

The runtime needs to know:

- that tool calling is expected
- that GPT-OSS output uses a family-specific tool-call parsing flow
- that the tokenizer and runtime should agree on the GPT-OSS tool format

## Current Capability Declarations

The built-in GPT-OSS cards now declare:

- tool-calling support
- GPT-OSS tool-call format
- GPT-OSS output parser selection

That moves GPT-OSS support out of “only infer it from the model id” territory
and into the same declarative-plus-resolved model that Gemma 4 uses.

## Current Gaps

Phase 1 does not attempt to make every GPT-OSS behavior card-driven yet.

It focuses on:

- parser selection
- tool-call format selection
- API/dashboard capability surfacing

## Why This Matters

GPT-OSS is the proof that the capability system is not just a Gemma 4 special
case. It shows that model-card-backed runtime behavior can cleanly cover a
second specialized family with different parser and tool semantics.

## Related

- [Model Cards](../model-cards)
- [Model Capabilities](../model-capabilities)
