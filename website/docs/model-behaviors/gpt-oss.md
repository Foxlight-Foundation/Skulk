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

- reasoning support with a default effort of `medium`
- non-toggleable reasoning semantics
- tool-calling support
- GPT-OSS tool-call format
- GPT-OSS output parser selection
- the generic `web_search` builtin tool contract

That moves GPT-OSS support out of “only infer it from the model id” territory
and into the same declarative-plus-resolved model that Gemma 4 uses.

## Current Gaps

Current GPT-OSS support is intentionally narrow:

- Harmony parsing stays GPT-OSS-specific
- reasoning effort is explicit, but a true “thinking off” mode is not promised
- browsing is search-style tool use, not browser automation
- the dashboard executes GPT-OSS `web_search` tool calls and feeds the result
  back into the conversation loop

## Why This Matters

GPT-OSS is the proof that the capability system is not just a Gemma 4 special
case. It shows that model-card-backed runtime behavior can cleanly cover a
second specialized family with different parser, reasoning, and tool semantics
without changing generic-model behavior.

## Related

- [Model Cards](../model-cards)
- [Model Capabilities](../model-capabilities)
