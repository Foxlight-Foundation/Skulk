---
id: gpt-oss
title: GPT-OSS
sidebar_position: 2
---

<!-- Copyright 2025 Foxlight Foundation -->

GPT-OSS is one of the first non-Gemma families treated as a first-class
capability-system consumer.

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
- builtin browser tools:
  - `web_search`
  - `open_url`
  - `extract_page`

That moves GPT-OSS support out of “only infer it from the model id” territory
and into the same declarative-plus-resolved model that Gemma 4 uses.

## Current Gaps

Current GPT-OSS support is intentionally narrow:

- Harmony parsing stays GPT-OSS-specific
- reasoning effort is explicit, but a true “thinking off” mode is not promised
- browsing is static fetch + extraction, not browser automation
- the dashboard executes GPT-OSS browser tool calls and feeds the result back
  into the conversation loop

## Why This Matters

GPT-OSS is the proof that the capability system is not just a Gemma 4 special
case. It shows that model-card-backed runtime behavior can cleanly cover a
second specialized family with different parser, reasoning, and tool semantics
without changing generic-model behavior.

## On both engines

GPT-OSS harmony parsing works on the MLX engine (token level) and on the
llama.cpp engine (string level). llama.cpp hands back already-detokenized text,
so the runner reparses the harmony channel markers from strings into a separate
reasoning channel, with a dependency-free parser that runs on non-Mac GPU nodes.
The same approach covers plain `<think>`-delimited reasoning models, so a
reasoning model's chain-of-thought lands in `reasoning_content` and the answer
stays clean regardless of which engine serves it.

## Related

- [Model Cards](../model-cards)
- [Model Capabilities](../model-capabilities)
