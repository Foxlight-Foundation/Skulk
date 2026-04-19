<!-- Copyright 2025 Foxlight Foundation -->

# DeepSeek V3.2 Runtime Notes

## What It Is

These notes cover the validated DSML path for:

- `mlx-community/DeepSeek-V3.2-4bit`
- `mlx-community/DeepSeek-V3.2-8bit`

DeepSeek V3.2 is not just another token-delimited thinking model. In Skulk it
uses a dedicated prompt renderer plus a DSML-aware output parser for reasoning
and tool calls.

## What Is Unusual

- prompt construction goes through the MLX DeepSeek V3.2 DSML encoder instead
  of the generic tokenizer chat template
- tool calls are emitted inside DSML `<｜DSML｜function_calls>` blocks that may
  arrive split across multiple stream chunks
- thinking traces still need to be routed away from visible assistant text even
  while DSML parsing is active
- the MLX prompt encoder can emit an orphan assistant + `</think>` sequence
  that needs normalization before generation starts

## What Failed In Skulk

### 1. Runtime Support Lived Mostly In Family Fallbacks

Observed symptom:

- DeepSeek V3.2 behaved correctly only because the resolved capability layer
  recognized the family and patched in DSML defaults
- the built-in cards themselves did not say that DSML rendering, DSML tool
  parsing, and toggleable thinking were part of the trusted contract

Why that mattered:

- the validated runtime contract stayed implicit instead of visible in the card
  corpus
- model metadata in `/v1/models` could not show the declared DSML sections for
  these built-in cards
- internal docs lagged behind the actual runtime path

Current fix:

- add explicit `[reasoning]`, `[tooling]`, and `[runtime]` sections to the
  validated DeepSeek V3.2 quantizations
- keep the resolved-family fallback in place for compatibility and future alias
  coverage

### 2. DSML Markers Can Cross Token Boundaries

Observed symptom:

- tool-call markers such as `<｜DSML｜function_calls>` or invoke blocks can be
  split across chunks during streaming
- naive per-token parsing would either leak raw DSML text or miss the tool call

Current fix:

- accumulate text until the DSML marker is complete
- parse the finished function-calls block as one unit
- fall back to plain text only if DSML parsing truly fails

### 3. DSML Prompt Encoding Needed A Small Normalization Shim

Observed symptom:

- the upstream DeepSeek V3.2 encoder can produce an orphan assistant-prefixed
  thinking terminator

Current fix:

- normalize that prompt fragment before it reaches generation so the thinking
  envelope stays structurally valid

## Current Trusted Runtime Envelope

For the validated DeepSeek V3.2 quantizations listed above:

- prompt renderer: `dsml`
- output parser: `deepseek_v32`
- tool-call format: `dsml`
- thinking behavior: toggleable
- reasoning markers: token-delimited think blocks handled by the DeepSeek V3.2
  parser path
- tool-call handling: DSML `function_calls` accumulation plus structured parse

## Known Failure Signatures

If any of these show up again, DeepSeek V3.2 is probably off the trusted path:

- raw `<｜DSML｜function_calls>` text appears in visible assistant output
- tool calls stop being surfaced as structured tool requests
- visible assistant output contains reasoning text that should have stayed in
  the thinking stream
- DeepSeek V3.2 requests fall back to the generic tokenizer chat template

## Useful Debugging

DeepSeek V3.2 issues usually reduce to one of two questions:

1. did the request take the DSML prompt-renderer path?
2. did the stream take the DeepSeek V3.2 parser path?

Focused tests that currently exercise those contracts:

- `tests/test_gemma_vision.py`
- `src/exo/shared/tests/test_model_capabilities.py`
- `src/exo/worker/tests/unittests/test_runner/test_finish_reason_sse.py`
