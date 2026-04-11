<!-- Copyright 2025 Foxlight Foundation -->

# Nemotron Runtime Notes

## Why This Model Is Weird

The Nemotron v2 chat models are reasoning models that hide part of their work
behind token-delimited thought traces, but they do not use the same runtime
shape as Gemma 4 or DeepSeek.

For `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, the official model card says:

- reasoning is on by default when no signal is provided
- `/think` and `/no_think` can be placed in system or user turns
- the chat template inserts `<think>\n` when reasoning is enabled
- the chat template inserts `<think></think>` when reasoning is disabled

That means Skulk has to treat Nemotron as a real thinking model even though its
reasoning markers are just regular text tokens rather than a custom channel
format.

## What Failed In Skulk

### 1. Thinking Leaked Into Visible Assistant Text

Observed symptom:

- chat output includes raw `<think>` markers
- reasoning text is streamed as if it were normal assistant content

What was wrong:

- the built-in Nemotron model cards did not declare reasoning or toggle support
- the generic token-delimited parser only handled marker tokens when they
  arrived as exact standalone chunks
- split or fused markers like `<th` + `ink>` or `</think>Answer` leaked through

Current fix:

- declare Nemotron Nano v2 quantizations as `thinking` + `thinking_toggle`
- mark the reasoning format as `token_delimited`
- make the generic token-delimited parser buffer partial markers so split and
  fused `<think>` / `</think>` tags are swallowed correctly

### 2. Batch Decode Exposed A Generic MLX Cache Compatibility Bug

Observed symptom:

- `ArraysCache.make_mask() got an unexpected keyword argument 'return_array'`

This turned out to be infrastructure rather than a Nemotron-only semantic bug.
Nemotron's batch path exercised an older `mlx_lm` cache signature that Skulk had
not patched yet.

Current fix:

- class-level compatibility shim for `ArraysCache.make_mask`

## Current Trusted Runtime Envelope

For `mlx-community/NVIDIA-Nemotron-Nano-9B-v2-{4bits,6bit}`, the currently
trusted baseline is:

- prompt renderer: tokenizer chat template
- thinking markers: token-delimited `<think>...</think>`
- toggle behavior: `/think` and `/no_think` via chat template
- KV backend baseline: `default`
- clustered prefill: explicit pipeline prefill when pipelined

## Known Failure Signatures

If any of the following reappear, Nemotron is probably off the trusted path:

- visible assistant output begins with `<think>`
- reasoning text appears in the normal `content` stream instead of reasoning
- first batched decode crashes with `ArraysCache.make_mask(... return_array=...)`

## Sources

- Official NVIDIA Nemotron Nano 9B v2 model card:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
