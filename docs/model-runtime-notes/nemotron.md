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

It also means there are two separate contracts to honor:

- prompt-time reasoning controls (`/think` and `/no_think`)
- stream-time reasoning markers (`<think>...</think>`)

If either one is mishandled, the user-visible chat experience becomes
misleading even when the model itself is behaving as designed.

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
- parser activation depended on tokenizer metadata exposing think markers
- in clustered runs, Nemotron could resolve to a token-delimited reasoning
  model without tokenizer metadata, so the parser never attached at all

Current fix:

- declare Nemotron Nano v2 quantizations as `thinking` + `thinking_toggle`
- mark the reasoning format as `token_delimited`
- make the generic token-delimited parser buffer partial markers so split and
  fused `<think>` / `</think>` tags are swallowed correctly
- fall back to literal `<think>` / `</think>` markers when the tokenizer does
  not expose reasoning metadata

### 2. Thinking-Off In The UI Still Produced `<think>`

Observed symptom:

- the dashboard thinking toggle was off
- rendered Nemotron prompts still ended with `Assistant\n<think>\n`

What was wrong:

- Skulk was passing `enable_thinking=False` as a generic chat-template kwarg
- Nemotron's actual contract is stricter: it expects literal `/think` or
  `/no_think` control text in system or user turns
- without a literal `/no_think`, Nemotron defaults to reasoning on

Current fix:

- for Nemotron only, explicit `enable_thinking=False` injects `/no_think`
- explicit `enable_thinking=True` injects `/think`
- if the conversation already contains one of those controls, Skulk leaves it
  alone instead of duplicating it

### 3. Batch Decode Exposed A Generic MLX Cache Compatibility Bug

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
- explicit thinking toggle translation: inject `/think` or `/no_think`
- thinking markers: token-delimited `<think>...</think>`
- toggle behavior: `/think` and `/no_think` via chat template
- parser activation: declared token-delimited reasoning can fall back to
  literal `<think>` markers when tokenizer metadata is missing
- KV backend baseline: `default`
- clustered prefill: explicit pipeline prefill when pipelined

## Known Failure Signatures

If any of the following reappear, Nemotron is probably off the trusted path:

- visible assistant output begins with `<think>`
- reasoning text appears in the normal `content` stream instead of reasoning
- trace logs show `stage=raw` and `stage=post-all-parsers` but never
  `stage=post-thinking-parser`
- the rendered prompt shows `Assistant\n<think>` while the dashboard toggle is off
- first batched decode crashes with `ArraysCache.make_mask(... return_array=...)`

## Useful Debugging

For live clustered debugging, enable the thinking-stream trace:

```bash
SKULK_TRACE_THINKING_STREAM=1 uv run skulk -v
```

The decisive lines are:

- `stage=raw`
- `stage=post-thinking-parser`
- `stage=post-all-parsers`
- `stage=chat-completions`

If `post-thinking-parser` never appears for Nemotron, the token-delimited
reasoning parser was never attached.

## Sources

- Official NVIDIA Nemotron Nano 9B v2 model card:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
