<!-- Copyright 2025 Foxlight Foundation -->

# Qwen 3.5 Runtime Notes

## Why This Model Is Weird

Qwen 3.5 is easy to underestimate because some local cards only marked it as a
"thinking" model without saying that thinking is enabled by default and can be
explicitly disabled.

For `Qwen/Qwen3.5-9B`, the official model card says:

- Qwen 3.5 operates in thinking mode by default
- the model emits `<think>\n...\n</think>` reasoning traces before the final answer
- direct non-thinking responses require `enable_thinking=False`
- unlike newer Qwen families, Qwen 3.5 does not use `/think` or `/nothink`
  soft-switch prompt controls

That means Skulk should treat Qwen 3.5 as a token-delimited thinking model with
real toggle support, but the toggle lives in API/chat-template parameters, not
in literal prompt commands.

## What Failed In Skulk

Observed symptom:

- `mlx-community/Qwen3.5-9B-4bit` was not clearly surfaced as toggleable in the
  project metadata even though the model definitely reasons by default
- `mlx-community/Qwen3.5-9B-MLX-4bit` appeared as a separate model id without a
  built-in card, so it fell back to generic metadata despite being the same
  validated runtime family

What was wrong:

- the local `Qwen3.5-{9B,27B}` model cards only declared `thinking`
- they omitted `thinking_toggle` and did not include explicit reasoning
  metadata
- the `Qwen3.5-9B-MLX-4bit` alias did not have its own built-in card

Current fix:

- declare `thinking_toggle`
- mark the reasoning format as `token_delimited`
- set `supports_toggle = true`
- keep conservative defaults of `default_effort = "medium"` and
  `disabled_effort = "none"`
- add a built-in card for `mlx-community/Qwen3.5-9B-MLX-4bit` that matches the
  validated `Qwen3.5-9B-4bit` runtime metadata

## Current Trusted Runtime Envelope

For the Qwen 3.5 text variants currently covered by local cards:

- prompt renderer: tokenizer chat template
- thinking markers: token-delimited `<think>...</think>`
- toggle behavior: `enable_thinking=True/False` via template/API parameters
- no soft-switch `/think` / `/no_think` prompt controls
- `mlx-community/Qwen3.5-9B-MLX-4bit` is treated as the same runtime family as
  `mlx-community/Qwen3.5-9B-4bit`

## Sources

- Official Qwen 3.5 9B model card:
  https://huggingface.co/Qwen/Qwen3.5-9B
