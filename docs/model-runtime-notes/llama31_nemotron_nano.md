<!-- Copyright 2025 Foxlight Foundation -->

# Llama 3.1 Nemotron Nano Runtime Notes

## What This Model Actually Is

`mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-*` is not just a plain Llama
checkpoint and not the same family as the newer Nemotron Nano v2 reasoning
models.

Per NVIDIA's model card, `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` is:

- derived from `Llama-3.1-Minitron-4B-Width-Base`
- ultimately descended from Meta Llama 3.1 8B through NVIDIA compression
- post-trained by NVIDIA for chat, reasoning, RAG, and tool-use style tasks
- exposed with explicit Reasoning On / Reasoning Off modes through the system
  prompt contract

So the best current mental model is:

- Llama-derived base
- NVIDIA Nemotron post-training
- reasoning-capable chat model with toggleable reasoning

## What Failed In Skulk

Observed symptom:

- the built-in cards only declared `["text"]`
- the model therefore looked like a plain non-thinking text model despite the
  upstream model card describing explicit reasoning modes

Current fix:

- mark the 4bit, 8bit, and bf16 cards as `thinking` + `thinking_toggle`

## Current Trusted Runtime Envelope

For `mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-{4bit,8bit,bf16}`:

- family stays `llama`
- tensor support stays `true`
- reasoning support is declared as present
- toggle support is declared as present

## What We Still Need To Learn

We have not yet verified from live traces:

- the exact stream markers, if any, used for reasoning content
- whether the toggle is best expressed via generic `enable_thinking` template
  kwargs or a more specific system-prompt convention
- whether tool-calling should be declared explicitly in Skulk metadata

Until that is validated, this model should be treated as:

- definitely reasoning-capable
- probably straightforward to run
- not yet fully characterized in clustered behavior

## Sources

- Official NVIDIA model card:
  https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-4b-v1_1/modelcard
- MLX conversion page:
  https://huggingface.co/mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit
