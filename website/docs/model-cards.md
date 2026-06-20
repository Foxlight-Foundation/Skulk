---
id: model-cards
title: Model Cards
sidebar_position: 4
---

<!-- Copyright 2025 Foxlight Foundation -->

Model cards are Skulk's durable source of truth for model metadata.

They are how Skulk knows things like:

- what a model is called
- what task types it supports
- how large it is for placement and download planning
- whether it supports tensor sharding
- whether it has vision support
- whether it declares advanced model-specific behavior

## What Model Cards Do

Model cards sit at the boundary between static metadata and runtime behavior.

They drive:

- placement and memory calculations
- model browsing in the API and dashboard
- custom model registration
- modality hints such as vision
- advanced capability declarations for model-specific runtime behavior

## Where They Live

Built-in cards are shipped in:

- [`resources/inference_model_cards`](https://github.com/Foxlight-Foundation/Skulk/tree/main/resources/inference_model_cards)
- [`resources/image_model_cards`](https://github.com/Foxlight-Foundation/Skulk/tree/main/resources/image_model_cards)
- [`resources/embedding_model_cards`](https://github.com/Foxlight-Foundation/Skulk/tree/main/resources/embedding_model_cards)

Custom cards are stored under the user data directory and synced through the cluster event flow.

## The card interface (source of truth)

The authoritative definition of the model-card interface is the `ModelCard`
type in
[`src/skulk/shared/models/model_cards.py`](https://github.com/Foxlight-Foundation/Skulk/tree/main/src/skulk/shared/models/model_cards.py).
Every field is documented in that model, and the exhaustive, always-current field
reference is the generated API schema (`ModelCard` and its nested
`PlacementCardConfig` / `RuntimeCapabilityCardConfig` / `VisionCardConfig` /
`ReasoningCardConfig` / `ModalitiesCardConfig` / `ToolingCardConfig` /
`ComponentInfo`) in the [API reference](api). This page is the curated narrative;
when in doubt about an exact field, the schema is canonical.

Cards are camelCase on the wire and strict (unknown fields are rejected), so every
node in a cluster must run the same Skulk version.

## Core Fields

### Identity and size

- `model_id`
  - Hugging Face / MLX model identifier
- `storage_size`
  - total model size used for store/download/placement planning
- `n_layers`
  - number of transformer layers used for pipeline sharding
- `hidden_size`
  - hidden dimension used for placement and compatibility checks
- `num_key_value_heads`
  - optional KV head count for tensor compatibility decisions
- `gguf_file`
  - for GGUF (llama.cpp) models only: the repo-relative weights file the runner loads (the selected quant's first shard), resolved once at card creation; `null` for safetensors/MLX cards
- `components`
  - for multi-component models (such as a diffusion stack): the per-component weight layout; `null` for a single-weights model

### Runtime and placement

- `supports_tensor`
  - whether tensor-style placement is allowed (GGUF/llama.cpp cards set this `false`)
- `tasks`
  - supported task families such as `TextGeneration`, `TextEmbedding`, or image tasks
- `trust_remote_code`
  - whether the loader may enable remote-code behavior for this model
- `uses_cfg`
  - whether the model uses classifier-free guidance (relevant to some image/diffusion models)

### Catalog metadata

- `family`
  - coarse family label such as `gemma`, `qwen`, `deepseek`
- `quantization`
  - human-facing quantization label
- `base_model`
  - display-friendly base model name
- `context_length`
  - advertised context length if known
- `capabilities`
  - coarse capability list such as `text`, `vision`, `thinking`, `embedding`

These coarse capabilities remain useful for browsing, badges, and basic compatibility, but they are not expressive enough for model-specific runtime behavior on their own.

## Vision Section

`[vision]` is the existing structured section for multimodal text-generation models.

Fields include:

- `image_token_id`
  - token used to represent image slots in prompt tokenization
- `model_type`
  - MLX-VLM model family identifier such as `gemma4`
- `weights_repo`
  - optional alternate weights repository for the vision tower
- `image_token`
  - optional literal image token string
- `processor_repo`
  - optional alternate processor repository
- `boi_token_id`
  - optional begin-of-image token id
- `eoi_token_id`
  - optional end-of-image token id

## Placement Section

`[placement]` declares where a model is allowed to run and which backend is
preferred. It is what the planner reads to route a model to suitable nodes, and
it is how a heterogeneous cluster keeps each model on hardware that can run it.

Backends are named `<engine>-<compute>` tags (for example `mlx-metal`,
`llama_cpp-vulkan`, `llama_cpp-rocm`, `llama_cpp-cpu`). The engine selects the
worker runner; the compute names the accelerator.

- `compatible_backends`
  - the hard filter: the set of backend tags this model may run on. The planner excludes any node whose advertised backends do not intersect this set. An MLX card lists MLX tags (default `{"mlx"}`); a GGUF card lists the llama.cpp tags. This is what keeps an MLX model off an AMD node and a GGUF model off a Mac.
- `backend_preference`
  - the soft score: an ordered list of preferred tags. When several compatible nodes qualify, the planner prefers the node whose backend ranks earliest, with graceful fallback to the rest. This lets a card say "fastest on Vulkan, but ROCm is fine."
- `min_vram_gib`
  - optional minimum accelerator memory a node must have to be eligible.
- `max_context_tokens`
  - optional cap on the admission context for this model, independent of the model's trained context length.

## Extended Capability Sections

Skulk now supports optional structured sections that declare refined model behavior.

Existing cards do **not** need these sections. If they are absent, Skulk falls back to generic behavior.

### `[reasoning]`

Declares advanced reasoning behavior:

- `supports_toggle`
  - whether thinking/reasoning can be explicitly enabled or disabled
- `supports_budget`
  - whether the model supports a reasoning budget control
- `format`
  - reasoning marker format such as `channel_delimited` or `token_delimited`
- `default_effort`
  - reasoning effort used when thinking is enabled without an explicit effort
- `disabled_effort`
  - reasoning effort used when thinking is explicitly disabled

### `[modalities]`

Declares refined modality support:

- `supports_audio_input`
  - whether the model supports audio input
- `supports_native_multimodal`
  - whether the model uses a native multimodal path rather than generic text-only prompting

### `[tooling]`

Declares tool-calling behavior:

- `supports_tool_calling`
  - whether tool calling is supported
- `builtin_tools`
  - optional list of builtin platform tool contracts such as `web_search`, `open_url`, or `extract_page`
- `tool_call_format`
  - expected tool-call output format such as `generic`, `gemma4`, `gpt_oss`, or `dsml`

### `[runtime]`

Declares runtime integration preferences:

- `prompt_renderer`
  - prompt renderer to use, such as `tokenizer`, `gemma4`, or `dsml`
- `output_parser`
  - output parser to use, such as `generic`, `gemma4`, `gpt_oss`, or `deepseek_v32`
- `metal_fast_synch`
  - per-model override for the MLX `MLX_METAL_FAST_SYNCH` flag; set to `false` for models that deadlock under FAST_SYNCH on the ring backend
- `mtp_heads`
  - set to `true` when the model has native MTP prediction heads available via sidecar; required alongside `mtp_sidecar_repo` to enable speculative decoding
- `mtp_max_depth`
  - maximum draft depth the MTP heads support; start at `1` for Apple Silicon (deeper values rarely amortize on Metal due to near-linear verify-pass scaling)
- `mtp_sidecar_repo`
  - Hugging Face repo ID containing the published `mtp.safetensors` sidecar (e.g. `"FoxlightAI/qwen3-5-7b-instruct-mtp-q4k"`); produced by SWP from the original BF16 checkpoint
- `mtp_norm_convention`
  - how the MTP heads normalize hidden states (`zero_centered` or `actual_scale`); must match how the sidecar was produced
- `mtp_concat_order`
  - the order the MTP heads concatenate the embedding and hidden state (`embed_first` or `hidden_first`); must match the sidecar
- `speculative_multi_node`
  - set to `false` to forbid speculative decoding when the model is sharded across multiple nodes (it stays single-node speculative); the runner and the generation loop read this to make the same rank-symmetric decision
- `assistant_model_repo`
  - Hugging Face repo of a small companion model used as an external drafter for speculative decoding (the Gemma 4 path), as opposed to native MTP heads

## MTP Speculative Decoding

Some models include native multi-token prediction (MTP) heads baked into their checkpoint weights. Skulk uses these heads for speculative decoding: the MTP heads draft candidate tokens cheaply, a full forward pass verifies them, and accepted tokens are emitted in bulk, substantially increasing throughput with no accuracy loss.

### Why a sidecar

Standard quantization pipelines (including mlx-lm's `sanitize()`) strip `mtp.*` tensor keys at conversion time. The MLX-quantized checkpoint you download from Hugging Face typically does not contain MTP weights.

SWP (Skulk Weights Publisher) solves this by re-extracting the `mtp.*` tensors from the original BF16 checkpoint, quantizing only those tensors, and publishing the result as `mtp.safetensors` to a dedicated sidecar repo on Hugging Face. This is the same pattern Skulk uses for vision encoder weights.

### How it works

When a model card declares `mtp_heads = true` and `mtp_sidecar_repo`, Skulk:

1. Downloads `mtp.safetensors` from the sidecar repo alongside the base model weights.
2. Loads the sidecar at model load time and makes the weights available to the runner.
3. Uses the MTP heads during generation for speculative decoding (on runners that support it).

If the sidecar is declared but the file is not found locally, Skulk logs a warning and continues with standard autoregressive generation.

### Models with native MTP heads

- Qwen3.5 variants (7B, 14B, 32B, 72B)
- Qwen3.6 variants
- DeepSeek V3 / R1

Gemma 4 does **not** use native MTP heads; it uses an external drafter model (`assistant_model_repo`) for speculative decoding, which is a separate feature.

### Adding MTP to a card

```toml
[runtime]
mtp_heads = true
mtp_max_depth = 1
mtp_sidecar_repo = "FoxlightAI/qwen3-5-7b-instruct-mtp-q4k"
```

The sidecar repo must be published by SWP before adding these fields. See the [SWP documentation](https://foxlight-foundation.github.io/skulk-weights-publisher/) for how sidecars are produced and published.

## Declarative vs Resolved

The model card is the **declarative** capability source.

At runtime, Skulk resolves the card plus tokenizer/model-family facts into a normalized execution profile. That resolved profile is what prompt rendering, reasoning defaults, and output parsing consume.

That gives Skulk three good properties:

- old cards remain valid
- advanced cards unlock refined behavior
- runtime code can rely on normalized values instead of ad hoc optional checks

## Resolution Precedence

When Skulk resolves a runtime capability profile, it uses this order:

1. explicit advanced model-card declarations
2. conservative family/model heuristics
3. generic fallback behavior

That means a custom or built-in card can refine behavior without breaking old
cards that only declare coarse metadata.

## Extended Card Example

This is a minimal example of a custom card that opts into refined runtime
behavior:

```toml
model_id = "custom/gemma-compatible"
n_layers = 10
hidden_size = 1024
supports_tensor = false
tasks = ["TextGeneration"]
family = "gemma"
capabilities = ["text", "vision", "thinking"]

[storage_size]
in_bytes = 1073741824

[reasoning]
supports_toggle = true
format = "channel_delimited"
default_effort = "medium"
disabled_effort = "none"

[modalities]
supports_native_multimodal = true

[tooling]
tool_call_format = "gemma4"

[runtime]
prompt_renderer = "gemma4"
output_parser = "gemma4"
```

The card stays declarative. Skulk still resolves it into a normalized runtime
profile before execution code consumes it.

## When to Extend a Card

Extend a card when:

- the model needs special prompt rendering
- the model uses a non-generic reasoning format
- the model supports modalities or controls that generic metadata cannot express
- the dashboard or API needs to expose richer behavior safely

For concrete examples, see [Model Capabilities](model-capabilities) and the per-family notes in [Model Behaviors](model-behaviors/gemma4).
