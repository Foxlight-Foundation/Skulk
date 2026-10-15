---
id: native-image-models
title: Native Image Models
sidebar_position: 6
---

<!-- Copyright 2026 Foxlight Foundation -->

Skulk serves Qwen Image 2.1, Qwen Image 2.1 Turbo, FLUX.2 Klein 4B,
Z-Image Turbo and SDXL through the on-demand `stable_diffusion_cpp` engine.
Each quantization or reference artifact has its own immutable model card.
SDXL metadata exposes its mixed-width UNet through `image.unetStructure`;
its generic dimensions count verified modules and maximum module output width
rather than provisional transformer placeholders.

## Choose and place a model

Find the cards through `GET /v1/models` or the dashboard's model selector.
Native cards expose an `image` object with their canvas limits, reference-image
limit, alpha support and sampling defaults. Their engine requirement is
`stable_diffusion_cpp`; it is not a list of GPU models permitted to run them.

| Model | Catalog aliases |
| --- | --- |
| Qwen Image 2.1 | `sd-cpp/Qwen-Image-2.1-Q4_K`, `sd-cpp/Qwen-Image-2.1-BF16` |
| Qwen Image 2.1 Turbo | `sd-cpp/Qwen-Image-2.1-Turbo-Q4_K_M`, `sd-cpp/Qwen-Image-2.1-Turbo-BF16` |
| FLUX.2 Klein 4B | `sd-cpp/FLUX.2-Klein-4B-Q4_0`, `sd-cpp/FLUX.2-Klein-4B-BF16` |
| Z-Image Turbo | `sd-cpp/Z-Image-Turbo-Q4_0`, `sd-cpp/Z-Image-Turbo-BF16` |
| SDXL | `sd-cpp/SDXL-Base-1.0-Q8_0-0.9VAE`, `sd-cpp/SDXL-Base-1.0-Reference` |

Use a Skulk build that implements the native image engine and typed image cards.
A catalog entry does not prove that an older Skulk installation has that runner.
Inspect [placement previews](api-guide.md#preview-valid-placements) or place the model
through the dashboard to resolve the actual available engine and memory fit.
When preparation is needed, the preview's `engine_install` describes its additional
engine download. Accepted placement installs the exact engine package and pinned
model components before load. Browsing the catalog does not download them.

Native inference currently runs on one host and one GPU. Managed packages cover
macOS ARM64 Metal, Linux ARM64/x86_64 CUDA, and Linux x86_64 Vulkan/ROCm.
Actual wheel ABI, CPU helper instructions, compiled GPU targets and runtime
requirements determine package eligibility. Model-specific memory fit uses the
selected artifact and runtime configuration. CPU-only native serving and
multi-node native image splitting are not implemented.

**Compatible untested hardware remains eligible.** Representative tests establish
observed behavior for their exact configuration; absent, failed or differently
scoped qualification does not exclude a separately compatible configuration.
See [Model and engine compatibility](engine-compatibility.md).

## Generate and edit

After placing the model, send an ordinary image-generation request:

```bash
curl http://localhost:52415/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sd-cpp/Qwen-Image-2.1-Turbo-Q4_K_M",
    "prompt": "a fox curled up in autumn leaves",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json"
  }'
```

The existing `/v1/images/generations` and multipart `/v1/images/edits` routes
serve these models. Requests retain OpenAI-style response formats and Skulk's
`advanced_params`, `stream` and `partial_images` extensions. Native requests
support terminal-only SSE with `stream=true` and zero previews; closing the
stream cancels the owned task. Follow the [image API contract](api-guide.md#image-generation-and-editing)
for request fields, reference ordering, limits and streaming events.

Qwen Turbo fixes eight steps, guidance 1 and an empty negative prompt; conflicting
overrides return 400. Z-Image does not accept reference images. Editing and alpha
requests follow the selected card's declared operations and the implemented
runner. PNG/WebP preserve supported alpha output; JPEG discards it. Qualification
results remain scoped evidence rather than permission to select a model.

Model users review and accept the applicable model license. The catalog exposes
license information independently of engine availability and hardware test coverage.
