# The vLLM engine (GPU concurrent serving)

vLLM is one of Skulk's **served** engines: instead of loading the model
in-process, the worker launches an external `vllm serve` subprocess and proxies
its OpenAI HTTP API, the same managed-server-plus-proxy shape as the
`llama_server` engine. Its continuous batching and paged attention support concurrent GPU workloads.
Latency and throughput depend on the model, GPU, context, request mix and memory
budget; benchmark the intended workload before choosing an engine.

It **coexists** with the other engines rather than replacing them: MLX owns
Apple Silicon, and the llama.cpp engines remain the GGUF paths. vLLM is
GPU-only in Skulk's scope (`vllm-cuda` on NVIDIA, `vllm-rocm` on AMD CDNA).

## When a model runs on vLLM

Model support and live node support must agree:

- **The model card** declares and ranks the engines that can serve it in
  `compatible_backends`. A card that lists a vLLM backend is a vLLM candidate.
- **The node** advertises a vLLM backend, which it does only when
  `SKULK_VLLM_BIN` points at a usable `vllm` CLI and a GPU backend resolves
  (declared via `SKULK_VLLM_BACKENDS`, or inferred from the observed GPU
  vendor). A node without the binary is never a placement candidate for vLLM
  cards.

Signed engine-support claims can also establish compatibility for an exact
artifact, engine build, capability and hardware class. Skulk applies its runner
limits after that match. See [Model capabilities](model-capabilities.md).

When several nodes qualify, placement prefers the card's higher-ranked
backend, so the card is where the "this model is better on vLLM than on
llama.cpp here" judgment lives.

## Current scope

The engine serves **single-node text generation with tool calling**. Its
boundaries are enforced loudly rather than degraded silently:

- **Tool calling works on parser-pinned cards.** A card that pins
  `vllm_tool_call_parser` in its `[runtime]` section (the registry's Qwen2.5
  vLLM cards pin `hermes`) launches the server with vLLM's native
  tool-call parsing, and a tool-enabled request runs non-streamed so the
  caller receives the assembled call, the same shape as the llama.cpp
  engines. A tool attempt cut short by `max_tokens` reports `length`
  instead of surfacing an incomplete call. Cards without a pinned parser
  reject tool requests with a clear error rather than silently dropping
  them; there is no family-default fallback, because one model family can
  span generations with different tool wire formats.
- **Reasoning is split on parser-pinned cards.** vLLM only separates
  `reasoning_content` from `content` when the server is launched with a
  reasoning parser, so a card pins `vllm_reasoning_parser` in its
  `[runtime]` section (a Muse Glimmer card pins `muse_glimmer` for both
  parsers) and the runner passes it as `--reasoning-parser`. Explicit only,
  no family fallback, for the same reason as the tool-call parser. An
  unpinned reasoning model streams its thinking inline. Muse Glimmer's
  always-on reasoning is steered through the template's
  `reasoning_strength` kwarg, which the runner derives from the request's
  `reasoning_effort`.
- **Per-token logprobs are rejected** with a clear error: the OpenAI SSE proxy
  does not surface them, and Skulk refuses to silently omit what you asked
  for.
- **Multi-node placement is refused.** vLLM's own tensor and pipeline
  parallelism are not wired into Skulk placement.
- **Reasoning is best-effort.** Thinking controls (`enable_thinking`,
  `reasoning_effort`) are forwarded so the model behaves as requested, and
  separated reasoning deltas are parsed into thinking chunks when the server
  emits them; on models where vLLM needs a family-specific reasoning parser to
  split thinking from content, the thinking text can arrive inline in the
  content stream instead.

The served context window is sized to the memory the cluster admitted for the
instance (passed as `--max-model-len`), never blindly to the model's full
trained context.

## Setup

The easiest path is the one-command installer's flag on an NVIDIA Linux node:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash -s -- --with-vllm
```

This creates a **dedicated virtual environment** at `~/.skulk/vllm-env` with
Skulk's validated dependency matrix (a pinned vLLM release, a compatible
`transformers`, and the matching CUDA torch backend; several GB of wheels) and
records `SKULK_VLLM_BIN=~/.skulk/vllm-env/bin/vllm` in `~/.skulk/skulk.env`,
which the service wrappers source. The separate venv is not an accident: Skulk's
own environment and vLLM currently require conflicting dependency versions, so
vLLM must never be installed into Skulk's venv. Skulk drives its CLI purely as
an external process.

Already have vLLM installed some other way? Point `SKULK_VLLM_BIN` at its CLI
before launching Skulk. Confirm the supported GPU backend, engine build, model
parser pins and compiler/Python development prerequisites with `skulk doctor`;
a CLI path alone does not prove that the server can load a given model.

## Concurrency behavior and knobs

The vLLM runner **dispatches concurrently**: it keeps multiple requests in flight
against the one `vllm serve` process at once, which is what lets the server's
continuous batching actually engage and decode them together.

- `SKULK_VLLM_MAX_CONCURRENT_REQUESTS` (default 32) bounds how many
  generations the runner keeps in flight; requests beyond it queue in the
  runner's bounded pool. This is a client-side admission bound, not the
  server's batch width (vLLM batches up to its own `--max-num-seqs`).
- `SKULK_VLLM_GPU_MEMORY_UTILIZATION` (default 0.90) sets the fraction of GPU
  VRAM vLLM may use for weights plus KV cache, passed through as
  `--gpu-memory-utilization`.

Operationally: server startup on a large model can take a couple of minutes
(weight load, compilation, CUDA-graph capture) and is allowed a generous health
deadline; the server's own log is written to a deterministic per-runner file
under the system temp directory for postmortems. Cancelling a request aborts
its proxied HTTP connection, which stops the server-side generation; if the
runner process itself dies, the kernel reaps the `vllm serve` child so it never
orphans GPU memory.

## Honest performance framing

Continuous batching can improve aggregate throughput under concurrency, but
it does not promise constant latency as load rises. Single-request performance
also depends on quantization, hardware and speculative decoding. Use the card's
backend ranking with measurements from the workload you expect to serve.

vLLM runs card-driven speculative decoding for checkpoints that ship
native multi-token-prediction heads (Qwen3.6 among them): the card's
`vllm_spec_method = "mtp"` and `vllm_spec_num_tokens` map to vLLM's
`--speculative-config`, engaging the model's own prediction heads with no
separate draft model. Measured on an A100-80GB, this roughly doubles
single-stream decode on Qwen3.6-27B-FP8 (2.01x at depth 2, 77%
acceptance). Vendor schemes with a separately published speculator use the
same fields: `vllm_spec_method = "dflash"` plus `vllm_spec_draft_repo`
pairs Poolside's Laguna models with their block-parallel DFlash drafter
(vLLM 0.25.1 or later), with the drafter repo resolved through vLLM's own
Hugging Face cache at engine start (measured 1.35x single-stream on an
A100-80GB; gains on other GPUs require measurement). Deep speculative depths need more scheduler budget
than vLLM's defaults provide, so for carded depths of 8 or more the runner
raises `--max-num-batched-tokens` automatically; shallow MTP depths run
with vLLM's defaults untouched. DFlash speculators also JIT their kernels
through NVRTC at engine start and need a CUDA 12.8+ toolchain on the node.
See [Speculative Decoding](speculative-decoding.md) for the other engines'
mechanisms.
