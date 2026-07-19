# The vLLM engine (GPU concurrent serving)

vLLM is one of Skulk's **served** engines: instead of loading the model
in-process, the worker launches an external `vllm serve` subprocess and proxies
its OpenAI HTTP API, the same managed-server-plus-proxy shape as the
`llama_server` engine. It exists for one reason: vLLM's continuous batching and
paged attention hold latency flat and grow aggregate throughput under
**concurrent load**, where a single-stream engine collapses. In a benchmark on
an A100 at 64-way concurrency, llama.cpp's time-to-first-token reached ~31
seconds while vLLM's stayed at ~0.5 seconds.

It **coexists** with the other engines rather than replacing them: MLX owns
Apple Silicon, and the llama.cpp engines remain the GGUF paths. vLLM is
GPU-only in Skulk's scope (`vllm-cuda` on NVIDIA, `vllm-rocm` on AMD CDNA).

## When a model runs on vLLM

Two things have to line up, the same rule as every engine:

- **The model card** declares and ranks the engines that can serve it in
  `compatible_backends`. A card that lists a vLLM backend is a vLLM candidate.
- **The node** advertises a vLLM backend, which it does only when
  `SKULK_VLLM_BIN` points at a usable `vllm` CLI and a GPU backend resolves
  (declared via `SKULK_VLLM_BACKENDS`, or inferred from the observed GPU
  vendor). A node without the binary is never a placement candidate for vLLM
  cards.

When several nodes qualify, placement prefers the card's higher-ranked
backend, so the card is where the "this model is better on vLLM than on
llama.cpp here" judgment lives.

## Current scope

The engine serves **single-node, streamed text generation**. Its boundaries
are enforced loudly rather than degraded silently:

- **Tool calling is rejected** with a clear error: retry without `tools` or
  use a model carded for `llama_cpp` / `llama_server`.
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
before launching Skulk and the node advertises the engine; nothing else is
required.

## Concurrency behavior and knobs

Unlike the in-process runners, which serialize one generation at a time, the
vLLM runner **dispatches concurrently**: it keeps multiple requests in flight
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

vLLM's win is **concurrency, not single-stream speed**. Under concurrent load
it holds time-to-first-token flat and grows aggregate throughput where the
single-stream engines queue and collapse. For one request at a time, the
in-process engines can be as fast or faster depending on the GPU generation
(on GPUs without native FP4 support, in particular, a single stream can favor
them). Skulk keeps the engines side by side precisely so the choice is made
per model and per hardware rather than by ideology; the model card's backend
ranking is where that choice is recorded.

vLLM does not run Skulk's speculative decoding; see
[Speculative Decoding](speculative-decoding.md) for which engines do.
