# The vLLM engine (GPU concurrent serving)

vLLM is one of Skulk's **served** engines: instead of loading the model
in-process, the worker launches an external `vllm serve` subprocess and proxies
its OpenAI HTTP API, the same managed-server-plus-proxy shape as the
`llama_server` engine. Its continuous batching and paged attention support concurrent GPU workloads.
Latency and throughput depend on the model, GPU, context, request mix and memory
budget; benchmark the intended workload before choosing an engine.

It **coexists** with the other engines rather than replacing them: MLX owns
Apple Silicon, and the llama.cpp engines remain the GGUF paths. vLLM is
GPU-only in Skulk's scope (`vllm-cuda` on NVIDIA, `vllm-rocm` on AMD with a matching ROCm runtime).

## When a model runs on vLLM

Model support and live node support must agree:

- **The model card** declares and ranks the engines that can serve it in
  `compatible_backends`. A card that lists a vLLM backend is a vLLM candidate.
- **The node** advertises an installed or installable vLLM runtime matching its
  OS, CPU ABI, GPU runtime and operator policy. An engine-free eligible node
  reports the backend in `onDemandBackends`; placement installs it before load.
  Explicit `SKULK_VLLM_BIN` runtimes retain precedence, including their diagnostics.

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
trained context. An oversized prompt retains vLLM's bounded explanation and
the shared `context_length_exceeded` classification. If response headers have
already been sent, the API returns that error in the response body, following
the [API error contract](api-guide.md).

Placement, worker admission and server sizing reserve at least 3 GiB beyond
weights and cache for vLLM runtime and cold compilation workspace. Larger
models retain the larger proportional overhead estimate. This reserve covers
the first start as well as later starts that reuse compiled kernels. Native
MTP additionally reserves graph/recurrent workspace from the model geometry,
actual `SKULK_VLLM_MAX_CONCURRENT_REQUESTS` and card-pinned draft depth. The
conservative reference reserve is 8 GiB at width 5120, 64 layers, 32 sequences
and two draft tokens; it scales with those dimensions and does not shrink
with weight quantization. These remain estimates, rather than universal
measurements. The text runner passes `--language-model-only` so unused media
encoders cannot consume its text/cache reservation. Model modality claims are
retained; this describes the current runner limitation.

## Setup

A normal Linux GPU installation needs no separate vLLM setup. Select a published
vLLM model through the dashboard or `POST /place_instance`. The placement preview
and response expose `engine_install`, including the approximate 8 GiB first-use
download. The worker installs the runtime alongside the model download and holds
model loading until installation succeeds.

Managed runtimes use **vLLM 0.32.0** and uv-managed **Python 3.12.13** (including
compiler headers), in an isolated directory under Skulk's engines cache. vLLM
never joins Skulk's own Python environment. Three published recipe wheels bind
complete transitive dependency versions and hashes:

| Runtime | CPU | Minimum host libc | Torch |
| --- | --- | --- | --- |
| CUDA 12.9 | x86_64 / aarch64 | glibc 2.28 | 2.13.0+cu129 |
| ROCm 7.2.3 | x86_64 | glibc 2.39 | 2.13.0+git733fca1 |

Both recipes pin Transformers 5.19.0 and tokenizers 0.23.3. The upstream vLLM
wheels are mirrored byte for byte on `wheels.foxlight.ai`; the small
`skulk-vllm-runtime-cuda` / `skulk-vllm-runtime-rocm` wheels carry the complete
hashed dependency recipe. Other dependencies download from their recorded
upstream channels. Compatible hardware remains eligible without a qualification
record; actual model support also depends on the runtime's kernels and resources.
On a mixed AMD/NVIDIA host, `SKULK_VLLM_BACKENDS=cuda` or `rocm` selects
the corresponding managed recipe; installed tags describe only the verified
runtime, rather than every GPU the machine can see.

Allow at least **20 GiB free disk** for first-use installation, plus model storage
and compilation caches. The host supplies a compatible GPU driver. The upstream ROCm wheel also requires
the ROCm 7.2.3 userspace SDK and OpenMPI 4.x libraries (`libmpi_cxx.so.40` and
`libmpi.so.40`); Skulk checks these native prerequisites before advertising a
first-use download. Install the SDK through AMD's documented ROCm installation
[path for the host OS](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/install/quick-start.html); the recipe does not replace system GPU libraries.
Model-specific
JIT kernels can also require a CUDA compiler/toolkit. Installation checks the
recorded hashes, every locked dependency version, the interpreter and live GPU
before publishing a completion receipt. A failure reports `engine_install_failed`
and leaves earlier runtime versions available. Retry by placing the model again
after correcting the reported prerequisite.

Startup only discovers a completed cache and never downloads this engine. An
installed runtime remains available on an offline restart. A fresh offline node,
management-only node, or node with `SKULK_NO_ENGINE_AUTOPROVISION=1` does not
advertise a download it cannot perform. Explicit operator runtimes remain usable
under those policies when their normal runtime requirements are met.

`--with-vllm` remains an optional eager preparation flag. It uses the same managed
recipe and cache as first use:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash -s -- --with-vllm
```

Already have vLLM installed? Set `SKULK_VLLM_BIN` to its executable before launching
Skulk, and optionally declare `SKULK_VLLM_BACKENDS=cuda` or `rocm`. The override
wins; Skulk does not upgrade that environment or hide an invalid executable behind
a managed fallback. Unset it to use the current managed runtime. Older managed
recipe directories remain intact when a new runtime pin is introduced.

### Cache, parsers and API compatibility

The runner explicitly supplies `--kv-cache-dtype auto`. Placement and the worker
retain the shared full-precision cache estimate for this policy; stored weight
quantization remains a separate input. This upgrade does not expose vLLM's new
NVFP4 or ROCm four-bit cache modes. Such a mode needs matching runner controls and
shared memory accounting before Skulk can claim its savings.

Card-pinned `vllm_tool_call_parser` and `vllm_reasoning_parser` remain authoritative.
A checkpoint with the appropriate response template may explicitly select the new
`hf` parser; the upgrade does not migrate existing cards automatically. Upstream
endpoints are not automatically forwarded: Skulk retains its documented API,
request limits, streaming semantics and cancellation behavior.

### Known issue: H100 GPUs in Confidential Computing mode

From vLLM 0.29 every model runs on vLLM's newer (V2) model runner. On an H100
running in NVIDIA Confidential Computing mode, as in confidential cloud VMs,
that runner currently decodes about 3x slower and can return repeated
garbage tokens or crash its engine
([vllm-project/vllm#57680](https://github.com/vllm-project/vllm/issues/57680)).
Ordinary H100, H200 and other GPUs are not affected. Until the upstream fix
ships in a release Skulk pins, add this line to `~/.skulk/skulk.env` on such a
node and restart Skulk; the runner passes it to `vllm serve`, which then uses
its previous runner:

```bash
VLLM_USE_V2_MODEL_RUNNER=0
```

## Concurrency behavior and knobs

The vLLM runner **dispatches concurrently**: it keeps multiple requests in flight
against the one `vllm serve` process at once, which is what lets the server's
continuous batching actually engage and decode them together.

- `SKULK_VLLM_MAX_CONCURRENT_REQUESTS` (default 32) bounds how many
  generations the runner keeps in flight; requests beyond it queue in the
  runner's bounded pool. The runner also passes it as the server's
  `--max-num-seqs`, since vLLM never sees more concurrent sequences than the
  runner admits. A larger cap only costs memory, and vLLM 0.31 refuses to
  start a hybrid model (Qwen3.5 and Qwen3.6) whose cap exceeds its Mamba
  cache, one block per sequence.
- vLLM's share of GPU memory (`--gpu-memory-utilization`) is sized to the
  instance's placement. The runner passes the fraction of the device that
  holds the memory Skulk reserved for the model (weights with overhead, the KV
  cache for the served window, and a small floor), rounded up to the next
  ten-thousandth of the device total CUDA reports. vLLM spends whatever part of that share weights and runtime
  memory leave on KV cache. A vLLM model can therefore share a GPU with other
  models and stays within its reservation. The share never drops below the
  weights, the served window's KV cache at the model's own geometry (read
  from its `config.json`, counting every layer at its widest attention head)
  and about a gigabyte of vLLM's own runtime memory, so wide-head models and
  very small models still fit their window. It never exceeds the previous
  fixed share of 0.90.
- `SKULK_VLLM_GPU_MEMORY_UTILIZATION` pins a fixed share instead. Set it on a
  GPU dedicated to vLLM to give the KV cache the rest of the device, for more
  concurrent long requests; other models then find that GPU full. A model
  whose `config.json` does not name its layer count, head width and KV-head
  count, or a GPU whose memory total cannot be read, gets the fixed 0.90
  share with a warning in the node log.

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
pins `--max-num-batched-tokens` against the pinned sequence cap, since vLLM's
own default varies by version and hardware; shallow MTP depths keep vLLM's
batched default. DFlash speculators also JIT their kernels
through NVRTC at engine start and need a CUDA 12.8+ toolchain on the node.
See [Speculative Decoding](speculative-decoding.md) for the other engines'
mechanisms.

Upstream vLLM inputs live under `https://wheels.foxlight.ai/runtime-inputs/vllm/`,
selected directly by each hashed recipe. They are deliberately excluded from the
generic PEP 503 index so CUDA and ROCm distributions cannot be confused by a
package resolver. The Foxlight recipe packages remain indexed normally.
