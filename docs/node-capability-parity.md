# Node Capability Parity

Status: design / gap inventory (no code changes). Baseline: `dev` @ `de2e217`.

## Principle

Every node type supports as much functionality as possible. A capability may
be absent on a node type only when the platform genuinely cannot do the work
(no hardware, no upstream implementation) — never merely because Skulk has
not written the glue. This is the existing "model truth vs platform truth"
doctrine (`src/skulk/shared/backends.py`) applied as a roadmap discipline:
every platform gate must be classified as **physics** (upstream/hardware
impossibility) or **glue** (upstream supports it; the gap is ours), and glue
gaps are debt to be paid down.

## Capability × node-type matrix (today)

Node types by serving engines available to them:

- **macOS / Apple Silicon** — `mlx`, `mlx_audio`, `llama_cpp`, `llama_server`
- **Linux CUDA** — `llama_cpp-cuda`, `llama_server`, `vllm-cuda`
- **Linux ROCm** — `llama_cpp-rocm`, `llama_server`, `vllm-rocm`
- **Linux Vulkan** (AMD/Intel without ROCm) — `llama_cpp-vulkan`, `llama_server`
- **CPU-only** — `llama_cpp-cpu`, `llama_server`

| Capability | macOS | CUDA | ROCm | Vulkan | CPU | Gap class |
|---|---|---|---|---|---|---|
| Text generation (single-node) | ✓ mlx | ✓ | ✓ | ✓ | ✓ | — |
| Text generation (multi-node) | ✓ mlx ring | ✓ llama_server RPC | ✓ | ✓ | ✓ | — |
| Speculative decoding | ✓ MTP/sidecar | ✓ served MTP/DFlash, vLLM MTP/DFlash | ✓ | ✓ served | partial | — |
| Vision input (chat) | ✓ mlx | ✓ llama_cpp only | ✓ llama_cpp only | ✓ llama_cpp only | ✓ llama_cpp only | **glue** on `llama_server` (upstream `--mmproj`; comment in `backends.py` says "the gap is ours") and on `vllm` (upstream serves VLMs) |
| Tool calling | ✓ | ✓ except vllm | ✓ except vllm | ✓ | ✓ | **glue** on `vllm` (runner comment: "follow-up") |
| Per-token logprobs | ✓ | ✓ except vllm | ✓ except vllm | ✓ | ✓ | **glue** on `vllm` (runner comment: "follow-up") |
| Embeddings (`/v1/embeddings`) | ✓ mlx | ✗ | ✗ | ✗ | ✗ | **glue** — embedding cards default to `{"mlx"}`; llama.cpp and vLLM both serve embeddings upstream |
| Image generation / edit | ✓ mlx engine | ✗ | ✗ | ✗ | ✗ | **glue (large)** — engine is MLX-only; no served diffusion backend exists in Skulk; upstream options exist (diffusers/ComfyUI-class servers) |
| Speech STT (batch) | ✓ mlx_audio | ✗ | ✗ | ✗ | ✗ | **glue** on CUDA/ROCm — upstream vLLM serves `/v1/audio/transcriptions` (Whisper, Voxtral, Gemma3n, Qwen3-Omni). Vulkan/CPU: whisper.cpp server (glue, second tier) |
| Speech translation | ✓ mlx_audio | ✗ | ✗ | ✗ | ✗ | **glue** on CUDA/ROCm — upstream vLLM serves `/v1/audio/translations` |
| TTS (`/v1/audio/speech`) | ✓ mlx_audio | ✗ | ✗ | ✗ | ✗ | **glue (new served engine)** — no vLLM path; OpenAI-compatible TTS servers exist (Kokoro; Orpheus via llama.cpp GGUF) |
| Realtime STT (`WS /v1/realtime`) | ✓ mlx_audio | ✗ | ✗ | ✗ | ✗ | **hard** — bidirectional frame streaming is coupled to runner internals; proxied HTTP servers don't expose it. Defer; document as macOS-only |
| Chat audio input (omni LLM) | ✗ | ✗ | ✗ | ✗ | ✗ | **glue everywhere (large)** — no `input_audio` content part, no task-param field. Upstream: vLLM serves Qwen3-Omni audio-in today; mlx-vlm has omni support. Could land on Linux (vllm) before MLX |
| Video input | ✗ | ✗ | ✗ | ✗ | ✗ | absent everywhere; no upstream consensus path yet |

The notable inversion: macOS is the flagship for everything Skulk has built
in-process, but the **omni-LLM future may arrive on Linux first**, because
vLLM already serves audio-in models upstream while the MLX path requires new
engine work.

## Mechanism: one serving matrix instead of per-capability frozensets

`platform_compatible_backends()` currently takes a single
`card_serves_vision: bool`, and speech has its own
`_SPEECH_SERVING_ENGINES` set. Each new capability added this way grows a
new parallel gate. Before closing the gaps below, consolidate into a single
per-capability serving table in `src/skulk/shared/backends.py`:

```
_SERVING_MATRIX: dict[ServingCapability, frozenset[EngineType]]
```

keyed by what the card declares (vision, stt, stt_translation, tts,
realtime_stt, embeddings, image_generation, tools, logprobs, audio_input),
so "when a runner gains a capability, flip the code table" stays a
one-line change and the placement/bootstrap call sites stop growing
boolean parameters.

## Roadmap, ordered by leverage

Phases 1–4 are pure glue against upstream capabilities that already exist,
using the managed-server-plus-proxy shape Skulk already ships twice
(`llama_server`, `vllm`).

1. **vLLM speech-to-text + translation** (CUDA/ROCm). Add `vllm-*` to the
   speech-serving gate for `SpeechToText`/`SpeechTranslation`; teach
   `worker/runner/vllm/runner.py` to proxy `AudioTranscriptionTask` as
   multipart; add HF-format Whisper/Voxtral speech cards with
   `compatible_backends = ["vllm-cuda", "vllm-rocm"]`. Closes the loudest
   gap ("a GPU-only Linux cluster has zero speech capability") with no new
   external dependency. Requires the serving-matrix generalization.
2. **`llama_server` vision** — stage and pass `--mmproj`; delete the
   engine from the vision gate. The code comment already declares this
   debt.
3. **vLLM text parity** — tool calling, per-token logprobs, then vision
   (upstream serves VLMs). All three are declared follow-ups in the
   runner.
4. **Embeddings on GPU/CPU nodes** — llama.cpp embeddings for GGUF
   embedding cards; optionally vLLM embeddings. Removes the silent
   MLX-only default on `resources/embedding_model_cards/`.
5. **Served TTS engine** (new engine tag, e.g. `tts_server`) — wrap an
   OpenAI-compatible `/v1/audio/speech` server (Kokoro first; Orpheus via
   llama.cpp as a GGUF-ecosystem alternative). Third instance of the
   served-engine pattern; `vllm/orphan_sweep.py` is the lifecycle
   template.
6. **Chat audio input (omni phase 1)** — `input_audio` content part across
   the four API dialects, `audios` on `TextGenerationTaskParams`, reuse of
   the existing `AudioInputChunk` transport; serve first via vLLM
   (Qwen3-Omni/Voxtral upstream), then MLX via mlx-vlm's omni path.
7. **Image generation on Linux** — served diffusion engine. Largest new
   surface; do last unless demand says otherwise.
8. **Long tail** — whisper.cpp served STT for Vulkan/CPU; realtime STT on
   Linux; LLM-native speech output (blocked upstream in both MLX and
   llama.cpp ecosystems).

## Non-goals

- Mixed-version clusters remain unsupported; every phase above is a
  whole-fleet upgrade when it touches wire types (`extra="forbid"`).
- Parity does not mean identical quality: MLX-side prefix caching,
  batching, and multi-node sharding remain richer than served proxies.
  The bar is "the capability exists on the node type," not "identical
  performance envelope."
