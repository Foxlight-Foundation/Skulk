# TTS Integration Plan for Skulk (MLX-first)

This document maps Skulk's current inference pipeline and proposes concrete approaches for adding text-to-speech (TTS) support with minimal architectural risk.

## Current Inference Pipeline (as implemented)

### 1) API compatibility layer receives requests

- FastAPI endpoints in `src/exo/api/main.py` convert OpenAI/Claude/Ollama-style payloads into internal command types.
- Text goes through `/v1/chat/completions` and adapters that produce `TextGeneration` commands.
- Images already have first-class task families (`ImageGeneration`, `ImageEdits`) and dedicated chunk queues.

### 2) Commands are event-sourced by the master

- `Master._command_processor()` turns incoming commands into `TaskCreated` events bound to an eligible instance.
- For each command family (text/image/embedding), the master creates a typed task (`TextGeneration`, `ImageGeneration`, `ImageEdits`, `TextEmbedding`) and stores `command_id -> task_id` mapping for cancellation/finish tracking.

### 3) Worker planner advances runner state machine

- Worker loop runs `plan()` to emit operational tasks in this order:
  1. cancel/kill stale runner
  2. create runner
  3. ensure model download
  4. connect distributed group (if multi-node)
  5. load model
  6. warmup
  7. dispatch pending inference task
- This sequencing is exactly what allows Skulk to treat inference as a state machine over event-sourced state.

### 4) Runner supervisor manages subprocess and task IPC

- `RunnerSupervisor` owns one subprocess per bound shard and forwards events back to the worker/master streams.
- It handles acknowledgement, completion, cancellations, and failure fallback (including emitting `ErrorChunk` on runner crash for user-visible inference errors).

### 5) Runner entrypoint selects model family implementation

- `entrypoint()` currently routes to one of three runner implementations:
  - image runner when `BoundInstance.is_image_model`
  - embedding runner when `BoundInstance.is_embedding_model`
  - default MLX LLM inference runner otherwise
- This dispatch point is the cleanest insertion point for adding a dedicated TTS runner family.

## Why TTS fits the existing architecture cleanly

Skulk already supports **heterogeneous inference families** (text, image, embedding) with:

- typed command/task definitions,
- family-specific runner implementations,
- family-specific chunk streaming,
- common planning/scheduling lifecycle.

TTS can reuse that same pattern with comparatively small surface-area changes.

## External MLX ecosystem findings relevant to Skulk

### 1) `mlx-audio` is the strongest template for API + web UX

Repository: <https://github.com/Blaizzy/mlx-audio>

Useful properties for adaptation:

- Broad MLX TTS model coverage (Kokoro, Qwen3-TTS, etc.).
- Explicit OpenAI-compatible speech endpoint example (`POST /v1/audio/speech`).
- Includes both API server and web UI flow (separate backend + frontend process), which maps well to Skulk's existing dashboard + FastAPI setup.
- Demonstrates streaming-oriented CLI/server ergonomics for audio generation.

### 2) `f5-tts-mlx` is a focused MLX inference reference

Repository: <https://github.com/lucasnewman/f5-tts-mlx>

Useful properties for adaptation:

- Clean, compact MLX implementation of F5-TTS.
- Practical voice-cloning/reference-audio flow.
- Helpful for designing an internal "engine adapter" layer in Skulk, even if Skulk does not expose every model-specific knob initially.

### 3) `Kokoro-FastAPI` is a good API contract reference

Repository: <https://github.com/remsky/Kokoro-FastAPI>

Useful properties for adaptation:

- OpenAI-style speech endpoint expectations in real deployments.
- Production-style UX expectations: voices, language selection, optional captions/timestamps, and web integrations.

### 4) Mobile-first angle (`mlx-audio-swift`)

Repository: <https://github.com/Blaizzy/mlx-audio-swift>

Useful properties for adaptation:

- Confirms demand for low-latency local TTS and strongly typed audio APIs.
- Suggests Skulk should keep model/task schema expressive enough to carry future speech controls (voice/style/speed/reference audio), not just plain text input.

## Recommended approach

## Recommendation A (preferred): Add first-class TTS task family in Skulk

### API contract

Add OpenAI-compatible endpoint:

- `POST /v1/audio/speech`

Initial request shape:

- `model: str`
- `input: str`
- `voice: str | None`
- `response_format: Literal["wav", "mp3", "flac", "pcm"] | None`
- `speed: float | None`
- `stream: bool | None` (optional phase-2)

### Internal typing/event model

Add new internal types parallel to existing families:

- `TtsGenerationTaskParams`
- `commands.TextToSpeech`
- `tasks.TextToSpeech`
- audio chunk types (e.g., `AudioChunk`, `AudioDoneChunk`, `AudioErrorChunk` or reuse `ErrorChunk` + new audio payload chunk)

### Runner path

- Add `ModelTask.TextToSpeech` to model-card task enum.
- Extend `BoundInstance` with `is_tts_model`.
- Extend runner bootstrap dispatch to `runner/tts/runner.py`.
- Implement `exo.worker.runner.tts.runner.Runner` with the same supervisor contract (ack, status updates, task completion).

### Engine adapter

Introduce `exo/worker/engines/tts/` with provider adapters:

- `mlx_audio_adapter.py` (primary)
- optional future adapters (`f5_tts_mlx_adapter.py`, etc.)

This isolates model/runtime differences from orchestration.

### Streaming model

- Phase 1: non-streaming binary response for compatibility and simplicity.
- Phase 2: progressive chunk streaming (SSE or chunked transfer) once chunk envelope is stable.

## Recommendation B (fallback): API proxy mode to external TTS service

If timeline is tight, add a temporary adapter that forwards `/v1/audio/speech` to a colocated TTS service (e.g., `mlx-audio` server), while keeping the final Skulk API contract stable.

Pros:

- Very fast time-to-demo.

Cons:

- Bypasses Skulk placement/state machine and weakens distributed scheduling consistency.

Use only as an interim milestone.

## Proposed implementation phases

### Phase 0 — Design + schema

1. Add TTS API models in `src/exo/api/types/`.
2. Add endpoint declaration in `src/exo/api/main.py` with tags/summary/description.
3. Add command/task/chunk types.
4. Update `docs/api.md` for the new endpoint.

### Phase 1 — Single-node happy path

1. Implement TTS runner and adapter using one baseline model (Kokoro-82M MLX).
2. Return `audio/wav` output from `/v1/audio/speech`.
3. Add unit tests for:
   - request validation,
   - command->task mapping,
   - runner completion/error behavior,
   - cancel flow.

### Phase 2 — Distributed and operational hardening

1. Validate placement behavior with TTS-tagged model cards.
2. Add download/store integration tests for TTS artifacts.
3. Add timeout and memory-pressure protections tuned for long audio outputs.

### Phase 3 — UX and advanced controls

1. Dashboard TTS panel (model + voice + speed + output format).
2. Optional streaming playback in UI.
3. Add reference-audio / voice-clone fields once baseline is stable.

## Key engineering decisions to make early

1. **Audio chunk format:** PCM frames vs encoded chunks.
2. **Sampling boundary:** enforce single output sample rate or expose model-native rates.
3. **Cancellation granularity:** stop-at-next-chunk vs immediate abort.
4. **Model card metadata:** minimal voice/language metadata in cards vs dynamic runtime discovery.
5. **Storage policy:** whether generated audio should be persisted similarly to image outputs.

## Suggested first milestone (1 PR)

"OpenAI-compatible `/v1/audio/speech` with one MLX model, non-streaming WAV output, fully integrated with command/task/event pipeline."

That milestone proves architecture fit and unblocks incremental model additions.
