<!-- Copyright 2026 Foxlight Foundation -->

# Higgs TTS 3 (bosonai/higgs-tts-3-4b) Support Evaluation

Status: evaluation only — no Higgs TTS 3 support ships yet. This note records
what the model needs, where Skulk (dev) stands, and the concrete integration
plan, so the implementation PRs can be scoped without re-deriving any of it.

## What The Model Is

- ~4B autoregressive conversational TTS from Boson AI, built on a Qwen3-4B
  decoder: `model_type: higgs_multimodal_qwen3`
  (`HiggsMultimodalQwen3ForConditionalGeneration`), 36 layers, hidden 2560,
  GQA 32/8, context length 8,192.
- Audio is generated as discrete tokens over 8 codebooks x 1,026 vocabulary,
  staggered with a delay pattern (Higgs tokenizer, integrated — no separate
  codec repo to stage). Output is 24 kHz mono, 25 fps (40 ms/frame).
- Zero-shot voice cloning from reference audio plus transcript; the reference
  transcript materially improves fidelity and should be treated as required.
  There are no built-in speaker IDs — voice identity comes entirely from the
  reference audio.
- Inline control tokens with `<|category:value|>` syntax: 21 emotions,
  3 styles (singing/shouting/whispering), 10 prosody controls, 9 sound
  effects. Delivery-level tokens (emotion/style/speed/pitch/expressive) go at
  sentence start; `sfx` and pause tokens are positional, and `sfx` must be
  immediately followed by matching onomatopoeia (`<|sfx:laughter|>Haha`).
- 102 languages (85 at production quality, WER/CER < 5%).
- Upstream serving: SGLang-Omni (recommended), vLLM-Omni (OpenAI-compatible
  `/v1/audio/speech`, SSE streaming of base64 WAV chunks), or Transformers.
  Reported H100 numbers at concurrency 16: 14.74 req/s, RTF 0.262.
- Artifacts: bf16 safetensors, 9.31 GB, plus `chat_template.jinja` and a
  `PROMPTING.md` documenting the control-token vocabulary.
- License: "Boson Higgs TTS 3 Research and Non-Commercial License" with a
  creator-use grant (monetized creator content allowed with attribution).
  Hosted-API/commercial embedding requires a separate license from Boson.

## Where Skulk Dev Stands

The speech stack is hard-gated to one engine, `mlx_audio`, at three
independent layers that must stay in agreement:

1. `_SPEECH_SERVING_ENGINES = frozenset({"mlx_audio"})` in
   `src/skulk/shared/backends.py` (platform truth applied by master placement
   and the worker fallback probe).
2. The speech branch in `src/skulk/worker/runner/bootstrap.py` fires before
   the text-engine chain and raises unless the resolved engine is
   `mlx_audio`.
3. The bundled-card CI invariant in
   `src/skulk/shared/tests/test_bundled_model_cards.py`: "speech card must
   list the mlx_audio engine until another speech runner exists".

Within that gate, the good news:

- **The speech runner is family-agnostic by design.** It loads through
  `mlx_audio.tts.utils.load_model` and calls `model.generate(text, **kwargs)`
  with signature introspection to drop unsupported kwargs. The candidate set
  already includes everything Higgs needs: `ref_audio`, `ref_text`, `stream`,
  `temperature`, `top_k`, `max_tokens`. A new family normally needs zero
  runner code.
- **The voice-cloning contract already matches Higgs exactly** (reference
  audio + transcript). Request-scoped uploads arrive via multipart
  `POST /v1/audio/speech` over the node-addressed `SPEECH_MEDIA` Zenoh path
  (25 MiB cap, SHA-256 verified, never in State or the event log), and the
  bundled shared reference-voice catalog
  (`resources/speech_reference_voices/`, angus…sylvie) provides managed
  voices for a model with no built-in speakers.
- **Control tokens need no plumbing.** They are inline text in `input` and
  pass through the API, task params, and runner untouched.

The blockers, as of this evaluation:

- **The pinned mlx-audio fork cannot load this model.** `pyproject.toml` pins
  a Foxlight fork of mlx-audio at the 0.4.3 tag plus upstream #693. Its
  `higgs_audio` family targets Higgs Audio **v2** (Llama-3.2-3B based) only —
  it does not know `higgs_multimodal_qwen3`. Upstream mlx-audio **v0.4.4**
  added Higgs v3 (upstream PRs #770 model, #802 batch generation, #804
  continuous batching, #808 decode-sync reduction), and #693 merged upstream
  between 0.4.3 and 0.4.4, so a pin advance also retires the carried fork
  patch.
- **No pinned MLX artifact.** Community MLX conversions of higgs-tts-3-4b
  exist on the Hub (bf16 and quantized), but none is validated or
  revision-pinned by us. Card auto-generation never emits `[audio]` cards, so
  the card must be hand-authored regardless.
- **No GPU/Linux path.** Upstream's production serving is SGLang-Omni /
  vLLM-Omni. Skulk's `vllm` runner is explicitly text-only (rejects non-text
  tasks, emits only `TokenChunk`, parses chat-completions SSE), and no served
  speech engine exists.

## Phase A (recommended first): MLX path on Apple Silicon

Smallest end-to-end support; expected to need **no new engine code**.

1. **Advance the mlx-audio pin** from 0.4.3+#693 to >= 0.4.4 (rebase the
   Foxlight fork or return to an upstream tag — #693 is upstream now). This
   is fleet-affecting: regression-validate all seven bundled speech cards
   (3 TTS + 4 STT), not just Higgs.
2. **Select and pin an artifact.** Prefer a Foxlight- or mlx-community-
   controlled MLX conversion (bf16 is ~9.3 GiB; a validated 6/8-bit quant is
   the likelier daily driver on laptops), with `source_revision` pinned to a
   full commit hash per card policy.
3. **Hand-author the bundled card** in `resources/speech_model_cards/`,
   modeled on `mlx-community--Qwen3-TTS-12Hz-0.6B-Base-6bit.toml` (the
   closest structural analogue: LLM-style autoregressive TTS with
   reference-audio cloning and streaming):
   - `tasks = ["TextToSpeech"]`, `capabilities = ["tts"]`,
     `family = "higgs_audio"` (mlx-audio family name), `context_length = 8192`
   - `[audio]`: `kind = "tts"`, `response_formats = ["wav", "mp3", "pcm"]`
     with `default_response_format = "wav"`, `sample_rates = [24000]`,
     `supports_reference_audio = true`, `supports_voice_listing = true`,
     `default_voice = "angus"`, and the full 10-profile shared reference
     catalog (CI enforces: `voice_catalog` ids equal `voices` exactly and in
     order, each id equals its `reference_profile`, and reference-capable
     cards expose the shared catalog — add the new card to
     `test_reference_capable_cards_expose_shared_voice_catalog`).
   - `supports_streaming` only after live validation of chunked mp3/pcm
     streaming through the runner's segment encoder (see risks). WAV itself
     is not in `_STREAMABLE_AUDIO_RESPONSE_FORMATS`; mp3/pcm cover streaming
     and the `tts@1.0.0` facade is mp3-only, so streaming truth on the card
     is what unlocks the dashboard voice chain.
   - `[placement] compatible_backends = ["mlx_audio", "mlx_audio-metal"]`.
4. **Document control tokens** in `website/docs/model-behaviors/` (syntax,
   sentence-start vs positional placement, the sfx+onomatopoeia pairing) so
   users don't need Boson's PROMPTING.md. No schema slot is needed for v1;
   tokens ride the `input` text.
5. **Validation gauntlet** before flipping `supports_streaming` or shipping
   the card: non-streaming wav/mp3/pcm; streaming mp3/pcm stability across
   long outputs; cloning fidelity with and without transcript via multipart
   upload and via each bundled profile; control-token acoustic realization
   (emotion, style, sfx pairing); `/v1/audio/voices` and dashboard Auto voice
   selection; `tts@1.0.0` facade; memory headroom for the pinned quant on the
   smallest supported node.

## Phase B (separate project): served GPU engine for Linux nodes

Serving Higgs on NVIDIA/AMD nodes means a fourth engine tag (e.g.
`vllm_omni` or `sglang_omni`) and Skulk's first served *speech* engine. The
managed-subprocess-plus-OpenAI-proxy shape ports from the `vllm` runner
almost verbatim (spawn/health/teardown/orphan-sweep, `SKULK_<X>_BIN` env
pattern), but the work fans out across every layer that currently spells
`mlx_audio`:

1. `EngineType`/`_ENGINES` + env knobs + GPU-only compute allowlist in
   `shared/backends.py`; add the engine to `_SPEECH_SERVING_ENGINES`.
2. Facts probe + tag derivation (`facts/probe.py`, `facts/derive.py`) and a
   doctor check for the binary.
3. Restructure the bootstrap speech branch to dispatch on the resolved
   engine instead of asserting `mlx_audio`.
4. New runner: translate the upstream SSE base64-WAV stream into
   `AudioChunk`s (the speech runner's module-level
   `_emit_audio_chunks`/`_emit_streaming_audio_chunks` are engine-agnostic
   and reusable), honor `reference_audio_data`/`reference_text` from
   `SpeechSynthesisTaskParams` (upstream takes `references` with audio path +
   text — a request-scoped temp file mirrors the MLX runner), and wrap
   admission in `ServedConcurrentDispatch` since continuous batching is the
   entire point of this path.
5. Relax the bundled-card CI invariant and extend `_VALID_TAGS`; update the
   `"runtime": "mlx_audio"` annotations in `extensions/speech.py`; give the
   engine a non-darwin dependency story (mlx-audio is `sys_platform ==
   'darwin'` only).
6. Docs: architecture.md + architecture-reference.md entries, api-guide
   updates, and a `vllm-engine.md`-style operator page.

Before investing here, resolve the **licensing question**: the model's
license restricts hosted-API/commercial serving, which is exactly what a
GPU serving lane is for. Phase B should be justified by a model we can serve
commercially or by an operator with a Boson license.

## Risks And Open Questions

- **Streaming stability is unproven.** The card-level `supports_streaming`
  gate exists precisely because chunked encode of autoregressive TTS is
  where models fall over. Higgs v3's delay-pattern ramp-in/out must be
  validated under mlx-audio's segment streaming before the card claims it.
- **Memory.** bf16 is ~9.3 GiB of weights before activations; placement
  admission for speech cards leans on `storage_size`. Validate the quant tier
  and record honest `storage_size`/`min_vram_gib` numbers.
- **License.** Non-commercial + creator grant. Bundling a card is
  distribution of metadata, not weights, but the operator-facing model docs
  must state the license terms plainly.
- **Control-token discoverability.** A per-card control-token vocabulary
  (for a dashboard emotion/style picker) would be a new `AudioCardConfig`
  field — a wire-format change on an `extra="forbid"` model requiring a
  same-version fleet. Defer until a second control-token model justifies the
  schema.
- **Language surface.** Skulk's per-voice `preferred_languages` plus the
  dashboard's script-sniffing (`ko`/`ja`/`zh`/`ru` only) undersell a
  102-language model; fine for v1, worth revisiting if multilingual TTS use
  materializes.
- **Upstream kwarg drift.** The runner's signature introspection makes new
  families cheap but silent: if mlx-audio's Higgs v3 `generate` names a
  parameter differently (e.g. reference handling), the kwarg is dropped, not
  errored. The validation gauntlet must confirm cloning actually conditions
  on the reference, not just that audio comes out.
