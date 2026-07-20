---
id: speech-fabric-realtime
title: Speech Providers and Realtime Transcription
sidebar_position: 31
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk serves mounted speech models through OpenAI-compatible REST endpoints and
first-party Fabric provider capabilities. Speech model placement, loading,
health, cancellation, and model-store staging remain owned by the normal Skulk
model lifecycle.

## HTTP APIs

Skulk currently exposes:

- `POST /v1/audio/speech` for text-to-speech synthesis;
- `POST /v1/audio/transcriptions` for bounded uploaded audio clips;
- experimental `POST /v1/audio/translations` for speech-to-English translation;
- `GET /v1/audio/voices` for a mounted model's static built-in voice catalog;
- `WS /v1/realtime?model=<model-id>` for serialized multi-turn realtime
  transcription, including optional bounded server VAD and automatic commit.
- `WS /v1/fabric/chains/speech?stt_model=<model-id>` for the same typed
  STT-to-chat-to-TTS composition under an explicit Fabric endpoint.

`/v1/audio/speech` returns an encoded audio response. Its stable `stream=true`
path is available when the mounted TTS card declares
`audio.supports_streaming = true`, every routable instance is ready, and the
request format resolves to MP3 or raw PCM. PCM responses declare sample rate,
channel count, and sample format in HTTP headers. Other encoded formats remain
batch-only.

`/v1/audio/transcriptions` accepts a bounded multipart upload. Its stable
`stream=true` path is available when the mounted STT card declares
`audio.supports_streaming = true` and every routable instance is ready. It
returns typed SSE delta/completed/usage/error events, while explicit
`response_format=ndjson` preserves progressive NDJSON chunk framing. Batch
cards and non-streaming requests retain the completed-response formats.

### Batch versus realtime routing

The two transcription surfaces are routed differently and are not
interchangeable:

- `POST /v1/audio/transcriptions` is the **batch** path: the caller uploads a
  complete audio clip and receives its transcript. The `stream=true` and
  `response_format=ndjson` variants (available where the card has proven
  streaming support) stream the **response** progressively; the input is still
  one complete upload.
- `WS /v1/realtime` is the **live** path: a bidirectional session where audio
  frames stream in while partial and final transcripts stream out, backed by a
  true streaming session on the mounted realtime STT model. It requires a card
  that declares both streaming and realtime support.

### Voice discovery

`GET /v1/audio/voices?model=<model-id>` is a Skulk extension that lists a
mounted TTS model's static built-in voices. It serves only models whose card
declares `audio.supports_voice_listing = true`; the identifiers come from the
card's `audio.voices` declaration, and a card may also declare a validated
`audio.default_voice`, which the API applies only when a request omits `voice`.
See [Model cards](model-cards.md) for the card-side declarations.

### Reference-audio uploads

`/v1/audio/speech` also accepts bounded reference audio for mounted cards
declaring `audio.supports_reference_audio = true`, sent as a
`multipart/form-data` request whose binary part is the `reference_audio` field
alongside the normal speech request fields. The upload is request-scoped for
its whole lifecycle: it travels over node-addressed Zenoh `SPEECH_MEDIA`, stays
out of State and the event log, is assembled only in bounded process-local
memory on the serving worker, and the runner deletes its request-scoped
temporary file after generation. The API rejects reference uploads when Zenoh
is unavailable instead of broadcasting private media through gossipsub.

The dashboard shows a reference-audio picker only when the selected mounted TTS
card declares this capability. The optional transcript and selected browser
`File` stay local until synthesis; the dashboard sends them as multipart form
data and reuses the same clip for every sentence-sized request in one response.
Changing TTS models clears the selection, and the browser never turns the clip
into a persistent voice profile.

`WS /v1/realtime?model=<model-id>` accepts OpenAI-style base64 PCM16 append and
commit events over a WebSocket. It emits transcript delta, final, and failure
events from the mounted realtime STT model. An optional response configuration
routes final transcripts through a mounted chat model and can stream the visible
answer through a mounted TTS participant.

Server VAD on the realtime endpoint is configured through the session's
`turn_detection` object (sent in a `session.update` event, or the legacy
`transcription_session.update` form). Setting `"type": "server_vad"` enables
server-owned WebRTC voice activity detection with these bounded fields, all
optional:

- `aggressiveness` (0 to 3, default 2): WebRTC VAD aggressiveness, least to
  most restrictive.
- `prefix_padding_ms` (default 200): speech preroll subtracted from the
  reported turn-start boundary.
- `silence_duration_ms` (default 400): trailing silence required to end the
  current utterance.
- `minimum_speech_ms` (default 120): continuous speech required before a turn
  start is announced.
- `maximum_utterance_ms` (default 30000): hard duration limit that ends a
  continuous utterance.

With server VAD active the endpoint commits each detected utterance
automatically as its own serialized provider call; barge-in during an active
automatic response cancels the underlying model and TTS commands.

The Fabric chain endpoint deliberately reuses this contract and implementation.
Clients select optional chat/TTS participants through `session.update`; normal
capability discovery, health, locality, cancellation, and data-plane rules stay
authoritative instead of being duplicated in a second orchestration runtime.

See [API Guide](api-guide.md) for request fields, response formats, limits, and
errors.

## Provider Capabilities

Eligible nodes advertise these built-in provider contracts while compatible
mounted capacity is healthy:

| Capability | I/O mode | Behavior |
| --- | --- | --- |
| `tts@1.0.0` | `server_streaming` | Accepts text controls and emits raw MP3 media frames. |
| `stt@1.0.0` | `client_streaming` | Accepts bounded encoded audio, starts inference on input half-close, and emits one final transcript. |
| `stt.realtime@1.0.0` | `bidirectional` | Accepts mono PCM16 frames and emits model-provided partial and final transcripts. |

The provider descriptor is the public interface contract. Model cards remain
the source of model capability truth, while Skulk's backend filters and runner
code determine whether the platform can serve that capability.

Provider calls use the shared lifecycle states `started`, `chunk`, `completed`,
`failed`, and `cancelled`. Each admitted call owns exactly one terminal outcome.
Caller and provider directions have independent monotonic sequence numbers.
Input `completed` is a half-close, not cancellation.

## Realtime Audio Transport

Realtime PCM frames do not use event-sourced State. The API owner sends them to
the selected serving worker through node-addressed `REALTIME_AUDIO` transport:

- same-node calls use an in-process short circuit;
- remote calls use bounded Zenoh ingress;
- remote realtime capacity is not advertised when Zenoh is unavailable;
- worker-to-runner IPC is bounded and cancellation-aware;
- transport rejection is routed to the source API call rather than failing
  unrelated sessions.

A realtime call is pinned to one single-host speech model instance. It is not
migrated between runners during an utterance. Disconnect, timeout, explicit
cancellation, runner failure, and transport failure all terminate the provider
call and release its reservation.

Binary audio remains on the data path. It is not included in State, the event
log, or structured logs. Batch REST transcription uses a separate bounded
upload path and does not claim the same no-retention property.

Reference-audio TTS uses a separate `SPEECH_MEDIA` data family. The API selects
and pins one ready single-host TTS instance, chunks the upload with a terminal
digest, and the target worker assembles it only in bounded process-local memory.
Cancellation, transport failure, checksum failure, dispatch, and expiry clear
the worker buffer. Only the worker-local runner task receives the bytes.

## Dashboard Behavior

The dashboard exposes microphone controls only when the selected transcription
model and local node advertise the required capabilities.

- Realtime models use an `AudioWorklet` to capture mono Float32 browser audio.
- A stateful browser resampler produces 24 kHz PCM16.
- Capture callbacks are aggregated into bounded 100 ms transport frames.
- Realtime mode keeps one multi-turn socket open, uses server VAD for automatic
  turn boundaries, and shows partial transcripts in the editable chat draft.
- Auto-send optionally routes final transcripts to the selected chat model;
  assistant text remains visible while it streams and barge-in cancels active
  response playback.
- Batch-only models retain the `MediaRecorder` upload flow.
- Transcription results populate the chat draft for review or submission.
- Streaming-capable TTS models use sentence-sized raw PCM requests and a bounded
  AudioWorklet playback queue. The dashboard pauses HTTP reads under pressure,
  preserves sentence order, and propagates stop to queued and active requests.
- For cards with voice discovery, the dashboard lists the mounted catalog and
  defaults to automatic language matching. It chooses the first catalog voice
  whose preferred language matches the response text, then pins that voice for
  every sentence request in the response. An explicit user selection overrides
  automatic matching but remains pinned for the same response.
- Batch-only TTS models retain complete-response encoded playback.

The dashboard falls back to batch transcription when realtime model or node
capabilities are absent. Speech controls remain hidden when no suitable mounted
speech model is available.

## Pressure and Diagnostics

Provider transport applies bounded admission and per-call/per-owner queues. A
slow or disconnected caller cannot grow an unbounded stream or consume all
provider capacity.

`NodeDiagnostics.provider` exposes aggregate and per-capability snapshots for:

- active and reserved concurrency;
- admissions and overload rejections;
- caller-input queue depth;
- input/output frame and inline-media byte counts;
- admission-to-first-output latency;
- total stream lifetime;
- terminal outcomes, cancellation requests, and missing terminals.

Diagnostics retain counters and bounded aggregates, not audio payloads,
transcripts, or completed call identifiers.

## Model Requirements

Speech serving runs on the `mlx_audio` engine, which Skulk probes and
advertises on **macOS nodes only** (it is backed by the upstream `mlx-audio`
package). The GPU text engines (`llama_cpp`, `llama_server`, `vllm`) do not
serve speech models, so a cluster needs at least one Mac with mounted speech
capacity to offer these endpoints.

Speech cards use the `mlx_audio` backend vocabulary and are single-node. A card
may declare synthesis, transcription, streaming, realtime, voice-listing,
reference-audio, and translation capabilities. Runtime admission remains
conservative: a feature is exposed only when card metadata, platform support,
mounted health, and node transport all agree.

Models must be downloaded into the model store and staged through Skulk's normal
mount lifecycle. Direct filesystem placement is not a supported serving path.
