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
- `WS /v1/realtime?model=<model-id>` for one-utterance realtime transcription.

`/v1/audio/speech` returns an encoded audio response. Its `stream=true` path is
experimental and is available only when all of these conditions are true:

- the node runs with `SKULK_ENABLE_EXPERIMENTAL_MODE`;
- cluster settings enable `experiments.tts_streaming`;
- the mounted TTS card declares `audio.supports_streaming = true`;
- the serving runner is ready and advertises the required capability.

`/v1/audio/transcriptions` accepts a bounded multipart upload and returns one
completed transcription. It does not provide progressive REST transcription.

`WS /v1/realtime?model=<model-id>` accepts OpenAI-style base64 PCM16 append and commit events over a
WebSocket. It emits transcript delta, final, and failure events from the mounted
realtime STT model. The route is a transcription compatibility surface, not a
full speech-to-speech conversation API.

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

## Dashboard Behavior

The dashboard exposes microphone controls only when the selected transcription
model and local node advertise the required capabilities.

- Realtime models use an `AudioWorklet` to capture mono Float32 browser audio.
- A stateful browser resampler produces 24 kHz PCM16.
- Capture callbacks are aggregated into bounded 100 ms transport frames.
- Batch-only models retain the `MediaRecorder` upload flow.
- Transcription results populate the chat draft for review or submission.
- TTS playback uses `/v1/audio/speech` and currently begins after the complete
  encoded response is available.

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

Speech cards use the `mlx_audio` backend vocabulary and are single-node. A card
may declare synthesis, transcription, streaming, realtime, voice-listing,
reference-audio, and translation capabilities. Runtime admission remains
conservative: a feature is exposed only when card metadata, platform support,
mounted health, and node transport all agree.

Models must be downloaded into the model store and staged through Skulk's normal
mount lifecycle. Direct filesystem placement is not a supported serving path.
