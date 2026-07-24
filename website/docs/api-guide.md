---
id: api-guide
title: Skulk API
sidebar_position: 2
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk serves an API at `http://localhost:52415`.

That API has two jobs:

- compatibility endpoints for tools that already speak OpenAI, Claude, or Ollama-style APIs
- Skulk-specific control endpoints for placement, downloads, config, tracing, and model-store workflows

A model must be placed and running before chat requests for it succeed; calling
`/v1/chat/completions` for an unplaced model returns a 404 `No instance found`.
Text-generation endpoints require the mounted card to declare `TextGeneration`;
targeting a TTS-only or STT-only model returns **400 Bad Request** before any
runner command is dispatched. This applies to Chat Completions, Responses,
Claude, Ollama chat/generate, and benchmark adapters through their shared
admission path.
The [First Success Flow](#first-success-flow) below walks from placement to first
token.

## Quick Navigation

- First working request: [First Success Flow](#first-success-flow)
- OpenAI-compatible chat: [OpenAI Chat Completions](#openai-chat-completions)
- OpenAI Responses format: [OpenAI Responses API](#openai-responses-api)
- OpenAI embeddings: [OpenAI Embeddings API](#openai-embeddings-api)
- OpenAI text-to-speech: [OpenAI Audio Speech API](#openai-audio-speech-api)
- Image generation: [Image Generation and Editing](#image-generation-and-editing)
- Claude format: [Claude Messages API](#claude-messages-api)
- Ollama compatibility: [Ollama API](#ollama-api)
- Placement and launch: [Placement and Instance Management](#placement-and-instance-management)
- Store and config: [Model Store Endpoints](#model-store-endpoints) and [Configuration Endpoints](#configuration-endpoints)
- Debugging: [State, Events, and Tracing](#state-events-and-tracing)

## First Success Flow

### 1. Start Skulk

```bash
uv run skulk
```

### 2. Preview placements

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Llama-3.2-1B-Instruct-4bit"
```

This shows what Skulk can actually place on the current node or cluster.

### 3. Launch a placement

```bash
curl -X POST http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1
  }'
```

### 4. Send a chat request

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello from Skulk"}]
  }'
```

If this fails with `404 No instance found for model ...`, the placement is not ready yet or never launched successfully.

## Endpoint Overview

### Compatibility APIs

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/audio/voices`
- `WS /v1/realtime`
- `WS /v1/fabric/chains/speech`
- `POST /v1/messages`
- `POST /v1/cancel/{command_id}`
- `POST /ollama/api/chat`
- `POST /ollama/api/generate`
- `GET /ollama/api/tags`
- `POST /ollama/api/show`
- `GET /ollama/api/ps`
- `GET /ollama/api/version`

The Ollama group also serves alias paths (`/ollama/api/api/...`,
`/ollama/api/v1/...`, and `HEAD` version probes); see
[Ollama API](#ollama-api).

### Images

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `GET /images`
- `GET /images/{image_id}`

### Benchmarking

- `POST /bench/chat/completions`
- `POST /bench/images/generations`
- `POST /bench/images/edits`

### Skulk Control APIs

- `GET /v1/models`
- `GET /models`
- `POST /v1/tools/web_search`
- `POST /v1/tools/open_url`
- `POST /v1/tools/extract_page`
- `GET /models/search`
- `POST /models/add`
- `DELETE /models/custom/{model_id}`
- `POST /place_instance`
- `POST /instance`
- `GET /instance/placement`
- `GET /instance/previews`
- `GET /instance/{instance_id}`
- `DELETE /instance/{instance_id}`
- `GET /state`
- `GET /events`
- `POST /download/start`
- `DELETE /download/{node_id}/{model_id}`
- `GET /config`
- `PUT /config`
- `GET /store/health`
- `GET /store/registry`
- `GET /store/downloads`
- `POST /store/models/{model_id}/download`
- `GET /store/models/{model_id}/download/status`
- `DELETE /store/models/{model_id}`
- `POST /store/purge-staging`
- `GET /store/storage`
- `POST /store/models/{model_id}/optimize`
- `GET /store/models/{model_id}/optimize/status`
- `GET /filesystem/browse`
- `GET /node/identity`
- `GET /node_id`
- `POST /admin/restart`
- `GET /onboarding`
- `POST /onboarding`
- `GET /v1/tracing`
- `PUT /v1/tracing`
- `GET /v1/telemetry/preview`
- `GET /v1/traces`
- `GET /v1/traces/cluster`
- `POST /v1/traces/delete`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`
- `GET /v1/diagnostics/node`
- `GET /v1/diagnostics/telemetry`
- `GET /v1/diagnostics/performance-envelopes`
- `GET /v1/diagnostics/performance-envelopes/cluster`
- `POST /v1/diagnostics/node/capture`
- `POST /v1/diagnostics/node/runners/{runner_id}/cancel`
- `GET /v1/diagnostics/cluster`
- `GET /v1/diagnostics/cluster/timeline`
- `GET /v1/diagnostics/cluster/{node_id}`
- `POST /v1/diagnostics/cluster/{node_id}/capture`
- `POST /v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel`
- `GET /v1/capabilities`
- `POST /v1/capabilities/call`
- `POST /v1/capabilities/stream`
- `POST /v1/capabilities/stream/cancel`

The node diagnostics bundle includes the node's own Tailscale state
(`tailscale`: running flag, tailnet IP, hostname, MagicDNS name), probed on
the node the bundle describes, so the per-node cluster endpoint reports the
selected node's tailnet identity rather than whichever node served the HTTP
request. The probe is best-effort: a node without a working `tailscale` CLI
reports `running: false`, and `null` marks only an unexpected probe failure.

For the full interactive reference with request/response schemas, see the [API Reference](/api/skulk-api).

## OpenAI Chat Completions

**POST** `/v1/chat/completions`

This is the main chat-generation endpoint for both text-only and multimodal
models.

Requests are validated before dispatch: an empty `messages` array or a
non-positive `max_tokens` returns **400 Bad Request** rather than being
accepted and failing during generation. (This applies across the Claude,
Ollama, and Responses wire formats too, which share the same dispatch path.)
The mounted model must also declare `TextGeneration`; speech-only cards return
**400 Bad Request** without affecting their speech runner.

### Context-length limits

Every placed instance has a usable context limit: the smaller of the model's
advertised context length and the number of KV-cache tokens that fit in memory
next to the model weights on the hosting node(s). Requests are admitted
against that limit instead of growing the KV cache until the node runs out of
memory:

- A `max_tokens` value that cannot fit in the limit at all returns
  **400 Bad Request** immediately (`context_length_exceeded: ...`).
- After tokenization on the serving instance, a prompt that fills the window,
  or a prompt plus an explicit `max_tokens` that exceeds the limit, is
  rejected with an OpenAI-style `invalid_request_error` whose message starts
  with `context_length_exceeded:`. For streaming requests this arrives as the
  first SSE `data:` event; for non-streaming requests the response body is the
  error envelope (the HTTP status is already committed when the rejection is
  computed on the serving node).
- When `max_tokens` is omitted, the server default output budget is clamped to
  the remaining window, so generation ends with `finish_reason: "length"`
  instead of overrunning the context.

### OpenAI Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Curl Example

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Example

```python
stream = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Common Request Fields

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required. Must match a placed and running model. |
| `messages` | array | Required. Supports `system`, `user`, `assistant`, `developer`, `tool`, `function`. |
| `stream` | boolean | Use `true` for SSE streaming. |
| `temperature` | number | Sampling temperature. |
| `top_p` | number | Nucleus sampling. |
| `top_k` | integer | Top-k sampling. |
| `min_p` | number | Minimum-probability threshold. |
| `max_tokens` | integer | Max generated tokens. When omitted, Skulk uses a backend default of 4096 generated tokens (`DEFAULT_MAX_OUTPUT_TOKENS`); operators can override that default with `SKULK_MAX_OUTPUT_TOKENS` (or the legacy `SKULK_MAX_TOKENS`). |
| `stop` | string or array | Stop sequences. |
| `seed` | integer | Reproducibility helper. |
| `frequency_penalty` | number | Frequency penalty. |
| `presence_penalty` | number | Presence penalty. |
| `repetition_penalty` | number | Repetition penalty. |
| `repetition_context_size` | integer | Context window for repetition handling. |
| `logprobs` | boolean | Return token logprobs when supported. |
| `top_logprobs` | integer | Number of top logprobs to include. |
| `tools` | array | OpenAI-style tool definitions. |
| `tool_choice` | string or object | `auto`, `none`, or a specific tool selection. |
| `parallel_tool_calls` | boolean | Accepted for compatibility. |
| `enable_thinking` | boolean | Skulk extension for reasoning-capable models. |
| `reasoning_effort` | string | Reasoning hint when supported. |
| `response_format` | object | Accepted for compatibility, not strictly enforced. |
| `stream_options` | object | Includes `include_usage`. |
| `user` | string | Optional caller identifier. |

### Message Format

```json
{
  "role": "user",
  "content": "hello"
}
```

Assistant messages may include `tool_calls`.
Tool response messages should include `tool_call_id`.

User messages may also be sent as structured content parts. Skulk accepts
OpenAI-style image inputs for vision-capable models:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "What is in this image?" },
    {
      "type": "image_url",
      "image_url": { "url": "data:image/png;base64,..." }
    }
  ]
}
```

Notes:

- inline `data:` URLs are supported for image inputs
- Anthropic-compatible requests can also carry image content for multimodal models
- image understanding depends on the selected model exposing the `vision` capability
- request-scoped encoded image media is limited to 32 MiB; multipart image-edit
  uploads are read through the corresponding 24 MiB raw-image limit
- image bytes are sent only after authoritative task placement, directly to the
  selected worker rank or ranks; they are bounded, integrity-checked, and each
  rank must acknowledge task-owner verification before the transfer deadline
- image bytes are never written to the event log or replicated `State`

### Finish Reasons

| Value | Meaning |
|-------|---------|
| `stop` | Natural stop or stop sequence reached |
| `length` | `max_tokens` limit reached |
| `tool_calls` | Model is requesting a tool call |
| `content_filter` | Reserved for compatibility |
| `function_call` | Reserved for compatibility |
| `error` | Generation failed |

## Tool Use

Skulk supports OpenAI-style function calling.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)
```

Typical flow:

1. Send messages and tool definitions.
2. Inspect `finish_reason`.
3. If it is `tool_calls`, execute the tool in your app.
4. Send the tool result back as a `tool` message.
5. Request the final model response.

## Thinking / Reasoning

Skulk supports reasoning-aware chat for compatible models.

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "What is 127 * 43?"}],
    enable_thinking=True,
)

msg = response.choices[0].message
print(msg.reasoning_content)
print(msg.content)
```

Notes:

- `enable_thinking` is a Skulk extension.
- Reasoning support depends on model capabilities.
- Use `GET /v1/models` response `data[].resolved_capabilities` to decide whether a model supports thinking and whether clients should render a thinking toggle.
- Treat `resolved_capabilities` as the default tool-free request path; request-specific options such as tools can change prompt rendering and related resolved values for mixed-mode model families.
- Thinking-control semantics are model-aware:
  - if `supports_thinking_toggle` is `true`, send `enable_thinking=true` or `false` explicitly
  - `reasoning_effort="none"` disables thinking for toggleable models
  - if a model does not support toggleable thinking, Skulk ignores explicit toggle overrides but still preserves explicit non-disabled reasoning-effort hints when the model family supports them

## Builtin Browser Tools

**POST** `/v1/tools/web_search`

Execute Skulk's generic `web_search` tool and return structured search results.

```bash
curl -X POST http://localhost:52415/v1/tools/web_search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "foxlight skulk distributed inference",
    "top_k": 5
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `query` | string | Required search query. |
| `top_k` | integer | Optional max results, `1` to `10`, default `5`. |

Response fields:

| Field | Type | Notes |
|-------|------|-------|
| `query` | string | Original search query. |
| `provider` | string | Search backend identifier. |
| `results` | array | Ordered search results with `title`, `url`, and `snippet`. |

This endpoint is designed for client-executed tool loops. GPT-OSS can request
`web_search`, the client can call this endpoint, then send the JSON result back
as a `tool` message.

**POST** `/v1/tools/open_url`

Fetch one HTTP or HTTPS URL, follow redirects, and return structured metadata.

```bash
curl -X POST http://localhost:52415/v1/tools/open_url \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com/article"
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Required absolute `http://` or `https://` URL. |

Response fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Original requested URL. |
| `final_url` | string | Final URL after redirects. |
| `title` | string or null | Best-effort page title. |
| `status_code` | integer | Final HTTP status code. |
| `content_type` | string or null | Normalized response content type. |
| `provider` | string | Backend provider identifier. |

**POST** `/v1/tools/extract_page`

Fetch one HTTP or HTTPS URL and return bounded readable text extracted from the
response body.

```bash
curl -X POST http://localhost:52415/v1/tools/extract_page \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com/article",
    "max_chars": 12000
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Required absolute `http://` or `https://` URL. |
| `max_chars` | integer | Optional maximum characters, `500` to `50000`, default `12000`. |

Response fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Original requested URL. |
| `final_url` | string | Final URL after redirects. |
| `title` | string or null | Best-effort page title. |
| `text` | string | Readable extracted text. |
| `truncated` | boolean | Whether the text was clipped to `max_chars`. |
| `provider` | string | Backend provider identifier. |

These browser-tool endpoints are designed for client-executed tool loops. In
dashboard chat, GPT-OSS can request `web_search`, `open_url`, or
`extract_page`; the dashboard executes the endpoint call, then sends the JSON
result back as a `tool` message.

## Structured Output

`response_format` is accepted for compatibility, but Skulk does not currently enforce strict JSON mode or JSON schema validation.

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "Return valid JSON with three colors"}],
    response_format={"type": "json_object"},
)
```

For the best results, explicitly instruct the model to return valid JSON.

## OpenAI Responses API

**POST** `/v1/responses`

Use this when your client expects the OpenAI Responses format instead of Chat Completions.

```bash
curl -X POST http://localhost:52415/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "input": "Hello from the Responses API"
  }'
```

## OpenAI Embeddings API

**POST** `/v1/embeddings`

Generates embeddings with a mounted embedding model. The model must be placed
and running, and its card must declare `TextEmbedding`: a non-embedding model
returns **400 Bad Request**, and an unplaced model returns **404 No instance
found**.

```bash
curl -X POST http://localhost:52415/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-small-en-v1.5",
    "input": ["Skulk connects devices into one cluster"]
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required mounted embedding model id |
| `input` | string or array | Required text or list of texts; an empty list returns **400 Bad Request** |
| `encoding_format` | string | `float` (default) returns number arrays; `base64` returns each embedding as base64-encoded little-endian float32 bytes |
| `dimensions` | integer | Not supported. Any value returns **400 Bad Request**; embeddings come back at the model's native dimensionality |
| `user` | string | Optional caller identifier, accepted for compatibility |

The response is the OpenAI list shape: one `data[]` entry per input in input
order, the resolved `model`, and `usage` with prompt and total token counts.

## OpenAI Audio Speech API

**POST** `/v1/audio/speech`

Generates speech audio from a mounted text-to-speech model. The model must be
placed and running, and its resolved capabilities must include
`supports_speech_synthesis`.

```bash
curl -X POST http://localhost:52415/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output speech.wav \
  -d '{
    "model": "mlx-community/kokoro-test",
    "input": "Hello from Skulk speech serving",
    "voice": "af_heart",
    "response_format": "wav"
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required mounted TTS model id |
| `input` | string | Required text to synthesize |
| `voice` | string or null | Optional model-specific voice name. When omitted, Skulk applies the mounted model card's `audio.default_voice` when declared. |
| `speed` | number or null | Optional positive speaking speed multiplier |
| `response_format` | string or null | Optional output format: `mp3`, `wav`, `flac`, `ogg`, `opus`, or raw `pcm`. When omitted or set to `null`, Skulk uses `mp3` for `stream=true`; otherwise it uses the mounted model card default when declared and falls back to `mp3`; supported values are constrained by the model card when declared |
| `stream` | boolean | Optional. When `true`, Skulk returns a chunked HTTP response and yields MP3 or raw PCM bytes as the speech runner emits them; accepted only when the mounted TTS card explicitly declares `audio.supports_streaming = true` and every routable instance of the requested model has a ready runner |
| `streaming_interval` | number or null | Optional positive model-specific streaming cadence hint, accepted only with `stream=true` |
| `instruct`, `lang_code` | string or null | Optional model-specific generation hints |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens` | number or integer | Optional model-specific sampling controls |
| `reference_audio` | multipart file or null | Optional request-scoped voice-conditioning audio. Accepted only as a multipart upload for a mounted card declaring `audio.supports_reference_audio = true`; server-local paths are rejected |
| `reference_text` | string or null | Optional transcript of `reference_audio`; accepted only when the multipart upload is present |

The response body is raw audio bytes with a matching audio media type
(`audio/mpeg`, `audio/wav`, `audio/flac`, `audio/ogg`, `audio/opus`, or
`audio/pcm`).
For `stream=true`, the mounted TTS card must explicitly declare
`audio.supports_streaming = true`. The response format must currently
resolve to `mp3` or `pcm`; when a streaming request omits `response_format`, Skulk
requests `mp3` instead of the model card's non-streaming default. Skulk returns
`audio/mpeg` or `audio/pcm` with chunked HTTP bytes. Raw `pcm` is mono signed
16-bit little-endian audio; `X-Audio-Sample-Rate`, `X-Audio-Channels`, and
`X-Audio-Sample-Format` define its framing. Admission returns `503` if any routable
instance of the requested model lacks a ready runner. This is TTS output streaming, not a
realtime session: the request text is still a complete bounded input,
cancellation closes the command stream, and each chunk follows the mounted
model's generation cadence. The bundled Qwen3 TTS card declares MP3 and PCM streaming
support after live validation; Fish Audio and the other bundled speech cards
remain non-streaming. Streaming support is enabled card-by-card only when
the runtime can provide the encoder and the model has passed streaming
validation.

JSON requests remain text-only. To condition a supporting model with reference
audio, send the same scalar fields as multipart form values and include a
`reference_audio` file of at most 25 MiB. Skulk validates the mounted model and
audio metadata, pins the request to one ready single-host instance, and sends
the bytes over the node-addressed Zenoh data plane. Reference media is never
written to State or the event log, and the serving runner deletes its temporary
file when generation ends or fails. Reference-audio requests return **503
Service Unavailable** when the Zenoh data plane is unavailable; Skulk never
broadcasts private reference media through the gossipsub fallback.

`streaming_interval` without `stream=true`, `reference_text` without a
multipart reference upload, and JSON `reference_audio` path strings return
**400 Bad Request**.

## Skulk Audio Voices API

**GET** `/v1/audio/voices?model=<model-id>`

Returns stable built-in voice identifiers declared by one mounted TTS model.
This is a Skulk extension, not an OpenAI compatibility route. The model must
declare `audio.supports_voice_listing = true`; otherwise Skulk returns **400 Bad
Request**.

```bash
curl 'http://localhost:52415/v1/audio/voices?model=org/tts-model'
```

The response is `{ "object": "list", "data": [...] }`. Each item contains the
voice `id`, display `name`, mounted `model`, `kind = "builtin"`, and an ordered
`preferred_languages` array of BCP 47 tags when the model card declares
language preferences. Version 1 does not create or persist voice profiles.

## OpenAI Audio Transcriptions API

**POST** `/v1/audio/transcriptions`

Transcribes a multipart audio upload with a mounted speech-to-text model. The
model must be placed and running, and its resolved capabilities must include
`supports_transcription`.

```bash
curl -X POST http://localhost:52415/v1/audio/transcriptions \
  -F model=mlx-community/whisper-test \
  -F file=@sample.wav \
  -F response_format=verbose_json
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `file` | file | Required audio upload. Common WAV, MP3, FLAC, OGG, Opus, WebM, MP4/M4A containers are accepted up to 25 MiB |
| `model` | string | Required mounted STT model id |
| `language` | string or null | Optional input language hint |
| `prompt`, `context`, `text` | string or null | Optional model-specific transcription context |
| `response_format` | string | Optional output format: `json`, `text`, `verbose_json`, `srt`, `vtt`, or `ndjson`; default `json` |
| `temperature`, `max_tokens`, `chunk_duration`, `frame_threshold`, `prefill_step_size` | number or integer | Optional model-specific generation controls passed through only when the runner supports them |
| `word_timestamps` | boolean | Optional request for word timestamp metadata when supported |
| `timestamp_granularities` | string | Optional comma-separated or JSON-list timestamp granularity hints |
| `stream` | boolean | Optional. Requires a mounted card declaring `audio.supports_streaming = true` and ready runners. Returns typed SSE events by default; explicit `response_format=ndjson` retains progressive NDJSON framing. |

Response formats:

| Format | Media type | Shape |
|--------|------------|-------|
| `json` | `application/json` | `{ "text": "..." }` |
| `text` | `text/plain` | Plain transcript text |
| `verbose_json` | `application/json` | Transcript text plus language and segment metadata when the model returns it |
| `srt` | `application/x-subrip` | Subtitle output from model segments, with a zero-length fallback segment when timestamps are absent |
| `vtt` | `text/vtt` | WebVTT subtitle output from model segments |
| `ndjson` | `application/x-ndjson` | One JSON line per transcription chunk |

The endpoint never accepts server-local file paths. The API retains the bounded
multipart upload until the master selects the authoritative task placement,
then sends raw audio frames directly to the selected worker over
`SPEECH_MEDIA`. Only control-sized metadata and the task lifecycle enter the
ordered event log. The worker verifies the owner, frame count, and SHA-256
before injecting the payload into the serving runner; the runner writes a
temporary local audio file only while inference executes. With `stream=true`,
supported models yield their actual decoded text deltas. The default
`text/event-stream` response emits typed
`transcription.delta`, `transcription.completed`, `transcription.usage`, and
`transcription.error` events. Disconnecting before a terminal event cancels the
core command and releases its bounded output queue. An explicit
`response_format=ndjson` streams the existing per-chunk JSON shape one line at
a time. Cards without proven streaming support fail before response headers.
On Zenoh, upload frames are addressed to the selected worker. The gossipsub
fallback broadcasts target-tagged frames across the trusted cluster fabric;
non-target workers discard them before speech assembly.

## OpenAI Audio Translations API

**POST** `/v1/audio/translations`

Experimentally translates a multipart speech upload into English. The request
uses the same 25 MiB bounded upload path and response formats as transcription.
The mounted card must declare `audio.supports_translation = true`.

```bash
curl -X POST http://localhost:52415/v1/audio/translations \
  -F model=org/canary-model \
  -F file=@sample-fr.wav \
  -F language=fr \
  -F response_format=json
```

| Field | Type | Notes |
|-------|------|-------|
| `file` | file | Required bounded audio upload |
| `model` | string | Required mounted translation-capable STT model id |
| `language` | string or null | Optional source-language hint; required by the bundled Canary model |
| `prompt` | string or null | Optional model-specific translation context |
| `response_format` | string | `json`, `text`, `verbose_json`, `srt`, `vtt`, or `ndjson`; default `json` |
| `temperature` | number or null | Optional model-specific sampling temperature |

Translation target is English. The only gates are model truth and instance
availability: the mounted card must declare `audio.supports_translation =
true`, matching every other speech endpoint. Skulk maps the generic request to
model-family arguments inside the speech runner. The bundled
`CogniSoftOrg/canary-1b-v2-mlx-bf16` card is the initial supported model;
requests for that model return **400 Bad Request** when `language` is omitted.
Its upstream CC-BY-4.0 terms and NVIDIA attribution continue to apply.

## Claude Messages API

**POST** `/v1/messages`

Use this when your client expects Anthropic-style request and response shapes.

```bash
curl -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'
```

## Ollama API

Skulk supports several Ollama-compatible endpoints so tools like OpenWebUI can connect with minimal glue code.

### Chat

```bash
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Generate

```bash
curl -X POST http://localhost:52415/ollama/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "prompt": "Write a haiku about foxes"
  }'
```

### List models

```bash
curl http://localhost:52415/ollama/api/tags
```

### Show model details

```bash
curl -X POST http://localhost:52415/ollama/api/show \
  -H 'Content-Type: application/json' \
  -d '{"name": "mlx-community/Llama-3.2-1B-Instruct-4bit"}'
```

### Alias routes

Ollama clients differ in how they join a configured base URL with API paths, so
Skulk also serves alias routes that map onto the same handlers:

- `POST /ollama/api/api/chat` and `POST /ollama/api/v1/chat` alias
  `POST /ollama/api/chat`
- `GET /ollama/api/api/tags` and `GET /ollama/api/v1/tags` alias
  `GET /ollama/api/tags`
- `HEAD /ollama/` and `HEAD /ollama/api/version` answer the version probe some
  clients send before their first real request

## Image Generation and Editing

Skulk serves OpenAI-style image generation and editing from placed image
models (for example the bundled FLUX cards).

Availability note: these routes are always registered, but they return
**404 No instance found** until an instance of the requested image model is
placed and running. Image model cards are hidden from the model catalog
(`GET /v1/models`, placement previews, and the dashboard) unless the node runs
with `SKULK_ENABLE_IMAGE_MODELS=true`, so in practice serving image models
requires setting that environment variable before launching one.

### Generate images

**POST** `/v1/images/generations`

```bash
curl -X POST http://localhost:52415/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "exolabs/FLUX.1-schnell-4bit",
    "prompt": "a fox curled up in autumn leaves",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required placed image model id |
| `prompt` | string | Required generation prompt |
| `n` | integer | Number of images; default `1` |
| `size` | string | `auto` (default), `512x512`, `768x768`, `1024x768`, `768x1024`, `1024x1024`, `1024x1536`, or `1536x1024` |
| `quality` | string | `high`, `medium` (default), or `low` |
| `output_format` | string | `png` (default), `jpeg`, or `webp` |
| `response_format` | string | `b64_json` (default) returns inline base64 image data; `url` stores each image on this API node and returns a fetchable URL instead |
| `stream` | boolean | With `partial_images > 0`, returns an SSE stream of partial and final images instead of one JSON response |
| `partial_images` | integer | Number of intermediate previews per image when streaming; default `0` |
| `advanced_params` | object | Optional `seed`, `num_inference_steps` (1-100), `guidance` (1.0-20.0), `negative_prompt`, and `num_sync_steps` (1-100). When `seed` is omitted, Skulk assigns one so multi-node generation stays deterministic |

The non-streaming response is `{ "created": ..., "data": [...] }` with one
`b64_json` or `url` entry per image.

### Edit images

**POST** `/v1/images/edits`

Image-to-image editing. Unlike generations, this endpoint takes a multipart
form because it carries the input image:

```bash
curl -X POST http://localhost:52415/v1/images/edits \
  -F image=@input.png \
  -F prompt='make it snow' \
  -F model=exolabs/FLUX.1-Kontext-dev-4bit \
  -F response_format=b64_json
```

Form fields mirror the generation fields (`n`, `size`, `quality`,
`output_format`, `response_format`, `stream`, `partial_images`, and a JSON
`advanced_params` string), plus:

| Field | Type | Notes |
|-------|------|-------|
| `image` | file | Required input image, at most 24 MiB raw; larger uploads return **413 Request Entity Too Large** |
| `input_fidelity` | string | `low` (default) or `high`; controls how strongly the input image constrains the edit |

The input image travels to the selected worker over the bounded vision media
path described under chat image inputs; it is never written to the event log
or replicated `State`. The response shape matches image generation.

### Stored images

**GET** `/images`

Lists the images this API node currently stores for `response_format: "url"`
responses. Each entry carries `image_id`, `url`, `content_type`, and
`expires_at`.

**GET** `/images/{image_id}`

Returns one stored image as raw bytes with its stored content type.

```bash
curl http://localhost:52415/images
curl -o out.png http://localhost:52415/images/<image_id>
```

Stored images are node-local and expire one hour after creation; a missing or
expired id returns **404 Image not found or expired**. Use
`response_format: "b64_json"` when you need the image bytes to outlive the
cache.

## Benchmark Endpoints

Benchmark variants of the generation endpoints run the same admission and
validation as their non-bench counterparts, force a non-streaming run
(`stream=false`, and `partial_images=0` for images), and flag the task so the
serving runner collects generation statistics. The response extends the normal
response shape with two extra fields:

- `generation_stats`: runner-reported timing/throughput statistics for the run
- `power_usage`: per-node and total system power sampled from live cluster
  telemetry while the request ran

Endpoints:

- **POST** `/bench/chat/completions` takes the same body as
  `POST /v1/chat/completions` and returns the chat completion plus stats. The
  same `TextGeneration` admission applies.
- **POST** `/bench/images/generations` takes the same body as
  `POST /v1/images/generations`.
- **POST** `/bench/images/edits` takes the same multipart form as
  `POST /v1/images/edits`.

```bash
curl -X POST http://localhost:52415/bench/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Benchmark me"}]
  }'
```

## Cancel an In-Flight Command

**POST** `/v1/cancel/{command_id}`

Requests cancellation of one in-flight generation command by its command ID.
It covers text generation, image generation, embeddings, and speech synthesis
or transcription commands owned by the API node you call: Skulk closes the
local response stream and sends a task cancellation so the serving runner
stops instead of generating into the void.

Finding the command ID:

- streaming chat completions open with the SSE comment line
  `: command_id <id>`, and every streamed chunk's `id` field carries the same
  value
- non-streaming chat responses use the command ID as the response `id`
- Skulk control responses such as `POST /place_instance` return an explicit
  `command_id` field (those placement commands complete immediately and are
  not cancellable here)

```bash
curl -X POST http://localhost:52415/v1/cancel/<command_id>
```

A cancelled command returns
`{"message": "Command cancelled.", "command_id": "..."}`. An unknown or
already-completed command returns **404 Command not found or already
completed**. Command streams are node-local, so call the same API node that
accepted the original request. Simply disconnecting from a streaming response
triggers the same cancellation path implicitly.

## Model Discovery

### List models

**GET** `/v1/models`

```bash
curl http://localhost:52415/v1/models
```

This returns known model cards, not just running instances. `GET /models`
serves the same catalog through the same handler; prefer the `/v1/models` path
for OpenAI-compatible clients.

### Search Hugging Face

**GET** `/models/search?query=...&limit=...&mlx_only=...`

```bash
curl "http://localhost:52415/models/search?query=qwen3&limit=5"
```

Behavior note:

- `mlx_only=true` restricts results to the `mlx-community` author; the default
  searches all Hugging Face model repositories.
- Ordinary text queries use Hugging Face repository search.
- A query ending in `.gguf` also performs a bounded filename-aware fallback:
  Skulk broadens the model-name prefix, inspects those candidate repositories'
  manifests, and returns only exact filename matches. Exact matches carry a
  `matched_file` repo-relative path so the dashboard can preserve that quant.

### Add a Hugging Face model

**POST** `/models/add`

```json
{
  "model_id": "satgeze/Hy3-1M-GGUF",
  "gguf_file": "hy3-1M-MTP-IQ3_XXS.gguf",
  "source_revision": "0123456789abcdef0123456789abcdef01234567"
}
```

Fetches metadata and adds a custom model card to the cluster catalog. A
generated GGUF card is compatible with both llama.cpp engines and prefers
the served `llama_server` tags, so on a node running llama-server it gets
that engine's concurrency slots and is eligible for multi-node pooling via
RPC; nodes without a served binary fall through to the in-process engine.
The
`model_id` field is required. `gguf_file` is optional; when supplied it must be
an exact repo-relative GGUF weight path and the card pins that quant instead of
using Skulk's default GGUF preference. If the selected quant is split, Skulk
stores its first shard as the backend entrypoint while downloading the full
shard group. `source_revision` is also optional; when supplied it must be a full
40-character Hugging Face commit hash, and both card metadata and subsequent
artifact downloads are pinned to that immutable revision.

### Per-node storage breakdown

**GET** `/store/storage`

Returns the local node's storage picture: every staged model with its size,
last-use time, and whether a live instance (or one of its companion repos:
MTP sidecar, assistant, vision weights) currently depends on it, plus
event-log usage and free disk on the models volume. Cluster-wide views query
each node's API.

```bash
curl http://localhost:52415/store/storage
```

Staged copies are managed automatically when the model store is on: when an
instance shuts down (and at node startup, which reconciles copies orphaned
by a crash), not-in-use staged models are kept newest-first up to the
`staging_keep_recent_gb` grace budget (default 40 GiB) and evicted beyond
it. Set `cleanup_on_deactivate: false` in the staging config to keep every
staged copy and manage cleanup manually via `POST /store/purge-staging`.

## Placement and Instance Management

These endpoints are the heart of the Skulk control plane.

### Quick launch

**POST** `/place_instance`

```bash
curl -X POST http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/Qwen3.5-9B-4bit",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1,
    "excluded_nodes": []
  }'
```

| Field | Meaning |
|-------|---------|
| `model_id` | Hugging Face-style model ID |
| `sharding` | `Pipeline` or `Tensor` |
| `instance_meta` | `MlxRing`, `MlxJaccl`, or `LlamaRpc` (multi-node GGUF pooling: one driver node holds the model and each donor node lends GPU memory over the network) |
| `min_nodes` | Minimum nodes required for the placement |
| `excluded_nodes` | Optional. Node IDs the master should treat as if absent when scoring this placement. Already-running instances on those nodes are unaffected (exclusion is per-placement, not cluster-wide), and automatic repair re-placements of this instance (memory refusal, download failure) keep honoring the same exclusions. Default: `[]`. Note: node IDs are per-session, so they change when a cluster session restarts. |

The placement is validated against the current cluster state **before** the
command is forwarded, so an impossible placement fails at the API instead of
silently failing on the master:

- **400** with the specific reason: no connected cycle of `min_nodes` nodes,
  exclusions removed every candidate, the model does not support Tensor
  sharding, or a node cannot fit its weight shard plus runtime headroom
  (the error names the node and the GB arithmetic).
- **503** when cluster info is still being gossiped (a cluster that just
  formed): connection edges lag node identities by a few gossip rounds, and
  per-node memory info lags the edges. The request internally waits up to
  15 seconds for the info to arrive before giving up, so retry shortly on 503.

Memory fitting is checked **per node, not summed across the cycle**: Tensor
sharding splits weights evenly, Pipeline allocates layers proportionally to
each node's available memory, and every node must hold its share times a
runtime-overhead factor (KV cache, activations, runner) on top of the raw
weight bytes. A model that exactly equals a node's free memory is rejected,
because that placement would thrash, not run.

### Preview valid placements

**GET** `/instance/previews?model_id=...`

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit"
```

This is usually the best first Skulk-specific endpoint to call. It shows which combinations of sharding mode, networking mode, and node count are valid, and why invalid combinations fail.

Each preview's `instance_meta` reports the shape placement would *actually*
mint for that combination, not the shape that was asked about: for example a
GGUF model previewed at two GPU nodes reports `LlamaRpc` (driver plus memory
donors) even though the request enumerates the generic metas. Trust the
preview's reported meta when constructing a follow-up `POST /place_instance`.

Besides the planner's ranked pick per shape, the response also contains
per-host single-node previews marked `"alternative": true` for every other
host that passes admission. On a heterogeneous fleet the ranked winner is
typically the node with the most free accelerator memory; the alternatives
expose the full set of valid hosts so an operator can choose by cost,
locality, or to keep the big GPU free. Alternatives are omitted when
`node_ids` already constrains the hosts.

| Query parameter | Meaning |
|-----------------|---------|
| `model_id` | Required. Hugging Face-style model ID. |
| `node_ids` | Optional, repeatable. Restricts previews to candidate cycles that contain *all* of these node IDs (subset matching). |
| `excluded_node_ids` | Optional, repeatable. Excludes the listed node IDs from candidate cycles for every previewed combination. Mirrors the `excluded_nodes` field on `POST /place_instance` so dashboards can render an accurate preview against the post-exclusion topology. |

```bash
# Preview with one node excluded:
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit&excluded_node_ids=12D3KooWAbc..."
```

### Build a placement manually

**GET** `/instance/placement`

Use this when you want a specific combination and want to inspect the exact instance shape before launch.

### Create an instance from a fully specified placement

**POST** `/instance`

Use this when you already have an `instance` object and want exact control.

### Inspect one instance

**GET** `/instance/{instance_id}`

### Delete an instance

**DELETE** `/instance/{instance_id}`

## Download Management

### Start a node download

**POST** `/download/start`

Lower-level endpoint for explicit node download control.

### Delete a node download

**DELETE** `/download/{node_id}/{model_id}`

## Model Store Endpoints

These endpoints are available when the model store is configured.

If it is not configured, Skulk returns `503 Store not configured`.

### Store health

**GET** `/store/health`

Use this to confirm whether the store is configured and reachable.

### Store registry

**GET** `/store/registry`

Use this to inspect which models the shared store knows about.

The dashboard combines registry results with `GET /v1/models` metadata so it can
display derived tags such as `vision`, `thinking`, `embedding`, `tensor`, and
`optiq` in the Store list.

### Store downloads

**GET** `/store/downloads`

Use this to inspect in-progress shared-store download activity.

### Request a store download

**POST** `/store/models/{model_id}/download`

Use this when you want the store host to fetch and register a model.

The optional JSON body accepts `gguf_file` and `source_revision`:

```json
{
  "gguf_file": "<repo-relative path>",
  "source_revision": "0123456789abcdef0123456789abcdef01234567"
}
```

`gguf_file` pins which quant the store fetches for a multi-quant GGUF repo (that
file's shard group plus `config.json`). `source_revision` pins all repository
reads to a full immutable Hugging Face commit. When either value is omitted and
a curated model card declares it, the endpoint uses the card value. Otherwise,
omitting `source_revision` follows mutable `main`. A GGUF pin naming a file not
present in the selected revision falls back to the default at the store protocol
layer; the `/models/add` card-building endpoint validates exact pins before
requesting a download.

### Store download status

**GET** `/store/models/{model_id}/download/status`

### Delete a model from the store

**DELETE** `/store/models/{model_id}`

Removes the model from the store host (registry + disk) and broadcasts a
cluster-wide eviction so every node also drops its locally-staged copy, freeing
disk fleet-wide instead of leaving worker copies until they age out under
staging pressure. Returns `404` if the model is not registered in the store. (To
clear staged copies without deleting the store copy, use
`POST /store/purge-staging`.)

### Purge staging caches

**POST** `/store/purge-staging`

Use this to remove staged model artifacts from nodes without deleting the store copy itself.

### Start optimization

**POST** `/store/models/{model_id}/optimize`

Use this for workflows such as model optimization or alternate artifact generation.

## Models Endpoint

### List models

**GET** `/v1/models`

Returns the known model catalog, including downloaded models and catalog-backed
entries. Each item includes nullable `source_revision` metadata identifying the
qualified Hugging Face commit when its card pins immutable artifacts.

Important fields:

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Canonical model ID |
| `capabilities` | array | Functional capabilities such as `text`, `vision`, `thinking`, `code`, `embedding`, `tts`, or `stt` |
| `tags` | array | UI-friendly derived labels such as `vision`, `thinking`, `embedding`, `tts`, `stt`, `tensor`, and `optiq` |
| `supports_tensor` | boolean | Whether tensor parallel launch is supported |
| `base_model` | string | Base family or upstream source model when known |
| `audio` | object | Declared speech metadata from the model card, including `kind`, audio response formats, streaming/realtime flags, built-in `voices`, `default_voice`, voice/reference-audio flags, translation support, and sample rates |
| `resolved_capabilities.supports_speech_synthesis` | boolean | Whether clients should treat the model as a text-to-speech model |
| `resolved_capabilities.supports_transcription` | boolean | Whether clients should treat the model as a speech-to-text model |
| `resolved_capabilities.supports_speech_translation` | boolean | Whether clients should treat the model as supporting speech translation |
| `resolved_capabilities.supports_audio_output` | boolean | Whether the model produces audio output |
| `resolved_capabilities.supports_realtime_audio` | boolean | Whether the model declares realtime audio support |
| `resolved_capabilities.audio_response_formats` | array | Encoded audio formats the model can produce for speech synthesis |
| `runtime.mtp_sidecar_repo` | string | Repo of this model's MTP sidecar (prediction heads), when it declares one |
| `runtime.assistant_model_repo` | string | Repo of this model's speculative-decoding assistant (drafter), when it declares one |
| `runtime.served_spec_draft_repo` | string | Repo of this model's separate served-engine draft GGUF, when it declares one |

The dashboard uses `tags` for compact badges and `capabilities` for filtering
and richer tooltips. The `audio` and `resolved_capabilities.*speech*` fields
identify speech-capable models; `supports_speech_synthesis` models can serve
non-streaming `/v1/audio/speech` when mounted, and `supports_transcription`
models can serve non-streaming `/v1/audio/transcriptions`. The chat dashboard
uses the same metadata to show TTS playback and STT microphone controls only
for ready mounted speech models. Browser microphone capture is a browser
security feature, so STT recording controls require a secure origin such as
HTTPS or localhost even though the API endpoint itself is ordinary multipart
HTTP. Speech translation metadata remains reserved for later audio endpoints.
The three `runtime.*_repo` fields name a model's
speculative-decoding companions (a draft model or an MTP-head sidecar). Those
companion repos are downloaded and loaded automatically with their parent and
are not independently placeable, so the dashboard marks any store entry matching
one of these repos as a companion (a "Drafter" or "Sidecar" badge) rather than
offering it launch, placement, or optimize actions.

## Configuration Endpoints

### Get config

**GET** `/config`

Returns the current cluster config and config path. Sensitive values (`hf_token`) are stripped from the response.

The response also carries an `effective` block describing runtime-resolved values that are not part of the persisted file:

- `kv_cache_backend`: the KV cache backend actually in effect (config value or `SKULK_KV_CACHE_BACKEND` override)
- `has_hf_token`: whether a HuggingFace token is configured (via the file or `HF_TOKEN`), without exposing the token
- `experimental_mode_enabled`: whether this node runs with `SKULK_ENABLE_EXPERIMENTAL_MODE` set; when a release carries active experiments, the dashboard uses it to reveal the gated Experiments settings section

The persisted `experiments` section is deprecated compatibility surface: every
speech feature that incubated there has graduated to standard, and no built-in
experiment is currently active. The fields remain accepted (the strict config
would otherwise refuse an existing `skulk.yaml` that carries them) but are all
ignored:

- `experiments.tts_streaming`: deprecated compatibility field. Stable TTS
  streaming ignores this value and follows mounted model capability metadata.
- `experiments.stt_realtime`: deprecated compatibility field. It remains
  accepted in existing configuration but is ignored; realtime STT is selected
  from card truth, reachable transport, and ready mounted capacity.
- `experiments.speech_translation`: deprecated compatibility field. Speech
  translation is a standard capability; `/v1/audio/translations` serves for
  any mounted card declaring `audio.supports_translation = true`, and this
  value is accepted but ignored.

### Update config

**PUT** `/config`

Updates cluster-wide config. Important behavior:

- if you omit `hf_token`, Skulk preserves the existing value
- if you omit `logging`, Skulk preserves the existing logging config
- if you omit `experiments`, Skulk preserves the existing experiment toggles
- `hf_token` is not broadcast over gossipsub; it stays on the local node's `skulk.yaml`
- logging changes (enable/disable) take effect immediately on all nodes
- inference changes affect future launches
- model-store location changes generally require restart

### Filesystem browse

**GET** `/filesystem/browse`

Used by the dashboard to browse a safe subset of the filesystem when selecting config paths.

### Node identity

**GET** `/node/identity`

Returns hostname, preferred IP, and node identity information used by the dashboard.

`GET /node_id` is the minimal companion route: it returns only this node's ID
(the same value `/node/identity` reports, without the hostname and IP fields).
Node IDs are per-session and change when the process restarts.

### Restart a node

**POST** `/admin/restart?node_id=<optional node id>`

Gracefully restart the Skulk process on this or a remote node. When `node_id` is omitted or matches the local node, replaces the current process image in-place via `os.execv` (same PID). When `node_id` targets a remote node, sends a `RestartNode` command via pub/sub.

- GPU/Metal memory is released when the process image is replaced
- the node rejoins the cluster automatically on startup
- active inference is interrupted

Returns `{"status": "restarting", "node_id": "..."}` for local restarts, or `{"status": "restart_sent", "node_id": "..."}` for remote restarts.
If a local restart is already scheduled, returns HTTP 409 with `{"status": "restart_already_pending"}`.

### Onboarding status

**GET** `/onboarding`

Returns whether the dashboard onboarding flow has been completed on this node:

```json
{"completed": false}
```

**POST** `/onboarding`

Marks the local onboarding flow as complete and returns `{"completed": true}`.
The request takes no body. The flag is a node-local marker file, not cluster
state: each node tracks its own onboarding status, and the dashboard uses it to
decide whether to show the first-run setup flow.

```bash
curl http://localhost:52415/onboarding
curl -X POST http://localhost:52415/onboarding
```

## State, Events, and Tracing

### Cluster state

**GET** `/state`

Returns the cluster state as Skulk currently sees it.

The response also carries a derived `nodeHealth` map (keyed by node id) so a
problem on a node is visible rather than silent. Each entry is a `level`
(`ok`, `warn`, or `error`) plus a list of `reasons`, where each reason has a
`code`, a `message` describing what is wrong, and a `remediation` describing how
to fix it. It is computed read-only from state already in the response (terminal
download failures, low or full models-volume disk, and late liveness signals),
so it adds no new polling. Liveness uses the freshest of the dedicated
telemetry heartbeat, ordinary telemetry fallback, and `lastSeen`. The
`lastSeen` response field is only the last indexed control-plane event and may
be stale for a healthy node; it must not be interpreted as a heartbeat. A node
with no problems reports `level: "ok"` with an empty `reasons` list.

When known live node identities report different Skulk package versions or
source commits, every topology entry receives the warning-level
`version_mismatch` reason. This marks a staggered deployment as degraded until
all nodes converge. Operational visibility remains available, but events,
commands, state, and inference are not cross-version-compatible; finish the
deployment before starting new inference work.

The response carries a live `nodeResources` map as well. Each node entry includes
its placement `backends`, declared `participation`, resolved `dataTransport`
(`gossipsub` or `zenoh`), `zenohConnectedPeers` (the node's live Zenoh
peer-transport count, sampled at each advertisement; `null` when the node runs
gossipsub or while the count is not yet trustworthy after startup), and
`capabilityConflicts`: loud
observation-vs-declaration disagreements from backend derivation, each with a
`code`, `message`, and `remediation`. Conflicts also surface as `nodeHealth`
reasons on the same response: `gpu_serving_disabled` (error level: a visible
GPU that no engine would use, so serving would run far below hardware speed),
and the warning-level `gpu_detection_degraded` (an NVIDIA device present but
not fully detectable), `invalid_engine_binary` (an engine binary override
pointing at an unusable path), and `backend_override_conflict` (a declared
backend the observed hardware cannot support; the declaration is still
honored). A live fleet that advertises both transports receives
the error-level `data_transport_mismatch` reason in every `nodeHealth` entry.
Mixed DATA transports are unsupported: the signal is diagnostic and does not
bridge traffic. Configure and restart every node uniformly before serving
inference. A node advertising Zenoh with a trustworthy peer-transport count of
exactly 0 while at least one other live node also advertises Zenoh receives
the error-level `zenoh_isolated` reason: its control plane looks healthy but
every remote model or provider stream through it will fail. The typical cause
is a node that cannot reach peers via local multicast (for example one joined
over a routed or overlay network); the remediation is an explicit
`SKULK_ZENOH_CONNECT` peer endpoint plus a dialable `SKULK_ZENOH_LISTEN`
address. The API includes fresh telemetry-only management nodes, local or
remote, even when replicated worker membership does not carry their entries.

The `topology` map lists each node's connections. A socket edge carries the
peer's `sinkMultiaddr` plus a boolean `session` annotation distinguishing its
two sources: `session: false` (the default) marks an HTTP-probe-verified
advertised address, which is dialable and eligible as a placement host, while
`session: true` marks a live, authenticated fabric connection recorded as a
path in its own right. Session edges are what keep a NAT'd or proxied remote
member visible and placeable when none of its advertised addresses are
reachable; their recorded address is the connection's observed remote
endpoint, so placement host selection never uses them as a dial target.
Consumers rendering or analyzing the graph should treat `session` edges as
proof of connectivity, not as routable addresses.

Operational note:

- a follower may briefly report a local view that is behind the elected master
  while it is catching up
- on newer builds, catch-up can start from a snapshot plus retained replay tail
  instead of always rebuilding from event `0`
- if your cluster is mixed-version during rollout, upgrade all nodes before you
  rely on bounded replay retention on the master; an older restarted node may
  not be able to fully resync after old history has been compacted away

### Event log

**GET** `/events`

Returns stored events from the API-side event log.

### Diagnostics

- `GET /v1/diagnostics/node`
- `GET /v1/diagnostics/telemetry`
- `GET /v1/diagnostics/performance-envelopes`
- `GET /v1/diagnostics/performance-envelopes/cluster`
- `POST /v1/diagnostics/node/capture`
- `POST /v1/diagnostics/node/runners/{runner_id}/cancel`
- `GET /v1/diagnostics/cluster`
- `GET /v1/diagnostics/cluster/timeline`
- `GET /v1/diagnostics/cluster/{node_id}`
- `POST /v1/diagnostics/cluster/{node_id}/capture`
- `POST /v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel`

Use these endpoints when a node appears stuck loading, warming up, decoding, or
shutting down and you need a read-only snapshot without SSHing into every node.

Behavior notes:

- `GET /v1/diagnostics/telemetry` takes no parameters and returns aggregate
  metrics for the API node's isolated telemetry transport: fixed admission and
  network-queue capacities, current and maximum depth, offered/coalesced/dropped
  readings, successful publishes, publish failures and bytes, no-peer publish
  count (`noPeerPublishes`: publishes that found no peers subscribed on the
  telemetry protocol; sustained growth on a connected node means its
  heartbeats reach nobody and it will not appear in membership, typically a
  build/wire mismatch), plus oldest pending and last-successful-publish age. It never returns telemetry payloads, node/model maps,
  or completed attempt identifiers. Query each node directly for its local
  counters; this endpoint is deliberately separate from the node diagnostics
  bundle so additive telemetry instrumentation does not change that bundle's
  rolling-window schema.
- `GET /v1/diagnostics/performance-envelopes` returns this node's observe-only
  performance envelopes: for each `(hardware class, model, engine+backend,
  quantization)` it has served, a throughput-and-latency-versus-concurrency
  curve. Each envelope lists per-concurrency buckets (request count, mean/p50
  decode tokens/second, aggregate decode tokens/second, p50/p90
  time-to-first-token) and a simple `kneeConcurrency` estimate: the concurrency past which
  aggregate throughput stops rising. It is data only (no serving behavior is
  driven from it), kept in bounded memory, and never touches State, the event
  log, or the telemetry gossip plane. Concurrency is the serving instance's own
  in-flight load when a generation began: the served engines (llama.cpp server,
  vLLM) report their true in-flight count, so the curve is accurate across
  replicas and when several API nodes drive one instance; the single-stream
  engines report none and fall back to this API node's outstanding-request count.
  `GET /v1/diagnostics/performance-envelopes/cluster` fans out to every
  reachable member and returns each one's report, with unreachable members
  listed as explicit failures. The dashboard's Performance tab renders these.
- `GET /v1/diagnostics/node` returns the local node's runtime/config facts,
  resources, process tree, live runner-supervisor state, flight-recorder phase
  state, placement analysis, and `dataPlane` plus `provider` blocks. DATA diagnostics include
  transport/reorder mode; active and terminal lifecycle counts; first-byte and
  stream-span timing; duplicate, reordered, skipped, late, idle-timeout,
  transport-failure, and missing-lifecycle counters; plus router egress queue
  depth, independent command-queue count, per-owner pressure, drops, publish
  failures, idle stream reclamations, byte volume, and enqueue/publish latency.
  `dataPlane.egress.idleStreamReclaims` and each owner's matching counter
  increase when a remote command queue emits no frame for its 30-minute resource
  lease and is forcibly terminated and released. The dashboard Node tab renders
  the operational subset and highlights non-zero failure counters.
  Provider diagnostics report active unary calls and streams, concurrency
  limits and high-water marks, admissions and overload rejections, caller input
  queue depth, input/output frame and inline-media byte volume, first-output and
  total stream latency, terminal outcomes, cancellation requests, and
  missing-terminal streams. The same counters are grouped by qualified
  capability ID without retaining call IDs, audio, transcripts, or payloads.
- `POST /v1/diagnostics/node/capture` collects an on-demand local diagnostic
  bundle. Body fields are `runnerId`, `taskId`, `includeProcessSamples`, and
  `sampleDurationSeconds`; all are optional. When a runner/task is provided,
  the response includes that runner's bounded flight recorder, latest MLX
  memory snapshot, and best-effort macOS `sample`, `vmmap -summary`, and
  `footprint -p` output. Sampling failures are returned as structured partial
  failures instead of failing the bundle.
- `POST /v1/diagnostics/node/runners/{runner_id}/cancel` requests cooperative
  cancellation for one task that the local runner supervisor still knows about.
- `GET /v1/diagnostics/cluster` fans out to reachable peer APIs and returns
  partial results when some peers are unavailable. The sweep uses a fail-fast
  probe budget (single attempt, short timeout per advertised address) so one
  unroutable address cannot stall the response. Every topology member appears
  in `nodes`: peers with no reachable API route are explicit `ok: false`
  entries with a `no reachable API route` error rather than being omitted, so
  an overlay-joined node always has an observability presence. Peer diagnostic
  reads ignore unknown additive fields recursively and use compatibility
  defaults for additive counters. The response returns aggregate
  `versionStatus` (`consistent`, `mixed`, or `unknown`) and per-node
  `versionStatus` (`current`, `version_mismatch`, or `unknown`). This tolerance
  applies only to operational diagnostics, not correctness-bearing wire types.
- `GET /v1/diagnostics/cluster/timeline` stitches every reachable node's
  runner-supervisor diagnostics into one cross-rank chronological view. The
  response carries a per-runner synopsis sorted by `(modelId, deviceRank)`
  and every flight-recorder entry across all ranks merged and sorted by `at`.
  Use this when debugging a distributed deadlock: the rank-disagreement
  signature ("rank 0 entered `pipeline_last_eval_output` at T while rank 1
  is still in `pipeline_first_recv_like`") is invisible from any single
  node's local diagnostics but obvious top-to-bottom in the merged timeline.
  Unreachable peers are returned in `unreachableNodes` instead of failing
  the request.
- `GET /v1/diagnostics/cluster/{node_id}` proxies one reachable peer bundle or
  returns the local bundle if `node_id` is the current API node.
- `POST /v1/diagnostics/cluster/{node_id}/capture` proxies the same on-demand
  capture request to a reachable peer node.
- `POST /v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel` proxies
  the same cooperative live-runner cancellation request to a reachable peer.
- Placement diagnostics explicitly include whether the current master is part of
  each model placement, which helps investigate hangs where the master is not
  one of the inference ranks.
- The dashboard node inspect icon uses these endpoints to open live diagnostics
  for any reachable node. DATA pressure appears in the `DATA Plane` section.
- The diagnostics drawer prefers `Capture bundle` before cancellation so
  operators can collect phase, MLX memory, and process samples before changing
  the runner state.
- Runner cancellation is best-effort only. A wedged native/MLX runner may
  ignore the request and still require stronger intervention.
- Diagnostics endpoints do not currently kill or restart runners. Capture is
  read-only; the only mutating diagnostics action is the cooperative task-cancel
  request above.

Example:

```bash
curl http://localhost:52415/v1/diagnostics/node
curl http://localhost:52415/v1/diagnostics/telemetry
curl http://localhost:52415/v1/diagnostics/cluster
curl http://localhost:52415/v1/diagnostics/cluster/timeline
curl http://localhost:52415/v1/diagnostics/cluster/<node_id>
curl -X POST http://localhost:52415/v1/diagnostics/node/capture \
  -H 'content-type: application/json' \
  -d '{"runnerId":"<runner_id>","taskId":"<task_id>"}'
curl -X POST http://localhost:52415/v1/diagnostics/cluster/<node_id>/capture \
  -H 'content-type: application/json' \
  -d '{"runnerId":"<runner_id>","includeProcessSamples":true}'
curl -X POST http://localhost:52415/v1/diagnostics/node/runners/<runner_id>/cancel \
  -H 'content-type: application/json' \
  -d '{"taskId":"<task_id>"}'
curl -X POST http://localhost:52415/v1/diagnostics/cluster/<node_id>/runners/<runner_id>/cancel \
  -H 'content-type: application/json' \
  -d '{"taskId":"<task_id>"}'
```

### Field telemetry

- `GET /v1/telemetry/preview`

**GET** `/v1/telemetry/preview`

Returns the field-telemetry consent state and the exact pending sample batch
that would next be sent to the ingest service, so operators can inspect
precisely what leaves the cluster before or after opting in. Collection is
opt-in (dashboard consent flow; `telemetry:` in `skulk.yaml`) and
content-free: samples carry model ids, canonical hardware classes, timing,
token counts, and failure-class enums only. No parameters.

```json
{
  "enabled": false,
  "consent": "unasked",
  "pending": [],
  "dropped_since_start": 0,
  "install_id": "",
  "ingest_url": "https://..."
}
```

### Traces

- `GET /v1/tracing`
- `PUT /v1/tracing`
- `GET /v1/traces`
- `GET /v1/traces/cluster`
- `POST /v1/traces/delete`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`

Use these endpoints when you are debugging generation behavior, cluster execution, or performance.

Behavior notes:

- `GET /v1/tracing` returns whether runtime tracing is currently enabled for new
  requests across the live cluster session.
- `PUT /v1/tracing` toggles tracing cluster-wide for new requests only. It does
  not retroactively trace in-flight work.
- `GET /v1/traces*` reads local trace artifacts stored on the current node.
- `GET /v1/traces/cluster*` fans out to reachable peer APIs, deduplicates by
  `task_id`, and proxies read-only trace access from any reachable node.
- `POST /v1/traces/delete` remains local-only in v1 even when cluster browsing
  is enabled.

### Runtime tracing control

**GET** `/v1/tracing`

Returns the current cluster tracing state:

```json
{"enabled": false}
```

**PUT** `/v1/tracing`

Enable or disable tracing for new requests across the current cluster session.

Request body:

```json
{"enabled": true}
```

Response body:

```json
{"enabled": true}
```

Operational notes:

- this is a runtime toggle, not a restart-required config edit
- it applies to new requests only
- it does not retroactively trace work already in flight
- the dashboard traces page uses this same API

### Local trace endpoints

These endpoints operate on trace artifacts stored on the current node:

- `GET /v1/traces` lists local trace artifacts with metadata such as task kind,
  model, source nodes, and tool-activity tags
- `GET /v1/traces/{task_id}` returns structured trace events for one task
- `GET /v1/traces/{task_id}/stats` returns aggregated timing summaries
- `GET /v1/traces/{task_id}/raw` downloads Chrome-trace-compatible JSON
- `POST /v1/traces/delete` deletes one or more local trace artifacts

Example:

```bash
curl http://localhost:52415/v1/traces
curl http://localhost:52415/v1/traces/<task_id>/stats
curl -OJ http://localhost:52415/v1/traces/<task_id>/raw
```

### Cluster trace endpoints

These endpoints let a dashboard or script on any reachable node browse traces
across the cluster:

- `GET /v1/traces/cluster`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`

Operational notes:

- cluster browsing is read-only in v1
- the API fans out to reachable peer APIs and deduplicates traces by `task_id`
- if some peers are unreachable, cluster results may be partial
- source node metadata in responses tells you which nodes contributed trace content

Example:

```bash
curl http://localhost:52415/v1/traces/cluster
curl http://localhost:52415/v1/traces/cluster/<task_id>/stats
curl -OJ http://localhost:52415/v1/traces/cluster/<task_id>/raw
```

## Extension Capabilities

### List a node's served capabilities

```
GET /v1/capabilities
GET /v1/capabilities?node_id=<id>
```

Returns the self-describing capability descriptors served by a node's
provider extensions (see [Extensions](extensions)). Without `node_id` it
describes the node serving the request; with a peer's `node_id` it proxies
that peer's describe surface (empty when the peer is unreachable). Each
descriptor carries the capability `id`, semantic `version`, a human/LLM-readable
description, JSON Schemas for input and output, the call's I/O mode, and the
response maps each `id@version` to a content revision digest so callers can pin
the exact shape they discovered. Production nodes also include first-party
provider descriptors, including the mounted-model speech providers and stable
`vad@1.0.0`, so descriptor presence alone is not always a liveness claim.
Extensions consume this through
`describe_node`; the light discovery layer (which nodes offer which capability
tag) rides the telemetry plane and appears as `nodeCapabilities` in
`GET /state`.

### Invoke a capability on this node

```
POST /v1/capabilities/call
```

Dispatch one unary capability call to this node's provider extensions. The
body is the typed call envelope: `call_id`, `capability_id`, exact `version`,
the `descriptor_revision` pinned at discovery, `caller_node`, `target_node`,
`timeout_seconds`, and the opaque `payload`. The payload is validated against
the descriptor's input schema before the provider runs, the result against
its output schema after, and payloads are capped at 1 MiB in each direction.
A syntactically valid envelope always gets HTTP 200 with a typed result
(`call_id`, `ok`, `result`, `error`); failures arrive as machine-readable
codes (`not_found`, `version_mismatch`, `revision_mismatch`,
`invalid_payload`, `invalid_result`, `payload_too_large`, `overloaded`,
`timeout`, `provider_error`), so callers switch on `error.code` rather than
transport status. A body that does not parse as the envelope at all
(malformed JSON, missing fields, out-of-range values) gets the standard 422,
since there is no call id to correlate a typed result to. Extensions
normally use this through their context's `call_capability` rather than
calling the endpoint directly.

### Open a streaming capability on this node

```
POST /v1/capabilities/stream
```

This is the control-sized node-to-node opening verb for a provider descriptor
whose `io_mode` is `server_streaming`, `client_streaming`, or `bidirectional`.
The request body is the same pinned
`CapabilityCall` envelope used by unary calls. Skulk checks target identity,
handler/version/revision, the 1 MiB request limit, the descriptor's input
schema, the per-node stream concurrency bound, and the single deadline budget.
Providers may then perform dynamic admission, such as checking that a requested
model is mounted and healthy, inside those same bounds and before lifecycle
creation.
It then returns a typed `CapabilityResult`: `ok: true` with
`{"admitted": true, "io_mode": "..."}` means the stream was admitted; a
pre-admission rejection uses the same typed call errors as the unary endpoint
and creates no stream.

Output is **not** an HTTP response stream. After admission, the provider emits
`started`, ordered `chunk` frames, and exactly one `completed`, `failed`, or
`cancelled` terminal on the provider DATA topic. The handler must return after
yielding that terminal; Skulk withholds it until iterator exhaustion so handler
cleanup finishes before dependent calls can observe completion. Malformed or
trailing handler output closes a closable iterator before Skulk publishes its
synthetic failure terminal. Structured frame metadata is JSON-schema validated
against `output_chunk_schema`; realtime media is an optional raw binary
attachment capped at 1 MiB per frame, while large immutable results use staged
blob references. The topic is node-addressed to
`caller_node`, short-circuits same-node calls, and uses the DATA plane's bounded
per-owner/call/direction Zenoh queues for remote calls. Extensions consume the
flow through `ExtensionContext.stream_capability(...)`, which returns a
`CapabilityStreamSession` containing the typed opening result, one output
iterator, and an `input` sink only for client-streaming/bidirectional calls.
`input.send_chunk()` moves structured metadata plus optional raw media to the
provider; `input.complete()` half-closes caller input without closing provider
output. Input cancellation, invalid schema, unresolved sequence gaps, and queue
pressure produce a typed terminal for only that call.

### Cancel an admitted capability stream

```
POST /v1/capabilities/stream/cancel
```

Accepts `call_id`, `caller_node`, `target_node`, and an optional cancellation
message. Only the caller identity that opened the active stream can cancel it.
Cancellation is idempotent: an active handler is cancelled and emits one typed
`cancelled` terminal; an already-terminal or unknown call returns
`{"cancelled": false}`. `stream_capability` sends this request automatically
when its iterator closes before a terminal frame.

### Stream speech through the built-in TTS provider

Production nodes describe a first-party `tts@1.0.0` server-streaming
capability. It is a facade over core `mlx_audio` serving, not a second model
runtime: the provider translates the generic call into the existing mounted
model `SpeechSynthesis` command and translates `AudioChunk` output into raw
binary provider media frames.

Its payload accepts:

| Field | Type | Meaning |
|-------|------|---------|
| `model` | string | Required mounted TTS model id |
| `text` | string | Required non-empty text to synthesize |
| `response_format` | string | Optional; only `mp3` is accepted in version 1 |
| `voice` | string | Optional model-specific voice |
| `streaming_interval` | number | Optional positive generation cadence hint |
| `speed`, `instruct`, `lang_code` | model-specific | Optional speech controls |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens` | number | Optional model-specific sampling controls |

Each `chunk` payload reports `model`, `format: "mp3"`, `chunk_index`,
`is_partial`, and optional `sample_rate`; the MP3 bytes are carried beside it
as an `InlineMediaAttachment` with `media_type: "audio/mpeg"`.

The descriptor is always available for contract discovery, while the `tts`
telemetry tag is advertised when at least one eligible model is mounted and
every routable instance of an eligible model has a ready runner.
Dynamic admission rechecks the requested model before `started`. A
caller cancellation propagates to the underlying synthesis command.

### Transcribe a bounded clip through the built-in STT provider

Production nodes describe a first-party `stt@1.0.0` client-streaming
capability and advertise its `stt` telemetry tag while a ready, single-host STT
runner is mounted. The operation is batch inference: client streaming is used
only so encoded audio remains binary provider media instead of base64 in the
unary JSON envelope.

The opening payload requires `model` and optionally accepts `filename`,
`content_type`, `language`, `prompt`, `temperature`, `max_tokens`,
`chunk_duration`, `frame_threshold`, `context`, `prefill_step_size`, `text`,
`word_timestamps`, and `timestamp_granularities`. Send the complete clip as one
or more ordered `InlineMediaAttachment` values, each at most 1 MiB and at most
25 MiB in aggregate, then call the input sink's `complete()` method. Empty,
oversized, cancelled, or non-inline input fails only that provider call.

After input half-close, Skulk runs the existing mounted-model
`AudioTranscription` path. The provider emits no partial transcript chunks; its
single `completed` payload contains `model`, `text`, and optional `language`
and `segments`. Managed blob references are not accepted until Skulk has a
general immutable blob service.

### Transcribe realtime PCM through the built-in STT provider

Production nodes also describe a first-party `stt.realtime@1.0.0`
bidirectional capability. It is advertised only when eligible mounted capacity
is ready and reachable. The owning API may differ from the speech runner node:
same-node input short-circuits locally, while remote input requires the
node-addressed Zenoh data plane. Remote capacity is not advertised when Zenoh
is unavailable.

The opening payload accepts:

| Field | Type | Meaning |
|-------|------|---------|
| `model` | string | Required mounted realtime STT model id |
| `sample_rate` | integer | Input PCM sample rate from 8000 through 96000 Hz |
| `temperature` | number | Optional decode temperature; defaults to `0` |
| `transcription_delay_ms` | integer | Optional upstream cadence from 80 through 2400 ms, in 80 ms steps; defaults to `480` |

Each caller `chunk` must carry an `InlineMediaAttachment` containing mono,
signed little-endian 16-bit PCM. Its payload and attachment metadata must agree
on `format: "pcm_s16le"`, `sample_rate`, and `channels: 1`. The input sink's
`complete()` method half-closes audio input and lets final decoding finish.
Provider output `chunk` frames contain `model`, transcript `text`, and
`is_partial: true`; the `completed` payload contains the accumulated final text
with `is_partial: false`. Skulk withholds that terminal until core cleanup has
sent `TaskFinished` and replicated state reports the task terminal or deleted,
so a following turn cannot be rejected against stale busy state.

Admission pins a `RealtimeAudioTranscription` task to one selected single-host
model instance, and the master reserves that instance against concurrent
admission. Audio bypasses event-sourced State and travels through bounded raw
PCM packets to the serving worker plus a bounded worker-to-runner channel; only
transcript output uses the existing core DATA lifecycle. The mounted upstream
model must expose a true `create_streaming_session` interface. Batch STT cards
are never promoted to realtime by buffering a complete recording.

### Detect speech turns through the built-in VAD provider

Every production API advertises `vad@1.0.0`. Open a bidirectional capability
stream with `sample_rate` set to 8000, 16000, 32000, or 48000 and send ordered
mono `pcm_s16le` inline media. Optional settings are `aggressiveness` (0-3),
`frame_ms` (10, 20, or 30), `minimum_speech_ms`, `silence_hangover_ms`,
`preroll_ms`, and `maximum_utterance_ms`; the descriptor publishes their exact
bounds. Output chunks contain `event` (`speech_started` or `speech_stopped`),
`timestamp_ms`, `reason`, and `preroll_ms`. The completed payload reports the
turn count. Input must end on an exact classifier-frame boundary. Media is
processed within the call and is not retained.

### Realtime transcription WebSocket compatibility edge

```
WS /v1/realtime?model=<mounted-realtime-stt-model>
```

This transcription-only WebSocket is an API-edge adapter over the same
`stt.realtime@1.0.0` provider described above. It does not own model placement,
runner sessions, or a second speech implementation. The API node accepting the
socket owns the provider call, which may select a speech runner on another node.
The same truthful-card, runner-readiness, and Zenoh remote-capacity gates apply.
OpenAPI does not model WebSocket operations, so this manual section is the
normative edge contract;
the underlying provider opening remains represented by the documented HTTP
capability endpoints.

The wire contract implements a bounded subset of OpenAI Realtime transcription:

| Direction | Event | Behavior |
|---|---|---|
| server to client | `session.created` | Reports a `type: transcription` session with the selected model and fixed PCM input configuration. |
| client to server | `session.update` | Confirms the current nested `audio.input` configuration. `turn_detection` may be null or a bounded `server_vad` configuration. Optional `response` selects a mounted chat `model`, optional mounted `tts_model`, optional `voice`, `max_output_tokens` from 1 through 4096 (default 256), and `enable_thinking` (default false); attempts to change the input model/codec, enable noise reduction/language hints, or add unsupported fields are rejected. |
| server to client | `session.updated` | Confirms an accepted current session update. |
| client to server | `input_audio_buffer.append` | Appends one base64 PCM16 frame and immediately forwards its decoded bytes as binary Fabric media. |
| client to server | `input_audio_buffer.commit` | Half-closes the current utterance and triggers final provider drain. Empty commits and duplicate manual commits are rejected. A manual commit racing after server VAD has already auto-committed the same utterance is an idempotent no-op. The next turn may begin after its completed event. |
| server to client | `input_audio_buffer.speech_started` | Reports the detected start timestamp and current item when server VAD is enabled. |
| server to client | `input_audio_buffer.speech_stopped` | Reports the detected end timestamp immediately before server VAD commits the utterance. |
| server to client | `input_audio_buffer.committed` | Confirms the input half-close. |
| server to client | `conversation.item.input_audio_transcription.delta` | Carries one provider transcript delta after commit. |
| server to client | `conversation.item.input_audio_transcription.completed` | Carries the accumulated final transcript, completes the current item, and leaves the socket ready for another turn. |
| server to client | `conversation.item.input_audio_transcription.failed` | Carries a provider/transport/cancellation terminal failure. |
| server to client | `response.created` | Announces automatic assistant work after a final transcript when `session.response` is configured. |
| server to client | `response.output_text.delta` / `response.output_text.done` | Streams visible assistant text and its bounded final value. Reasoning tokens and tool calls are not exposed or synthesized. |
| server to client | `response.audio.delta` / `response.audio.done` | Streams base64 MP3 chunks from the selected mounted `tts_model`. |
| client to server | `response.cancel` | Cancels active model generation or TTS. New speech detected by server VAD performs the same cancellation before starting the next turn. |
| server to client | `response.done` | Terminates one assistant response with `completed`, `cancelled`, or `failed` status. |
| server to client | `error` | Reports invalid client events, unsupported configuration, or response failures. Policy and transport errors may close the socket; response failures are non-terminal to the socket and are followed by `response.done`. |

Version 1 accepts JSON text WebSocket messages and base64-encoded mono,
signed little-endian PCM16 at 24 kHz. A decoded audio frame is capped at 1 MiB,
the encoded WebSocket event at 2 MiB, and one session at 64 MiB of decoded
audio. Provider transcript text is capped at 1 MiB per event and in the
pre-commit buffer; overflow emits a typed transcription failure and closes the
socket with `1011`. `input_audio_buffer.clear` is deliberately unsupported because the API
forwards audio incrementally and retains no replay buffer that could safely
retract already-delivered media. Browser connections must be same-origin; SDK
clients without an `Origin` header remain supported.

`turn_detection: {"type":"server_vad"}` enables server-owned WebRTC VAD.
Optional settings are `aggressiveness` (0-3), `prefix_padding_ms` (0-2000),
`silence_duration_ms` (20-5000), `minimum_speech_ms` (20-5000), and
`maximum_utterance_ms` (100-120000). The edge incrementally resamples the
24 kHz input to the classifier's 16 kHz frame contract, emits typed speech
boundaries, and commits on silence or the maximum utterance duration. The edge
forwards VAD-enabled input in 20 ms source-rate slices and stops at the
detected boundary, so the unprocessed remainder of a large append cannot leak
into the committed utterance. The socket serializes turns: each utterance opens
one bounded Fabric provider call,
and audio appended while a committed turn is still draining receives a
non-terminal `turn_in_progress` error. Completed turns rotate `item_id`, link
the next commit through `previous_item_id`, reset VAD state, and release their
provider capacity. The 64 MiB decoded-audio bound applies across the complete
WebSocket session.

The dashboard chat microphone uses this edge only when both the selected model
declares streaming/realtime audio and the API node currently advertises the
stable `stt.realtime` provider. An `AudioWorklet` captures mono browser
samples, the dashboard continuously resamples them to 24 kHz PCM16, and the
client aggregates worklet callbacks into 100 ms transport frames before the
mic control commits the socket when recording stops. Realtime mode can retain
the socket across server-VAD turns, show partial transcripts in the editable
draft, and optionally auto-send final transcripts through the selected mounted
chat model. If either capability truth is absent, chat retains the batch `MediaRecorder` plus
`POST /v1/audio/transcriptions` path.

When `response` is configured, the API node that owns the WebSocket retains the
bounded text-only conversation history for that socket, routes each final
transcript through the selected mounted chat model with the configured
`max_output_tokens` ceiling. Hidden reasoning is disabled by default so the
bounded budget produces speech-ready visible text; clients may opt in with
`enable_thinking`. The edge then optionally opens a normal `tts@1.0.0` Fabric
provider stream for the visible final answer. Explicit
`response.cancel`, a new non-VAD audio turn, or VAD speech detection cancels the
active model/TTS command before the replacement turn proceeds. Media bytes are
not added to conversation history or State.

The edge does not implement noise reduction, G.711, ephemeral session-token
creation, client-created conversation items, or tool execution.
Provider capacity failures close with retryable WebSocket code `1013`; client
protocol/policy violations use `1003`, `1008`, or `1009`; internal provider
failures use `1011`. Disconnecting before a terminal event cancels the provider
input and output directions.

For compatibility with clients written against the earlier transcription beta,
the edge also accepts `transcription_session.update` with
`input_audio_format`, `input_audio_transcription`, `turn_detection`, and
`input_audio_noise_reduction`, replying with `transcription_session.updated`.

### Compose a typed Fabric speech chain

**WS** `/v1/fabric/chains/speech?stt_model=<mounted-realtime-stt-model>`

This first-class composition surface uses the same hardened event contract as
`/v1/realtime`, but names the endpoint by its Fabric role. After
`session.created`, send `session.update` to select server VAD and an optional
`response` containing mounted `model`, `tts_model`, and `voice` participants
plus optional bounded `max_output_tokens` and `enable_thinking` controls.
Input PCM, transcript events, assistant text, TTS audio, cancellation, bounded
history, and terminal status retain the contracts documented above.

The chain resolves every participant through normal mounted capability and
health checks. It does not create a second runtime, persist audio or transcripts
in State, perform graph search, or introduce prompt-level authority. Remote
participants continue to use the normal bounded provider data plane, and socket
disconnect or `response.cancel` reaches the active provider/model commands.

```bash
curl http://localhost:52415/v1/capabilities
```

```json
{
  "node_id": "12D3KooW...",
  "capabilities": [
    {
      "id": "echo",
      "version": "1.0.0",
      "title": "Echo",
      "description": "Returns the input text unchanged.",
      "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
      "output_schema": {"type": "object"},
      "io_mode": "unary",
      "input_chunk_schema": null,
      "output_chunk_schema": null,
      "annotations": null
    }
  ],
  "revisions": {"echo@1.0.0": "5a1c9e327b6f4d08"}
}
```

## Connectivity Endpoints

### Tailscale status

```
GET /v1/connectivity/tailscale
GET /v1/connectivity/tailscale?node_id=<id>
```

Returns whether tailscaled is running on a node and, if so, the node's Tailscale IP, hostname, DNS name, and tailnet. All fields except `running` are `null` when tailscaled is not installed or not running.

Pass `node_id` to proxy the request to a specific cluster node. Omit it to query the local node directly. Returns `404` if the target node is not reachable.

**Response fields:**

| Field | Type | Description |
| --- | --- | --- |
| `running` | boolean | `true` when tailscaled reports `BackendState == "Running"` |
| `selfIp` | string \| null | Node's Tailscale IPv4 address (100.x.x.x range) |
| `hostname` | string \| null | Node hostname as registered in the tailnet |
| `dnsName` | string \| null | Fully-qualified Tailscale MagicDNS name, e.g. `my-node.tailnet-abc.ts.net` |
| `tailnet` | string \| null | Tailnet name derived from `dnsName` |
| `version` | string \| null | Tailscale client version string |

```bash
# Local node
curl http://localhost:52415/v1/connectivity/tailscale

# Specific cluster node
curl "http://localhost:52415/v1/connectivity/tailscale?node_id=<node-id>"
```

### Remote access info

```
GET /v1/connectivity/remote-access
```

Returns aggregated remote access information for the local node: LAN address, Tailscale address, and a `preferredUrl` (Tailscale if running, otherwise LAN). When Tailscale is running, `preferredUrl` uses the node's MagicDNS name (`my-node.tailnet-abc.ts.net`) if available, falling back to the raw `100.x.x.x` IP. `operatorUrl` appends `/operator` to `preferredUrl` (suitable for QR code generation so mobile users land directly on the operator panel).

**Response fields:**

| Field | Type | Description |
| --- | --- | --- |
| `local.ip` | string \| null | Preferred LAN IPv4 address |
| `local.port` | integer | API/dashboard port |
| `local.url` | string \| null | `http://{ip}:{port}` |
| `tailscale.running` | boolean | `true` when tailscaled is connected |
| `tailscale.ip` | string \| null | Tailscale IPv4 address (100.x.x.x) |
| `tailscale.dnsName` | string \| null | MagicDNS fully-qualified name, e.g. `my-node.tailnet-abc.ts.net` |
| `tailscale.port` | integer | API/dashboard port |
| `tailscale.url` | string \| null | `http://{dnsName or ip}:{port}` if running |
| `preferredUrl` | string \| null | MagicDNS URL if available, else Tailscale IP URL, else LAN URL |
| `operatorUrl` | string \| null | `preferredUrl + /operator` |

```bash
curl http://localhost:52415/v1/connectivity/remote-access | python3 -m json.tool
```

Example response when Tailscale is running with MagicDNS:

```json
{
  "local": { "ip": "192.168.1.5", "port": 52415, "url": "http://192.168.1.5:52415" },
  "tailscale": {
    "running": true,
    "ip": "100.101.102.103",
    "dnsName": "my-node.tailnet-abc.ts.net",
    "port": 52415,
    "url": "http://my-node.tailnet-abc.ts.net:52415"
  },
  "preferredUrl": "http://my-node.tailnet-abc.ts.net:52415",
  "operatorUrl": "http://my-node.tailnet-abc.ts.net:52415/operator"
}
```

## Operator App Integration

The operator panel at `/operator` is designed for mobile access and can also be driven by a native app. The relevant API endpoints are:

### Node and cluster state

| Endpoint | Description |
| --- | --- |
| `GET /state` | Full cluster state: nodes, instances, runners, memory, GPU |
| `GET /node_id` | Local node's ID |
| `GET /node/identity` | Node ID, hostname, and preferred LAN IP |

### Remote access and connectivity

| Endpoint | Description |
| --- | --- |
| `GET /v1/connectivity/remote-access` | LAN + Tailscale addresses, preferred URL, operator URL for QR |
| `GET /v1/connectivity/tailscale` | Tailscale status for local node |
| `GET /v1/connectivity/tailscale?node_id=<id>` | Tailscale status for a specific peer node |

### Node management

| Endpoint | Description |
| --- | --- |
| `POST /admin/restart?node_id=<id>` | Send a restart command to any node in the cluster |

### Typical operator app workflow

1. Call `GET /v1/connectivity/remote-access` on the initially discovered node to get the `preferredUrl`, then use that as the base URL for subsequent calls.
2. Poll `GET /state` every 5 seconds for node health (memory, GPU, temperature).
3. Show per-node cards with restart buttons that call `POST /admin/restart?node_id=<id>`.
4. On first launch or settings screen, show the `operatorUrl` as a QR code so users can hand it off to another device.

## Helpful Next Docs

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- [Tracing and debugging](tracing)
- [Model store guide](model-store)
- [Architecture overview](architecture)
- [API Reference](/api/skulk-api)
