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

## The Most Important Rule

For text generation, Skulk is not just a stateless HTTP server.

**A model has to be placed and running before chat-style requests will work.**

The dashboard enforces this too: it will not let you chat until a model is placed and ready.

If you call `/v1/chat/completions` too early, Skulk will usually return something like:

```json
{
  "error": {
    "message": "No instance found for model mlx-community/...",
    "type": "Not Found",
    "code": 404
  }
}
```

## Start Here

If you want your first successful API call, use this flow:

1. Start Skulk with `uv run skulk`.
2. Preview valid placements for a model.
3. Launch a placement.
4. Wait for the model to be ready.
5. Send a chat request.

## Quick Navigation

- First working request: [First Success Flow](#first-success-flow)
- OpenAI-compatible chat: [OpenAI Chat Completions](#openai-chat-completions)
- OpenAI Responses format: [OpenAI Responses API](#openai-responses-api)
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
- `POST /v1/messages`
- `POST /ollama/api/chat`
- `POST /ollama/api/generate`
- `GET /ollama/api/tags`
- `POST /ollama/api/show`
- `GET /ollama/api/ps`
- `GET /ollama/api/version`

### Skulk Control APIs

- `GET /v1/models`
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
- `POST /store/models/{model_id}/optimize`
- `GET /store/models/{model_id}/optimize/status`
- `GET /filesystem/browse`
- `GET /node/identity`
- `GET /v1/traces`
- `POST /v1/traces/delete`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`

For the full interactive reference with request/response schemas, see the [API Reference](/api/skulk-api).

## OpenAI Chat Completions

**POST** `/v1/chat/completions`

This is the main chat-generation endpoint for both text-only and multimodal
models.

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
| `max_tokens` | integer | Max generated tokens. |
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
- Phase 2 semantics are model-aware:
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

## Model Discovery

### List models

**GET** `/v1/models`

```bash
curl http://localhost:52415/v1/models
```

This returns known model cards, not just running instances.

### Search Hugging Face

**GET** `/models/search?query=...&limit=...`

```bash
curl "http://localhost:52415/models/search?query=qwen3&limit=5"
```

Behavior note:

- Skulk searches `mlx-community` first.
- If that returns nothing, it falls back to a broader Hugging Face search.

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
    "min_nodes": 1
  }'
```

| Field | Meaning |
|-------|---------|
| `model_id` | Hugging Face-style model ID |
| `sharding` | `Pipeline` or `Tensor` |
| `instance_meta` | `MlxRing` or `MlxJaccl` |
| `min_nodes` | Minimum nodes required for the placement |

### Preview valid placements

**GET** `/instance/previews?model_id=...`

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit"
```

This is usually the best first Skulk-specific endpoint to call. It shows which combinations of sharding mode, networking mode, and node count are valid, and why invalid combinations fail.

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

### Store download status

**GET** `/store/models/{model_id}/download/status`

### Delete a model from the store

**DELETE** `/store/models/{model_id}`

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
entries.

Important fields:

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Canonical model ID |
| `capabilities` | array | Functional capabilities such as `text`, `vision`, `thinking`, `code`, or `embedding` |
| `tags` | array | UI-friendly derived labels such as `vision`, `thinking`, `embedding`, `tensor`, and `optiq` |
| `supports_tensor` | boolean | Whether tensor parallel launch is supported |
| `base_model` | string | Base family or upstream source model when known |

The dashboard uses `tags` for compact badges and `capabilities` for filtering
and richer tooltips.

## Configuration Endpoints

### Get config

**GET** `/config`

Returns the current cluster config and config path. Sensitive values (`hf_token`) are stripped from the response.

### Update config

**PUT** `/config`

Updates cluster-wide config. Important behavior:

- if you omit `hf_token`, Skulk preserves the existing value
- if you omit `logging`, Skulk preserves the existing logging config
- `hf_token` is not broadcast over gossipsub — it stays on the local node's `skulk.yaml`
- logging changes (enable/disable) take effect immediately on all nodes
- inference changes affect future launches
- model-store location changes generally require restart

### Filesystem browse

**GET** `/filesystem/browse`

Used by the dashboard to browse a safe subset of the filesystem when selecting config paths.

### Node identity

**GET** `/node/identity`

Returns hostname, preferred IP, and node identity information used by the dashboard.

### Restart a node

**POST** `/admin/restart?node_id=<optional node id>`

Gracefully restart the exo process on this or a remote node. When `node_id` is omitted or matches the local node, replaces the current process image in-place via `os.execv` (same PID). When `node_id` targets a remote node, sends a `RestartNode` command via pub/sub.

- GPU/Metal memory is released when the process image is replaced
- the node rejoins the cluster automatically on startup
- active inference is interrupted

Returns `{"status": "restarting", "node_id": "..."}` for local restarts, or `{"status": "restart_sent", "node_id": "..."}` for remote restarts.
If a local restart is already scheduled, returns HTTP 409 with `{"status": "restart_already_pending"}`.

## State, Events, and Tracing

### Cluster state

**GET** `/state`

Returns the cluster state as Skulk currently sees it.

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

### Traces

- `GET /v1/traces`
- `POST /v1/traces/delete`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`

Use these endpoints when you are debugging generation behavior, cluster execution, or performance.

## Helpful Next Docs

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- [Model store guide](model-store)
- [Architecture overview](architecture)
- [API Reference](/api/skulk-api)
