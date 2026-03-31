<!-- Copyright 2025 Foxlight Foundation -->

# Skulk API

Skulk exposes an API server at `http://localhost:52415`.
It serves both compatibility APIs for existing tools and Skulk-specific control endpoints for placement, downloads, store management, config, and debugging.

## Start Here

If you are new to Skulk, remember this rule:

**A chat request only works after the target model has been placed and is running.**

The dashboard enforces this too: it will not let you chat until a model is placed and ready.

If you call `/v1/chat/completions` too early, Skulk will usually return:

```json
{
  "error": {
    "message": "No instance found for model mlx-community/...",
    "type": "Not Found",
    "code": 404
  }
}
```

The happy path is:

1. Start Skulk.
2. Preview or create a placement.
3. Launch the model.
4. Then send chat, responses, Claude, or Ollama-style requests.

## Quick Navigation

- **I want a first successful API call**: [First Success Flow](#first-success-flow)
- **I want OpenAI SDK compatibility**: [OpenAI Chat Completions](#openai-chat-completions)
- **I want Claude compatibility**: [Claude Messages API](#claude-messages-api)
- **I want OpenAI Responses compatibility**: [OpenAI Responses API](#openai-responses-api)
- **I want Ollama compatibility**: [Ollama API](#ollama-api)
- **I want placement and launch endpoints**: [Model Placement and Instance Management](#model-placement-and-instance-management)
- **I want model store or config endpoints**: [Model Store Endpoints](#model-store-endpoints) and [Configuration Endpoints](#configuration-endpoints)
- **I want debugging endpoints**: [State, Events, and Tracing](#state-events-and-tracing)

## First Success Flow

### 1. Start Skulk

```bash
uv run exo
```

### 2. Preview placements for a model

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Llama-3.2-1B-Instruct-4bit"
```

### 3. Launch a simple placement

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
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

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

### Skulk Control Plane APIs

- `GET /v1/models`
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

## OpenAI Chat Completions

**POST** `/v1/chat/completions`

This is the main text-generation endpoint.

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
| `model` | string | Required. Must match a running instance. |
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
| `reasoning_effort` | string | Reasoning hint when supported by the model or adapter. |
| `response_format` | object | Accepted, but not strictly enforced. |
| `stream_options` | object | Includes `include_usage`. |
| `user` | string | Optional caller identifier. |

### Message Format

```json
{
  "role": "user",
  "content": "hello"
}
```

Assistant messages may also include `tool_calls`.
Tool response messages should include `tool_call_id`.

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

### Example

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
- Models with `thinking` or `thinking_toggle` metadata are the likeliest candidates.

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

Skulk also supports streaming on this endpoint.

## Claude Messages API

**POST** `/v1/messages`

Use this when you want Anthropic-style request and response shapes.

```bash
curl -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'
```

Claude-style streaming events are supported.

## Ollama API

Skulk supports several Ollama-compatible endpoints to make tools like OpenWebUI easier to connect.

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

### Running models

```bash
curl http://localhost:52415/ollama/api/ps
```

## Model Discovery

### List models

**GET** `/v1/models`

```bash
curl http://localhost:52415/v1/models
```

This returns known model cards, not just currently running instances.

### List downloaded models only

```bash
curl "http://localhost:52415/v1/models?status=downloaded"
```

### Search Hugging Face

**GET** `/models/search?query=...&limit=...`

```bash
curl "http://localhost:52415/models/search?query=qwen3&limit=5"
```

Behavior note:

- Skulk searches `mlx-community` first.
- If that returns nothing, it falls back to a broader Hugging Face search.

### Add a custom model card

**POST** `/models/add`

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mlx-community/my-custom-model"}'
```

### Delete a custom model card

**DELETE** `/models/custom/{model_id}`

```bash
curl -X DELETE http://localhost:52415/models/custom/mlx-community/my-custom-model
```

## Model Placement and Instance Management

These endpoints are Skulk-specific and matter a lot for first-time users.

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

Fields:

| Field | Meaning |
|-------|---------|
| `model_id` | Hugging Face-style model ID |
| `sharding` | `Pipeline` or `Tensor` |
| `instance_meta` | `MlxRing` or `MlxJaccl` |
| `min_nodes` | Minimum nodes required for the placement |

### Preview all valid placements

**GET** `/instance/previews?model_id=...`

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit"
```

This is one of the most useful endpoints in Skulk.
It shows which combinations of sharding mode, networking mode, and node count are valid.
Invalid combinations include an error message.

### Build a placement manually

**GET** `/instance/placement`

Use this if you want to request a specific combination and inspect the exact instance shape.

### Create an instance from a fully specified placement

**POST** `/instance`

Use this when you already have an `instance` object and want exact control.

### Inspect one instance

**GET** `/instance/{instance_id}`

### Delete an instance

**DELETE** `/instance/{instance_id}`

```bash
curl -X DELETE http://localhost:52415/instance/YOUR_INSTANCE_ID
```

## Download Management

### Start a node download

**POST** `/download/start`

This is a lower-level endpoint used when you want explicit download control.

### Delete a node download

**DELETE** `/download/{node_id}/{model_id}`

## Model Store Endpoints

These endpoints are available when the model store is configured.
If it is not configured, Skulk returns `503 Store not configured`.

### Store health

**GET** `/store/health`

```bash
curl http://localhost:52415/store/health
```

### Store registry

**GET** `/store/registry`

```bash
curl http://localhost:52415/store/registry
```

### Active store downloads

**GET** `/store/downloads`

### Request a store download

**POST** `/store/models/{model_id}/download`

```bash
curl -X POST http://localhost:52415/store/models/mlx-community/Qwen3.5-9B-4bit/download
```

### Check store download status

**GET** `/store/models/{model_id}/download/status`

### Delete a model from the store

**DELETE** `/store/models/{model_id}`

### Purge staging caches

**POST** `/store/purge-staging`

```bash
curl -X POST http://localhost:52415/store/purge-staging \
  -H 'Content-Type: application/json' \
  -d '{}'
```

You can also target one model:

```bash
curl -X POST http://localhost:52415/store/purge-staging \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mlx-community/Qwen3.5-9B-4bit"}'
```

### Start OptiQ mixed-precision optimization

**POST** `/store/models/{model_id}/optimize`

```bash
curl -X POST http://localhost:52415/store/models/mlx-community/Qwen3.5-9B-4bit/optimize \
  -H 'Content-Type: application/json' \
  -d '{"target_bpw": 4.5, "candidate_bits": [4, 8]}'
```

### Check optimization status

**GET** `/store/models/{model_id}/optimize/status`

Common responses:

- `404 No optimization job found`
- `409 ...` if an optimization job is already in progress
- `503 Model optimizer not available` when the store is not configured

## Configuration Endpoints

### Get config

**GET** `/config`

```bash
curl http://localhost:52415/config
```

Behavior:

- Returns the current config file contents.
- Strips `hf_token` from the visible config for safety.
- Includes effective runtime values like `kv_cache_backend`.

### Update config

**PUT** `/config`

```bash
curl -X PUT http://localhost:52415/config \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "inference": {"kv_cache_backend": "optiq"},
      "hf_token": "hf_your_token_here"
    }
  }'
```

Behavior notes:

- Skulk validates the payload against `exo.yaml` schema.
- Config is written locally and broadcast to the cluster.
- If you omit `hf_token`, Skulk preserves the existing saved token.
- Inference config changes affect future launches.
- Model store changes still require restart.

## State, Events, and Tracing

### Cluster state

**GET** `/state`

This is the best general debugging endpoint.
It includes topology, nodes, instances, runners, and downloads.

### Event stream snapshot

**GET** `/events`

Returns the stored event log as JSON.

### Traces

- `GET /v1/traces`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`
- `POST /v1/traces/delete`

Use tracing when you need performance investigation rather than normal day-to-day usage.

## Filesystem and Identity Helpers

### Browse filesystem

**GET** `/filesystem/browse?path=/Volumes`

Used by the dashboard settings UI for choosing store paths.

Behavior notes:

- Browsing is restricted to allowed roots.
- Typical allowed roots include `/Volumes`, `/home`, `/mnt`, `/tmp`, and `/opt`.
- Returns `400` for invalid paths and `403` for permission problems.

### Node identity

**GET** `/node/identity`

Returns:

- `nodeId`
- `hostname`
- preferred IPv4 LAN address when available

## Image Endpoints

Skulk also supports image-generation and image-editing APIs.
These only work when image models are enabled and the relevant image model is already placed.

Endpoints:

- `POST /v1/images/generations`
- `POST /bench/images/generations`
- `POST /v1/images/edits`
- `POST /bench/images/edits`
- `GET /images`
- `GET /images/{image_id}`

## Cancellation

**POST** `/v1/cancel/{command_id}`

Use this to cancel an active text or image command when you still know the command ID.

## What Is Not Implemented Yet

| Feature | Status |
|---------|--------|
| `/v1/embeddings` | Not implemented |
| Strict JSON mode enforcement | Not implemented |
| JSON schema enforcement | Not implemented |
| API key authentication | Not implemented |
| Rate limiting | Not implemented |
| `/v1/audio` | Not implemented |
| `/v1/files` | Not implemented |
| `/v1/fine_tuning` | Not implemented |
| `/v1/batches` | Not implemented |

## Practical Tips

- Use `/instance/previews` early and often. It explains why a placement is invalid.
- Use `/state` when you are not sure what the cluster thinks is running.
- Use `/v1/models?status=downloaded` to see what is locally available.
- Use streaming for better UX in chat clients.
- Use the dashboard if you are learning the system and the API if you are integrating code.

## Related Docs

- [README.md](../README.md)
- [docs/model-store.md](model-store.md)
- [docs/kv-cache-backends.md](kv-cache-backends.md)
- [docs/architecture.md](architecture.md)
