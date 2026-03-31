<!-- Copyright 2025 Foxlight Foundation -->

# Skulk API

Skulk exposes an OpenAI-compatible API at `http://localhost:52415`. You can use it with the OpenAI Python/JS SDK, curl, or any tool that speaks the OpenAI API.

## Quick Start

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="unused",  # Skulk doesn't require an API key
)

response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Using curl

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```python
stream = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Chat Completions

**POST** `/v1/chat/completions`

This is the main inference endpoint. It's compatible with the OpenAI Chat Completions API.

### Request

```json
{
  "model": "mlx-community/Qwen3.5-9B-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model ID (e.g., `mlx-community/Qwen3.5-9B-4bit`) |
| `messages` | array | required | Conversation messages (see Message Format below) |
| `stream` | boolean | `false` | Stream response as Server-Sent Events |
| `temperature` | float | `1.0` | Sampling temperature (0.0 = deterministic, 2.0 = very random) |
| `top_p` | float | `1.0` | Nucleus sampling — only consider tokens with cumulative probability above this |
| `top_k` | integer | `-1` | Only consider the top K tokens. -1 = disabled |
| `min_p` | float | `0.0` | Minimum probability threshold. 0.0 = disabled |
| `max_tokens` | integer | null | Maximum tokens to generate. null = model default |
| `stop` | string/array | null | Stop generation when these strings are encountered |
| `seed` | integer | null | Random seed for reproducible output |
| `frequency_penalty` | float | `0.0` | Penalize tokens based on frequency in output so far |
| `presence_penalty` | float | `0.0` | Penalize tokens that have appeared at all in output so far |
| `repetition_penalty` | float | null | Alternative repetition penalty (multiplicative) |
| `logprobs` | boolean | `false` | Return log probabilities for each token |
| `top_logprobs` | integer | null | Number of top log probabilities to return per token |
| `tools` | array | null | Tool/function definitions for function calling (see Tool Use below) |
| `tool_choice` | string/object | null | Control tool selection (`auto`, `none`, or specific tool) |
| `enable_thinking` | boolean | `false` | Enable thinking/reasoning mode for capable models |
| `response_format` | object | null | Requested output format (see Structured Output below) |

### Message Format

Each message in the `messages` array has:

```json
{
  "role": "system" | "user" | "assistant" | "tool",
  "content": "message text",
  "tool_calls": [...],      // assistant messages with tool calls
  "tool_call_id": "...",    // tool response messages
  "name": "..."             // optional function name
}
```

### Response (Non-Streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711500000,
  "model": "mlx-community/Qwen3.5-9B-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2+2 equals 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

### Response (Streaming)

When `stream: true`, the response is a stream of Server-Sent Events:

```
data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":"+2"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":" equals 4."},"finish_reason":"stop"}]}

data: [DONE]
```

### Finish Reasons

| Value | Meaning |
|-------|---------|
| `stop` | Natural end of response or stop sequence hit |
| `length` | Hit `max_tokens` limit |
| `tool_calls` | Model wants to call a tool |
| `error` | Generation failed |

---

## Tool Use / Function Calling

Skulk supports OpenAI-style tool calling. Define tools in the request, and the model can choose to call them.

### Example: Weather Tool

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto",
)

# Check if the model wants to call a tool
choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    for tool_call in choice.message.tool_calls:
        print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
```

### Tool Response Flow

1. Send messages with tool definitions
2. Model responds with `finish_reason: "tool_calls"` and tool call details
3. Execute the tool locally
4. Send the result back as a `tool` message:

```python
messages.append(choice.message)  # The assistant's tool call message
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": '{"temperature": 22, "condition": "sunny"}',
})
# Send again to get the final response
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=messages,
    tools=tools,
)
```

**Note:** Tool calling quality depends on the model. Models like Qwen3, DeepSeek V3, and GLM-4 have good tool calling support. Smaller models may produce unreliable tool calls.

---

## Thinking / Reasoning Mode

Some models support "thinking" mode where the model shows its reasoning process before answering.

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "What is 127 * 43?"}],
    enable_thinking=True,  # Skulk extension
)

# Thinking content is in reasoning_content
msg = response.choices[0].message
if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
    print("Thinking:", msg.reasoning_content)
print("Answer:", msg.content)
```

**Supported models:** Models with `thinking` or `thinking_toggle` capability (e.g., Qwen3 with thinking_toggle, Nemotron with always-on thinking).

---

## Structured Output

The `response_format` parameter requests structured output:

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"},
)
```

**Current limitations:** The `response_format` field is accepted but not strictly enforced. The model may or may not produce valid JSON depending on the prompt and model capability. For reliable structured output, include explicit JSON formatting instructions in your prompt.

---

## Models

### List Models

**GET** `/v1/models`

```bash
curl http://localhost:52415/v1/models
```

Returns all known models with metadata:

```json
{
  "data": [
    {
      "id": "mlx-community/Qwen3.5-9B-4bit",
      "name": "Qwen3.5-9B-4bit",
      "storage_size_megabytes": 5977,
      "context_length": 262144,
      "capabilities": ["text", "thinking", "thinking_toggle"],
      "family": "qwen",
      "quantization": "4bit",
      "supports_tensor": true,
      "tags": ["thinking", "tensor"]
    }
  ]
}
```

### List Downloaded Models Only

```bash
curl "http://localhost:52415/v1/models?status=downloaded"
```

### Search HuggingFace

**GET** `/models/search?query=llama&limit=10`

```bash
curl "http://localhost:52415/models/search?query=qwen3&limit=5"
```

### Add Custom Model

**POST** `/models/add`

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mlx-community/my-custom-model"}'
```

---

## Instance Management

Before you can chat with a model, it needs to be "placed" (loaded onto cluster nodes).

### Quick Launch

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

| Parameter | Options | Description |
|-----------|---------|-------------|
| `model_id` | string | HuggingFace model ID |
| `sharding` | `Pipeline`, `Tensor` | How to split the model across nodes |
| `instance_meta` | `MlxRing`, `MlxJaccl` | Networking backend (Ring = TCP, Jaccl = RDMA/TB5) |
| `min_nodes` | integer | Minimum number of nodes to use |

### Preview Placements

See what placement options are valid before launching:

**GET** `/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit`

Returns an array of placement previews showing which combinations of sharding, networking, and node count will work, with error explanations for invalid combinations.

### Delete Instance

**DELETE** `/instance/{instance_id}`

```bash
curl -X DELETE http://localhost:52415/instance/YOUR_INSTANCE_ID
```

### Check Running Instances

**GET** `/state`

Returns the full cluster state including all running instances, their runners, and download status.

---

## Cluster State

### Get State

**GET** `/state`

Returns everything about the cluster: topology, nodes, instances, runners, downloads.

### Get Events

**GET** `/events`

Returns the event log (for debugging).

---

## Model Store

### Store Registry

**GET** `/store/registry`

Lists all models in the store with sizes and file counts.

### Download a Model

**POST** `/store/models/{model_id}/download`

Triggers a download from HuggingFace to the store.

### Delete a Model

**DELETE** `/store/models/{model_id}`

Removes a model from the store and disk.

### Store Health

**GET** `/store/health`

Returns store disk usage and path info.

### Optimize Model (OptiQ)

**POST** `/store/models/{model_id}/optimize`

Starts an OptiQ mixed-precision optimization job:

```bash
curl -X POST http://localhost:52415/store/models/mlx-community/Qwen3.5-9B-4bit/optimize \
  -H 'Content-Type: application/json' \
  -d '{"target_bpw": 4.5, "candidate_bits": [4, 8]}'
```

### Optimization Status

**GET** `/store/models/{model_id}/optimize/status`

Poll for optimization progress.

---

## Configuration

### Get Config

**GET** `/config`

Returns the current `exo.yaml` configuration and effective runtime values.

### Update Config

**PUT** `/config`

Update configuration and sync to all cluster nodes:

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

---

## Alternative API Formats

### Claude Messages API

**POST** `/v1/messages`

Compatible with Anthropic's Claude API format. Supports streaming with Claude-specific event types (`message_start`, `content_block_delta`, etc.) and thinking blocks.

### OpenAI Responses API

**POST** `/v1/responses`

Compatible with OpenAI's newer Responses API format. Supports streaming with response-specific events and reasoning items.

### Ollama API

For tools like OpenWebUI:

```bash
# Chat
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3.5-9B-4bit", "messages": [{"role": "user", "content": "Hello"}]}'

# List models
curl http://localhost:52415/ollama/api/tags

# Model info
curl -X POST http://localhost:52415/ollama/api/show \
  -d '{"name": "mlx-community/Qwen3.5-9B-4bit"}'
```

---

## Image Generation

**POST** `/v1/images/generations`

```bash
curl -X POST http://localhost:52415/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a fox in a server room",
    "model": "exolabs/FLUX.1-dev",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

Requires an image model to be placed. See [README](../README.md) for supported image models.

---

## Benchmarking

**POST** `/bench/chat/completions`

Same as `/v1/chat/completions` but returns performance metrics:

```json
{
  "generation_stats": {
    "prompt_tps": 245.3,
    "generation_tps": 42.1,
    "prompt_tokens": 128,
    "generation_tokens": 256,
    "peak_memory_usage": 8589934592
  }
}
```

---

## What's NOT Supported (Yet)

| Feature | Status |
|---------|--------|
| `/v1/embeddings` | Not implemented — planned |
| JSON mode enforcement | `response_format` accepted but not enforced |
| JSON schema validation | Not implemented — planned |
| `/v1/batches` | Not implemented |
| `/v1/audio` | Not implemented |
| `/v1/files` | Not implemented |
| `/v1/fine_tuning` | Not implemented |
| API key authentication | Not implemented — all requests accepted |
| Rate limiting | Not implemented |

---

## Tips

- **Finding model IDs**: Use `/v1/models?status=downloaded` to see what's available
- **Streaming is recommended**: For chat use cases, `stream: true` gives much better UX
- **Cancel stuck requests**: Close the connection or call `/v1/cancel/{command_id}`
- **Check cluster health**: `GET /state` shows everything — nodes, instances, runners, downloads
- **HuggingFace token**: Set via Settings panel or `HF_TOKEN` env var for faster downloads and gated model access
