---
title: Extensions (Plugins)
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk can load separately installed Python packages as extensions and call
them at well-defined points in the serving path. Extensions are how
deployment-specific behavior (an audit logger, a request policy filter, a
prompt annotator, a memory layer) rides the fabric without forking Skulk.

An extension is a normal Python package installed into the same environment
as Skulk. At node startup Skulk discovers every package that registers an
entry point in the `skulk.extensions` group, version-checks it, and loads it.
No configuration file, no registration API call: install the package and
restart the node.

This is primarily a Python-side contract. Provider discovery and node-to-node
call admission also have control-sized endpoints in the
[HTTP API reference](api-guide.md), but extension authors normally use the
typed surfaces exported from `skulk.extensions` and documented below.

## The contract

An extension provides three things (`src/skulk/extensions/types.py`):

- **`name`**: a short unique name, used in logs.
- **`skulk_requires`**: a [PEP 440](https://peps.python.org/pep-0440/)
  version specifier for the Skulk versions it supports, for example
  `>=1.4,<1.5`. An extension whose specifier does not match the running
  Skulk is refused at load time with a loud error. Mixed plugin/fabric
  versions are the same anti-pattern as mixed-version clusters; upgrade the
  fleet and its extensions together.
- **`chat_middleware()`**: returns the extension's chat middleware, or
  `None` if it has none.

Chat middleware gets two hooks, both `async`:

- **`transform_chat_request(context, task_params)`** runs on the API node
  after the OpenAI adapter has normalized the request and before it is
  dispatched to the cluster. It returns (possibly modified) task params, so
  it can rewrite or augment the prompt, adjust sampling, or annotate the
  request.
- **`observe_chat_response(context, task_params, summary)`** runs as a
  background task after the response has finished streaming. The summary is
  immutable (final text, thinking text, finish reason, error flag); an
  observer can log, index, or learn from it, but can never touch the stream.

Each hook receives an **`ExtensionContext`** carrying the node identity, the
running Skulk version, and `embed_texts`, programmatic in-process access to
the cluster's embedding serving (the equivalent of `POST /v1/embeddings`).
`embed_texts` returns `None` when no embedding instance is available;
extensions must degrade gracefully on `None`, never raise.

### Reading the cluster (`read_cluster`)

`ExtensionContext.read_cluster()` returns an immutable snapshot of the telemetry
plane: one `ClusterNodeView` per node the local node currently sees, each with
`node_id`, `friendly_name`, `backends`, `participation`, `skulk_version`,
`accelerator_vendor`, `ram_total_bytes`, `last_telemetry` (its liveness signal),
and `capabilities` (the tags peers have advertised; see below). This is how an
extension discovers the cluster it belongs to instead of being blind to
everything beyond the request in front of it.

The call is cheap and side-effect free (an in-memory snapshot, no network I/O),
so it is safe from an inline hook. It is a **read**: an extension can observe the
cluster but never mutate telemetry. Every field beyond `node_id` may be `None`
(or an empty tuple) when that reading has not yet arrived (telemetry is
last-write-wins and partial), so treat missing values as "not known yet".

```python
for node in context.read_cluster():
    if node.accelerator_vendor == "amd" and node.last_telemetry is not None:
        ...  # e.g. prefer an AMD node for a GGUF-friendly task
```

### Advertising a capability (`advertise_capability`)

`ExtensionContext.advertise_capability(tag)` is the write half of the telemetry
plane: it publishes an opaque capability tag this node offers so peers discover
it the same way native nodes advertise their backends. The tag then appears in
every peer's `read_cluster()` snapshot under `ClusterNodeView.capabilities`.
Tags are free-form strings owned by your extension (for example `"memory"` or
`"embeddings:bge-m3"`); Skulk core neither interprets nor validates them.

Advertising is additive and idempotent, so the natural place to call it is once,
when the extension is constructed or on its first hook:

```python
context.advertise_capability("memory")
# ... later, on any node in the cluster:
peers_with_memory = [
    node for node in context.read_cluster() if "memory" in node.capabilities
]
```

Notes:

- The tag is gossiped on the node's normal telemetry poll, so peers see it
  within a second or two, not instantly. A node that advertised is discoverable
  by nodes that join later (the plane re-gossips it).
- `withdraw_capability(tag)` is the counterpart: it stops advertising the tag
  so callers stop selecting this node for it. When the last tag is withdrawn,
  one final empty reading is published so peers clear their entry; a node's
  tags also disappear when the node leaves the cluster.
- A node must run a worker to gossip its advertisement (the worker owns the
  telemetry emit path). The mainstream node runs both an API and a worker, so
  this is automatic; a rare API-only (`--no-worker`) node records the tag but
  does not gossip it.

## Serving a capability (providers)

Beyond observing chat traffic, an extension can be a **provider**: a plugin
that serves a capability of its own (a memory service, a speech backend,
anything not yet imagined). Skulk cannot enumerate future capabilities, so it
standardizes the *description* instead: a provider publishes one
`CapabilityDescriptor` per capability, a fixed self-describing shape that tells
any caller, human or LLM, how to call it.

A descriptor carries:

- `id` and `version`: the negotiation key (`echo@1.0.0`). The `id` doubles as
  the telemetry discovery tag and is auto-advertised for you.
- `title` and `description`: written for both humans and generative callers
  (an LLM reads the description plus the schemas at runtime to call a
  capability it has never seen, the tool-use model).
- `input_schema` / `output_schema`: JSON Schemas for the call payload and
  result.
- `io_mode`: how the call moves data: `unary`, `server_streaming`,
  `client_streaming`, or `bidirectional`, with chunk schemas for the streaming
  modes.

To become a provider, implement `capabilities()` on your extension; the
`on_start` startup hook is optional and independent (any extension can use it):

```python
class MyExtension:
    # ... name, skulk_requires, chat_middleware() as usual ...

    def capabilities(self) -> list[CapabilityDescriptor]:
        # This method alone makes the extension a provider.
        return [MY_DESCRIPTOR]

    def on_start(self, context: ExtensionContext) -> None:
        # Optional: startup registration with the live context; runs once at
        # node startup. Must be fast; heavy init belongs in background work
        # you own. A pure provider has no chat hook, so this is how it
        # reaches the context without waiting for a chat request.
        ...
```

Discovery then has two layers, cheap and heavy:

1. **Tag** (telemetry): peers see `"echo"` in `read_cluster()` capabilities.
2. **Descriptor** (describe): `await context.describe_node(node_id)` fetches
   the node's full descriptors; the same list is served over
   `GET /v1/capabilities` on every node.

```python
for node in context.read_cluster():
    if "echo" in node.capabilities:
        descriptors = await context.describe_node(node.node_id)
        # descriptors[0].input_schema tells you exactly what to send
```

A complete reference provider lives at `examples/extensions/echo-provider/` in
the repository.

## Calling a capability (`call_capability` / `handle_call`)

The generic call verb completes the unary loop. On the provider side, an
extension that also implements `handle_call` becomes callable:

```python
class MyExtension:
    # ... capabilities() as above ...

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        # call.payload has already been validated against your input_schema.
        return {"text": call.payload["text"]}
```

On the caller side, any extension invokes a discovered capability through its
context:

```python
descriptors = await context.describe_node(node.node_id)
echo = next(d for d in descriptors if d.qualified_id == "echo@1.0.0")
result = await context.call_capability(
    node.node_id, echo.id, echo.version, descriptor_revision(echo),
    {"text": "hello"},
)
if result.ok:
    print(result.result)
else:
    print(result.error.code, result.error.message)  # typed, never parse prose
```

The call contract:

- **Typed results, never exceptions.** Every failure arrives as a
  machine-readable code on the result: `not_found`, `version_mismatch`,
  `revision_mismatch` (the provider's descriptor drifted since you discovered
  it; re-describe and retry), `invalid_payload`, `invalid_result`,
  `payload_too_large`, `overloaded`, `timeout`, `provider_error`,
  `unreachable`.
- **Pinned contract.** A call carries the exact `id@version` plus the
  descriptor revision digest from discovery, so discovery and invocation can
  never silently disagree.
- **Schema-validated both ways.** The payload is validated against the
  descriptor's `input_schema` before your handler runs, and your result
  against `output_schema` after. Validation never fetches remote schema
  references.
- **Bounded.** Calls have a deadline (default 30s) that spans the whole call,
  including target resolution on the caller and payload validation on the
  provider; payloads and results are capped at 1 MiB, and each node bounds
  concurrent in-flight provider calls; excess calls are rejected as
  `overloaded` rather than queued. Handlers are
  `async`: move CPU-heavy or blocking work off the event loop yourself (a
  worker thread), or you will stall the API node the handler runs on.
- **Direct and off the log.** Calls go node-to-node; the master is never in
  the hot path and calls are never event-sourced. Calling your own node is an
  in-process fast path with identical guards.

## Streaming a capability (`stream_capability` / `handle_stream`)

The first executable streaming mode is `server_streaming`: one validated call
payload followed by provider output. This is the shape needed by TTS. A
provider implements `handle_stream` and begins at sequence one because Skulk
owns the admission lifecycle and emits `started` at sequence zero:

```python
from collections.abc import AsyncIterator

from skulk.extensions import (
    CapabilityCall,
    CapabilityStreamFrame,
    ExtensionContext,
    InlineMediaAttachment,
)


async def handle_stream(
    self,
    context: ExtensionContext,
    call: CapabilityCall,
) -> AsyncIterator[CapabilityStreamFrame]:
    yield CapabilityStreamFrame(
        call_id=call.call_id,
        direction="provider_to_caller",
        sequence=1,
        kind="chunk",
        payload={"format": "pcm_s16le"},
        media=InlineMediaAttachment(
            data=audio_frame,
            media_type="audio/pcm",
            codec="pcm_s16le",
            sample_rate=24000,
            channels=1,
        ),
    )
    yield CapabilityStreamFrame(
        call_id=call.call_id,
        direction="provider_to_caller",
        sequence=2,
        kind="completed",
    )
```

The caller opens the stream through its context and checks the typed admission
result before consuming frames:

```python
session = await context.stream_capability(
    node.node_id,
    tts.id,
    tts.version,
    descriptor_revision(tts),
    {"text": "Speak this sentence."},
)
if not session.open_result.ok:
    print(session.open_result.error.code)
else:
    async for frame in session.frames:
        if isinstance(frame.media, InlineMediaAttachment):
            play(frame.media.data)
```

The opening request is control-sized and direct to the provider node. Output
does not stream over HTTP: it uses the separate `PROVIDER_DATA` type family,
off the master, State, and event log. Same-node output short-circuits locally;
remote output uses bounded independent per-owner/per-call queues. Structured
payloads are validated against `output_chunk_schema`; inline media remains raw
bytes outside JSON and is capped at 1 MiB per frame. Use a staged
`BlobMediaAttachment` for large immutable objects.

Skulk enforces exact call identity and sequence, one deadline across admission
and streaming, exactly one terminal, bounded reorder/gap handling, and explicit
cancellation when the caller closes the iterator early. A raising or malformed
handler fails only its own stream with a typed terminal.

`client_streaming` and `bidirectional` descriptors remain discoverable but are
not executable yet. Realtime STT follows with ordered input media frames and an
explicit caller half-close; a provider must not advertise progressive output
until its underlying model actually produces it.

## Guarantees

Three invariants shape the design, and Skulk's call sites enforce them:

1. **A raising extension never breaks inference.** Every extension call is
   guarded: an exception is logged loudly and skipped, and the request
   proceeds as if the extension did not exist. Be precise about the scope,
   though: request transforms run inline before dispatch, so a *slow or
   hanging* transform delays the request it is transforming (keep transforms
   fast and bounded). Observers run as background tasks after the stream ends
   and can never affect request latency.
2. **Extensions never own the response stream.** Skulk accumulates the
   response and hands observers a summary, so a buggy extension cannot
   corrupt, reorder, or stall token delivery.
3. **No extension installed means Skulk unchanged.** All hooks are inert
   when nothing is loaded.

## A complete example

A minimal extension that stamps a system-prompt suffix onto every chat
request and logs completions:

```python
# my_skulk_extension/extension.py
from skulk.extensions import (
    BaseChatMiddleware,
    ChatResponseSummary,
    ExtensionContext,
)
from skulk.shared.types.text_generation import TextGenerationTaskParams


class AuditMiddleware(BaseChatMiddleware):
    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        # Modify and return the params; return them unchanged to no-op.
        return task_params

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        print(f"[audit] finish={summary.finish_reason} chars={len(summary.text)}")


class AuditExtension:
    name = "audit-example"
    skulk_requires = ">=1.4,<1.5"

    def chat_middleware(self) -> AuditMiddleware:
        return AuditMiddleware()
```

Register the zero-argument factory in the package's `pyproject.toml`:

```toml
[project.entry-points."skulk.extensions"]
audit-example = "my_skulk_extension.extension:AuditExtension"
```

Install it next to Skulk on each node and restart:

```bash
uv pip install ./my-skulk-extension
```

The startup log lists every discovered extension and whether it loaded or
was refused (with the reason).

## Operational notes

- **Install on every node.** Chat middleware runs on the API node that owns
  the request, and any node can serve API traffic, so install extensions
  fleet-wide (the same discipline as Skulk versions).
- **Kill switch:** `SKULK_EXTENSIONS_DISABLE=1` skips discovery entirely on
  that node.
- **`BaseChatMiddleware`** is a no-op base class; subclass it and override
  only the hook you need.
- Extension hooks currently cover the chat serving path. The surface will
  grow deliberately; anything an extension can reach is a public contract
  Skulk has to honor across versions.
