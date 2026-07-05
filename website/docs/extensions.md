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

This is a Python-side contract, not an HTTP API. Nothing here appears in the
[HTTP API reference](api-guide.md); the public types live in
`skulk.extensions` and are documented below.

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
