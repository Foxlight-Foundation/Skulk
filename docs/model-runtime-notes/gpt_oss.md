<!-- Copyright 2025 Foxlight Foundation -->

# GPT-OSS Runtime Notes

## What It Is

`mlx-community/gpt-oss-20b-MXFP4-Q8` is a Harmony-format reasoning model with
native tool-calling expectations.

In Skulk it should be treated as its own runtime island rather than folded into
generic reasoning or generic tool parsing.

## What Is Unusual

- reasoning is driven by Harmony channels, not generic `<think>` tags
- tool calls are emitted through the GPT-OSS Harmony parser path
- reasoning effort matters (`low`, `medium`, `high`)
- explicit on/off thinking toggle should **not** be treated as the primary
  control surface

## Current Safe Runtime Contract

- `output_parser = "gpt_oss"`
- `tool_call_format = "gpt_oss"`
- default reasoning effort is `medium`
- explicit non-disabled `reasoning_effort` values are preserved even though the
  model is not marked toggleable
- builtin browsing exposure is currently one generic tool:
  - `web_search(query, top_k?) -> { query, provider, results[] }`

## Browsing Support

Current browsing support is intentionally search-style retrieval only.

- no page navigation
- no browser session state
- no click-following loop

Dashboard chat handles GPT-OSS tool calls client-side:

1. advertise the `web_search` tool to GPT-OSS
2. receive a Harmony tool call
3. execute `/v1/tools/web_search`
4. send the tool result back as a `tool` message
5. continue generation

That keeps GPT-OSS support working without changing generic parser behavior for
other model families.

## What Remains Out Of Scope

- a verified true “thinking off” mode
- generic server-side tool execution for every model family
- interactive browser automation
