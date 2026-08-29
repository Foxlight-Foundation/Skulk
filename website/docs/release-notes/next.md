---
id: release-next
title: Next release
sidebar_position: 0
---

## Tool calling across more model families

A model that calls several tools at once now returns all of them. Models that
write each call separately previously came back with only the first.


A model that reasons before calling a tool now has its call recognized. Its
reasoning previously made the request look like an ordinary answer, so the call
that followed was returned as raw markup in the message content. A call a model
only thought about while reasoning is also no longer carried out.


The `tool_choice` option now behaves the same way whichever engine serves the
model. It previously reached only the engines that run an inference server of
their own, so a request that sent `"none"` could still come back with a tool
call. Sending `"none"` now guarantees no call, and naming a single function
guarantees the model cannot call a different one.


A request that offers no tools can no longer receive tool-call control markup
as answer text, whichever engine serves the model. Models sometimes write a
call nobody asked for; on the engines whose tool parsing only runs when tools
are in the request, that markup previously passed straight through to the
caller. The markers are now removed from the answer on every engine.


Tool calls are now recognized when a model's opening marker arrives split
across several streamed pieces, which is the normal case rather than the
exception, and when the model writes a sentence before calling ("I'll check
that.") rather than opening with the call. Previously the caller received the
raw markup as message content with an ordinary stop reason, so a well formed
call from the model looked like a refusal to call anything.


Tool calling now works for Llama models served by the MLX engine, and Skulk
recognizes the ways more model families write a call. Llama ends a message that
hands off to a tool with a token it does not declare as a stop token, so
generation used to continue past the end of the call and write the next turn's
opening into the answer. It also writes the call as a plain JSON object with no
opening marker, which nothing recognized as a call, so a request that offered a
tool came back with JSON in the message content and no tool call at all. Both
are fixed, and the recognized formats now cover Llama calls, Mistral tool-call
arrays, and GLM argument pairs alongside the formats already supported.

Two rules make the result predictable when you send tools. A response that
names no tool you offered comes back as ordinary content rather than as a tool
call, because models sometimes reach for a built-in of their own that you have
no implementation for. And text that opens like a call but does not parse as
one, which is what a model answering in JSON looks like when tools are also
available, is returned as content rather than reported as a generation error.

## Exact artifact bundles

Signed registry-v2 cards may now identify one complete executable artifact or
quant inside a repository containing many alternatives. Skulk fetches only the
exact required files, validates immutable file metadata, preserves nested
layouts, and loads from the declared artifact root. Bundle identity keeps
several aliases from one repository and revision distinct in the model store
and installed sidecars. Existing v1 cards remain compatible.

Trusted operator workflows may install one exact candidate card before registry
publication and exercise the normal store, placement, and runner path. Skulk
removes registry trust claims from this temporary custom card, so qualification
does not pretend that unpublished content is signed. A dedicated high-entropy
service bearer can authorize only the temporary install and cleanup calls for a
headless registry worker; it grants no broader operator or inference access.
Skulk requires an immutable source revision and marks lifecycle ownership so
the worker cannot replace or delete any pre-existing non-qualification card;
operator installs do not receive that marker. The elected master rechecks the
ownership precondition when ordering every service mutation and reconciles
later signed-registry refreshes into that view. Install and cleanup responses
wait for their exact indexed command acknowledgements; cleanup retains model
bytes without allowing a temporary installed sidecar to shadow signed truth.
The store request can also pin the v2 artifact-bundle identity end to end, so
qualification fails rather than downloading a replacement behind the same
alias.

Cleanup sends the complete original candidate back to Skulk. The elected master
deletes the alias only while that exact temporary card still owns it, preventing
an older or retried job from removing a newer qualification replacement.

## Served GGUF vision

GGUF vision cards can now pin one exact multimodal projector and run through
the external llama-server engine on CUDA, ROCm, Vulkan, or CPU. Skulk stages
only that projector, verifies its manifest digest before load, and accounts for
its fixed memory cost. Homogeneous llama.cpp RPC placements are supported with
the projector and image input owned by the driver. Vision and native MTP work
together with serial serving as the initial compatibility mode; legacy cards
without a projector pin continue through the in-process runner.

When a signed replacement changes only card metadata for the same immutable
artifact, Skulk now refreshes the installed-card sidecar before treating a
staged cache hit as launchable. The unchanged model bytes remain in place while
runner trust sees the newly approved exact card identity.

## Native CUDA serving on Linux ARM64

The managed CUDA llama-server wheel now ships for Linux aarch64 as well as
x86_64. The ARM64 lane is built natively with CUDA 12.9 for compute capability
12.1, allowing Grace Blackwell and GB10 nodes to use the CUDA served engine
instead of falling back to Vulkan.

## Skulk fabric identity and spoken answers

Intelligent Fabric now appears as Skulk itself rather than as a separate
Steward character. It answers in the first person as the intelligent
distributed AI fabric. When a ready streaming TTS model exposes Skulk's
signature voice, the dashboard can speak fabric answers as they stream and
pins that voice for every sentence without changing ordinary chat voice
selection.

## Durable local model cards and cache reconciliation

Every complete model artifact now retains its full effective card and a hashed
file manifest beside the bytes. Existing node caches can converge into the
central model store without another Hugging Face download, while unmarked
legacy artifacts remain usable and are labeled honestly as revision-unverified.

Air-gapped restarts load installed cards before registry access. The currently
installed generation remains active until a replacement has completely
transferred and verified; newer registry truth appears as an available update.
The dashboard shows central-store presence, node cache locations, companion
artifacts, verification state, reconciliation progress, and signed security
advisories. Advisories are warnings only and cannot disable user workloads.
Store deletions also persist reconciliation tombstones, so a stale cache on an
unreachable node cannot silently recreate an intentionally removed model.

## Adaptive model and engine capability truth

Signed registry cards now retain open architecture and intrinsic capability
claims independently of what Skulk can serve today. A separately signed support
matrix can make a newly discovered architecture placeable without replacing its
card, but only when an active positive claim matches the exact engine build,
artifact format, quantization, capability, and hardware class advertised by a
node. Existing card backend declarations remain compatible, while stale,
experimental, unsupported, other-artifact, or incomplete evidence fails closed.
Empirical load and feature claims are pinned to the exact card tested. Model
APIs, placement previews, and the dashboard expose the evidence source and
actionable compatibility gaps.

## Adaptive placement and model authorization

Models may advertise several compatible engines and an ordered preference;
Skulk keeps the final choice in the planner and falls through to another
currently admissible engine or node when the preferred option is unavailable.
Placement failures retain readable messages and now expose a stable category
for operator clients.

Repository-code authorization now follows the action that introduced the exact
card. Signed registry publication authorizes every provenance class, explicitly
adding an external model authorizes its pinned card, and bundled cards are
authorized by the Skulk release. A Hugging Face addition without an explicit
revision resolves `main` once to a full immutable commit before creating the
card, then waits for the exact ordered mutation to appear in the responding
API's catalog before returning. Historical executable custom cards with no
immutable revision fail closed until re-added. Read and launch requests cannot
implicitly fetch an unknown Hub card, and
fully specified placements must carry the exact current catalog card rather
than caller-selected content under a matching alias. There is no second Model
trust ceremony in Settings, and vision metadata
alone no longer creates an approval blocker. Historical approval config, state,
wire fields, and endpoints remain deprecated and inert for rolling upgrades.
Image, embedding, and speech inference requests for an unknown catalog alias
return HTTP 404 rather than an internal server error.
The elected master repeats exact-card validation at command ordering for quick
and caller-specified placements, so a card replacement or deletion that wins a
race after API lookup also prevents stale content from launching.
Executable bundled fallback cards must pin their repository revision. Retained
installed sidecars continue to describe custom artifacts, but they no longer
restore catalog authorization after an operator deletes the custom card.
External processor, vision-weight, assistant, MTP, and speculative-draft
repositories likewise require their own immutable companion revisions.
The explicit low-level download route now requires operator access and rejects
embedded shard cards that differ from current authorized catalog truth.
Snapshot-only republication no longer rejects an otherwise identical signed
card during master-ordered placement.
The runner still verifies signed-card identity, immutable revisions, installed
sidecars, and artifact manifests before executing repository code.

Mistral-family models can now call tools on Apple Silicon. Their chat
templates write calls as `[TOOL_CALLS]` arrays rather than `<tool_call>`
blocks, which the engine previously did not recognize, so the call arrived as
raw markup in the answer. A card for Ministral 8B is bundled so smaller
machines can serve a tool-calling Mistral out of the box.


A cluster configured with the model store enabled but no store host named now
refuses to start with a message that says what to fix, and the dashboard
refuses to save that shape. Previously such a cluster looked healthy while no
node could download any model that was not already staged.


Gemma 4 models now call tools on the in-process llama.cpp engine. Their call
format was previously unreadable on that engine, so calls arrived as raw
markup in the answer; the format is now recognized by the same parser the
Apple Silicon engine uses.
