---
id: release-next
title: Next release
sidebar_position: 0
---

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
operator installs do not receive that marker.

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

## Adaptive placement and model-scoped trust

Models may advertise several compatible engines and an ordered preference;
Skulk keeps the final choice in the planner and falls through to another
currently admissible engine or node when the preferred option is unavailable.
Placement failures retain readable messages and now expose a stable category
for operator clients.

Repository-code trust is now one operator decision per exact immutable model
card, managed in cluster Settings rather than repeated on individual nodes.
Revision-pinned Foxlight-provenance cards from the signed registry are already
Foxlight's trust decision. Agent, community, custom, and unsigned cards remain
loadable after explicit approval, and any changed card identity must be
evaluated again. Trust and custom-card mutations require the direct-local
dashboard or an authenticated operator-gateway credential. Config convergence
also preserves each node's private Hugging Face token and owner-only config
permissions. Approval and revocation are serialized by the elected master and
replicated as durable cluster state, so concurrent operator actions and
unrelated Settings saves cannot overwrite or resurrect trust decisions.
