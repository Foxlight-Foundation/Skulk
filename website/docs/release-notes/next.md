---
id: release-next
title: Next release
sidebar_position: 0
---

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
