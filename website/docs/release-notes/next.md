---
id: release-next
title: Next release
sidebar_position: 0
---

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
