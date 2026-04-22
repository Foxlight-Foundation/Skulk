---
title: Release Notes Unreleased
sidebar_position: 1
---

<!-- Copyright 2025 Foxlight Foundation -->

## Recovery And Retention

- Added snapshot bootstrap for follower recovery so a restarted node can load a
  master-published state snapshot and then replay only the retained tail after
  that snapshot.
- Added bounded live replay retention on the master so long-running sessions do
  not keep growing the active replay log without limit.

## Upgrade Guidance

Mixed-version clusters are acceptable during rollout, but operators should
upgrade every node before relying on bounded replay retention as the normal
steady state.

Why this matters:

- a new follower can fall back to full replay when talking to an older master
- a new master can still serve replay to older followers while the relevant
  history is retained
- but once old replay history has been compacted away, an older restarted node
  that only knows how to rebuild from event `0` may no longer be able to fully
  resync

The practical rule is simple: complete the cluster upgrade before treating
snapshot bootstrap plus compacted replay history as the stable post-rollout
configuration.
