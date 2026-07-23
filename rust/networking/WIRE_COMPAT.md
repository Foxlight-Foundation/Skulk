# Wire-compatibility log for rust/networking

Every change to this crate's wire behavior (gossipsub protocol ids, topics,
message framing, behaviour composition, transport upgrades) MUST bump
`NETWORK_VERSION` in `src/swarm.rs` in the same commit and add an entry
here. The pnet pre-shared key derives from `NETWORK_VERSION`, so a bump
makes wire-incompatible builds refuse to connect, loudly, instead of
half-working.

A `NETWORK_VERSION` bump MUST also bump the bindings package version in
`rust/skulk_pyo3_bindings/pyproject.toml` (plus `uv lock`). That version is
the rollout forcing function: `uv sync` rebuilds the cached bindings wheel
only when it moves, and every installed startup script runs `uv sync`, so
the bump makes auto-updating nodes rebuild their bindings on the FIRST
restart instead of running new Python against stale wire code.

CI enforces the pairing: a PR touching anything under this crate's `src/`
tree, this crate's `Cargo.toml`, or the workspace `Cargo.toml`/`Cargo.lock`
at the repo root (a libp2p or zenoh dependency bump changes protocol
behavior without touching our source) must either change the
`NETWORK_VERSION` line or add an entry below explicitly recording that the
change is wire-neutral (timing, comments, logging, refactors that provably
keep protocol behavior identical). Deciding wire-neutrality is a review
judgment; recording it here makes that judgment auditable.

Why this exists: the telemetry-isolation change (31e3f333, 2026-07-14)
moved TELEMETRY onto its own gossipsub protocol without a bump. A fresh
build connecting to a stale fleet half-worked: events and election flowed
(main protocol and the election legacy copy), telemetry silently reached
nobody, and the node was fully event-log-synced yet invisible to
membership. Eight days later the live fleet was still running the stale
bindings while `versionStatus` reported "consistent".

## Entries (newest first)

- **v0.0.2** (2026-07-22, #659): retroactive bump covering the
  telemetry-isolation protocol split (31e3f333: TELEMETRY moved to
  `/skulk/telemetry/meshsub`, ELECTION to `/skulk/election/meshsub` with a
  temporary legacy dual-publish). Establishes this log and the CI pairing
  guard.
- **v0.0.1**: initial versioned network (pnet key seeded from
  `skulk_discovery_network`, #324).
