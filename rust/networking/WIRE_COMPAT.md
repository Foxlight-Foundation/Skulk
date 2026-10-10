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

- **wire-neutral** (2026-10-10): the lockfile moves to security-fixed
  patch releases. `libp2p-gossipsub` 0.49.2 to 0.49.5 ignores invalid and
  bounds remote PRUNE backoff values, caps subscriptions at 2,000 topics per
  peer and per request, and discards a GRAFT for a topic the peer never
  subscribed to. `yamux` 0.13.8 to 0.13.10, `quinn-proto` 0.11.13 to
  0.11.15, `libp2p-quic` 0.13.0 to 0.13.1, `rustls` 0.23.35 to 0.23.45 and
  `rustls-webpki` 0.103.8 to 0.103.15 fix panics and unbounded memory on
  malformed input. `libp2p-rendezvous`, `bytes`, `time`, `serde_with`,
  `rand` and `keccak` move with them. Protocol ids, topics, framing,
  behaviour composition, key derivation and session configuration are
  unchanged; the new behavior differs only toward malformed or abusive
  peers, and Skulk subscribes to far fewer than 2,000 topics, so
  `NETWORK_VERSION` remains v0.0.3. Bindings advance to 0.2.7 so ordinary
  upgrades rebuild them instead of keeping a cached wheel without the fixes.
  Not covered: `hickory` 0.25 (via `libp2p-dns`) and `lz4_flex` 0.10 (via
  `zenoh-transport`) need their parents upgraded, and `yamux` 0.12.1 is
  compiled only for `libp2p-yamux`'s legacy mode, which Skulk does not use.

- **v0.0.3** (2026-10-08): the Zenoh data plane is scoped to and
  authenticated by the cluster key. Everything derives one-way from the
  pnet pre-shared key (`src/data_plane_trust.rs`):
  - Multicast scouting keeps the `224.0.0.224` group but uses a port
    derived from the key (20000 to 26999, never Zenoh's shared 7446), so
    clusters with different keys normally scout apart. The port is a hash
    into 7,000 values, so two keys can share one (about 1 in 7,000 pairs);
    such clusters exchange scout packets but the mutual TLS handshake
    refuses the foreign key, so the port is not the security boundary. Explicit
    `SKULK_ZENOH_CONNECT` endpoints still turn multicast off.
  - Every link is TLS 1.3 with mutual authentication: each node derives the
    same Ed25519 certificate authority (HKDF-SHA256 over the key) and each
    process issues itself an ephemeral member certificate. `tls/` is the
    only link protocol the session opens or accepts (`tcp/` endpoints in
    environment files are rewritten to `tls/` by Python), with one link per
    peer.
  - Zenoh's TLS dialer also trusts the public Web PKI roots, so member
    certificates carry a key-derived common name under `.invalid` and a
    default-deny access-control policy admits traffic only on links whose
    peer presents it.
  - `connected_peer_count()` counts only peers whose links present that
    name.

  The bump re-keys pnet and the Zenoh session together: upgraded nodes
  refuse old nodes on libp2p, and on Zenoh they scout on a different port
  and speak only TLS, so old and new meshes cannot exchange data. Upgrade a cluster as
  a whole. On the default namespace the key is public, exactly as on libp2p,
  so links are encrypted against passive observers but any Skulk node can
  join. Bindings advance to 0.2.6.

- **wire-neutral** (2026-10-09): comment wording in `src/data_plane_trust.rs`
  and `src/zenoh_session.rs` now says the key-derived scouting port normally
  separates clusters and that a rare port collision is refused by mutual TLS.
  No code, protocol, key derivation or session configuration changes.

- **wire-neutral** (2026-09-12): bound-listener queries read `swarm.listeners()`
  and `session.info().locators()` through the local PyO3 boundary. The new
  `ToSwarm::ListenAddresses` variant stays on an in-process channel; it is not
  serialized onto the network. No protocols, topics, framing, namespace/key
  derivation, connection policy or session configuration change, so
  `NETWORK_VERSION` remains v0.0.2. Bindings advance to 0.2.5 so ordinary
  upgrades rebuild the Python-visible methods instead of retaining a cached
  wheel without them.

- **wire-neutral** (2026-07-24): `ZenohSession` gains `connected_peer_count()`,
  a read-only introspection of the local session's live peer transports via
  `session.info().peers_zid()`, exposed to Python for data-plane isolation
  health. Nothing about the session posture (mode/scouting/namespace), Zenoh
  keys, QoS, framing, topics, or payloads changes, and no gossipsub or pnet
  surface is touched, so no `NETWORK_VERSION` bump. The bindings package
  version bumps to 0.2.4 so auto-updating nodes rebuild and expose the new
  method on first restart.
- **wire-neutral** (2026-07-23): Zenoh multicast scouting can now be enabled for
  zero-config local peer discovery. Explicit-endpoint fleets retain multicast
  off, and Zenoh keys, namespace derivation, QoS, framing, topics, and payloads
  are unchanged. Old and new Zenoh peers therefore remain wire-compatible; this
  changes default discovery/configuration, not the protocol.
- **wire-neutral** (2026-07-22, #662): `FromSwarm::Discovered` gains
  `remote_ip`/`remote_tcp_port` fields describing the connection's observed
  remote endpoint. This enum crosses only the in-process PyO3 boundary to
  Python (`PyFromSwarm.Connection`); nothing about gossipsub protocols,
  topics, framing, transports, or the pnet key changes, so no
  `NETWORK_VERSION` bump.
- **v0.0.2** (2026-07-22, #659): retroactive bump covering the
  telemetry-isolation protocol split (31e3f333: TELEMETRY moved to
  `/skulk/telemetry/meshsub`, ELECTION to `/skulk/election/meshsub` with a
  temporary legacy dual-publish). Establishes this log and the CI pairing
  guard.
- **v0.0.1**: initial versioned network (pnet key seeded from
  `skulk_discovery_network`, #324).
