<!-- Copyright 2026 Foxlight Foundation -->

# Subscription reannouncement backport

This is the crates.io `libp2p-gossipsub` 0.49.5 source archive, SHA-256
`3573f3d8e30bd62cda336df5c7c1041a1caa40a648376b8e1e274d585c0ed25c`.
`upstream-provenance.json` records the original member hashes and source commit.
The workspace uses it through `[patch.crates-io]`; it is excluded from workspace
membership so upstream development targets do not become Skulk targets.

The upstream shared send queue gives each connection its own `Peekable`
receiver. A handler can prefetch `Subscribe` while opening its substream. If
that connection closes, its announcement disappears while another connection
to the same peer survives. Announcements sent only on the first connection do
not repair this state. The peer can receive traffic while having no topic
subscribers to which it can publish.

The backport repeats the current local subscriptions on every additional
connection and when a connection closes with another still established. It
keeps first-connection peer scoring, blacklisting, final-disconnect cleanup,
local subscriptions, queue bounds, authentication and wire formats intact.
Repeated announcements retain one remote topic membership. No new protocol,
topic, message field, namespace or key derivation is introduced.

Regression tests drive actual behaviour events and actual handler prefetch
without sockets, for ordinary control and Skulk's isolated election and
telemetry protocols. They cover the closing handler taking the replacement
announcement too, blacklisted peers, and the final disconnect. Run them with:

```sh
cargo test --manifest-path rust/vendor/libp2p-gossipsub/Cargo.toml \
  --lib subscription_reannouncement_tests
```

The networking crate declares this backport as a direct path dependency and uses
its crate alias; the workspace patch keeps libp2p's re-export on the same package.
Maturin does not discover registry overrides when collecting source dependencies.
The direct declaration makes its archive self-contained without changing resolved
features or other dependency versions. The original `Cargo.toml.orig` and
`.cargo_vcs_info.json` remain in Git for provenance but are excluded from
repackaging because Cargo regenerates these reserved files. The original VCS
identity is also recorded in `upstream-provenance.json`. CI extracts the real
bindings source archive, resolves it with the
production lock, and compares the backport's Rust source and provenance files.

Only `src/behaviour.rs` changes runtime behavior. `src/handler.rs` registers the
new tests; the manifest enables Ed25519 only for tests and restores the packaged-away
QuickCheck development dependency needed by upstream library tests. Two generated files lose only their extra final blank line for repository
whitespace checks. Original copyright and
license notices remain in the upstream files. Review these local changes when
updating the dependency and remove the backport only after equivalent upstream
behavior passes the regressions. Source tests do not establish full-cluster
cold-start qualification.
