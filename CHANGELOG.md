<!-- Copyright 2025 Foxlight Foundation -->

# Changelog

This project records release notes here and mirrors public-facing notes in
`website/docs/release-notes/`.

## [Unreleased]

## [1.1.0] - 2026-05-03

### Added

- Headless-resilience deployment kit: a systemd user unit
  (`deployment/systemd/skulk.service`) with `Restart=on-failure` plus
  start-limit backoff, a macOS LaunchAgent
  (`deployment/launchd/foundation.foxlight.skulk.plist`) with conditional
  `KeepAlive`, and one-shot installers for each
  (`deployment/install/install-systemd.sh`, `install-launchd.sh`). The
  Linux installer enables user lingering so headless boxes autostart Skulk
  across reboots without an active login session.
- Startup port preflight (`exo.startup_recovery.preflight_api_port`) runs
  before component boot and exits with `EX_TEMPFAIL` (75) when the API
  port is held by a previous instance, so the service supervisor can
  retry with backoff instead of producing a confusing bind error mid-run.
- Tailscale connectivity layer (`exo.connectivity.tailscale`): detection via
  `tailscale status --json`, `TailscaleConnectivityConfig` in `skulk.yaml`,
  `GET /v1/connectivity/tailscale` API endpoint, and a status row in the
  dashboard's Node tab Runtime section. Cluster nodes can now span multiple
  physical networks over Tailscale (or Headscale).
- Operator panel — a mobile-first `/operator` route in the dashboard for
  remote cluster control: cluster-wide memory/GPU/temperature summary, per-node
  health cards, and a tap-twice-to-confirm node restart button that calls
  `POST /admin/restart`.
- `copyToClipboard()` helper (`dashboard-react/src/utils/clipboard.ts`) with
  a `document.execCommand` fallback so copy affordances work over plain HTTP,
  not just `localhost` or HTTPS.
- Operator-facing "Run Skulk as a service" guide at
  `website/docs/run-skulk-as-a-service.md` — quickstart-first,
  copy-paste install per platform, day-to-day operations table, reboot
  verification, troubleshooting, uninstall, and an advanced
  system-level systemd variant for niche server setups.
- Tailscale setup and troubleshooting guide at `website/docs/tailscale.md`.

### Fixed

- Dashboard `crypto.randomUUID()` replaced with the `uuid` npm package so chat
  session IDs generate correctly over plain HTTP (secure-context restriction).
- `StartLimitBurst` / `StartLimitIntervalSec` moved from `[Service]` to `[Unit]`
  in `skulk.service` — these directives are silently ignored in `[Service]`.
- API port preflight now gated behind `spawn_api` so `--no-api` worker nodes
  don't fail a port check they'll never bind.
- macOS log directory corrected to `~/.skulk/logs` in both the installer script
  and the ops-table in the user guide (was incorrectly `~/.cache/skulk/logs`).
- Tailscale status fields serialized as camelCase (`selfIp`, `dnsName`) to
  match FastAPI's default `by_alias=True` encoding; dashboard hook updated to
  match.
- Removed the unwanted selected-bar highlight (blue stroke) from the trace
  waterfall renderer.

## [1.0.3] - 2026-05-02

### Added

- Per-placement node exclusion. `POST /place_instance` accepts an optional
  `excluded_nodes` array; the master's planner treats those nodes as if
  absent from the topology when scoring candidate cycles for that single
  placement. Already-running instances on the listed nodes are unaffected.
  The dashboard's placement modal exposes click-to-toggle pills under
  "Available Nodes" so operators can mark exclusions before launch.
- `excluded_node_ids` query parameter on `GET /instance/previews` so the
  preview endpoint produces previews against the post-exclusion topology
  and the dashboard's cluster preview reflects the operator's intent
  pre-launch.
- Observability surface consolidation under one panel: Live (cluster health
  + cross-rank flight-recorder timeline + tracing toggle), Node (per-node
  diagnostics with a node selector that defaults to the master), Traces
  (saved-trace browser with inline filtering, expandable rows, native
  waterfall renderer; legacy traces page deleted).
- API trace janitor — hourly background task that drops saved trace files
  older than `tracing.retention_days` (default 3 days; configurable via
  `skulk.yaml`).
- New theme tokens: `errorFill` / `errorOnFill` / `warningFill` /
  `warningOnFill` (palette-independent solid-callout colors for
  iconography) and `errorOnSurface` / `warningOnSurface` (palette-aware
  text colors for callout body copy).

### Changed

- Snapshot bootstrap for follower recovery so newer nodes can hydrate
  cluster state from a master-published snapshot and replay only the retained
  tail instead of rebuilding from event `0`.
- Bounded live master replay retention so long-lived sessions no longer
  need to grow the active `events.bin` without limit.
- Dashboard state migrated from Zustand to Redux Toolkit + RTK Query. Same
  shapes, same persistence, native dedup / polling / cache invalidation.

### Docs

- Architecture documentation overhauled: `architecture.md` (narrative)
  and `architecture-reference.md` (dense fact-sheet) now both exist and
  are kept in sync as architectural shape changes land.
- Documented the rollout caveat for snapshot bootstrap plus bounded retention:
  mixed-version clusters are acceptable during upgrade, but all nodes should be
  upgraded before operators rely on compacted replay history as the steady
  state.

## [1.0.2] - 2026-04-19

### Added

- Explicit runtime notes and capability metadata for DeepSeek V3.2 trusted quantizations.
- Public model-behavior documentation for DeepSeek V3.2.
- A first release-notes workflow for Skulk, including this changelog and public docs release pages.

### Changed

- Hardened model-by-model capability handling across Gemma 4, Nemotron, Qwen 3.5, GPT-OSS, Llama Nemotron Nano, and DeepSeek V3.2 so more model behavior now comes from explicit cards and normalized runtime contracts instead of family-only fallbacks.
- Clarified the macOS build contract: `uv` is the canonical runtime path, and Nix is now documented as the reproducible tooling and validation path rather than a hidden alternate MLX runtime.
- Unified the Darwin Nix environment with the `uv` runtime contract by removing the stale MLX source-build override that no longer matched Skulk's official `mlx` + `mlx-metal` wheel path.
- Updated the README logo asset and related top-level presentation.

### Fixed

- Drove branch-wide `basedpyright` to zero with production-code fixes and tighter tests instead of suppressions.
- Fixed multiple MLX runtime and test-contract issues uncovered during the strict-typing cleanup, including native vision wrapper behavior, parser typing, warmup-path narrowing, and realistic test doubles.
- Fixed download re-download behavior when the model search path has not yet been configured at coordinator startup.
- Restored Python/Rust constructor compatibility for `NetworkingHandle` by giving the Python-facing API the same default bootstrap/listen behavior used elsewhere in the stack.

### Docs

- Updated contributor and agent guidance so build, formatting, and release-note expectations are explicit.
- Added dedicated build/runtime documentation describing how `uv` and Nix align and where they intentionally differ.
