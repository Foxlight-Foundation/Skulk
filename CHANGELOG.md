<!-- Copyright 2025 Foxlight Foundation -->

# Changelog

This project records release notes here and mirrors public-facing notes in
`website/docs/release-notes/`.

## [Unreleased]

### Added

- Tensor-parallel speculative decoding (#201 Track 1): the #200
  single-node guard is lifted for TP placements after lockstep was
  validated on real hardware — greedy byte-parity and seeded-sampled
  trace-hash parity across ranks, on both a localhost ring and a real
  two-node cluster (kite1+kite2, Qwen3.5-2B TP=2, 150–300-token
  generations, multiple seeds, depths 1–2). The two invariants that make
  it safe: TP collectives give every rank identical logits (embeddings
  and lm_head are replicated, only layer internals shard), and
  `mlx_generate` already seeds the RNG per request from the shared task,
  so sampled accept/reject draws are aligned with zero extra
  communication — unseeded ranks fork on the first draw (measured), which
  is why the probe validates the production seeding contract. Pipeline
  placements still disengage speculation pending the distributed
  draft/verify design (#201 Track 2). A two-rank lockstep regression test
  (greedy + sampled) guards the invariant.

- Bonus-driven MTP rounds: the speculative loop was restructured to the
  cadence the reference implementations use — every round verifies
  `[bonus, drafts]` in one forward and the very next round drafts from
  the correction position, instead of skipping post-correction drafts
  (statistically the easiest ones; the old cadence forfeited ~25pp of
  acceptance on identical inputs). Two companion optimizations close the
  hybrid-SSM gap the new cadence exposed: *deferred replay* (on a reject,
  restored-but-committed tokens ride at the front of the next verify
  forward instead of paying a dedicated replay pass — extra verify width
  is free on memory-bound decode, measured 46.6ms 2-wide vs 47.8ms 1-wide
  on Qwen3.5-9B) and *quantize-on-load sidecars* (the builder quantizes
  the bf16 sidecar block + fc to the target's `(group_size, bits)`; the
  unquantized block was ~10.7ms of the round budget). Re-measured
  2026-06-05 on M4/24GB, superseding all earlier figures in this section:
  Qwen3.5-9B 79% acceptance / 1.38x greedy (1.43x at T=0.7), Qwen3.5-27B
  depth-2 82% / 10.5 tok/s (1.87x), gemma-4-26B-A4B 84% / 35.1 tok/s
  (~2.2x vs warm vanilla), gemma-4-E4B depth-3 1.86x — beating upstream
  mlx-vlm (1.43x/1.66x) on identical artifacts. This RETRACTS the earlier
  "chained depth does not pay on quantized targets" finding below: it was
  an artifact of the old cadence's skipped drafts, not of quantized
  hiddens (E4B's carded depth is now 3).

- Gemma 4 assistant speculative decoding (gemma4-mtp Phase C): the
  separate 4-layer assistant models Google publishes per Gemma 4 target
  now draft through Skulk's Drafter protocol — the assistant cross-attends
  over the target's KV cache (shared-KV extraction with RotatingKVCache
  temporal restore), consumes the target's post-norm hidden, and loads
  bf16-enforced when a card declares `assistant_model_repo` (single-node,
  same #200/#201 envelope; forces SequentialGenerator like MTP). Measured
  on M4/24GB: gemma-4-26B-A4B-it-4bit 55% acceptance, 28.8 tok/s vs
  15.5–17.8 vanilla (1.6–1.85×); gemma-4-e4b-it-8bit 48%, 1.26×. Notable
  finding: chained depth does NOT pay on quantized targets (the assistant
  is trained against bf16 hiddens; chain acceptance decays to ~30% and MoE
  verify cost grows with block size) — depth 1 is the default and the
  measured optimum for the carded quants. Also fixed en route: the
  pre-norm trunk wrapper is gated to qwen-shaped trunks (it would build
  wrong masks for gemma4's sliding/full layers), and the companion-repo
  download-completeness gap (#185 flag) — cached bases now fetch newly
  declared sidecars/assistants.
- Sampled-decoding support for MTP speculative decoding (issue #180 item 1):
  at temperature > 0 the loop switches from argmax-prefix acceptance to
  Leviathan-Chen probability-ratio rejection sampling over the *effective*
  sampler distributions (temp + top_p + min_p + top_k, computed by reusing
  mlx-lm's own filter functions so they cannot drift), with residual
  resampling on reject — distribution-preserving by construction and
  verified by a 40k-draw statistical unit test. Depth is forced to 1 under
  sampling (the drafter's internal chain is greedy). Measured at T=0.7
  with default min_p: 9B 87% acceptance / 1.33x, 2B 71% / 1.09x; greedy
  path regression-checked identical. MTP previously disengaged entirely
  for any temperature > 0 — this extends every speedup to default-
  temperature chat traffic.

- Depth-K chained MTP drafting: the speculative loop now verifies up to
  `mtp_max_depth` chained drafts in a single K+1-token forward, committing
  the longest matching prefix (plus the verifier's correction on partial
  rejects). The Qwen drafter chains by recursing its block on its own
  output hidden — measured conditional acceptance decays fast beyond one
  step (86.8% / 39.2% / 28.2% at depths 1-3 on Qwen3.5-9B), so depth 2 is
  the practical ceiling for the single trained block; deeper gains need
  heads trained for chaining (or the Gemma 4 assistant drafter). On a full
  accept the next main token now comes straight from the verify logits,
  eliminating a redundant lm_head pass per accepted cycle. Measured: the
  recompute fix alone lifts 9B depth-1 from 1.20x to 1.30x; depth 2 lifts
  Qwen3.5-27B to 1.92x (from 1.73x, 78% chained acceptance, parity OK) but
  is SLOWER than depth 1 on the 9B (1.15x vs 1.30x) — depth only pays when
  the trunk dwarfs the drafter, so `mtp_max_depth` is set per card (2 on
  27B-class, 1 elsewhere).

- Phase 2 MTP speculative decoding behind a modular `Drafter` protocol
  (`src/exo/worker/engines/mlx/drafters/`). The generation loop now talks to
  a mechanism-agnostic drafter seam (`begin_request` / `observe` / `draft`)
  so Qwen sidecar heads, DeepSeek heads, and the planned Gemma 4 assistant
  drafter all plug into the same verify/accept/reject machinery. The
  Qwen3.5 drafter applies the three empirically isolated fixes from issue
  #192 — +1.0 zero-centered norm shift, `embed_first` fc concat order, and
  running the sidecar's `mtp.layers.0` transformer block with a private KV
  cache — measured live at ~58–66% draft acceptance on Qwen3.5-2B (0%
  before). Model cards gain optional `mtp_norm_convention` /
  `mtp_concat_order` runtime overrides keyed to layout-detected family
  defaults, and the loop logs a periodic `MTP acceptance so far` line as
  the production acceptance signal. Model cards for Qwen3.5 2B-4bit (69%
  acceptance, 1.26x), 9B-MLX-4bit (88%, 1.20x), and 27B-4bit (1.75x) now
  declare MTP sidecars, validated by a per-model sweep plus a 10-prompt
  exact-attempt acceptance suite. All shipped sidecars use base heads: a
  750-draft/arm comparison measured base vs instruct heads as
  statistically indistinguishable (87.6% vs 87.3% on 9B), so one base
  sidecar serves every variant of a backbone. Qwen3.6-27B-4bit (88%
  acceptance, 1.73x) is carded too — Qwen3.6 ships model_type=qwen3_5
  and works through the existing stack with zero code changes. MTP is
  skipped when logits processors are active (repetition penalty, bench
  EOS ban): accepted drafts commit from raw verifier logits, so
  processor-aware verification is required first (tracked follow-up). Known property: on hybrid (GDN) models, MTP greedy output is
  semantically greedy but not guaranteed byte-identical to non-MTP decode —
  the batched verify/replay chunked-scan numerics drift the recurrent state
  and can flip near-tie tokens.

### Fixed

- MTP is explicitly single-node for now: distributed placements (any group
  size > 1) disengage speculation with a logged fallback. Pipeline sharding
  was already excluded; tensor-parallel would mechanically run but
  accept/reject decisions consume per-rank RNG and cross-rank lockstep is
  unvalidated — a divergent decision would silently corrupt every rank's
  cache. Distributed MTP (TP lockstep validation, then pipeline
  draft/verify) is the next workstream.
- MTP terminal responses (EOS / max-tokens break paths) now finalize the
  detokenizer exactly once before yielding, matching the non-MTP path —
  sentencepiece-backed tokenizers buffer partial byte sequences until
  finalize() and could drop the last token's tail bytes (#180 item 4;
  latent for current tiktoken-backed targets).
- MTP drafting consumed post-final-norm hidden states; the trunk accessor
  now returns pre-norm hiddens (what the heads were trained on) and folds
  the final norm into the head callable, keeping main-path logits
  unchanged. Also fixed the accept-path token-history divergence (a
  never-emitted sampled token entered logits-processor history — PR #191
  review finding) and the pure-KV reject path dropping the emitted main
  token from processor history.

- Boot-time auto-update for the Skulk service: the LaunchAgent now runs
  `git pull`, `uv sync`, and the dashboard build through a wrapper
  (`deployment/install/skulk-startup.sh`) before exec'ing skulk. Failures of
  the pull / sync steps are non-fatal (logged to
  `~/.skulk/logs/skulk.prep.log` and the service boots whatever revision is
  on disk); a missing `dashboard-react/dist/` is fatal because the API has
  no UI to serve. Toggle with `SKULK_AUTO_UPDATE=0` in `~/.skulk/skulk.env`.
- Operator-editable env file at `~/.skulk/skulk.env`, copied from
  `deployment/install/skulk.env.example` on first install and never
  overwritten on re-run. Surfaces `SKULK_LIBP2P_NAMESPACE`,
  `SKULK_VERBOSITY`, `PYTHONUNBUFFERED`, debug toggles, and external-logging
  knobs without requiring a plist edit.
- Separate `foundation.foxlight.skulk-vector` LaunchAgent that runs Vector
  as its own process (via `deployment/install/vector-startup.sh`). Vector
  tails the captured `~/.skulk/logs/skulk.stdout.log` instead of piping
  through Skulk's process, so a slow VictoriaLogs sink can no longer
  backpressure inference threads. Opt out with `--no-vector` on the
  installer.
- `SKULK_LOGGING_EXTERNAL=1` mode in `exo.shared.logging`: structured JSON
  goes to stdout for an external shipper to consume, and Skulk does not
  spawn its own internal Vector subprocess. The launchd installer turns
  this on by default. JSON sink is now `enqueue=True` so log producers are
  decoupled from the sink's I/O.
- New operator guide at `website/docs/external-logging.md` covering the
  full Vector + VictoriaLogs + Grafana stack: central-host install, per-node
  configuration, JSON schema, and troubleshooting.

### Fixed

- Phase 1 MTP speculative decoding repaired on the post-ladder stack (it had
  never been validated end-to-end): the GDN softplus patch no longer probes
  foreign lazy modules (transformers 5.10 resolved a `compute_g` probe into
  an aria image-processing import requiring torchvision, crashing every GDN
  runner at startup); tied-embedding models (Qwen3.5 small variants) locate
  their output head via `embed_tokens.as_linear`; the MTP loop feeds the
  trunk correctly-shaped token batches; rejects snapshot/restore SSM
  (ArraysCache) state instead of zeroing it — the bug that degenerated
  hybrid-model output; and `mtp.safetensors` sidecars resolve via a new
  `build_sidecar_path` (sidecar repos have no `config.json`, so the model
  resolver rejected them and MTP silently never engaged). Verified
  end-to-end: greedy parity is byte-exact with MTP on vs off
  (Qwen3.5-2B-4bit + the FoxlightAI bf16 sidecar). Draft acceptance is
  currently 0% — the Phase 1 head intentionally omits the sidecar's
  transformer block (Phase 2); tracked separately.

### Changed

- mlx-vlm 0.5.0 → 0.6.1 (Gemma 4 MTP initiative Phase B). 0.6.1 ships the
  speculative-drafter catalog Phase C consumes (`gemma4_assistant`,
  `gemma4_unified_assistant`, `gemma4_dflash` — plus upstream
  `qwen3_5_mtp` and `deepseek_v4_mtp` drafters relevant to #194 and the
  DeepSeek sidecar path). All Skulk touchpoints verified against 0.6.1
  (prompt_utils, load_image_processor, dynamic `mlx_vlm.models.*` imports);
  dependency floors were already satisfied by the #188/#190 ladder.

- Web framework migrated to starlette 1.x (1.2.1) / fastapi 0.136, unified
  across darwin and linux. Test code uses `httpx2` for starlette's
  `TestClient` (the httpx-backed client is deprecated in 1.x); production
  HTTP-client code remains on `httpx`. This unblocks the mlx-vlm 0.6.x bump
  (which floors `starlette>=1.0.1`).
- MLX dependency version ladder (darwin): `mlx` 0.31.1 → 0.31.2, `mlx-vlm`
  0.4.4 → 0.5.0, `transformers` cap lifted to `>=5.5,<6`, and the Foxlight
  `mlx-lm` fork reconciled onto upstream v0.31.3 (`0.31.3.post1`, rev
  `e2f7ddcd`). The fork now carries only two non-upstream fixes (ArraysCache
  leak, DeepSeek-V3.2 lightning-indexer batch>1); float32 logprobs, GDN
  precision, and left-padding eval are absorbed upstream. mlx-vlm 0.5.0
  brings the `gemma4_assistant` drafter as a maintained dependency. (The
  ladder initially held `starlette<1.0`; the starlette 1.x migration above
  landed as its own change and removed that constraint.)
- Distributed/prefix-cache slow tests now select their model by available
  GPU working-set size: GPT-OSS-20B on machines that fit it, otherwise
  Llama-3.2-1B (override with `SKULK_TEST_DISTRIBUTED_MODEL`). Previously
  the hardcoded 20B memory-exhausted 16 GB machines.
- `test_batch_generate` B=1 vs B=2 equivalence is now teacher-forced with a
  relative logit tolerance instead of bit-exactness. Root cause of the
  divergence: M5-class Neural Accelerators run float32 GEMM (batch ≥ 2) at
  TF32-style reduced precision by default while GEMV (batch 1) stays full
  fp32 (ml-explore/mlx#3534; `MLX_ENABLE_TF32=0` opts out and restores
  bit-exactness). The test asserts under default precision — what
  production runs.
- The `opt_batch_gen` top-logprobs precompute patch is version-gated: it
  no-ops with a warning on mlx-lm ≥ 0.31.3 (BatchGenerator split) and
  `extract_top_logprobs` falls back to its synchronous path. Re-port
  tracked in #187.
- Two Vector configs now exist for the two transport modes:
  `deployment/logging/vector.yaml` keeps the original `stdin` source for
  the in-process subprocess shipper (used by Linux systemd installs and
  macOS `--no-vector` installs); `deployment/logging/vector-external.yaml`
  carries a `file` source tailing `~/.skulk/logs/skulk.stdout.log` plus a
  `remap` transform that drops non-JSON lines, used by the launchd
  `skulk-vector` agent.
- `deployment/install/install-launchd.sh` now installs both agents by
  default, manages `~/.skulk/skulk.env` (auto-flipping
  `SKULK_LOGGING_EXTERNAL` to match the chosen mode on `--no-vector`),
  supports `--no-vector` and `--uninstall`, drops `bash -lc` from the
  plist (so repo paths with spaces work), and produces a more useful
  post-install summary.
- `deployment/install/install-systemd.sh` and
  `deployment/systemd/skulk.service` now use the same wrapper +
  `~/.skulk/skulk.env` integration as macOS, with
  `EnvironmentFile=-%h/.skulk/skulk.env` so the unit picks up env-file
  knobs.
- `website/docs/run-skulk-as-a-service.md` updated for the auto-update,
  env-file customization, and Vector agent flow.

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
