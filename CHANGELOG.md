<!-- Copyright 2025 Foxlight Foundation -->

# Changelog

This project records release notes here and mirrors public-facing notes in
`website/docs/release-notes/`.

## [Unreleased]

### Fixed

- **GPU-wedge runner deaths are no longer retried (wired-memory leak).**
  Contrary to every other crash class, a runner hard-exited by the warmup
  deadline watchdog while its main thread is parked in a faulted Metal eval
  does NOT get its wired GPU memory reclaimed on exit — measured live
  (4-node matrix testing, 2026-06-09): each wedge-exit left ~5GB wired
  behind, recoverable only by reboot, and two automatic retries cost a 24GB
  node ~10GB. Worse, wedges take ~300s each so the 3-failures-in-60s crash
  breaker never trips — unattended, the relaunch loop leaks the node to
  death. The watchdog now exits with a distinct code (`WEDGE_EXIT_CODE`),
  the supervisor marks the failure (`gpu-wedge-deadline` in the runner's
  failure message — a string marker keeps the gossiped status type
  wire-compatible during rolling upgrades), and the worker gives the
  instance up on the FIRST wedge death with a log that names the leak and
  the reboot remedy.
- **`MLX_METAL_FAST_SYNCH` now defaults OFF cluster-wide.** The old ON default
  had no measured upside (vanilla dense decode: 20.8 tok/s off vs 20.7 on) and
  a catastrophic failure mode for any model without a curated card pin:
  hybrid-SSM models wedge at warmup under the flag — gpt-oss hit the 300s
  warmup deadline (#236, card-pinned off on 2026-06-07) and NemotronH-9B did
  exactly the same (#259) — and the resulting deadline kill mid-GPU-work leaks
  ~5GB of wired memory per attempt and degrades the node until reboot. With
  the flag off, Nemotron-Nano-9B warms in seconds and decodes at 19+ tok/s.
  All NemotronH/Nemotron-3-hybrid and gpt-oss cards also carry an explicit
  `runtime.metal_fast_synch = false` pin now, and any model that measurably
  benefits from FAST_SYNCH can pin it on per card; the operator override
  (`--fast-synch`/`--no-fast-synch`) is unchanged.

### Added

- **The dashboard now shows speculative-decoding status per instance.** Active
  instances with an MTP sidecar or assistant drafter display an `MTP D{n}`
  badge next to the status badge — depth from the card's `mtp_max_depth`,
  with the drafter kind spelled out in a hover tooltip.
  The status is derived from the model card's runtime section already present
  in cluster state — the rank-invariant source of truth for whether drafting
  engages (#254) — so no new wire data is needed. Cards that block multi-node
  speculation (`speculative_multi_node=false`) show no badge on multi-node
  placements, matching the runtime behavior.

### Fixed

- **Tensor-parallel placements of sidecar-MTP models no longer crash on the
  first request (#263).** The decider-only sidecar load introduced with the
  explicit lockstep protocol (#254) regressed tensor placements: draft
  logits go through the TP-sharded lm_head, an all-rank collective the idle
  receiver ranks never join, so the lone TP decider GPU-timed-out inside its
  first draft round and SIGABRT'd in the Metal completion block while the
  receivers hung to the eval watchdog. Deterministic on a homogeneous M4
  pair (plain TP decode on the same instance worked; only the speculative
  path wedged). Tensor placements now load the sidecar on every rank and
  draft rank-symmetrically — the same envelope assistants use on TP and the
  configuration the published +31% TP benchmark measured — while the
  decider protocol remains in force for pipeline placements. The drafter
  agreement still disables speculation symmetrically if any TP rank fails
  to produce a working drafter.
- **Placement no longer refuses models whose weights sit in the macOS file
  cache (cache-deflated availability).** The gossiped `ram_available` came
  from mactop's `available` (free + inactive + speculative), which counts
  reclaimable file cache as *used* — so immediately after downloading a model,
  availability was deflated by roughly the model's full size and placement
  refused fits that run comfortably (observed on a 24 GB node: 11.6 GB of
  just-downloaded weights in cache dropped "available" to ~12 GB while
  ~14.6 GB was genuinely wireable). On macOS, `ram_available` is now the
  GPU-wireable figure `total − wired − anonymous − compressor`, taken from a
  `vm_stat` snapshot alongside each telemetry sample; macOS reclaims file
  cache the moment Metal wires pages, so this is what a runner can actually
  use. The metric deliberately does not credit compression of idle anonymous
  memory, preserving the conservative posture of the oversized-placement OOM
  hardening (#243). The worker's local pre-spawn fit guard judges with the
  same metric (it previously used psutil's free + inactive, which would veto
  the very placement the master had just correctly admitted). Value-only
  change to the gossiped figure — the wire shape is unchanged, so
  mixed-version clusters interoperate.
- **Dashboard deep links and browser refresh no longer 404.** The dashboard is
  a SPA that restores its active view from the URL path, but the API served
  `index.html` only at `/` — refreshing on `/chat` (or following a shared link
  to `/cluster`, `/model-store`, `/operator`) returned a bare
  `{"detail":"Not Found"}`. The API now serves the SPA shell for the four
  client routes (kept in sync with the dashboard's `NavRoute`).
- **Multi-node speculative decoding no longer crashes on heterogeneous
  clusters (explicit cross-rank lockstep, #252/#254).** Distributed MTP kept
  ranks in sync by *assuming* every rank independently recomputed bit-identical
  accept/reject decisions from its own logits. Heterogeneous chips break that
  assumption (M5 vs M4 GEMM kernels differ; M5 additionally runs reduced-
  precision B≥2 matmuls), so mixed-chip pipelines desynced: ranks committed
  different token counts, fell out of the collective schedule, and one rank
  SIGABRT'd inside the Metal command-buffer completion block while the other
  waited on a `MTLSharedEvent` forever. The protocol is now explicit: the
  decider (last) rank alone holds the drafter, drafts, and decides; draft
  tokens and the per-round accept outcome (`[prefix_len, bonus_token]`) are
  broadcast via fixed-shape `all_sum` collectives, and receiving ranks apply
  the broadcast decisions to their own cache slices without ever sampling or
  comparing logits. The same applies to the request's first sampled token, and
  the non-MTP pipeline fallback broadcasts each step's token the same way (its
  per-rank sampling silently desynced heterogeneous ranks). Sidecar drafter
  weights now load only on the decider rank (matching assistants), saving
  drafter memory on every other rank.
- **Crash circuit breaker now trips once per crash loop, not once per failure.**
  `CrashWindow.record()` is edge-triggered: it returns `True` only when the
  in-window failure count *crosses* the threshold and stays latched (returning
  `False`) until the window drains below it, and `_give_up_on_instance` no
  longer clears the window. Previously the trip was level-triggered and the
  window was cleared on give-up, so a doomed instance lingering in replicated
  state before its `DeleteInstance` landed could re-accumulate and re-trip,
  emitting duplicate `DeleteInstance` commands and "giving up on instance" logs.
  `InstanceId`s are unique, so the retained failure history can never collide
  with a future instance, and the worker reclaims breaker entries for deleted
  instances each planning tick (`CrashWindow.retain`) so the history can't grow
  unbounded. (Follow-up to #243.)
- **Oversized model placements no longer brick a node.** Placing a model whose
  shard does not fit a node's memory previously passed an over-optimistic
  admission check (1.05x weights against gossiped `ram_available` only),
  OOM-aborted during load, and orphaned wired GPU memory reclaimable only by
  reboot — then the worker relaunched the doomed runner every ~1.5s with no
  backoff, compounding the leak (the GLM-4.7-Flash incident). Three changes
  harden this: (1) placement estimates a realistic footprint — weights x 1.30,
  an explicit KV-cache reservation for an 8192-token planning budget, and a
  per-node cap at the Metal GPU working-set ceiling (~75% of RAM) — and shards
  proportionally to fit heterogeneous clusters rather than refusing what fits;
  (2) the worker refuses a shard that won't fit *local, current* memory before
  spawning the runner, failing cleanly instead of OOM-aborting; (3) a crash
  circuit breaker gives up after 3 runner failures within 60s and deletes the
  instance instead of looping. Estimation lives in one shared module
  (`skulk.shared.models.memory_estimate`) used by both the master admission
  check and the worker guard so the two never disagree.
- **macOS node telemetry no longer crashes MLX inference (macmon → mactop).**
  Skulk's `InfoGatherer` spawned `macmon` at 1 Hz for hardware metrics on every
  macOS node. macmon reads the GPU via IOKit/IOGPUFamily — the same interface
  Metal uses for command-buffer completion — so sampling it concurrently with
  an in-flight MLX command buffer put the GPU into an error state that
  `mlx::core::gpu::check_error` threw inside the Metal completion-dispatch
  block: either an uncaught `abort()` (SIGABRT) or a silent GPU hang. On macOS
  the wedged GPU then starved WindowServer past its watchdog and **rebooted the
  node**. (Confirmed upstream as exo-explore/exo#2088 / #1823.) Replaced macmon
  with [`mactop`](https://github.com/metaspartan/mactop), which reads Apple's
  IOReport/SMC counters (not IOGPUFamily), needs no root, emits newline-
  delimited JSON (`--headless --format json`), and exposes a superset of the
  metrics (GPU util %, power breakdown, temps, DRAM bandwidth, system RAM).
  Validated on M4 hardware running sustained MLX inference with zero crashes.
  Provisioning moved with it: `README`/`CONTRIBUTING` (`brew install mactop`),
  the nix dev shell + package wrapper (`pkgs.mactop`), and the PyInstaller
  bundle. When mactop is absent the gatherer still falls back to psutil for
  memory. (mactop's reported `available` RAM equals `total − used`, the same
  figure macmon derived, so placement margins are unchanged.) The gossiped
  `NodeGatheredInfo` event keeps a decode-only `MacmonMetrics` shim so a
  newly-upgraded node still applies telemetry from macOS workers on the
  pre-mactop build during a rolling upgrade. A blank or unparseable line from
  mactop is now skipped rather than tearing down and respawning the subprocess.
- **Topology GPU bar no longer renders 100× too high.** The dashboard treated
  `SystemPerformanceProfile.gpuUsage` (a 0–100 percent) as a 0–1 fraction and
  re-multiplied it by 100, so e.g. 8.66% GPU showed as 866%. It is now
  converted to a fraction when populating the node's monitoring snapshot.

### Changed

- **The codebase is now Skulk all the way down (exo -> skulk rename).**
  The Python package is `skulk`, the Rust bindings crate is
  `skulk_pyo3_bindings`, the wire identity fields are
  `skulkVersion`/`skulkCommit`, and environment variables use the
  `SKULK_*` prefix. Backward compatibility is explicit: legacy `EXO_*`
  environment variables are aliased at startup (an explicit `SKULK_*`
  value always wins), the legacy `exo.yaml` config name is still
  honored, a populated pre-rename `~/.exo/staging` directory keeps
  being used when staging is unconfigured, and the dashboard migrates
  saved favorites/recents once. The deprecated `uv run exo` alias is
  removed — the command is `uv run skulk`. Upstream attribution (the
  "forked from exo" acknowledgment and exo's license copyright) is
  deliberately preserved.

### Fixed

- **Empty `messages` (and non-positive `max_tokens`) are rejected with
  400 instead of crashing the runner.** An empty message array was
  accepted, then `apply_chat_template([])` raised `IndexError` inside the
  runner — taking down the process serving that instance. A single
  renderability guard at the shared text-generation dispatch chokepoint
  (covering chat, Claude, Ollama, and Responses wire formats) now returns
  400 before the request reaches a runner. Found by the post-rename
  torture battery (#233).

- **Requests no longer hang when a node dies mid-generation.** When an
  instance was lost (node disconnect, crash, or deletion with a request
  in flight), the master tore the instance down but left the task
  orphaned — the API never received a terminal chunk, so the open HTTP
  connection hung until the client's own timeout. The master's plan loop
  now emits `TaskFailed` for in-flight API tasks whose instance is gone,
  and the API turns that into a terminal error chunk: streaming
  responses close with an error event, non-streaming requests return a
  500. Found by the 2026-06-07 node-kill drill (#223).

- **Master failover no longer strands open requests.** Killing the
  master mid-generation starts a new cluster session that cannot carry
  the old session's tasks, and the API's session reset replaced its
  command-queue maps without closing the old streams — a guaranteed
  permanent hang that the orphaned-task sweep above structurally could
  not cover. The API now fails every open command stream at the session
  boundary with an error explaining the session changed and asking the
  client to retry.
  Verified end-to-end: clients receive the error within ~4–6 seconds of
  a node kill (master or worker rank), versus an indefinite hang before.

### Changed

- **Speculative-decoding draft depths are now per-card measured optima.**
  A production depth sweep (3×200-token greedy A/Bs per cell) moved the
  gemma E-series assistant cards from depth 3 to depth 2 — E2B-8bit
  37.7 → 54.0 tok/s (+43%, was +20% at depth 3), E4B-8bit 19.5 → 25.4
  (+30%) — and Qwen3.5-27B from depth 2 to depth 1 (6.3 → 10.5 tok/s on
  a 2-node pipeline, +67%; depth 2's run-to-run spread was the
  GDN/SSM deferred-replay tax). Mechanism: on M4-class GPUs verifying up
  to 2 candidate tokens per step is effectively free, but each candidate
  beyond width 2 costs ~36% of a full forward pass — so drafting deeper
  than depth 2 over-spends on every model measured, and SSM-hybrid
  models pay an additional replay tax even at depth 2. Rule of thumb:
  gemma assistant cards depth 2, Qwen GDN sidecar cards depth 1.

- **A bare `repetition_penalty` no longer crashes the runner.** Requests
  carrying `repetition_penalty` without `repetition_context_size` passed
  the request's None straight into mlx-lm's processor builder, overriding
  its default of 20; the penalty processor's `tokens[-None:]` slice then
  raised and killed the runner on the first penalized request. Both call
  sites now coerce None to the default. Found by the 2026-06-06
  before/after benchmark matrix; applies to every model and every client
  that sends a penalty alone (many do by default).

### Added

- **Staged model copies now have a lifecycle, and nodes can report their
  storage.** With the model store on, staged copies previously survived
  instance deletion and node crashes forever (58-70 GB piles; one node
  died of a full disk in the launch smoke). `cleanup_on_deactivate` now
  defaults to true with a recent-use grace budget
  (`staging_keep_recent_gb`, default 40 GiB): when an instance shuts
  down — and at node startup, which reconciles copies orphaned by a
  crash — not-in-use staged models are kept newest-first by last use up
  to the budget and evicted beyond it. In-use detection includes
  companion repos (MTP sidecar / assistant / vision weights) of active
  models, so eviction can never corrupt a live runner. The grace budget
  is deliberate: node deaths, restarts, and repeated place/delete cycles
  of the same model do not re-pay the staging copy. New
  `GET /store/storage` returns the local node's breakdown: staged models
  with size/last-use/in-use, event-log bytes, and disk free.

### Fixed

- **Event logs can no longer eat the disk or kill nodes on a full one.**
  The API-side event log — which records per-token chunk events and backs
  only the `GET /events` diagnostic — had NO retention and grew for the
  life of the session (54 MB in 9 idle hours on every node; the file a
  node died writing during the launch smoke). It now ring-compacts past
  256 MiB, keeping the most recent 20k events. Archive rotation is capped
  by total bytes (1 GiB) in addition to count — five archives of
  unbounded size defeated the count cap in practice (3.5 GB observed).
  The remaining unguarded ENOSPC sites (`DiskEventLog.__init__` and
  `compact()` — the former is exactly where a node died) now degrade to
  the counting-only mode instead of crashing, and a proactive free-space
  floor (2 GiB, checked every 1024 appends) degrades persistence BEFORE
  the disk hits zero — a master on a full disk previously throttled the
  whole cluster to ~0.5 tok/s before dying. Log noise that bloats piped
  logs was also trimmed (per-minute download-coordinator path dumps), and
  the speculative-decoding enable line now reports the card's actual
  draft depth instead of a hardcoded "(D=1)".
- **Speculative decoding now engages for models that were already on
  disk.** Three of the model-store downloader's four resolution paths
  (already-staged fast path, store staging, direct-from-store) returned
  the base model without fetching the card's companion repos (MTP
  sidecar / assistant model / vision weights) — a staged model would
  load and silently run without speculation (observed in the launch
  smoke). Every `ensure_shard` resolution now also ensures companions
  through the same store-first path (so sidecars are served from the
  store when present), optional-companion fetch failures (MTP
  sidecar / assistant) log loudly without failing the base load, while
  split vision weights stay load-bearing, and the previously triplicated companion
  construction in the HF downloader is shared via
  `companion_download_specs`.
- **A wedged warmup no longer silently disables a node.** A faulted
  Metal eval can park warmup forever at 0% CPU (uninterruptible from
  Python); the runner then sat in `RunnerWarmingUp` indefinitely while
  every API request queued and timed out with no surfaced error. Warmup
  now runs under a hard deadline (default 300s,
  `SKULK_WARMUP_DEADLINE_SECONDS`): on overrun the runner logs a
  CRITICAL diagnosis (including the reboot-if-GPU-wedged guidance) and
  exits, the supervisor reports `RunnerFailed`, and the node keeps
  dispatching.
- **Disabled speculation is no longer near-silent.** Requests carrying
  logits processors (typically a `repetition_penalty` — some client
  libraries send one by default) fall back to plain decode; that
  fallback now logs a WARNING naming the cause and the fix instead of
  an easy-to-miss INFO line. The gemma 4 E-series pipeline rejection
  also explains itself in operator terms (place on a single node)
  instead of internals-speak.

- **Multi-node placement is now reliable and placement failures are
  visible.** Four compounding issues fixed in the placement path:
  (1) memory admission is per node instead of summed across the cycle —
  Tensor sharding splits weights evenly, so a 16+24 GB pair whose *sum*
  covered the model could be admitted with the even split overloading the
  smaller node; (2) admission requires runtime headroom
  (weights x 1.05 + 256 MB per node) on top of raw weight bytes — an
  exact weights-equal-free-memory fit previously produced a silent
  thrash (observed: 12-token prefill in 1230 s) instead of a refusal;
  (3) placing immediately after cluster formation no longer fails with a
  false "insufficient memory" — cycles touching nodes whose memory info
  has not been gossiped yet are now reported as info-pending, and
  `POST /place_instance` waits up to 15 s for the info before returning
  503; (4) impossible placements now fail loudly at the API with the
  specific typed reason (400) instead of returning "Command received"
  and silently failing on the master, leaving clients with unexplained
  404s. The old catch-all "No cycles found with sufficient memory" error
  (which fired for topology gaps, exclusions, startup races, AND real
  shortfalls alike) is split into per-stage `PlacementError` messages
  that include the per-node GB arithmetic.
- **Production MTP no longer runs ~20-46x slower than plain decode.**
  `FAST_SYNCH_CLUSTER_DEFAULT = True` silently applied
  `MLX_METAL_FAST_SYNCH=1` to every MTP runner, collapsing the
  speculative loop (Qwen3.5-9B-4bit on M4, mlx 0.31.2: 27.7 tok/s with
  the flag off vs 0.6 tok/s with it on) while leaving vanilla decode
  untouched (20.8 vs 20.7 tok/s). Probe harnesses never set the flag,
  which is why isolated measurements showed +26-50% while the production
  stack inverted. `resolve_metal_fast_synch` now defaults to OFF for any
  card that declares a speculation mechanism (`mtp_heads`,
  `mtp_sidecar_repo`, or `assistant_model_repo`); operator overrides and
  explicit card pins keep their precedence. Validated end-to-end:
  production Qwen 9B MTP went from 9.5 to 27.7-28.5 tok/s on a 16GB M4
  (~+55% over plain decode, 82-84% acceptance).

### Added

- Distributed gemma4 assistant drafting + gemma4 pipeline sharding (#201
  Track 2b): assistant-model speculation now runs on pipeline placements
  via LAST-RANK drafting — the assistant cross-attends the target's last
  full-attention/sliding KV layers (resident on the final slice by
  construction) and post-norm hidden (already all-gathered), and every
  rank joins one fixed-shape `all_sum` per round carrying the draft
  tokens (plus the drafter's effective distribution under sampling, so
  ratio-acceptance runs identically everywhere; drafting-rank draws use
  explicit per-round keys to keep global RNG streams aligned). Assistants
  load on the last pipeline rank only. En route, gemma4 pipeline sharding
  itself was made to work at all: decoder layers return (hidden, kvs,
  offset) tuples the wrappers now carry, and layer_types/previous_kvs/
  make_cache are re-keyed per slice — slices cutting a KV-sharing edge
  (E-series) fail loud, since those models fit single-node anyway. Two
  cross-attention correctness bugs found and fixed (masked by mlx-vlm's
  native rollback): deferred replay starved assistant drafters of
  committed tokens (74% -> 28% acceptance; the Drafter protocol gains
  `reads_target_cache` and the loop flushes immediately for such
  drafters), and the drafter held a COPY of the cache list that froze its
  view at the first reject-restore (progressive 56% -> 26% decay; it now
  holds the live sequence). Gemma4 coverage grew three validated cards —
  12B (2.03x single-node, 95%-of-single across 2 nodes), 31B (2.48x
  single; the pipeline flagship: 2x16GB nodes lift vanilla 3.8 -> 5.6 and
  MTP reaches 7.75 tok/s), E2B (1.56x) — with assistant-pipeline lockstep
  regression tests (greedy + sampled) alongside the Track 1/2a ones.

- Pipeline speculative decoding (#201 Track 2a): sidecar MTP now runs on
  pipeline-sharded placements with NO new distributed protocol — pipeline
  decode was already rank-symmetric (`pipeline_auto_parallel` slices only
  layers; embed/norm/head load in full everywhere, and decode-mode
  `PipelineLastLayer` all-gathers the final hidden to every rank), so the
  existing bonus-driven loop runs identically on each rank. Lockstep
  validated like Track 1: greedy byte-parity (depths 1-2) and
  seeded-sampled trace parity over 300-token generations, on a localhost
  ring and on real two-node hardware; pipeline acceptance matches
  single-node (79% on the 2B — full-precision slices, unlike TP's
  resharded reductions). Drafting is rank-local against replicated
  embed/head and overlaps the pipeline's sequential bubble; the K+1-wide
  verify pays one hop-set regardless of width, so inter-node latency
  amortizes per committed token — the placement where speculation helps
  most. Safety rails: a per-request `all_sum` keeps the speculate-or-not
  choice symmetric when a rank's sidecar is missing, and mid-request
  drafter failures abort loudly on multi-rank placements instead of
  silently forking the collective schedule. Assistant drafters (gemma4)
  cross-attend the target's KV — which a pipeline shard only holds for
  its own layers — and stay single-node/TP (#201 Track 2b). Pipeline
  lockstep regression tests (greedy + sampled) join the TP ones.

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
  (`src/skulk/worker/engines/mlx/drafters/`). The generation loop now talks to
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
