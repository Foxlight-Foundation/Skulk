# Slice Placement & Vindex Publisher Plan

Status: **Draft** — open for review, no code committed against it yet.
Owner: TBD.
Branch of origin: `claude/understand-larql-repo-sJqA1`.

## TL;DR

Add a **second placement mode** to Skulk in which the master shards a model's
weights across the cluster instead of placing a whole instance on one node.
The head node stays on MLX (no change to the existing fast path); a new
runner type wraps [LARQL](https://github.com/chrishayuk/larql) servers as
the cold tier that holds dormant expert / FFN weights on commodity RAM.
Vindexes are not extracted inside Skulk — they are consumed from
HuggingFace, published by a separate `skulk-vindex-publisher` tool on a
weekly cron.

The economic argument: MoE sparsity (DeepSeek V3 activates ~37 B of 671 B
per token, Kimi K2 ~32 B of ~1 T) means the dormant majority can sit on
mmap'd DDR4/DDR5 in a cheap commodity box. A $2 k used server with
512 GB ECC carries weights that would otherwise need a second
$10 k+ Mac Studio. 2.5 GbE (≈$25 NICs, <$100 switches) is more than
adequate — the hot-path payload is the residual stream, ~28 KB round
trip per layer for h=7168, ≈35 MB/s at 20 tok/s. The real constraint
is round-trip latency, not bandwidth; wired LAN is fine, WiFi is not.

## Goals

- Run 100 B+ MoE and 200 B+ dense models on Skulk clusters that include
  one Apple Silicon head node plus one or more commodity RAM-rich
  Linux/Windows boxes.
- Preserve the current MLX fast path unchanged for any model that fits
  on the head node.
- Surface honest telemetry (measured RTT, predicted tok/s for the
  slice plan) and let the user ship at any throughput they want — no
  paternalistic gating.
- Make adding a new model to the supported set a manifest edit, not a
  code change.

## Non-goals

- **Replacing MLX.** The head node continues to run MLX. Slice mode
  is purely additive.
- **In-tree vindex extraction.** Skulk does not depend on LARQL at
  build time. Vindexes are downloaded artifacts, like model weights
  today.
- **General-purpose model server framework.** Slice mode targets the
  LARQL wire protocol specifically; we are not building a pluggable
  remote-FFN abstraction in v1.
- **Cross-cloud / WAN deployment.** Slice mode assumes a low-latency
  wired LAN between head and cold tier.
- **Browse / interpretability surface in Skulk.** LARQL's DESCRIBE /
  WALK / TRACE remain LARQL CLI features. Skulk consumes vindexes for
  inference only in v1.

## Architecture decisions (land as ADRs before code)

Three decisions are expensive to reverse once they touch `apply()` or
the runner taxonomy. They get their own short ADRs and merge first.

### ADR-A: New runner type (not sidecar, not in-tree port)

A new `LarqlRunner` subtype of `Worker` runner sits alongside the
existing MLX runner. The worker process supervises a child
`larql-server` binary, exposes its `/v1/expert/batch` and
`/v1/walk-ffn-q8k` endpoints to the rest of the cluster, and emits
heartbeat / health events through Skulk's existing event-sourcing
spine.

Rejected alternatives:

- **Sidecar.** Fastest to prototype but leaks process-lifecycle
  responsibility outside Skulk and forks the operator UX.
- **In-tree port.** Rewriting LARQL's slice protocol against MLX is
  months of work and forks us from upstream improvements.

### ADR-B: Head node stays MLX; cold tier is LARQL

The MLX runner on the head node computes attention locally, then for
designated layers issues an HTTP call to a `LarqlRunner` peer for the
FFN / expert step, receives the post-FFN residual, and continues.
This requires:

- A hook in the MLX forward pass to delegate per-layer FFN execution.
- A small HTTP client in the MLX runner that speaks LARQL's f16-wire
  protocol with `Accept`/`Content-Type` negotiation.
- Knowledge of which layer ranges live on which peer (passed in the
  slice plan).

The head MLX runner never loads a vindex; it loads the standard MLX
weights for the layers it owns (attention, embed, norms, router) and
calls out for the rest.

### ADR-C: Vindex provenance — consume only

Skulk reads vindexes from HuggingFace (`hf://…`) the same way it
reads MLX weights today. It does not extract them. A separate tool,
`skulk-vindex-publisher`, runs LARQL's `extract` + `publish`
pipeline on a cron and maintains the catalogue.

Rejected alternatives:

- **Extract inside Skulk.** Adds a Rust toolchain dependency and a
  slow first-run UX. Punts work from a build box to every user's
  laptop.
- **No catalogue.** Forces every user to either find a community
  vindex or run LARQL themselves to extract one — that's not the
  Skulk pitch.

## Workstreams

### W1 — Slice-aware placement

Files of interest: `src/exo/shared/types/state.py`,
`src/exo/shared/types/events.py`, `src/exo/shared/types/commands.py`,
`src/exo/shared/apply.py`, `src/exo/master/placement.py`.

- New type `SliceSpec` describing a planned partition:
  `{ model_id, head_node, slices: list[SliceAssignment] }` where
  `SliceAssignment = { peer_id, preset: "expert-server" | "server",
  layer_range: tuple[int, int] | None, expert_range: tuple[int, int]
  | None, vindex_uri: str }`.
- New event `SliceInstancePlaced` and command `PlaceSliceInstance`.
- Placement algorithm extension:
  1. If the model fits in the chosen head node's MLX-addressable
     memory, use the existing single-node placement.
  2. Otherwise compute a slice plan by bin-packing the slice catalogue
     (per-slice byte sizes, read from vindex manifests) against
     `Node.available_ram` for non-Apple peers.
  3. Refuse to plan if no plan exists; otherwise emit
     `SliceInstancePlaced` and let workers spin up runners.
- The bin packer is dumb in v1: first-fit by largest slice, no
  multi-objective optimisation.

### W2 — LARQL runner

Files of interest: new `src/exo/worker/runners/larql_runner.py`,
plus changes to whatever owns runner spawn in `src/exo/worker/`.

- Spawn `larql-server <vindex-path> --port <p> --ffn-only` (and
  `--experts <range>` for MoE) as a managed child.
- Health-check via `/v1/health` (or whatever LARQL exposes); fail the
  worker if the child exits.
- Pull vindexes from HF on first use; cache to a shared model store.
  Reuse Skulk's existing model-store abstraction
  (`docs/model-store.md`) — vindex is just another artifact kind.
- Emit `LarqlRunnerReady` event with the served slice presets and
  layer / expert ranges so placement can target it.

### W3 — MLX-side FFN delegation

Files of interest: wherever the MLX forward pass lives in
`src/exo/worker/runners/` and the inference engine docs at
`docs/inference-engine.md` (LARQL's, useful reference for the wire
shape).

- Investigate MLX's hook surface — does it expose per-layer
  pre/post-FFN hooks today? If not, we need a thin Python wrapper
  that splits the forward pass into per-layer steps.
- Build an async HTTP client for `/v1/expert/batch` (MoE) and
  `/v1/walk-ffn-q8k` (dense remote-FFN).
- Wire format: f16 default, i8 opt-in via env var, matching LARQL's
  current contract.
- Token-level latency budget: aim to stay within 2× the local-MLX
  baseline for a single-layer FFN delegation on a wired LAN.

**This is the riskiest workstream.** Land a feasibility spike before
committing to the rest of the plan — if MLX doesn't expose the right
hooks, ADR-B may need to revisit "head stays MLX."

### W4 — Telemetry & UX

Files of interest: heartbeat plumbing (`src/exo/shared/`), dashboard
node graph (`dashboard-react/`), `/v1/models` API endpoint.

- Piggyback link probes on the existing heartbeat: each heartbeat
  carries the round-trip latency to every other peer measured during
  the last interval.
- Predicted tok/s for a proposed slice plan: simple model that
  weights local compute + remote-call RTT × layer count. The user
  sees an estimate before confirming the plan.
- Dashboard: render the slice plan as an annotated node graph with
  per-peer assignments and predicted tok/s. No new screens — just
  fields on the existing model launch flow.
- `/v1/models` gains a `slice_plan` field when the model is launched
  in slice mode, listing the participating peers and per-peer
  responsibilities.

### W5 — Vindex publisher (separate repo)

Repo: `skulk-vindex-publisher` (new, sibling to Skulk).

- `models.yaml` manifest: list of `(model_id, quant, slices)`
  tuples.
- GitHub Actions workflow on a weekly cron + manual dispatch. Each
  entry runs `larql extract … --quant <q> -o <tmp>` then
  `larql publish <tmp> --repo skulk/<repo-name>`.
- Skip-if-unchanged is built into `larql publish` (SHA256 vs HF
  `lfs.oid`), so a "nothing new" run finishes in seconds.
- HF org `skulk/` holds the published vindexes and collections.
- Build infra: one self-hosted runner with NVMe scratch. Start with
  a small VPS (200 GB) for ≤Mixtral 8x22B; scale to a beefier box
  when the killer cases come online.

#### Catalogue v1

| Tier | Model | Size (Q4_K) | Slices |
|---|---|---|---|
| Smoke | Gemma 3 4B | ~2.5 GB | `full` |
| Smoke | Llama 3.2 3B | ~2 GB | `full` |
| Smoke | Qwen 2.5 7B | ~4.5 GB | `full` |
| Sweet spot (MoE) | Gemma 4 26B-A4B | ~15 GB | `full`, `expert-server` |
| Sweet spot (MoE) | Mixtral 8x7B | ~28 GB | `full`, `expert-server` |
| Sweet spot (MoE) | Mixtral 8x22B | ~85 GB | `full`, `expert-server` |
| Killer (MoE) | DeepSeek V3 671B | ~400 GB | `full`, `expert-server` |
| Killer (MoE) | Kimi K2 ~1T | ~600 GB | `full`, `expert-server` (when LARQL lands K2) |
| Killer (dense) | Llama 3.1 405B | ~240 GB | `full`, `server` |
| Stretch (dense) | Qwen 2.5 72B | ~44 GB | `full`, `server` |

Slice selection rationale: since the head node is MLX, Skulk does not
need LARQL's `client` / `attn` / `embed` slices — the MLX node holds
the residual itself. `full` is published for single-machine LARQL
deployments and for forward compatibility; `expert-server` / `server`
is the cold-tier slice Skulk actually consumes.

## Phasing

1. **Phase 1 — Decisions + publisher smoke tier.**
   Land ADR-A / ADR-B / ADR-C. Stand up `skulk-vindex-publisher` and
   ship the three smoke-tier models end-to-end. Proves the pipeline.
2. **Phase 2 — LARQL runner + MoE sweet spot.**
   W2 lands a `LarqlRunner` that can serve a Gemma 4 26B-A4B
   expert-server slice. Skulk can pull the vindex from HF, spawn the
   server, and pass health checks. No integration with the MLX side
   yet — this is a runner-in-isolation milestone.
3. **Phase 3 — MLX FFN delegation feasibility spike.**
   W3 spike: prove the MLX forward pass can delegate FFN per layer
   over HTTP for at least one model (Gemma 4 26B-A4B is the natural
   target). Decision point — does ADR-B hold? If MLX hooks don't
   exist, escalate before continuing.
4. **Phase 4 — Slice-aware placement.**
   W1 lands the state/event/command changes and the bin-packer.
   `LaunchModel` learns to emit a slice plan when the model doesn't
   fit on the head node. End-to-end: Skulk runs Mixtral 8x22B with
   experts on a commodity Linux peer.
5. **Phase 5 — Telemetry & UX.**
   W4 lands link probes, predicted tok/s, dashboard surface.
6. **Phase 6 — Killer use cases + dense remote-FFN.**
   Build infra scales to handle DeepSeek V3 / Llama 3.1 405B
   extractions in the publisher. Skulk's slice mode exercises dense
   `--ffn URL` topology.
7. **Deferred — Kimi K2 / 1T-class.**
   Gated on LARQL landing K2 support upstream.

## Open questions

1. **MLX hook surface.** Does MLX expose pre/post-FFN hooks per
   layer, or do we need to split the forward pass manually? This is
   the largest unknown and gates Phase 3.
2. **Model-store integration.** Vindexes are directory artifacts, not
   single files. Does the existing model store handle directory-shaped
   artifacts cleanly, or does it need an extension?
3. **Auth between peers and LARQL servers.** Skulk's libp2p gives
   us peer identity. LARQL's HTTP server doesn't authenticate today.
   Do we accept LAN-trust in v1, tunnel through libp2p, or add bearer
   tokens to LARQL? (Probably LAN-trust + a roadmap note.)
4. **Failure handling mid-generation.** If an expert server dies
   during a token, the request fails. Retry semantics, partial
   replay, or surface the error? Default to surface; revisit if it
   becomes a UX problem.
5. **Catalogue governance.** Who decides what enters `models.yaml`?
   Suggest: maintainer-curated for v1, community PRs accepted with
   the criterion "this model is requested and LARQL supports it."
6. **Cost-model accuracy.** Predicted tok/s in W4 is naive in v1.
   Calibrate against real runs once Phase 4 is up; revisit if users
   report large mispredictions.

## Issue breakdown (for follow-up)

Suggested split into GitHub issues, grouped by milestone:

- **Phase 1 issues**
  - `ADR-A: New LARQL runner type`
  - `ADR-B: Head stays MLX, cold tier is LARQL`
  - `ADR-C: Vindex provenance — consume only`
  - `skulk-vindex-publisher: scaffold repo + smoke tier`
- **Phase 2 issues**
  - `LarqlRunner: process supervision`
  - `LarqlRunner: vindex pull + model-store integration`
  - `LarqlRunner: health + readiness events`
  - `publisher: MoE sweet-spot tier`
- **Phase 3 issues**
  - `Spike: MLX per-layer FFN delegation`
  - `Spike report + go/no-go on ADR-B`
- **Phase 4 issues**
  - `SliceSpec type + event/command`
  - `placement: bin-packer for slice mode`
  - `LaunchModel: emit slice plan when head node is too small`
  - `e2e: Mixtral 8x22B on Mac + Linux peer`
- **Phase 5 issues**
  - `heartbeat: link probes`
  - `predicted tok/s estimator`
  - `dashboard: slice plan annotation`
  - `/v1/models: slice_plan field`
- **Phase 6 issues**
  - `publisher: scale build infra for 400 GB+ extractions`
  - `dense remote-FFN: Llama 3.1 405B e2e`

## References

- LARQL: <https://github.com/chrishayuk/larql>
- LARQL README sections on `slice`, `publish`, `pull`, expert-server
  topology, and wire format are the operational reference.
- LARQL ADR-0007 (vindex distribution) and ADR-0008 (embed-server)
  describe the slice taxonomy this plan consumes.
- Skulk model store: `docs/model-store.md`.
- Skulk inference engine and runtime notes:
  `docs/inference-engine.md` (LARQL's, kept for reference),
  `docs/model-runtime-notes/`.
