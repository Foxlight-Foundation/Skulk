<!-- Copyright 2025 Foxlight Foundation -->

# Implementation Plan: Decoupling the Inference Engine from the Inference Pipeline

**Status:** Proposed — ready for implementation handoff
**Audience:** An implementing agent/engineer who has not yet read the worker code
**Scope:** Refactor the worker runner subsystem so that the inference *engine*
(MLX-LM, a future MLX-vLLM, HF Transformers, image diffusion, …) is a
swappable implementation behind a stable interface, while the inference
*pipeline* (lifecycle state machine, IPC, scheduling, tracing, diagnostics)
becomes a single engine-agnostic host. The end goal is that different models
can run on different engines, and — because placement already assigns each
instance to its own subset of nodes — different engines can run concurrently
on different sub-clusters with no central conflict.

> **One decision is deliberately deferred** (see
> [§9 Open Decisions](#9-open-decisions)): *how* an engine is selected per
> model (an explicit `engine` field on the model card vs. a per-`ModelTask`
> default vs. a registry heuristic). Phases 1–3 below are intentionally
> structured so that this decision can be made late without rework. Do **not**
> hardcode a selection policy before that decision is made; route every
> selection call through the single resolver seam defined in Phase 3.

---

## 1. Why this is tractable today

Skulk **already runs more than one engine.** This is the single most important
fact for the implementer to internalize before touching anything:

- `src/skulk/worker/runner/embeddings/runner.py` loads models with **HuggingFace
  Transformers + torch** (MPS/CUDA/CPU), *not* MLX. It has no KV cache, no
  distributed group, no generation loop.
- `src/skulk/worker/runner/image_models/runner.py` runs **diffusion** models
  (FLUX/Qwen-Image) with CFG-parallel sharding.
- `src/skulk/worker/runner/llm_inference/runner.py` runs **MLX-LM** text
  generation with pipeline/tensor parallelism, KV cache, speculative decoding,
  and continuous batching.

So engine pluggability is not greenfield — it exists *de facto*. The problems
are purely structural:

1. **Dispatch is keyed on `ModelTask`, hardcoded in one `if/elif/else`** in
   `bootstrap.entrypoint` (`src/skulk/worker/runner/bootstrap.py:437-464`).
   One task ⇒ exactly one engine, forever. This is what makes "Qwen on MLX-LM,
   Llama on MLX-vLLM" (same `TextGeneration` task, different engine)
   **impossible** today.
2. **Each runner re-implements the entire lifecycle/IPC/tracing/diagnostics
   boilerplate.** The three runner classes share a shape but no base.
3. **The MLX-LM runner conflates pipeline and engine.** Its lifecycle state
   machine calls MLX module-level functions directly instead of through an
   interface.

The target is to name the seam that already implicitly exists, lift it out of
MLX, and host all engines on one generic pipeline.

---

## 2. Current architecture (ground truth, with file references)

### 2.1 Process model

A `Worker` (`src/skulk/worker/main.py`) spawns one **runner subprocess per
shard** of each placed instance. The parent-side object is
`RunnerSupervisor` (`src/skulk/worker/runner/runner_supervisor.py`); the child
process entrypoint is `entrypoint()` in
`src/skulk/worker/runner/bootstrap.py`.

```
Worker (main.py)
  └── _create_supervisor(CreateRunner)            # main.py:846, calls RunnerSupervisor.create (:874)
        └── RunnerSupervisor (runner_supervisor.py)        ── PARENT side, ALREADY engine-agnostic
              └── mp.Process(target=entrypoint, ...)       # runner_supervisor.py:171
                    └── entrypoint(...)  (bootstrap.py:385) ── CHILD side
                          ├── if  is_image_model     → image_models.runner.Runner
                          ├── elif is_embedding_model → embeddings.runner.Runner
                          └── else (MLX-LM)          → llm_inference.runner.Runner
```

**Key insight #1:** `RunnerSupervisor` is *already* engine-agnostic. It starts a
process, forwards `Task`s in and `Event`s/diagnostics out over `mp` channels,
watches liveness, handles wedge/crash classification, and forwards cancels. It
knows nothing about MLX. **It does not need to change** (beyond possibly passing
an engine id through, which it gets for free — see Key insight #3). The engine
coupling is entirely *inside the subprocess*, in the `Runner` classes.

### 2.2 The MLX-LM runner: where engine bleeds into pipeline

`llm_inference/runner.py` `class Runner` is a **lifecycle state machine** driven
by a sequence of lifecycle `Task`s sent from the worker:

```
RunnerIdle
  ─ConnectToGroup→ RunnerConnecting ─→ RunnerConnected     # only for world_size > 1
  ─LoadModel→      RunnerLoading    ─→ RunnerLoaded
  ─StartWarmup→    RunnerWarmingUp  ─→ RunnerReady
  ─TextGeneration→ RunnerRunning    ─→ (loop) ─→ RunnerReady
  ─Shutdown→       RunnerShuttingDown ─→ RunnerShutdown
```

The lifecycle/IPC/tracing parts are **engine-agnostic**, but the handlers call
MLX directly:

| Lifecycle handler (runner.py) | Engine-specific call it makes |
|---|---|
| `ConnectToGroup` (`:253-285`) | `initialize_mlx(bound_instance)` → `mx.distributed.Group` |
| `LoadModel` (`:288-352`) | `load_mlx_items(...)` → model, tokenizer, vision processor, MTP weights, assistant model |
| `LoadModel` (`:341`) | `Builder.build()` → chooses `SequentialGenerator` vs `BatchGenerator` |
| `StartWarmup` (`:354-410`) | `generator.warmup()` |
| `TextGeneration` (`:412`, `_run_generation_loop` `:554`) | `generator.submit()` / `generator.step()` |
| `Shutdown` (`:426`) | `generator.close()`, `mx.clear_cache()` |

The `Builder` dataclass (`runner.py:730-869`) holds the half-loaded engine
state (`inference_model`, `tokenizer`, `group`, `vision_processor`,
`mtp_weights`, `assistant_model`) and encodes **MLX-specific generator
selection** (KV backend → force sequential; Gemma4 → force sequential; MTP →
force sequential; else batch).

### 2.3 The interface that already 80% exists

`InferenceGenerator` (ABC) in
`src/skulk/worker/runner/llm_inference/batch_generator.py:87-122` is already a
clean *serve-phase* contract:

```python
class InferenceGenerator(ABC):
    def warmup(self) -> None: ...
    def submit(self, task: TextGeneration) -> None: ...
    def step(self) -> Iterable[tuple[TaskId, ToolCallResponse | GenerationResponse | Cancelled | Finished]]: ...
    def close(self) -> None: ...
```

Implemented by `SequentialGenerator` and `BatchGenerator`. **It is the right
shape**, but (a) it is typed in MLX terms (`mx.array`, `Model`,
`mx.distributed.Group`), (b) it physically lives among MLX imports, and (c) it
covers only the *serve* half — the *load/connect* half leaks into `Runner` and
`Builder`.

### 2.4 The worker lifecycle planner also encodes engine assumptions

**Key insight #2:** `src/skulk/worker/plan.py` is a pure function that decides,
from the gossiped statuses of all runners of an instance, which lifecycle
`Task` to emit next. It is **distributed-aware**:

- `_connect_to_group` (`plan.py:~193`) only emits `ConnectToGroup` for
  multi-node instances and encodes ring-formation ordering (accepting ranks
  first, then `rank == world_size - 1`).
- `_load_model` (`plan.py:198-236`) skips `ConnectToGroup` entirely for
  single-node instances (`is_single_node_instance` → straight to `LoadModel`,
  `:219-221`).
- `_ready_to_warmup` (`plan.py:239-289`) encodes the `rank != 0` then `rank == 0`
  warmup ordering and a per-family "independent distributed warmup" exception
  (`_uses_independent_distributed_warmup`, Gemma4).

Therefore an engine that does **not** form an `mx.distributed.Group` (HF
Transformers embeddings; a single-process engine; an engine that owns its own
collective like vLLM) needs the planner to be driven by **engine capability**,
not by hardcoded family/`world_size` logic alone. This is the second place
(besides the master fit-check) where "different engine" changes *correctness*,
not just plumbing.

### 2.5 Lifecycle tasks and statuses are gossiped (wire-compat constraint)

`ConnectToGroup`, `LoadModel`, `StartWarmup`, `Shutdown` (`tasks.py:48-101`) and
the `RunnerStatus` union (`shared/types/worker/runners.py`) are **event-sourced,
gossiped types** shared across master/worker/runner. The codebase deliberately
avoids wire-breaking changes during rolling upgrades (e.g. the
`WEDGE_FAILURE_MARKER` string is a comment-documented workaround to avoid adding
a `RunnerStatus` field). **Any change to these types must be additive and
backward-compatible.** Prefer driving new behavior from engine capability
resolved *locally* on each node over inventing new gossiped task/status types.

### 2.6 Engine identity already flows for free

**Key insight #3:** `ShardMetadata` (`shared/types/worker/shards.py`) embeds the
full `ModelCard` (`BaseShardMetadata.model_card`). `ModelCard` flows
master → placement → `Instance` → `ShardMetadata` → `BoundInstance` →
`RunnerSupervisor` → `entrypoint`. **So whatever identifies the engine, if it
lives on the `ModelCard`, already reaches every node and every runner with zero
new plumbing through placement.** This is why the selection-policy decision can
be deferred cheaply: the *transport* for the decision already exists.

### 2.7 Master-side coupling: the memory model

The README's "memory-safe placement, checked twice" invariant
(`master/placement.py`, `master/placement_utils.py`,
`shared/models/memory_estimate.py`, and a worker-side pre-spawn re-check at
`worker/main.py:820`) assumes the MLX memory footprint. Different engines have
different footprints (KV-cache layout, weight residency, framework overhead).
This is the **single highest-risk correctness surface** of the whole effort and
is intentionally isolated into its own phase (Phase 4).

---

## 3. Target architecture

```
worker/runner/host.py            # ONE generic pipeline host: lifecycle SM, IPC, tracing,
                                 #   diagnostics, cancel handling, the submit/step drain loop.
                                 #   Engine-agnostic. Replaces the per-runner duplication.
worker/engines/base.py           # InferenceEngine protocol + shared value types
                                 #   (EngineCapabilities, EngineTopology, EngineResult).
worker/engines/registry.py       # The SINGLE selection seam: (ModelCard) -> EngineFactory.
                                 #   Selection policy is deferred (§9) but lives ONLY here.
worker/engines/mlx_lm/           # today's llm_inference engine, behind the protocol
worker/engines/transformers/     # today's embeddings engine, behind the protocol
worker/engines/image/            # today's image engine, behind the protocol
worker/engines/mlx_vllm/         # FUTURE — the first net-new engine, proves the seam
```

The `host` owns everything that is *not* model math:

- the receive loop (`main()`), status transitions, `TaskAcknowledged`,
  `TaskStatusUpdated`, `RunnerStatusUpdated`
- trace sessions + flight-recorder phase records + diagnostics
- the `submit → step → finished` drain loop, cancel handling
- the deadline watchdogs (group-connect, warmup)

The `engine` owns only model-specific behavior, expressed as the lifecycle
verbs the host already calls today:

```python
# worker/engines/base.py  (illustrative — finalize signatures during Phase 1)

class EngineCapabilities(Protocol):
    supports_batching: bool
    supports_distributed: bool          # drives plan.py: emit ConnectToGroup or not
    forms_mlx_group: bool               # whether load needs an mx.distributed.Group
    independent_distributed_warmup: bool # replaces the Gemma4 special-case in plan.py
    # memory estimation hook used by the master fit-check (Phase 4)
    def estimate_footprint(self, shard: ShardMetadata, ctx: MemoryContext) -> Memory: ...

class InferenceEngine(Protocol):
    capabilities: EngineCapabilities

    # ---- LOAD phase (today's ConnectToGroup / LoadModel / StartWarmup) ----
    def connect(self, bound_instance: BoundInstance, *, on_progress) -> EngineTopology | None: ...
    def load(self, bound_instance: BoundInstance, topology: EngineTopology | None,
             *, on_layer_loaded) -> None: ...
    def warmup(self) -> None: ...

    # ---- SERVE phase (today's InferenceGenerator) ----
    def submit(self, task: TextGeneration) -> None: ...
    def step(self) -> Iterable[tuple[TaskId, EngineResult]]: ...

    # ---- teardown ----
    def close(self) -> None: ...
```

`EngineResult` is the engine-neutral spelling of today's
`GenerationResponse | ToolCallResponse | Cancelled | Finished`. For non-text
engines (embeddings, image) the serve phase produces their existing chunk
types; keep the result type a discriminated union and let the host translate to
`ChunkGenerated` exactly as the three runners do today.

**The host calls `connect`/`load`/`warmup` during the load phase and
`submit`/`step`/`close` during serve — these are precisely the calls currently
inlined in `llm_inference/runner.py`, made polymorphic.** Each existing runner
becomes an engine implementation minus ~150–250 lines of duplicated host
boilerplate.

### 3.1 How multi-engine on sub-clusters falls out

Placement already assigns each instance to a specific subset of nodes
(`Instance.shard_assignments`, `ShardMetadata.world_size`). Two instances of two
models on disjoint node sets already run independent runner subprocesses with no
shared state. Making them *different engines* requires only:

1. Engine identity reaching the node — **already true** (Key insight #3).
2. `bootstrap.entrypoint` resolving the engine via `registry.py` instead of the
   `is_*_model` ladder.
3. `plan.py` and the master fit-check consulting `EngineCapabilities`.

There is **no central engine registry that needs locking or coordination**: each
node resolves the engine for each shard locally from the card it already has.
Concurrency across sub-clusters is a property of the existing per-instance
isolation, not a new mechanism to build.

---

## 4. Phased work plan

Each phase is independently shippable, ordered by ascending risk. Phases 1–3 are
pure structure with **no intended behavior change** and are covered by existing
runner tests. Phase 4 is the genuine design risk. Phase 5 is the first
net-new engine.

### Phase 0 — Characterization tests (do this first)

**Goal:** Lock current behavior before refactoring so regressions are visible.

- Inventory and run the existing runner/engine tests:
  `src/skulk/worker/tests/`, `src/skulk/worker/engines/mlx/tests/`,
  `src/skulk/worker/runner/llm_inference/` tests, plus any
  `test_plan*`/lifecycle tests.
- Add a focused characterization test (if absent) that drives a single-node
  MLX-LM runner through `Idle→Loaded→Ready→Running→Shutdown` with a fake task
  channel and asserts the emitted `Event` sequence. This is the oracle for
  Phases 1–2.
- **Acceptance:** `uv run pytest` green; the lifecycle characterization test
  exists and passes.

### Phase 1 — Extract `InferenceEngine` from the MLX-LM path (no behavior change)

**Goal:** Define the interface and make today's MLX-LM path implement it, with
the `Runner` calling through the protocol instead of MLX functions directly.

1. Create `worker/engines/base.py` with `InferenceEngine`,
   `EngineCapabilities`, `EngineTopology`, `EngineResult`, `MemoryContext`.
   Keep types engine-neutral (no `mx` imports in this module).
2. Create `worker/engines/mlx_lm/engine.py` implementing `InferenceEngine`:
   - `connect()` wraps `initialize_mlx(bound_instance)` (today
     `runner.py:281`) and returns an `EngineTopology` carrying the
     `mx.distributed.Group`.
   - `load()` wraps `load_mlx_items(...)` + `Builder.build()` (today
     `runner.py:328-341`).
   - `warmup()/submit()/step()/close()` delegate to the existing
     `SequentialGenerator`/`BatchGenerator` (today `InferenceGenerator`).
   - `capabilities` reports `supports_distributed=True`,
     `forms_mlx_group=True`, and the batching/MTP/Gemma4 facts the `Builder`
     currently computes.
   - Move the `Builder.build()` sequential-vs-batch selection logic verbatim
     into the engine; it is engine-internal detail.
3. In `llm_inference/runner.py`, replace the direct MLX calls in the lifecycle
   handlers with calls to an `InferenceEngine` instance. The state machine and
   all event emission stay byte-for-byte the same.
4. **Do not** move IPC/host code yet — only swap the engine calls.

**Acceptance:** Phase 0 tests green, unchanged. `uv run basedpyright` clean
(the engine boundary must be fully typed). No new `mx` import appears outside
`worker/engines/mlx_lm/` and existing MLX modules.

### Phase 2 — Generalize the host; re-express all three runners as engines

**Goal:** One host, three engines, zero duplication.

1. Create `worker/runner/host.py` by lifting the engine-agnostic machinery out
   of `llm_inference/runner.py`: the `main()` receive loop, the lifecycle
   state machine, `update_status`/`send_task_status`/`acknowledge_task`, the
   `submit/step/finished` drain (`_run_generation_loop`), cancel handling,
   trace flush, the deadline watchdogs. The host holds an `InferenceEngine`
   and drives it.
2. Re-express `embeddings/runner.py` and `image_models/runner.py` as
   `InferenceEngine` implementations under `worker/engines/transformers/` and
   `worker/engines/image/`. Their load/serve logic moves wholesale; their
   bespoke lifecycle/IPC boilerplate is deleted in favor of the host.
   - Embeddings engine: `capabilities.supports_distributed=False`,
     `forms_mlx_group=False`, `supports_batching=False`.
   - Image engine: keep CFG-parallel sharding semantics; report capabilities
     accordingly.
3. The host must support engines whose serve phase is request/response
   (embeddings: single forward, no stream) vs. streaming (LLM) vs. image. Model
   this as the `EngineResult` union + a capability flag; do not special-case
   engine identity in the host.

**Acceptance:** Phase 0 tests green. Embeddings and image E2E paths behave
identically (add focused characterization tests if missing). The three former
`runner.py` files contain no lifecycle/IPC boilerplate — only engine logic (or
are deleted in favor of `worker/engines/*/engine.py`).

### Phase 3 — Registry + selection seam (policy still deferred)

**Goal:** Route engine selection through exactly one resolver, without yet
committing to a selection *policy*.

1. Create `worker/engines/registry.py` with:
   - an `EngineFactory` registration table keyed by an `engine_id: str`,
   - `resolve_engine(model_card: ModelCard) -> EngineFactory`.
2. Replace the `if/elif/else` in `bootstrap.entrypoint`
   (`bootstrap.py:437-464`) with `resolve_engine(card)(...)`.
3. For now, `resolve_engine` reproduces **today's exact behavior** via the
   `ModelTask`-based mapping (`TextEmbedding → transformers`,
   `TextToImage/ImageToImage → image`, else `mlx_lm`). This is the *fallback*
   that the deferred policy (§9) will layer on top of — keep it as the default
   branch so nothing breaks when the real selector lands.
4. Add `apply_mlx_patches()` invocation into the MLX engine factory rather than
   `bootstrap` (so non-MLX engines don't import/patch MLX).

**Acceptance:** Observable behavior identical to Phase 2. The `is_image_model`/
`is_embedding_model` ladder in `bootstrap.py` is gone; all selection flows
through `registry.resolve_engine`. Adding a new engine is now a registration,
not a `bootstrap` edit.

### Phase 4 — Make master + planner engine-capability-aware (the real risk)

**Goal:** Correctness for engines whose footprint/topology differ from MLX-LM.

1. **Planner (`worker/plan.py`):** drive `_connect_to_group`, `_load_model`,
   `_ready_to_warmup` from `EngineCapabilities` resolved locally from the card,
   not from hardcoded `world_size`/family logic:
   - `supports_distributed=False` ⇒ never emit `ConnectToGroup`; go straight
     to `LoadModel` (generalizes the current `is_single_node_instance` path).
   - `independent_distributed_warmup` ⇒ replaces the Gemma4-specific
     `_uses_independent_distributed_warmup` special-case.
   - Keep the gossiped `Task`/`RunnerStatus` types unchanged (see §2.5);
     capability lookup is local.
2. **Memory model (`master/placement.py`, `placement_utils.py`,
   `shared/models/memory_estimate.py`, worker re-check at
   `worker/main.py:820`):** route the fit-check through
   `EngineCapabilities.estimate_footprint(...)` so master and worker share one
   engine-aware memory model (preserving the "checked twice, never disagree"
   invariant). The MLX-LM estimator is the current logic, unchanged; new
   engines supply their own.
3. Audit other MLX assumptions that placement/fit-check make (KV-cache
   quantization backends in `engines/mlx/cache.py`, context-admission ceiling
   `instance_context_token_limit` / `context_admission.py`) and gate any that
   are MLX-specific behind capabilities.

**Acceptance:** A single-process, non-grouped engine (use the transformers
embeddings engine as the test vehicle) is placed and served **without** the
planner emitting `ConnectToGroup` and **without** the fit-check assuming an MLX
footprint. Existing MLX placements are byte-for-byte unaffected (regression
suite + a multi-node placement test).

### Phase 5 — Land the first net-new engine (proves the seam)

**Goal:** Add MLX-vLLM (or the chosen second engine) as the first real consumer
of the abstraction, exercising same-task/different-engine selection.

- Implement `worker/engines/mlx_vllm/engine.py` against the protocol.
- Wire the deferred selection policy (§9) so a `TextGeneration` model can route
  to either `mlx_lm` or `mlx_vllm`.
- Validate two instances of different engines running concurrently on disjoint
  node subsets (the sub-cluster scenario).

**Acceptance:** Two text-generation models, one per engine, serve concurrently
on disjoint nodes; the dashboard shows both; neither destabilizes the other.

---

## 5. Test strategy

- **Per-phase regression:** `uv run pytest` (full, including the runner/engine
  trees) must stay green at every phase boundary. Phases 1–3 assert *no
  behavior change* via the Phase 0 characterization oracle.
- **Typing gate:** `uv run basedpyright` must be clean — the engine boundary is
  the whole point and must be exhaustively typed (`Protocol`, `Literal` for
  `engine_id` sets if/when enumerated, no `Any` leakage across the seam).
- **Capability matrix test:** a parametrized test that, for each registered
  engine, asserts the planner emits the correct lifecycle task sequence for
  single-node and (where `supports_distributed`) multi-node placements.
- **Memory-model parity test (Phase 4):** assert master fit-check and worker
  pre-spawn re-check agree for each engine (the "checked twice" invariant).
- **Multi-engine concurrency test (Phase 5):** two engines, disjoint fake node
  sets, interleaved task streams, assert isolation.

---

## 6. Pre-commit / CI obligations (from `CLAUDE.md`)

Every commit on this work must pass, in sequence:

```bash
uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
```

Stage any files `nix fmt` rewrites before committing. CI runs `nix flake check`
(formatting + lint + Rust tests).

---

## 7. Documentation obligations (mandatory, same PR as the code)

This is an **architectural shape change** (a new component family + a new
selection seam), so per `CLAUDE.md`'s mandatory workflow rules:

- Update `website/docs/architecture.md` (human narrative): add the
  engine/host/registry split and the "engine = swappable, pipeline = shared
  host" model.
- Update `website/docs/architecture-reference.md` (LLM fact-sheet): add the
  `InferenceEngine` protocol, `EngineCapabilities`, the registry, and the
  `worker/engines/*` layout.
- Update `CLAUDE.md`'s Architecture section (Rust/engine/runner description) and
  `CONTRIBUTING.md` if directory structure changes.
- If a new env var or `ModelCard` field is added for engine selection (§9), it
  must be documented in both architecture docs and `CHANGELOG.md` +
  `website/docs/release-notes/`.
- No API endpoints change in Phases 1–4. If `/v1/models` later exposes engine
  identity, update `website/docs/api-guide.md` and the OpenAPI decorators.

---

## 8. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Memory model divergence between engines breaks the "memory-safe placement, checked twice" invariant (mid-load Metal OOM is the worst failure mode on Apple Silicon). | **High** | Isolate to Phase 4; route both checks through one `estimate_footprint` hook; parity test; MLX estimator unchanged. |
| Changing gossiped lifecycle `Task`/`RunnerStatus` types breaks rolling upgrades. | High | Keep those types additive-only; drive new behavior from locally-resolved capabilities (§2.5). |
| Planner refactor (`plan.py`) subtly changes multi-node ring-formation/warmup ordering and wedges a real cluster. | High | Phase 0 lifecycle oracle + a multi-node placement test; refactor `plan.py` as a behavior-preserving generalization, not a rewrite. |
| Hidden MLX assumptions outside the runner (KV backend, context admission, vision processor) leak into the "generic" host. | Medium | Phase 4 audit; gate each behind a capability; non-MLX engines must reach Ready without importing `mlx`. |
| Scope creep — trying to add the new engine before the seam is proven. | Medium | Strict phase ordering; Phases 1–3 ship with only MLX engines registered. |
| `Builder`'s sequential-vs-batch selection logic is subtle (KV backend, Gemma4, MTP rank-symmetry, #254/#217). | Medium | Move it **verbatim** into the MLX-LM engine in Phase 1; do not "clean it up" while moving it. |

---

## 9. Open Decisions

### 9.1 Engine selection policy (DEFERRED — do not pre-empt)

The user has deferred *how* an engine is chosen per model. The three candidates:

1. **Explicit `engine: str | None` on `ModelCard`**, with a per-`ModelTask`
   default fallback. Most flexible; the only option that expresses "two models,
   same task, different engine." Rides existing card transport for free
   (§2.6). Cost: a new gossiped/persisted card field (additive, documented).
2. **Keep `ModelTask` as the only key.** Simplest; cannot express
   same-task/different-engine — defeats part of the stated goal. Effectively the
   current behavior.
3. **Registry heuristic** (engine inferred from card facts: family, quant,
   components) with no explicit field. No card change; least predictable.

**Implementation constraint regardless of choice:** all selection must flow
through `registry.resolve_engine(model_card)` (Phase 3). The default branch
reproduces today's `ModelTask` mapping, so whichever policy is chosen later is a
*localized* change to one function plus (for option 1) one additive card field.
**Do not** scatter selection logic into `bootstrap`, `plan.py`, or the host.

> Recommendation when the decision is revisited: **option 1** — it is the only
> one that satisfies the "different models, different engines" requirement, and
> §2.6 makes it nearly free to plumb.

### 9.2 Distributed topology ownership

Engines that own their own collective (e.g. vLLM) vs. engines that rely on
`mx.distributed` need a clear contract for who forms the topology. Phase 4's
`forms_mlx_group` capability is the minimal seam; revisit whether
`EngineTopology` should be richer (engine-owned process groups, port
allocation) when Phase 5's concrete second engine lands.

### 9.3 Result-type neutrality

Whether to fully unify `EngineResult` across text/embedding/image now, or keep
per-engine result unions translated by the host. Lean toward the latter
(minimal churn) until a fourth engine forces generalization.

---

## 10. Quick-start orientation for the implementing agent

Read these, in order, before writing code:

1. `src/skulk/worker/runner/bootstrap.py:385-509` — subprocess entry + current
   dispatch ladder (the thing Phase 3 replaces).
2. `src/skulk/worker/runner/llm_inference/runner.py` — the lifecycle state
   machine + `Builder` (Phases 1–2 source material).
3. `src/skulk/worker/runner/llm_inference/batch_generator.py:87-122` — the
   `InferenceGenerator` ABC (the serve-phase contract to generalize).
4. `src/skulk/worker/runner/runner_supervisor.py` — the parent-side host
   (already engine-agnostic; mostly untouched).
5. `src/skulk/worker/plan.py:160-300` — the distributed-aware lifecycle planner
   (Phase 4 source material).
6. `src/skulk/worker/runner/embeddings/runner.py` and
   `image_models/runner.py` — the two non-MLX engines that prove the seam is
   real (Phase 2 source material).
7. `src/skulk/shared/types/worker/shards.py` and
   `src/skulk/shared/models/model_cards.py` — where engine identity already
   travels (Key insight #3) and where a selector field would live (§9.1).
8. `src/skulk/master/placement.py`, `placement_utils.py`,
   `shared/models/memory_estimate.py` — the fit-check (Phase 4, highest risk).

**Golden rule:** Phases 1–3 must not change observable behavior. If a test
changes, you changed behavior — stop and reconcile against the Phase 0 oracle.
