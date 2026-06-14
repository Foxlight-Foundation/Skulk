---
id: architecture-reference
title: Architecture Reference
sidebar_position: 6
---

<!-- Copyright 2025 Foxlight Foundation -->

Dense per-symbol fact-sheet for AI assistants and operators who prefer reference style. For narrative and design rationale, see [Architecture](architecture). This document is intentionally terse — every entry has a file:line so you can jump to the code.

This file is intentionally dense. If you find a stale fact, fix it inline rather than working around it. The AGENTS.md "Documentation" section requires updates here when architectural shape changes.

## Components

### Master

- **Role:** elects + acts as cluster coordinator; indexes events; plans instance placements; publishes snapshots
- **Lives in:** `src/skulk/master/main.py`
- **Owns:** the authoritative event log (via `DiskEventLog`); the indexer that assigns monotonic indices to events; the placement planner. Master identity itself lives outside the master process — each node tracks the current master independently via the election protocol (`src/skulk/shared/election.py`); the `_master_node_id` cache is held on the API side at `src/skulk/api/main.py:461`.
- **Communicates via:** `LOCAL_EVENTS` (consumes), `GLOBAL_EVENTS` (publishes indexed events), `COMMANDS` (consumes), `STATE_SYNC_MESSAGES` (publishes snapshots)
- **Election:** `src/skulk/shared/election.py` — bully algorithm; a single master at a time
- **Failover:** re-election picks a new master, which seeds its session from the node's prior replicated state (#273, `seed_state_for_new_session` in `src/skulk/shared/session_carryover.py`): **instances, downloads, node info maps, and tracing carry over**; in-flight tasks, runner statuses, topology, and liveness timestamps are deliberately dropped (tasks died with the old session's plumbing; runner processes are torn down by the worker re-creation; topology/liveness must come from live gossip — a carried topology would keep a dead node's out-edges forever). Workers re-create runners for the carried instances through the ordinary plan loop, so placements survive a master restart with a model-reload-sized gap instead of a silent permanent 404. The plan loop suppresses liveness-based instance pruning for `TOPOLOGY_SETTLE_GRACE_SECONDS` (60s) after master start so carried instances aren't deleted while topology is still rebuilding; instances whose ranks lived on the dead master are pruned after the grace. A freshly-booted node that wins election seeds empty (it has no prior view) — identical to the pre-#273 behavior. The seed is indexed as **event 0 of the new session** (a logged `StateSnapshotHydrated`, `Master._index_seed_event`): late bootstrappers receive it inside the snapshot, early bootstrappers (including the promoted node's own worker, whose bootstrap races the promotion) receive it as the live first event — one delivery path, no idx-(-1) hydration skip.

### Worker

- **Role:** receives indexed events, applies them locally, downloads model weights, spawns + supervises runner subprocesses, dispatches tasks
- **Lives in:** `src/skulk/worker/main.py`; planning at `src/skulk/worker/plan.py`
- **Owns:** local view of `State` (derived); per-model `RunnerSupervisor` instances
- **Communicates via:** `GLOBAL_EVENTS` (consumes), `LOCAL_EVENTS` (publishes via `event_router.py`), `DOWNLOAD_COMMANDS` (publishes; e.g. shard-download requests at `worker/main.py:392`)

### RunnerSupervisor

- **Role:** parent-side lifecycle for one runner subprocess; signal handling; flight recorder buffer; SIGTERM/SIGKILL cleanup chain
- **Lives in:** `src/skulk/worker/runner/runner_supervisor.py`
- **Spawns:** `mp.Process(target=entrypoint, daemon=True)` with the runner subtype's main loop
- **Cleanup chain:** `join(5s)` → `terminate()` (SIGTERM) → `join(5s)` → `kill()` (SIGKILL); plus parent-pid watchdog inside the subprocess for reparenting (SIGKILL of agent)

### Runner subprocess

- **Role:** owns one MLX model; serves inference tasks for it; participates in distributed collectives with peer runners across ranks
- **Entrypoint:** `src/skulk/worker/runner/bootstrap.py::entrypoint`
- **Subtypes:**
  - `src/skulk/worker/runner/llm_inference/runner.py` — text generation
  - `src/skulk/worker/runner/embeddings/runner.py` — embeddings
  - `src/skulk/worker/runner/image_models/runner.py` — image generation
- **Communicates via:** `mp.Queue` from worker (incoming tasks); `mp.Queue` to worker (outgoing events); `mlx.distributed` collectives with peer runners

### Drafters (speculative decoding)

`src/skulk/worker/engines/mlx/drafters/`. The loop runs on single-node, tensor-parallel, AND pipeline placements. Multi-node PIPELINE placements use an EXPLICIT decider protocol (#254): exactly one rank — the decider, the last rank — holds the drafter, makes every speculative decision, and fans the outcomes out via fixed-shape per-round collectives: one `all_sum` lands the draft tokens (`_exchange_drafts`; the drafter's effective distribution rides along under sampling), and after the verify forward a second tiny `all_sum` lands the accept length and the next bonus token (`mtp_accept_decision`). The first sampled token of the request is broadcast the same way. Receiving ranks never draft, sample, or compare logits — they apply broadcast decisions to their own cache slices — so correctness never depends on cross-rank numerical determinism (heterogeneous chips, e.g. M5 vs M4 GEMM kernels and NAX reduced-precision B≥2 matmuls, produce divergent per-rank logits; the previous rank-symmetric design desynced on exactly that and SIGABRT'd in the Metal completion block, #252). A per-request `all_sum` agreement settles that exactly one rank holds a working drafter (speculation disables symmetrically otherwise); mid-request drafter failures abort loudly on multi-rank placements instead of silently forking the collective schedule. Assistant drafters (gemma4) cross-attend the target's KV, which the decider seat owns by construction (#201 Track 2b); sidecar drafters draft from the all-gathered trunk hidden from the same seat, and only the decider rank loads drafter weights. Multi-node TENSOR placements do NOT use the decider protocol (#263): draft logits go through the TP-sharded lm_head, an all-rank collective idle receivers would never join, so a lone TP decider GPU-times-out mid-draft. Instead every TP rank loads the sidecar (`sidecar_load_eligible` in `src/skulk/worker/engines/mlx/utils_mlx.py` — same envelope assistants use) and drafts rank-symmetrically; the drafter agreement requires ready_count == group.size() on that path and disables speculation symmetrically on partial loads. Cross-attending drafters declare `reads_target_cache = True` and the loop keeps the target cache fully committed before every draft (no deferred replay) — and they must hold the LIVE cache sequence, since reject-restores replace rotating entries in the loop's list. Forces `SequentialGenerator`. Greedy requests use argmax-prefix acceptance; temperature > 0 uses Leviathan-Chen probability-ratio acceptance over the effective sampler distributions (`src/skulk/worker/engines/mlx/generator/speculative_sampling.py`, depth forced to 1). Draft depth comes from the card's `mtp_max_depth` (default 1). Rounds are *bonus-driven*: the loop carries an emitted-but-unforwarded bonus token, verifies `[bonus, drafts]` in one K+1-token forward (the round's only target forward), commits the longest matching prefix, samples the next bonus from the first non-matching row, and drafts the very next round from the correction position.

- **Protocol:** `protocol.py::Drafter` — `begin_request(prompt_cache)` / `observe(hiddens, next_tokens)` / `draft(hidden, next_token, depth=1) -> (K, vocab) logits`. The generation loop owns verify/accept/reject and cache reconciliation — preferring the model's native `rollback_speculative_cache` (gemma4), else SSM snapshot/restore with *deferred replay* (restored-but-committed tokens ride at the front of the next verify forward; capped, flushed at stream end), else plain KV trim. Drafters own only their private state. The loop feeds every committed position's `(hidden, next token)` pair exactly once, in order (the pair-stream contract); the hidden convention is per-family (pre-final-norm for qwen-shaped trunks, post-norm for gemma4).
- **Builder:** `builder.py::build_drafter(model, mtp_weights, runtime)` — detects sidecar key layout, resolves family facts (norm convention, fc concat order) from layout-keyed defaults with model-card `runtime` overrides, and quantizes the sidecar block + fc to the target's `(group_size, bits)` on load (bf16 targets keep bf16 sidecars).
- **Implementations:**
  - `qwen_sidecar.py::QwenSidecarDrafter` — Phase 2: +1.0 zero-centered norm shift, `embed_first` concat, sidecar `mtp.layers.0` block instantiated from the target family's own decoder-layer class (strict-loaded), private `KVCache`. Validated 79–85% acceptance / 1.38–1.90x on Qwen3.5 9B–27B (issue #192, bonus-driven cadence).
  - `deepseek_sidecar.py::DeepseekSidecarDrafter` — legacy projection-only head; conventions unverified against real weights.
  - `gemma4_assistant.py::Gemma4AssistantDrafter` — wraps mlx-vlm's chain-trained assistant model: cross-attends over the target's KV (shared-KV extraction incl. RotatingKVCache temporal restore), consumes post-norm hiddens, loads via `assistant_model_repo` (bf16-enforced). Validated 84% acceptance / 35.1 tok/s on gemma-4-26B-A4B-4bit (depth 1) and 1.86x on E4B-8bit (depth 3).
- **Observability:** the loop logs `MTP acceptance so far: A/N` every 32 drafts; the public `GenerationResponse` does not carry per-token draft provenance.

### Router (libp2p)

- **Role:** transport for all inter-node communication
- **Lives in:** `src/skulk/routing/` (Python wrapper); `rust/networking/` + `rust/skulk_pyo3_bindings/` (Rust libp2p impl + PyO3 bindings)
- **Topics:** see "Pubsub topics" below
- **Discovery:** mDNS by default; `--bootstrap-peers` multiaddrs for explicit static peers

### Election

- **Role:** picks the cluster master via the bully algorithm
- **Lives in:** `src/skulk/shared/election.py`
- **Communicates via:** `ELECTION_MESSAGES` topic
- **Triggers:** node startup, lost master heartbeat, explicit master abdication

### API

- **Role:** HTTP entry point; FastAPI app; OpenAI / Ollama / Claude / Responses / Skulk-native adapters; serves dashboard
- **Lives in:** `src/skulk/api/main.py`; adapters at `src/skulk/api/adapters/`
- **Default port:** 52415
- **Mounts:** dashboard at `/`; OpenAPI at `/api/openapi.json`
- **Background tasks:** `_apply_state` (consumes `GLOBAL_EVENTS` and persists merged traces), `_pause_on_new_election`, `_cleanup_expired_images` (image-store TTL), `_prune_old_traces` (hourly trace janitor backed by `prune_old_trace_files`; retention via `tracing.retention_days`)

### Dashboard

- **Role:** operator UI for the same Skulk runtime
- **Lives in:** `dashboard-react/` (source); served by API at `/`
- **Stack:** React + TypeScript + styled-components + Vite
- **State:** Redux Toolkit + RTK Query (`dashboard-react/src/store/`). Slices at `store/slices/uiSlice.ts` and `store/slices/chatSlice.ts`; query endpoints injected from `store/endpoints/cluster.ts`, `store/endpoints/config.ts`, `store/endpoints/observability.ts` into a single `apiSlice` (`store/api.ts`).
- **Routing:** activity-style enum (`activeRoute` in `uiSlice`); no react-router
- **Persistence:** sessionStorage for in-session UI; localStorage for cross-session preferences (theme, panel widths)

### Storage

- **Event log:** `src/skulk/utils/disk_event_log.py` — append-only length-prefixed msgpack records (`events.bin`, uncompressed live); rotated archives are zstd-compressed (`events.*.bin.zst`) on rotation/close. Disk is treated as bounded: archives are capped by count (5) AND total bytes (1 GiB); any persistence failure (ENOSPC at init, append, or compaction) drops the log into a degraded counting-only mode with one CRITICAL line — indices keep advancing so follower replay coherence survives — and a proactive free-space floor (2 GiB, checked every 1024 appends) degrades BEFORE the disk hits zero. The API-side log (`event_log/api/`, backs `GET /events` diagnostics only and records per-token chunk events) additionally ring-compacts: past 256 MiB of active file it keeps only the most recent 20k events.
- **Model cache:** `SKULK_MODELS_DIR` (default `SKULK_DATA_HOME/models`; on Linux that's `~/.local/share/skulk/models` via XDG, on macOS/Windows it's `~/.skulk/models`); `SKULK_HOME` and `SKULK_MODELS_DIR` env overrides apply
- **Custom cards:** `SKULK_CUSTOM_MODEL_CARDS_DIR` (default `SKULK_DATA_HOME/custom_model_cards`) as TOML
- **Built-in cards:** `resources/inference_model_cards/` as TOML
- **Optional model store:** shared host with rsync-style staging — `src/skulk/store/`

## Pubsub topics

Defined in `src/skulk/routing/topics.py`.

| Topic | Wire payload type | Inner payload | Publisher | Consumer |
|---|---|---|---|---|
| `GLOBAL_EVENTS` | `GlobalForwarderEvent` | indexed `Event` (post-master indexing) | Master | All nodes |
| `LOCAL_EVENTS` | `LocalForwarderEvent` | un-indexed `Event` | Workers (via `event_router.py`) | Master |
| `COMMANDS` | `ForwarderCommand` | `Command` (`PlaceInstance`, `DeleteInstance`, `TaskFinished`, `SetTracingEnabled`, etc.) | API | Master (command processor); Election (every node — observes commands to inform leader-changeover decisions) |
| `DOWNLOAD_COMMANDS` | `ForwarderDownloadCommand` | `DownloadCommand` (`StartDownload`, `DeleteDownload`, `CancelDownload`, `SyncConfig`, `PurgeStagingCache`, `RestartNode`) | API (download/restart/sync admin ops), Master, Workers | All nodes |
| `STATE_SYNC_MESSAGES` | `StateSyncMessage` | bidirectional: followers publish `kind="request"` for snapshot/config bootstrap; master publishes `kind="response"` with the requested payload (`StateSnapshotHydrated` etc.) | All nodes (request: followers; response: master) | All nodes |
| `ELECTION_MESSAGES` | `ElectionMessage` | bully election rounds | All nodes | All nodes |
| `CONNECTION_MESSAGES` | libp2p connection updates | peer arrivals / departures | Router | All nodes |
| `TELEMETRY` | `NodeTelemetry` | `GatheredInfo` (`NodeResources` — participation role + backends; `MemoryUsage`/`MactopMetrics`+`MacmonMetrics` — per-node memory + system profile) | Workers | All nodes (applied into `TelemetryView`) |

### Telemetry plane (#279)

`TELEMETRY` is the first slice of the control/telemetry/data plane separation (#279). Node readings that are **last-write-wins and not decisions** are gossiped on this topic instead of being event-sourced into `State`. They land in an in-memory `TelemetryView` (`src/skulk/shared/types/telemetry.py`), held per-`Node` and read by the planner (placement eligibility) and the API placement previews — they are **not** persisted in the event log or carried in snapshots.

Slice 1 moved `node_resources` (a node's `participation` role and `backends`); slice 2 moved `node_memory` and `node_system` (the highest-volume readings, carried together by `MactopMetrics`/`MacmonMetrics`). The worker forwards these `GatheredInfo` variants to the telemetry sender (`worker/main.py` `_TELEMETRY_PLANE_INFO`), and `apply_node_gathered_info` treats them as no-ops (`shared/apply.py`). The `TelemetryView` survives master re-election (Node-owned), so a freshly promoted master does not start blind. `GET /state` merges `node_memory`/`node_system` back in from the view (`API.get_cluster_state`) so the dashboard's wire shape is unchanged.

**Placement reads two views.** The memory-fit check and the context-admission ceiling read `node_memory` from the `TelemetryView`, not `State`. Because the ceiling must be identical across ranks (divergent verdicts deadlock the collectives) and telemetry is unordered last-write-wins, the master computes the ceiling **once at placement time** and stamps it onto the instance (`BaseInstance.context_token_limit`, event-sourced); every worker rank, and the API's admission pre-flight, then read that stamped value instead of recomputing it.

**Rolling-upgrade note:** an un-upgraded worker still emits these readings as `NodeGatheredInfo` events (which `apply` no-ops). The worker event applier **bridges** any telemetry-plane `NodeGatheredInfo` into the shared `TelemetryView` (`worker/main.py`), so the upgraded master/API can place on those nodes during the mixed-version window instead of seeing "memory not gathered"; the bridged entry is pruned when the legacy worker restarts under a new id and the old id times out. `State` also keeps `extra="forbid"`, so a pre-#279 snapshot carrying the old `nodeResources`/`nodeMemory`/`nodeSystem` keys is stripped by a before-validator on hydrate rather than rejected.

## Events

Discriminated union at `src/skulk/shared/types/events.py`. Selected events:

| Event | Emitted when | Applied by |
|---|---|---|
| `InstanceCreated` | Master places a model | All nodes (update `State.instances`) |
| `InstanceDeleted` | Master deletes a placement | All nodes |
| `RunnerStatusUpdated` | Runner subprocess transitions state | All nodes |
| `RunnerFailed` | Runner crashes or exits unexpectedly | All nodes |
| `TaskAcknowledged` | Worker accepts a task | All nodes |
| `TaskStatusUpdated` | Task transitions state (`Running`, `Failed`, `Cancelled`, `Complete`, `TimedOut` — the last emitted by the worker on shutdown timeouts, `worker/main.py:474`). The `Complete` variant is emitted by the runner / worker on natural finish (e.g. `worker/main.py:362,388,450`, runner `send_task_status(..., TaskStatus.Complete)`). The `TaskFinished` command sent by API on stream end triggers `TaskDeleted` only (`master/main.py:444-450`), not this event. The `Cancelled` variant (operator instance deletion via `get_transition_events`) additionally makes the API terminate that command's open stream with an error chunk. | All nodes |
| `TaskFailed` | Master plan loop fails in-flight API tasks (TextGeneration / ImageGeneration / ImageEdits / TextEmbedding) whose instance is gone or dying — `orphaned_task_failure_events` in `master/main.py`, emitted BEFORE `InstanceDeleted`/`NodeTimedOut` so it indexes ahead of the applies that delete the task. `apply_task_failed` sets `task_status=Failed` (terminal — makes re-emission idempotent) plus error fields. The API reacts by delivering a terminal `ErrorChunk` into the command's stream (`_terminate_command_stream`): streaming closes with an error event, non-streaming returns 500. On master failover the new session cannot carry old tasks, so the API's session `reset()` fails all open command streams directly instead (`_fail_open_command_streams_for_session_reset`). | All nodes |
| `TaskDeleted` | Task is purged from cluster state | All nodes |
| `ChunkGenerated` | Runner emits an output chunk (token, tool call, error) | API queue subscribers |
| `TracesCollected` | Runner emits trace events for one rank | Master (merges across ranks) |
| `TracesMerged` | Master merges + persists a complete trace | API (writes to disk) |
| `TracingStateChanged` | Cluster tracing toggle changes | All nodes |

Apply function: `src/skulk/shared/apply.py::apply` — pure `(State, IndexedEvent) -> State`.

## Commands

Two distinct command unions on two distinct topics:

### COMMANDS topic — `Command` union

Discriminated union at `src/skulk/shared/types/commands.py`. Carried as `ForwarderCommand` over the `COMMANDS` pubsub topic.

| Command | What it requests | Master action |
|---|---|---|
| `PlaceInstance` | Spin up a model on the cluster. Optional `excluded_nodes: list[NodeId]` — planner treats those nodes as if absent for *this placement only*; already-running instances on them are not affected. | Pick ranks based on memory + topology (filtered by `excluded_nodes`); emit `InstanceCreated`. Memory admission is per-node (Tensor = even split, Pipeline = proportional to available): a node's weight share x `MEMORY_OVERHEAD_FACTOR` (1.30) + an explicit KV-cache reservation for `KV_CONTEXT_BUDGET_TOKENS` (8192) + `MEMORY_OVERHEAD_FLOOR` (256 MB), each node capped at `GPU_WORKING_SET_FRACTION` (0.75) of `ram_total` (the Metal GPU working-set ceiling, since gossiped `ram_available` can exceed what the GPU may wire). On macOS the gossiped `ram_available` is itself the GPU-wireable figure `total − wired − anonymous − compressor` (vm_stat snapshot per telemetry sample; see `MemoryUsage` below) rather than the naive free-plus-inactive figure that counted reclaimable file cache as used. Estimation lives in `skulk.shared.models.memory_estimate`, shared with the worker's local pre-spawn OOM guard so the two checks never disagree. Failures raise typed `PlacementError`s, with `PlacementInfoPendingError` for the cluster-startup windows where cluster info has not finished gossiping (connection edges lag identities; memory info lags the edges). The API dry-runs placement before forwarding (400 on impossible, 503 after a 15s wait on pending info). |
| `DeleteInstance` | Tear down a placed model | Emit `InstanceDeleted`; workers tear down runners |
| `TaskFinished` | Mark a streaming task complete (sent by API on stream end) | Emit `TaskDeleted` (`TaskStatusUpdated(Complete)` is emitted earlier on the chunk path, not from `TaskFinished` directly) |
| `TaskCancelled` | Cancel an in-flight command (sent by API on `/v1/cancel`) | Emit `TaskStatusUpdated(Cancelled)` |
| `SetTracingEnabled` | Cluster-wide tracing toggle | Emit `TracingStateChanged` |
| `AddCustomModelCard` | User-added model card | Emit `CustomModelCardAdded`; nodes persist locally |
| `DeleteCustomModelCard` | Remove user card | Emit `CustomModelCardDeleted` |

### DOWNLOAD_COMMANDS topic — `DownloadCommand` union

Discriminated union at `src/skulk/shared/types/commands.py`. Carried as `ForwarderDownloadCommand` over the `DOWNLOAD_COMMANDS` pubsub topic. Used for cluster-wide config sync and model-store coordination — separated from the main command channel because these are typically larger payloads and have different retry semantics.

| Command | What it requests |
|---|---|
| `SyncConfig` | Broadcast cluster config (`auth.api_keys` and `hf_token` stripped); followers observe and persist locally |
| Model store ops | Download / staging coordination commands (see `src/skulk/store/`) |

### Tasks (not commands)

Note `CancelTask` is a **task** (`src/skulk/shared/types/tasks.py`), not a command. Tasks are work units the runner executes; commands are imperative requests to the master. Cooperative task cancellation is implemented as a `CancelTask` task delivered to the runner over the `mp.Queue`.

## API endpoints

Lives in `src/skulk/api/main.py` (route registration in `API.__init__`).

### Inference

| Endpoint | Method | What |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI Chat Completions; SSE when `stream=true` |
| `/v1/responses` | POST | OpenAI Responses |
| `/v1/messages` | POST | Anthropic Messages |
| `/v1/embeddings` | POST | OpenAI Embeddings |
| `/v1/images/generations` | POST | OpenAI Images Generation |
| `/v1/images/edits` | POST | OpenAI Images Edits |
| `/ollama/api/chat` | POST | Ollama chat |
| `/ollama/api/generate` | POST | Ollama generate |
| `/v1/cancel/{command_id}` | POST | Cancel an in-flight task |

### Models / placement

| Endpoint | Method | What |
|---|---|---|
| `/models`, `/v1/models` | GET | List available models |
| `/models/search` | GET | Search Hugging Face |
| `/models/add` | POST | Register a custom model card |
| `/models/custom/{model_id}` | DELETE | Remove a custom card |
| `/instance` | POST | Place an instance |
| `/place_instance` | POST | Place a model: master picks ranks. Takes `PlaceInstanceParams` (model id + placement preferences, optional `excluded_nodes: list[NodeId]` to exclude specific nodes from this placement); not interchangeable with `/instance`, which takes a fully-specified `CreateInstanceParams`. |
| `/instance/{instance_id}` | GET / DELETE | Fetch / delete an instance |
| `/instance/placement` | GET | Compute placement preview |
| `/store/storage` | GET | Local node's storage breakdown: staged models (size, last-use, in-use incl. companions), event-log bytes, disk free. Staging eviction: `cleanup_on_deactivate` default true; not-in-use staged models kept newest-first up to `staging_keep_recent_gb` (40 GiB default), enforced at deactivate AND node startup (crash-orphan reconciliation); `src/skulk/store/staging_eviction.py`. Companion repos (MTP sidecar / assistant / split vision weights) resolve through `companion_download_specs()` (`src/skulk/download/download_utils.py`) on every resolution path — required companions (vision) fail the load loudly, best-effort companions (sidecar/assistant) log and degrade to plain decode. |
| `/instance/previews` | GET | List candidate placements |

### State / events

| Endpoint | Method | What |
|---|---|---|
| `/state` | GET | Cluster state snapshot |
| `/events` | GET | Stream stored events (debug) |
| `/node_id` | GET | Local node identity |
| `/config` | GET / PUT | Cluster config (sanitized) |

### Tracing

| Endpoint | Method | What |
|---|---|---|
| `/v1/tracing` | GET / PUT | Cluster tracing on/off |
| `/v1/traces` | GET | List local traces |
| `/v1/traces/cluster` | GET | List traces from all reachable peers |
| `/v1/traces/{task_id}` | GET | Get one local trace |
| `/v1/traces/{task_id}/stats` | GET | Aggregated timing stats |
| `/v1/traces/{task_id}/raw` | GET | Raw Chrome-trace JSON |
| `/v1/traces/cluster/{task_id}` | GET | One trace, proxied if remote |
| `/v1/traces/cluster/{task_id}/stats` | GET | Stats for a cluster trace |
| `/v1/traces/cluster/{task_id}/raw` | GET | Raw JSON for a cluster trace |
| `/v1/traces/delete` | POST | Delete saved local traces |

### Diagnostics

| Endpoint | Method | What |
|---|---|---|
| `/v1/diagnostics/node` | GET | Local node diagnostics bundle |
| `/v1/diagnostics/node/capture` | POST | On-demand local capture (sample, vmmap, footprint) |
| `/v1/diagnostics/node/runners/{runner_id}/cancel` | POST | Cooperative runner-task cancel |
| `/v1/diagnostics/cluster` | GET | Fan-out: every reachable node's diagnostics |
| `/v1/diagnostics/cluster/timeline` | GET | Cross-rank merged flight recorder |
| `/v1/diagnostics/cluster/{node_id}` | GET | One peer's diagnostics |
| `/v1/diagnostics/cluster/{node_id}/capture` | POST | Capture proxied to peer |
| `/v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel` | POST | Peer runner cancel |

### Tools / store / store / admin

| Endpoint | Method | What |
|---|---|---|
| `/v1/tools/web_search` | POST | Built-in tool: web search |
| `/v1/tools/open_url` | POST | Built-in tool: fetch URL |
| `/v1/tools/extract_page` | POST | Built-in tool: extract page text |
| `/store/health` | GET | Model store health |
| `/store/registry` | GET | Model store registry |
| `/store/models/{model_id}/download` | POST | Request store download |
| `/store/models/{model_id}` | DELETE | Delete store model |
| `/admin/restart` | POST | Request node restart. Optional `node_id` query param targets a specific peer; without it, restarts the local node. |

### Bench

| Endpoint | Method | What |
|---|---|---|
| `/bench/chat/completions` | POST | Bench chat completions (separate code path for benchmarking) |
| `/bench/images/generations` | POST | Bench image generation |
| `/bench/images/edits` | POST | Bench image edits |

## Pydantic models

### Tasks

`src/skulk/shared/types/tasks.py`. Discriminated union of:

- `TextGeneration` — chat / responses / messages / ollama-chat
- `TextEmbedding` — embeddings
- `ImageGeneration` — images.generations
- `ImageEdits` — images.edits
- Sentinel: `Shutdown`, `CANCEL_ALL_TASKS`

### Chunks

`src/skulk/shared/types/chunks.py`. Per-token output:

- `TokenChunk` — text / tool / token-level metadata
- `ToolCallChunk` — tool calls
- `ErrorChunk` — error result; terminal
- `PrefillProgressChunk` — distributed prefill progress
- `ImageChunk` — image generation output
- `EmbeddingChunk` — embedding output

### State

`src/skulk/shared/types/state.py`. Treated as immutable by convention (replaced wholesale by `apply()` rather than mutated in place); the model itself is not declared `frozen=True` on `model_config`, so direct mutation is technically possible but considered a bug at every call site.

- `instances: Mapping[InstanceId, Instance]` — placed model instances (each carries shard assignments + per-runner state)
- `runners: Mapping[RunnerId, RunnerStatus]` — per-runner status union
- `downloads: Mapping[NodeId, Sequence[DownloadProgress]]` — in-flight model downloads per node
- `tasks: Mapping[TaskId, Task]` — in-flight or recently-completed tasks
- `last_seen: Mapping[NodeId, datetime]` — peer liveness timestamps
- `topology: Topology` — cluster-wide node graph + capabilities (encoded/decoded via `TopologySnapshot` for JSON round-tripping)
- `tracing_enabled: bool` — cluster-wide tracing flag
- `last_event_applied_idx: int` — water mark for the local apply
- `node_identities`, `node_disk`, `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge`, `node_rdma_ctl: Mapping[NodeId, *]` — granular per-node info that updates at independent frequencies on the event path
- `node_resources` (slice 1), `node_memory` + `node_system` (slice 2) are **not** `State` fields — they moved to the telemetry plane (`TelemetryView`, gossiped on `TELEMETRY`, see "Telemetry plane" above) as part of #279. `State` keeps `extra="forbid"`, so a pre-#279 snapshot carrying the old `nodeResources`/`nodeMemory`/`nodeSystem` keys is stripped by a before-validator on hydrate rather than rejected (rolling-upgrade compatibility).
- `thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]]` — detected Thunderbolt-bridge cycles where every node has it enabled (>2 nodes)

Note: there is no `master_node_id` field on `State`. Master identity lives outside the event-sourced state — each node tracks the current master independently via the election protocol (`src/skulk/shared/election.py`). `placements` is also not a field; placement information is derived from `instances` (each `Instance` has its own shard assignments).

### Diagnostics

`src/skulk/shared/types/diagnostics.py`. Major models:

- `NodeDiagnostics` — runtime + identity + resources + processes + supervisor_runners + placements + warnings. `warnings` includes a **leaked-wired-memory** alert (`_leaked_wired_warning` in `src/skulk/api/main.py`): emitted when `resources.current_wired` exceeds ~5GB with zero `process_alive` runners — the signature of wired memory leaked by an abnormal Metal termination that only a reboot reclaims (#239). Server-side counterpart of `tests/preflight_mem.sh`. To stop a doomed runner from compounding such a leak, the worker circuit-breaks runner crash loops (`CrashWindow` in `src/skulk/utils/crash_window.py`, 3 failures within 60s) and deletes the instance via `DeleteInstance` instead of relaunching it.
- `NodeResourceDiagnostics` — gathered_memory, current_memory, **current_wired** (OS-level wired in use; macOS-only via `read_wired_memory_bytes`/psutil — MLX's own accounting can't see leaked wired), disk, system, network. `current_wired` is read locally on the diagnostics path and deliberately kept OFF the gossiped `MemoryUsage` so the `NodeGatheredInfo` event wire format is unchanged across a mixed-version rollout.
- `MemoryUsage` — ram_total, ram_available, swap_total, swap_available. On macOS, `ram_available` is the GPU-wireable figure `total − wired − anonymous − compressor` from a `vm_stat` snapshot taken per telemetry sample (`MachMemoryCategories` / `parse_vm_stat_output` in `src/skulk/shared/types/profiling.py`), falling back to mactop's raw `available` (free+inactive+speculative, which counts reclaimable file cache as used) when `vm_stat` fails. Value-only change — the gossiped shape is unchanged, so mixed-version clusters interoperate.
- `RunnerSupervisorDiagnostics` — flight_recorder, status, phase, MLX memory, in_progress_tasks, milestones
- `RunnerFlightRecorderEntry` — at, phase, event, detail, attrs, context, mlxMemory
- `MlxMemorySnapshot` — active, cache, peak, wired_limit (MLX's configured limit, not OS wired usage)
- `ClusterDiagnostics` — fan-out wrapper
- `ClusterTimeline` — cross-rank merged: runners (synopsis) + timeline (entries sorted by `at`) + unreachableNodes
- `DiagnosticCaptureResponse` — capture bundle (process samples, flight recorder, MLX memory)

### Model card

`src/skulk/shared/models/model_cards.py::ModelCard`. Fields:

- `model_id`, `family`, `quantization`, `base_model`, `n_layers`, `hidden_size`, `num_key_value_heads`
- `tasks: list[ModelTask]` — what task types this model serves
- `capabilities: list[str]` — text / vision / thinking / thinking_toggle / embedding
- `context_length`, `storage_size`, `supports_tensor`, `trust_remote_code`, `is_custom`
- `vision: VisionCardConfig | None` — image_token_id, model_type, BOI/EOI tokens
- `reasoning: ReasoningCardConfig | None` — supports_toggle, supports_budget, format, default_effort
- `modalities: ModalitiesCardConfig | None` — supports_native_multimodal, supports_audio_input
- `tooling: ToolingCardConfig | None` — tool_call_format, supports_tool_calling, builtin_tools
- `runtime: RuntimeCapabilityCardConfig | None` — prompt_renderer, output_parser, metal_fast_synch, mtp_heads, mtp_max_depth, mtp_sidecar_repo, mtp_norm_convention, mtp_concat_order, assistant_model_repo, speculative_multi_node (set `false` where multi-node speculation measures slower than plain sharded decode — e.g. gemma-4-26B-A4B MoE, 2026-06-06 matrix: 30.2 plain vs 28.2 MTP on 2 nodes; single-node speculation unaffected; card-driven so the agreement collective stays rank-symmetric)

### Capability profile

`src/skulk/shared/models/capabilities.py::ResolvedCapabilityProfile`. Computed at request time from card + tokenizer + task params:

- `family` (string)
- `supports_thinking`, `supports_thinking_toggle`, `supports_thinking_budget`
- `supports_image_input`, `supports_audio_input`, `supports_native_multimodal`
- `supports_tool_calling`
- `thinking_format: ReasoningFormat` — None_ / TokenDelimited / ChannelDelimited
- `default_reasoning_effort`, `disabled_reasoning_effort`
- `prompt_renderer: PromptRendererType` — Tokenizer / Gemma4 / Dsml
- `output_parser: OutputParserType` — Generic / Gemma4 / GptOss / DeepseekV32
- `tool_call_format: ToolCallFormat` — Generic / Gemma4 / GptOss / Dsml
- `builtin_tools: tuple[BuiltinToolType, ...]`

## Pipeline-parallel sharding strategies

Family-specific in `src/skulk/worker/engines/mlx/auto_parallel.py`. Each is a class implementing `TensorParallelShardingStrategy`. Dispatched at lines 830-905 via `isinstance(model, X)` chain (consolidation tracked under #130):

| Strategy | Applies to | Lines |
|---|---|---|
| `LlamaShardingStrategy` | Llama, Ministral3 | 939+ |
| `DeepSeekShardingStrategy` | DeepseekV3, DeepseekV32, KimiK25 | 995+ |
| `GLM4MoeLiteShardingStrategy` | Glm4MoeLite | 1080+ |
| `MiniMaxShardingStrategy` | MiniMax | 1226+ |
| `QwenShardingStrategy` | Qwen3Moe, Qwen3Next, Qwen3_5Text, Qwen3_5Moe | 1267+ |
| `Glm4MoeShardingStrategy` | Glm4Moe | 1428+ |
| `GptOssShardingStrategy` | GptOss | 1476+ |
| `Step35ShardingStrategy` | Step35 | 1519+ |
| `NemotronHShardingStrategy` | NemotronH | 1564+ |

## Family-specific code locations

Inventory snapshot — see #130 for consolidation plan.

| Family | Total lines | Primary locations |
|---|---|---|
| Gemma 4 | ~600 | `gemma4_prompt.py`, vision tower wrapping in `utils_mlx.py:333-456`, `parse_gemma4_thinking_channels` in `model_output_parsers.py`, native-vision branches in `generate.py:1337-1900` |
| Qwen (5 variants) | ~350 | `QwenShardingStrategy` in `auto_parallel.py:1267-1567` |
| DeepSeek V3.2 | ~350 | `dsml_encoding.py`, `parse_deepseek_v32` in `model_output_parsers.py:374-516` |
| GLM-4 (Lite + MoE) | ~280 | Two strategies in `auto_parallel.py` |
| MiniMax | ~225 | `MiniMaxShardingStrategy` + custom attention wrapper in `auto_parallel.py:1148-1226` |
| NemotronH | ~210 | `NemotronHShardingStrategy` + Mamba2 hybrid cache |
| GPT-OSS | ~180 | `parse_gpt_oss` (Harmony parser) + `GptOssShardingStrategy` |
| Step 3.5 | ~95 | Sliding-window cache tracking in `auto_parallel.py:639-650` |
| Llama / Ministral | ~70 | `LlamaShardingStrategy` (default) |

## KV cache backends

Selectable per-cluster via `inference.kv_cache_backend` config or `SKULK_KV_CACHE_BACKEND` env:

| Backend | What | Trade-off |
|---|---|---|
| `default` | Standard MLX, fp16 | Highest memory; baseline |
| `mlx_quantized` | Upstream MLX quantized | Lower memory, decode overhead |
| `turboquant` | Random orthogonal rotation + scalar quant | Storage savings, no decode perf benefit |
| `turboquant_adaptive` | TurboQuant with FP16 edges | Slightly better quality |
| `optiq` | Rotated-space attention trick | Decode-time perf benefit; falls back to default for incompatible head dims |

RotorQuant (block rotations + deferred quant) is research and lives in PR #103; it is not yet in the merged backend set. Verify the current valid values against `src/skulk/worker/engines/mlx/constants.py`.

Selection logic: `src/skulk/worker/engines/mlx/cache.py::make_kv_cache`. Some backends fall back to `default` for incompatible models (e.g., `optiq` for non-divisible head_dim).

## Configuration knobs

### `skulk.yaml`

| Section | Field | What |
|---|---|---|
| `model_store` | `enabled`, `host`, `port`, `path` | Shared model store config |
| `model_store.staging` | `enabled`, `node_cache_path`, `cleanup_on_deactivate` | Staging behavior |
| `inference` | `kv_cache_backend` | KV cache selection |
| `logging` | `enabled`, `ingest_url` | Centralized logging opt-in |
| `tracing` | `retention_days` | Saved-trace retention for the API janitor (default 3 days; 0 disables pruning) |
| `hf_token` | (string) | Local-only Hugging Face token (stripped from cluster broadcast) |

### Environment variables

`SKULK_*` is preferred; `SKULK_*` accepted as legacy. Migration tracked under #110.

| Var | What |
|---|---|
| `SKULK_HOME` / `SKULK_HOME` | Override the base data directory used to derive `SKULK_DATA_HOME` (and from there `SKULK_MODELS_DIR`, `SKULK_CUSTOM_MODEL_CARDS_DIR`, `SKULK_EVENT_LOG_DIR`). Default base: XDG-derived `~/.local/share/skulk` on Linux; `~/.skulk` on non-Linux. See `src/skulk/shared/constants.py:34-149`. |
| `SKULK_FAST_SYNCH` / `SKULK_FAST_SYNCH` | Force `MLX_METAL_FAST_SYNCH` on (`"on"`) or off (`"off"`); overrides per-model card. Resolution order: operator override → card `metal_fast_synch` pin → OFF for speculative-decoding cards (`mtp_heads` / `mtp_sidecar_repo` / `assistant_model_repo`; FAST_SYNCH collapses the MTP loop ~46x, measured 2026-06-06) → cluster default (OFF since #261) |
| `SKULK_PIPELINE_EVAL_TIMEOUT_SECONDS` | Per-eval timeout in pipeline collectives (default 60s) |
| `SKULK_GROUP_CONNECT_DEADLINE_SECONDS` | Hard deadline for distributed group formation (`mx.distributed.init`, default 120s). Ring init with `strict=True` blocks forever when a neighbor socket fails the post-TCP rank handshake (#265); on expiry the runner exits via the wedge path, the worker gives the instance up on first failure (#260), and a fresh placement mints a new ring port (also clearing stale-socket handshake collisions) |
| `SKULK_WARMUP_DEADLINE_SECONDS` / `SKULK_WARMUP_DEADLINE_SECONDS` | Hard deadline for runner warmup (default 300s). A wedged Metal eval parks warmup forever at 0% CPU and silently blocks all dispatch; the watchdog hard-exits the runner instead (supervisor reports RunnerFailed, node keeps working) |
| `SKULK_MLX_HANG_DEBUG` / `SKULK_MLX_HANG_DEBUG` | Emit periodic stack traces from stuck phases |
| `SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS` | Interval for above (default 30s) |
| `SKULK_MAX_OUTPUT_TOKENS` / `SKULK_MAX_TOKENS` | Default `max_tokens` (cluster default 4096; `DEFAULT_MAX_OUTPUT_TOKENS` constant) |
| `SKULK_NO_BATCH` / `SKULK_NO_BATCH` | Disable continuous batching |
| `SKULK_KV_CACHE_BACKEND` / `SKULK_KV_CACHE_BACKEND` | KV cache backend selection (overrides config) |
| `SKULK_LIBP2P_NAMESPACE` / `SKULK_LIBP2P_NAMESPACE` | libp2p namespace for cluster isolation |
| `SKULK_SKIP_LLM_WARMUP` | Skip warmup synthesis (single-node debug only) |
| `SKULK_IMAGE_TRANSPORT_DEBUG` | Verbose logging in image-transport pipeline |
| `SKULK_VISION_DEBUG_SAVE_DIR` | Save debug image artifacts |
| `SKULK_NATIVE_VISION_REFERENCE_PATH` | Force native-vision reference path (Gemma 4) |
| `SKULK_OFFLINE` | Run without internet checks (no model fetching) |
| `SKULK_TEST_DISTRIBUTED_MODEL` | Tests only: force the distributed/prefix-cache slow-test model (`gpt-oss-20b` or `llama-3.2-1b`); default auto-selects by Metal working-set size |
| `MLX_METAL_FAST_SYNCH` | Set by Skulk based on resolved card preference; not for direct operator use |
| `MLX_HOSTFILE`, `MLX_RANK`, `MLX_RING_VERBOSE`, `MLX_IBV_DEVICES`, `MLX_JACCL_COORDINATOR` | MLX upstream env vars; auto-set by Skulk during distributed init. Ring hostfile addresses are chosen per neighbor pair from OBSERVED libp2p connections, ranked thunderbolt > maybe_ethernet > ethernet > wifi > unknown > VPN/overlay — Tailscale CGNAT (100.64/10, fd7a:115c:a1e0::/48) addresses are detected by ADDRESS (utun types don't gossip) and rank strictly last: the overlay exists for external reachability and may be DERP-relayed, so it is only used when a pair has no local candidate (#265). Selection lives in `_find_ip_prioritised` / `get_mlx_ring_hosts_by_node` (`src/skulk/master/placement_utils.py`) |

### CLI flags

| Flag | What |
|---|---|
| `-v` / `-vv` / `-vvv` | Increase log verbosity |
| `-q` | Decrease verbosity |
| `--force-master` / `-m` | Force this node into master role |
| `--api-port` | Override default 52415 |
| `--no-api` | Disable API server |
| `--no-batch` | Disable continuous batching |
| `--fast-synch` / `--no-fast-synch` | Force MLX_METAL_FAST_SYNCH on/off |
| `--offline` | Offline mode |
| `--bootstrap-peers` | Comma-separated libp2p multiaddrs |
| `--libp2p-port` | Fixed TCP port for libp2p |

## Diagnostic mechanisms

### Flight recorder

- **Lives at:** `src/skulk/worker/runner/runner_supervisor.py` (the bounded buffer); emit helpers at `src/skulk/worker/runner/diagnostics.py`
- **Capacity:** last 128 entries per runner
- **Always-on; local-only.** Not gossiped, but exposed via `/v1/diagnostics/*`
- **Emission helpers:**
  - `record_runner_phase(phase, event=..., detail=..., attrs=..., include_memory=False)` — fire one entry
  - `runner_phase(phase, detail=...)` — context manager: enter / exit pair

### Trace sessions

- **Lives at:** `src/skulk/shared/tracing.py`
- **API:**
  - `begin_trace_session(task_id, rank, node_id, model_id, task_kind, tags)` — create
  - `record_trace_marker(name, rank, task_id, attrs)` — emit one event
  - `trace(category, name, ...)` — context manager / decorator
  - `pop_trace_session(task_id)` — collect + remove
  - `clear_trace_session(task_id)` — remove without collecting
- **Storage:** module-level dict `_trace_sessions: dict[str, TraceSession]`
- **Cluster path:** runner emits `TracesCollected` per rank → master merges to `TracesMerged` → API persists Chrome-trace JSON to disk

### MLX memory snapshot

- **Lives at:** `src/skulk/worker/runner/diagnostics.py::capture_mlx_memory_snapshot`
- **Returns:** `MlxMemorySnapshot { active, cache, peak, wiredLimit, source }`
- **Best-effort:** returns None if MLX isn't loaded or the snapshot fails

### Process sampling (macOS only)

- **Lives at:** `src/skulk/api/main.py::_collect_process_samples`
- **Wraps:** `sample <pid> <duration>`, `vmmap -summary <pid>`, `footprint -p <pid>`
- **Per-command timeout:** ~5-8s
- **Returns:** `list[DiagnosticProcessSample]` with `ok`, `stdout`, `stderr`, `error`

### Per-eval timeout

- **Lives at:** `src/skulk/worker/engines/mlx/auto_parallel.py::eval_with_timeout`
- **Wraps:** any `mx.eval(...)` call with a daemon-thread watchdog
- **Default timeout:** 60s (`pipeline_eval_timeout_seconds()`, configurable via `SKULK_PIPELINE_EVAL_TIMEOUT_SECONDS`)
- **On timeout:** emits `pipeline_eval_timeout` flight-recorder event, then `os._exit(1)`
- **Used at:** every `mx.eval` in `PipelineFirstLayer`, `PipelineLastLayer`, `mx_barrier`

### Parent-pid watchdog

- **Lives at:** `src/skulk/worker/runner/bootstrap.py::_install_parent_death_watchdog`
- **Mechanism:** daemon thread inside runner that polls `os.getppid()`; on reparenting, calls `mx.clear_cache()` + `gc.collect()` + `os._exit(1)`
- **Why:** SIGKILL of the agent leaves daemon `mp.Process` runners orphaned holding GPU memory. The watchdog detects the reparent and self-exits

## Centralized observability stack

Local Vector → VictoriaLogs → Grafana. Configuration:

- `src/skulk/shared/logging.py` — loguru JSON sink to stdout
- `deployment/logging/vector.yaml` — Vector pipeline (stdin → VictoriaLogs)
- `deployment/logging/docker-compose.yml` — VictoriaLogs + Grafana stack
- `skulk.yaml` `logging.enabled` + `logging.ingest_url` — opt-in; cluster-synced

## File map quick reference

```
src/skulk/
├── api/                # FastAPI app + adapters
│   ├── main.py         # routes, app construction, fan-out helpers
│   ├── adapters/       # OpenAI, Ollama, Claude, Responses, Skulk-native
│   └── types/          # API-facing Pydantic types
├── master/main.py      # event indexing, placement
├── worker/
│   ├── main.py         # worker loop
│   ├── plan.py         # task dispatch decisions
│   ├── runner/
│   │   ├── bootstrap.py            # subprocess entrypoint
│   │   ├── runner_supervisor.py    # parent-side lifecycle
│   │   ├── diagnostics.py          # flight recorder, MLX memory snapshot
│   │   ├── llm_inference/runner.py # text generation
│   │   ├── embeddings/runner.py    # embeddings
│   │   └── image_models/runner.py  # image generation
│   └── engines/mlx/
│       ├── auto_parallel.py        # sharding strategies + dispatch
│       ├── generator/generate.py   # prefill + decode hot path
│       ├── vision.py               # vision processing
│       ├── utils_mlx.py            # large utility module (decomposition tracked under #130 Phase 6)
│       ├── cache.py                # KV cache factory
│       └── gemma4_prompt.py        # Gemma 4 prompt renderer
├── routing/            # libp2p topics, event router, peer discovery
├── shared/
│   ├── types/          # State, events, commands, tasks, chunks, diagnostics
│   ├── models/         # ModelCard, capabilities resolver
│   ├── apply.py        # State + IndexedEvent → State
│   ├── election.py     # bully algorithm
│   └── tracing.py      # trace sessions
├── store/              # config, model store, custom card management
├── utils/              # disk_event_log, channels, helpers
└── main.py             # CLI entrypoint

dashboard-react/        # operator UI
deployment/             # observability stack docker-compose
bench/                  # benchmark + repro harnesses
docs/                   # operator guides (this file in website/docs/)
website/                # Docusaurus site
resources/inference_model_cards/  # built-in TOML cards
rust/                   # libp2p (networking), PyO3 bindings, system_custodian
```

## Maintenance discipline

This file is intentionally dense. If you find a stale fact, fix it inline rather than working around it.

The AGENTS.md "Documentation" section requires updates here when architectural shape changes:

- New component → add to "Components"
- New pubsub topic → add to "Pubsub topics"
- New event / command type → add to "Events" / "Commands"
- New state field → update "State" Pydantic model section
- New major API endpoint → add to the right "API endpoints" sub-table
- New family adapter → update "Family-specific code locations"
- New environment variable → add to "Configuration knobs"

Keep entries terse. Narrative belongs in [Architecture](architecture).
