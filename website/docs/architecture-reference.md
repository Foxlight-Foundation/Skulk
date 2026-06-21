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
- **Failover:** re-election picks a new master, which seeds its session from the node's prior replicated state (#273, `seed_state_for_new_session` in `src/skulk/shared/session_carryover.py`): **instances, downloads, node info maps, and tracing carry over**; in-flight tasks, runner statuses, topology, and liveness timestamps are deliberately dropped (tasks died with the old session's plumbing; runner processes are torn down by the worker re-creation; topology/liveness must come from live gossip — a carried topology would keep a dead node's out-edges forever). Workers re-create runners for the carried instances through the ordinary plan loop, so placements survive a master restart with a model-reload-sized gap instead of a silent permanent 404. The election winner tears its own worker down and rebuilds it, which cancels its `RunnerSupervisor.run()` tasks; that teardown is **shielded from cancellation** (`runner_supervisor.py`) so each runner process is fully reaped (Metal reclaims its wired GPU memory on exit) before `worker.shutdown()` returns. Without the shield the join was cancelled, the old runner lingered holding its memory, and the rebuilt worker's pre-load memory guard saw the not-yet-reclaimed memory, falsely refused the re-creation, and the #290 re-place-wider path deleted the carried instance (the silent 404 this design exists to prevent). It only bit when the winner also hosted a rank of a carried instance and was memory-tight. The plan loop suppresses liveness-based instance pruning for `TOPOLOGY_SETTLE_GRACE_SECONDS` (60s) after master start so carried instances aren't deleted while topology is still rebuilding; instances whose ranks lived on the dead master are pruned after the grace. A freshly-booted node that wins election seeds empty (it has no prior view) — identical to the pre-#273 behavior. The seed is indexed as **event 0 of the new session** (a logged `StateSnapshotHydrated`, `Master._index_seed_event`): late bootstrappers receive it inside the snapshot, early bootstrappers (including the promoted node's own worker, whose bootstrap races the promotion) receive it as the live first event — one delivery path, no idx-(-1) hydration skip.

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
- **Mounts:** dashboard at `/` (skipped when the built assets are absent, e.g. a headless/non-Mac worker node with no `dashboard-react/dist`; `DASHBOARD_DIR` is then `None` and the API serves without the UI, #333); OpenAPI at `/api/openapi.json`
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
| `COMMANDS` | `ForwarderCommand` | `Command` (`PlaceInstance`, `DeleteInstance`, `RefuseInstancePlacement`, `TaskFinished`, `SetTracingEnabled`, etc.) | API, Worker (`RefuseInstancePlacement`) | Master (command processor); Election (every node — observes commands to inform leader-changeover decisions) |
| `DOWNLOAD_COMMANDS` | `ForwarderDownloadCommand` | `DownloadCommand` (`StartDownload`, `DeleteDownload`, `CancelDownload`, `SyncConfig`, `PurgeStagingCache`, `RestartNode`) | API (download/restart/sync admin ops), Master, Workers | All nodes |
| `STATE_SYNC_MESSAGES` | `StateSyncMessage` | bidirectional: followers publish `kind="request"` for snapshot/config bootstrap; master publishes `kind="response"` with the requested payload (`StateSnapshotHydrated` etc.) | All nodes (request: followers; response: master) | All nodes |
| `ELECTION_MESSAGES` | `ElectionMessage` | bully election rounds | All nodes | All nodes |
| `CONNECTION_MESSAGES` | libp2p connection updates | peer arrivals / departures | Router | All nodes |
| `TELEMETRY` | `NodeTelemetry` | `GatheredInfo` (`NodeResources` — participation role + backends; `MemoryUsage`/`MactopMetrics`+`MacmonMetrics` — per-node memory + system profile; `LinuxGpuMetrics` — AMD/Linux GPU system profile; `NodeDiskUsage`/`MiscData`/`StaticNodeInformation`/`RdmaCtlStatus` — disk + identity + rdma-ctl, slice 3) | Workers | All nodes (applied into `TelemetryView`) |
| `DATA` | `DataChunk` | `{command_id, GenerationChunk}` — per-token generation output (data plane, #279 Phase 2) | Serving rank-0 worker | API nodes only (demux by `command_id` into per-command stream queues); master does NOT consume it |

### Telemetry plane (#279)

`TELEMETRY` is the first slice of the control/telemetry/data plane separation (#279). Node readings that are **last-write-wins and not decisions** are gossiped on this topic instead of being event-sourced into `State`. They land in an in-memory `TelemetryView` (`src/skulk/shared/types/telemetry.py`), held per-`Node` and read by the planner (placement eligibility) and the API placement previews — they are **not** persisted in the event log or carried in snapshots.

Slice 1 moved `node_resources` (a node's `participation` role and `backends`); slice 2 moved `node_memory` and `node_system` (the highest-volume readings, carried together by `MactopMetrics`/`MacmonMetrics`); slice 3 moved the observational readings `node_identities`, `node_disk`, and `node_rdma_ctl`. The set that rides telemetry is `TELEMETRY_PLANE_INFO` in `shared/types/telemetry.py`; the worker forwards those `GatheredInfo` variants to the telemetry sender (`worker/main.py`), and `apply_node_gathered_info` treats them as no-ops (`shared/apply.py`). The `TelemetryView` survives master re-election (Node-owned), so a freshly promoted master does not start blind, and these readings are no longer carried in the failover seed (`session_carryover.py`). `GET /state` merges them back in from the view (`API.get_cluster_state`) so the dashboard's wire shape is unchanged. Node identity is assembled from two readings (`MiscData` friendly-name + `StaticNodeInformation` static fields), so `TelemetryView.apply` **merges** them into one `NodeIdentity` rather than overwriting.

**Normalized accelerator metrics (collector-agnostic GPU telemetry).** `SystemPerformanceProfile` carries an optional `accelerator: AcceleratorMetrics` block (`shared/types/profiling.py`): `vendor`/`name`/`utilization_ratio` (0..1)/`vram_total_bytes`/`vram_used_bytes`/`gtt_total_bytes`/`power_watts`/`temperature_celsius`/`clock_mhz`, each `None` when a collector cannot measure it (distinct from a real `0`). `gtt_total_bytes` is the GPU's GTT (graphics translation table) size, the amount of system RAM the GPU can map; on a unified-memory APU (AMD Strix Halo) it spans system memory, so placement counts it toward usable GPU memory (see the `PlaceInstance` row). The expression is the same regardless of collector; normalization happens at the collector boundary, never downstream. macOS fills it from `mactop` (`vendor="apple"`, `utilization_ratio = gpu_usage/100`; unified memory so `vram_*` stay `None`). AMD/Linux fills it from a new `InfoGatherer._monitor_gpu_linux` that reads passive amdgpu sysfs (`gpu_busy_percent`, `mem_info_vram_*`, `hwmon/power1_average`, `temp1_input`, `pp_dpm_sclk` via `utils/info_gatherer/linux_gpu.py`) and publishes a `LinuxGpuMetrics` telemetry variant carrying only the system profile (memory rides the separate `MemoryUsage` reading). Passive sysfs reads, never a GPU-colliding poll (the macmon/#249 lesson). The AMD `vram_total_bytes` exposes a Strix Halo's BIOS-carved GPU VRAM pool, which node memory does not report; placement admits GPU-offload nodes against it (see the `PlaceInstance` row).

**Connectivity readings stay on the control plane.** `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge`, and the derived `thunderbolt_bridge_cycles` are NOT telemetry: `apply()` builds the RDMA topology graph from `node_thunderbolt` (`MacThunderboltConnections` to `replace_all_out_rdma_connections`) and recomputes TB-bridge cycles from `node_network` + `node_thunderbolt_bridge`, and the planner reads `node_network` for host selection. Those define the graph placement runs on, so they must be ordered event-sourced state, not an unordered last-write-wins plane (#279 slice 3 scoping; refines the original "all of `NodeGatheredInfo` to telemetry" target).

**Placement reads two views.** The memory-fit check and the context-admission ceiling read `node_memory` from the `TelemetryView`, not `State`. Because the ceiling must be identical across ranks (divergent verdicts deadlock the collectives) and telemetry is unordered last-write-wins, the master computes the ceiling **once at placement time** and stamps it onto the instance (`BaseInstance.context_token_limit`, event-sourced); every worker rank, and the API's admission pre-flight, then read that stamped value instead of recomputing it.

**Capability-aware placement (heterogeneous nodes).** Backend tags are `<engine>-<compute>` (`mlx-metal`, `llama_cpp-vulkan`, `llama_cpp-rocm`, `llama_cpp-cuda`, `llama_cpp-cpu`); the engine selects the worker runner class, the compute names the accelerator (`src/skulk/shared/backends.py`). A node probes its own backends (`probe_node_backends`: macOS advertises `{mlx, mlx-metal}`; any node with importable `llama_cpp` adds `{llama_cpp}` + a compound tag per `SKULK_LLAMA_CPP_BACKENDS` entry, defaulting to `llama_cpp-cpu` when that env var is unset so a node never over-claims a GPU it may not have built) and gossips them in `NodeResources.backends` on the telemetry plane. A model card's `PlacementCardConfig` carries two axes orthogonal to memory/topology: `compatible_backends` (a **hard filter**: the planner excludes a node when `resources.backends & compatible_backends` is empty, `src/skulk/master/placement.py`) and `backend_preference` (an ordered **soft score**, `_cycle_backend_preference_score`). GGUF cards stamp the llama.cpp tags as compatible; MLX cards keep MLX. The worker re-derives the concrete engine for its node at spawn (`bootstrap._resolve_text_engine`: card `compatible_backends` ∩ node backends, ordered by `backend_preference`). See `website/docs/amd-strix-halo-nodes.md` for a non-Mac node.

### Data plane (#279 Phase 2)

`DATA` carries per-token **generation output** off the event log. The serving rank-0 worker publishes `DataChunk` (`{command_id, GenerationChunk}`) on this topic; `RunnerSupervisor._emit` diverts `ChunkGenerated` to the data sender while every other runner event (task status, acks, runner status) stays on the ordered control-plane event sender. The owning API node drains `DATA` in `API._apply_data` and demuxes by `command_id` into the per-command stream queues (`_dispatch_generation_chunk`), exactly as the event path did; `await send` preserves the same backpressure (a slow client throttles its producer). The master **never** sees these chunks: no index, no disk, no `GLOBAL_EVENTS` rebroadcast.

Output chunks never mutated `State` (their `apply()` was a no-op), so removing them from the ordered log is loss-free for *state* correctness while eliminating the per-token master hop and disk write that dominated event-log volume and was the #278 storm vector. **Ordering is NOT free, though** (that was the Phase 2a bug, fixed in #279 Phase 2b): the master event `idx` it replaced gave every chunk a cluster-wide total order, but the `DATA` topic has no ordering key, and when the producing rank-0 worker and the owning API node are different nodes the gossip mesh can deliver a command's chunks out of order. The API consumed them in arrival order, silently transposing tokens/sub-words in multi-node *sampled* speculative output (single-node is local/in-order; greedy emits steadily). The fix: `DataChunk` carries a per-command monotonic `sequence` stamped by the producing supervisor (`RunnerSupervisor._emit`, assigned in `_forward_events` generation order), and the API reorders by it in a small per-command buffer (`API._reorder_and_dispatch`): it releases strictly in order, drops duplicates, and skips past a genuinely dropped sequence once the buffer exceeds `_MAX_CHUNK_REORDER_BUFFER` so a best-effort drop can't stall. The buffer exists only while a command has a live stream queue (created lazily, cleared in `_finalize_command_stream`), so late chunks after finalize drop without leaking. A `--no-api` worker-only node registers `DATA` but has no receiver, so `TopicRouter` drains and drops its messages (no buffer leak). Inbound vision chunks (`InputChunkReceived`, low-volume, API → worker) stay on the control plane for now; a later phase would move output to true per-command unicast (stamp the owning API node onto the dispatched task) to also drop the cluster-wide gossip fan-out.

Because `DATA` is best-effort (no replay), a dropped *final* chunk would otherwise leave a streaming response blocked forever. `_token_chunk_stream` guards against that with a per-receive idle timeout (`_STREAM_IDLE_TIMEOUT_SECONDS`, 120s, #279 Phase 2b): genuine producer silence for that long closes the stream with a terminal error rather than hanging. The timeout wraps only the `receive()` (not the `yield`), so it measures producer silence and never trips on a slow client. The timer is armed **only after the first real output token**: time-to-first-token is unbounded, so a request queued behind a long decode, or in a slow prefill, never trips it (prefill-progress chunks are explicitly not treated as output and do not arm the timer). When the stall does fire, it disambiguates by the master's task status: a task that has already reached a terminal status means the runner finished and only the *final* chunk was dropped, so the stream cleans up via the normal `TaskFinished` path (sending `TaskCancelled` there would be a no-op on a completed runner and leak the master's task/command mapping); a still-active task means a genuinely stuck runner, which `TaskCancelled` tears down. The dead-instance case is already covered by the control plane (`TaskFailed` → `_terminate_command_stream`). Mid-stream chunk *reordering* is now handled by the per-command sequence number (above); a genuinely *dropped* chunk on the best-effort topic remains a tradeoff (the reorder buffer skips past it after `_MAX_CHUNK_REORDER_BUFFER`, and the idle backstop covers a dropped final chunk).

**Zenoh data-plane transport (soft default-on, #315).** The `DATA` topic can ride an Eclipse Zenoh peer session instead of gossipsub; control, telemetry, and election planes stay on libp2p. Transport selection is resolved by `_resolve_zenoh_enabled(SKULK_ZENOH_DATA_PLANE, SKULK_ZENOH_LISTEN)`: explicit `SKULK_ZENOH_DATA_PLANE` of `1`/`true`/`yes`/`on` forces Zenoh on (and still requires an explicit listen, #308), `0`/`false`/`no`/`off` forces gossipsub, and **unset** is soft default-on (Zenoh when `SKULK_ZENOH_LISTEN` is configured, else gossipsub) so a bare node with no Zenoh config stays on gossipsub instead of failing the #308 listen requirement. The `Router` holds an optional `ZenohHandle` (`skulk_pyo3_bindings`, backed by `rust/networking/src/zenoh_session.rs`); `Router.uses_zenoh(topic)` routes only the `DATA` topic to it (subscribe/publish plus a parallel `_zenoh_recv` drain loop). The session is a Zenoh `peer` with multicast scouting OFF and gossip + explicit `SKULK_ZENOH_CONNECT` endpoints (the macOS-Local-Network-Privacy-safe posture), publishing `Reliable` + `Block` on a single priority so a single rank-0 producer's chunks are FIFO per key. Zenoh's per-publisher per-priority ordering is what lets the app-layer reorder buffer be skipped: as of #279 Phase 3 the buffer is **transport-conditional**: kept for gossipsub (which reorders), skipped under Zenoh (output dispatches in arrival order). The API selects this via `data_plane_zenoh` (the resolved transport from `_resolve_zenoh_enabled(SKULK_ZENOH_DATA_PLANE, SKULK_ZENOH_LISTEN)`, so it is true under soft default-on when only the listen is set); `SKULK_DATA_REORDER_BUFFER` overrides. The `sequence` field and `API._reorder_and_dispatch` machinery remain in the code (gossipsub still needs them) until Zenoh becomes the default DATA transport, when they are deleted outright.

Key-addressed unicast (#279 Phase 2) replaces the cluster-wide fan-out. The owning API node stamps its own node id on the serving command (`TextGeneration`/`ImageGeneration`/`ImageEdits`/`TextEmbedding`, `owner_node` field), the master carries it onto the worker task, and the rank-0 supervisor records it (`_command_owner`) and stamps it onto each `DataChunk` (`owner_node`). The `DATA` `TypedTopic` carries a `routing_key` hook (`_data_owner_key`) returning that owner id; the networking channel is a 3-tuple `(topic, routing_key, data)`, the `Router` publishes to the Zenoh key `data/<owner_node>`, and each node subscribes only to `data/<own_node_id>` (threaded in via `Router.create(..., node_id=...)`), so output reaches just the owning API node instead of every node. Inbound, `_zenoh_recv` strips the `/<owner>` suffix back to the bare `data` topic to find the `TopicRouter`. On gossipsub (flag off) `owner_node` is ignored and the bare topic broadcasts as before, so the two transports stay interchangeable per-node behind the flag.

Hardening toward default-on (#308 + #309). Security (#308): the session sets a Zenoh `namespace` (`ZenohConfig.namespace`) that transparently prefixes every key, so a peer on a different namespace does not receive this fleet's `data`. The namespace is a collision-resistant SHA-256 hash (`_derive_zenoh_namespace`) of the exact token libp2p isolates on: `_libp2p_namespace_token` mirrors `swarm.rs`, using `SKULK_LIBP2P_NAMESPACE` when set and the `NETWORK_VERSION` default `v0.0.1` otherwise (the legacy `EXO_LIBP2P_NAMESPACE` is NOT read; deriving from a different source would split one libp2p cluster across two Zenoh namespaces). This namespace provides isolation between distinct clusters, NOT confidentiality against an adversary already on the same Zenoh network: with no TLS the prefix is the only barrier and its seed is non-secret operator config (it is also surfaced in `/v1/diagnostics/node`), so on an untrusted network use Zenoh TLS or a firewall. As hygiene, startup never logs the raw token (it also seeds libp2p's private-network PSK) or the derived namespace, only a short non-routing fingerprint plus whether the override env was set. `SKULK_ZENOH_LISTEN` is required explicitly (no `0.0.0.0` default). TLS/ACL remain operator-configurable for untrusted links. Robustness (#309): the DATA plane has its own outbound loop (`Router._zenoh_networking_publish` draining a dedicated channel), so its `Block` backpressure can't stall the shared `_networking_publish` control loop; that channel is **bounded** (`_ZENOH_DATA_OUTBOUND_BUFFER`, #312 review) so a stalled subscriber backpressures the producer (the rank-0 emit) rather than growing memory without limit and OOMing the node, and the bound deliberately backpressures rather than drops because the Zenoh plane is Reliable+ordered (dropping would break the reorder-buffer-skip assumption); and `ZenohSession::publish`/`subscribe` clone the publisher `Arc` / check-and-release before the `declare`/`put` await, so the publishers mutex never spans a network put and concurrent per-command publishes don't serialize.

**Version policy:** all cluster nodes must run the same Skulk version — **mixed-version clusters are unsupported** (see "Deployment & versioning" in [Architecture](architecture)). With `extra="forbid"` models there is no cross-version wire compatibility, so the telemetry plane is **not** engineered to bridge an un-upgraded worker's legacy `NodeGatheredInfo` telemetry into the view. There is no transition-hydration concession either: a node never reloads its own persisted `State` across restart (node identity is ephemeral; `State` is rebuilt from the event log / state-sync), so a pre-#279 snapshot carrying the removed `nodeResources`/`nodeMemory`/`nodeSystem` keys is simply rejected by `extra="forbid"`. (A `State` before-validator that stripped those keys was removed in #294 because it silently broke state-sync by forcing strict Python-mode validation, under which ISO datetime strings like `lastSeen` were rejected.) General mixed-version compatibility (a protocol-version handshake) is tracked in #293 as explicitly out of scope.

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
| `PlaceInstance` | Spin up a model on the cluster. Optional `excluded_nodes: list[NodeId]`: planner treats those nodes as if absent for *this placement only*; already-running instances on them are not affected. | Pick ranks based on memory + topology (filtered by `excluded_nodes`); emit `InstanceCreated`. Memory admission is per-node (Tensor = even split, Pipeline = proportional to available): a node's weight share x an engine-aware overhead factor (`memory_overhead_factor`, 1.30 for MLX / 1.10 for GGUF; see below) + an explicit KV-cache reservation for `KV_CONTEXT_BUDGET_TOKENS` (8192) + `MEMORY_OVERHEAD_FLOOR` (256 MB), each node capped at `GPU_WORKING_SET_FRACTION` (0.75) of `ram_total` (the Metal GPU working-set ceiling, since gossiped `ram_available` can exceed what the GPU may wire). On macOS the gossiped `ram_available` is itself the GPU-wireable figure `total − wired − anonymous − compressor` (vm_stat snapshot per telemetry sample; see `MemoryUsage` below) rather than the naive free-plus-inactive figure that counted reclaimable file cache as used. A node that reports **discrete GPU VRAM** (AMD/NVIDIA `vram_total_bytes` in `node_system`) is instead admitted against its usable VRAM (`min(vram_total − vram_used, GPU_VRAM_WORKING_SET_FRACTION (0.90) × vram_total)` via `usable_vram_by_node`), because a GPU-offload engine (llama.cpp/vLLM) allocates weights + KV from VRAM, not system RAM (e.g. a Strix Halo's 64 GB VRAM pool, separate from its 64 GB system RAM, which a 0.75×system-RAM cap would wrongly refuse). On a **unified-memory APU** node (the accelerator's GTT spans the whole system: `gtt_total_bytes > vram_total_bytes` AND `gtt_total_bytes ≥ ram_total`) usable GPU memory is the working-set-capped VRAM (`0.90 × vram_total`) plus the system RAM the GPU can map via GTT, minus `UMA_GPU_OS_HEADROOM` (16 GB), so a model larger than the BIOS VRAM carve-out runs through GTT (e.g. a 58.5 GB GGUF gpt-oss-120B on a 128 GB Strix Halo with a 64 GB carve-out). The dual gate matters because a discrete amdgpu card also reports a `gtt_total_bytes` (its default can equal VRAM); requiring GTT to cover all of system RAM keeps a dedicated card on the conservative VRAM-only path. Apple unified-memory nodes report no discrete VRAM and keep the system-RAM ceiling. The weight-overhead factor is engine-aware (`memory_overhead_factor`): GGUF/llama.cpp models use `LLAMA_CPP_MEMORY_OVERHEAD_FACTOR` (1.10), lighter than MLX's 1.30, because the C++ runtime carries no MLX buffer cache or Python interpreter overhead. Estimation lives in `skulk.shared.models.memory_estimate`, shared with the worker's local pre-spawn OOM guard so the two checks never disagree. Failures raise typed `PlacementError`s, with `PlacementInfoPendingError` for the cluster-startup windows where cluster info has not finished gossiping (connection edges lag identities; memory info lags the edges). The API dry-runs placement before forwarding (400 on impossible, 503 after a 15s wait on pending info). |
| `DeleteInstance` | Tear down a placed model | Emit `InstanceDeleted`; workers tear down runners |
| `RefuseInstancePlacement` | Worker → master: this node cannot fit its shard at load time (the live GPU-wireable reading sits below what the gossiped telemetry admitted). Carries `instance_id`, `node_id`, `reason`. | Delete the refused instance and **re-place the same model one node wider** (`min_nodes` = refused width + 1, via `replacement_command_for_refused_instance` + `place_instance`), so each node holds a smaller share. If even a full-width split will not fit, `place_instance` raises `PlacementError` and the master stops at the deletion; that terminal case bounds the refuse→re-place loop to at most the cluster size. Idempotent: a refusal for an already-removed instance is a no-op. Fixes #290 (place-then-silently-vanish on tight multi-node splits). |
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
- `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge: Mapping[NodeId, *]` — the **connectivity** per-node maps that stay on the event path because they define the topology graph (see "Connectivity readings stay on the control plane" under the Telemetry plane section). They update at independent frequencies via `NodeGatheredInfo`.
- `node_resources` (slice 1), `node_memory` + `node_system` (slice 2), and `node_identities` + `node_disk` + `node_rdma_ctl` (slice 3) are **not** `State` fields: they moved to the telemetry plane (`TelemetryView`, gossiped on `TELEMETRY`, see "Telemetry plane" above) as part of #279. `State` keeps `extra="forbid"`, so a pre-#279 snapshot carrying the old `nodeResources`/`nodeMemory`/`nodeSystem`/`nodeIdentities`/`nodeDisk`/`nodeRdmaCtl` keys is rejected, which is the intended behavior, since mixed-version clusters are unsupported and a node never reloads its own persisted `State` across restart anyway (identity is ephemeral; State is rebuilt from the event log / state-sync). An earlier before-validator that stripped those keys was removed in #294 because it broke state-sync (it forced strict Python-mode validation, rejecting ISO datetime strings like `lastSeen`).
- `thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]]` — detected Thunderbolt-bridge cycles where every node has it enabled (>2 nodes)

Note: there is no `master_node_id` field on `State`. Master identity lives outside the event-sourced state — each node tracks the current master independently via the election protocol (`src/skulk/shared/election.py`). `placements` is also not a field; placement information is derived from `instances` (each `Instance` has its own shard assignments).

### Diagnostics

`src/skulk/shared/types/diagnostics.py`. Major models:

- `NodeDiagnostics` — runtime + identity + resources + processes + supervisor_runners + placements + warnings. `warnings` includes a **leaked-wired-memory** alert (`_leaked_wired_warning` in `src/skulk/api/main.py`): emitted when `resources.current_wired` exceeds ~5GB with zero `process_alive` runners — the signature of wired memory leaked by an abnormal Metal termination that only a reboot reclaims (#239). Server-side counterpart of `tests/preflight_mem.sh`. To stop a doomed runner from compounding such a leak, the worker circuit-breaks runner crash loops (`CrashWindow` in `src/skulk/utils/crash_window.py`, 3 failures within 60s) and gives up rather than relaunching it. The give-up action depends on *why*: a genuine crash or GPU wedge deletes the instance via `DeleteInstance`, but a **memory fit refusal** (the pre-spawn guard rejecting the shard) sends `RefuseInstancePlacement` instead, so the master re-places the model one node wider rather than letting it silently vanish (#290).
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
- `placement: PlacementCardConfig` (the only section the planner reads directly, `master/placement.py`). `compatible_backends: frozenset[str]` is a **hard filter** (route only to nodes whose advertised `NodeResources.backends` intersect it; default `{"mlx"}`). `backend_preference: tuple[str, ...]` is a **soft, ordered** rank among those backends: the planner prefers a cycle that can serve an earlier-listed tag (`_cycle_backend_preference_score`) and the runner picks the earliest backend the node has. Backends use compound `<engine>-<compute>` tags (`mlx-metal`, `llama_cpp-vulkan`, `llama_cpp-rocm`, and so on; vocabulary + node probing in `src/skulk/shared/backends.py`); nodes also advertise the bare engine tag (`mlx`) for back-compat with original `{"mlx"}` cards. The split is deliberate: filter answers "which nodes are allowed", preference answers "fastest for *this* model" (Vulkan vs ROCm performance is model-dependent), so a Vulkan-preferring model still degrades gracefully onto a ROCm-only node. Also `min_vram_gib` (hard) and `max_context_tokens` (soft KV-budget cap).

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

Only `SKULK_*` names are read. The legacy `EXO_*` deprecation runway was removed in #324; typed-config migration is tracked under #110. (Some rows below still show a duplicated `SKULK_X / SKULK_X` artifact from that rename and will be de-duplicated in the #110 sweep.)

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
| `SKULK_LLAMA_CPP_BACKENDS` | Comma-separated llama.cpp compute backends this node was built with, e.g. `vulkan` or `vulkan,rocm` (valid: `vulkan`/`rocm`/`cuda`/`cpu`; `metal` is MLX-only and ignored). Authoritative operator policy because the compiled build (not installed libraries) decides what llama.cpp can use, and the binding does not cleanly expose it. Read by `probe_node_backends` (`src/skulk/shared/backends.py`) to advertise compound tags like `llama_cpp-vulkan` in `NodeResources.backends`; unset claims only `llama_cpp-cpu` (never over-claims GPU). Inert until a node has `llama_cpp` importable. A declared GPU backend is cross-checked against the actual build via `llama_cpp.llama_supports_gpu_offload()`: if the installed wheel has no GPU offload compiled in (the classic case where `uv sync` restored the CPU-only PyPI wheel over a source-built GPU wheel), the GPU tags are dropped and the node advertises only `llama_cpp-cpu`, so GPU GGUF work is not routed to a degraded build. The service entrypoint (`deployment/install/skulk-startup.sh`) runs `uv sync --inexact` when this declares a GPU backend, so a routine sync does not prune the source-built wheel in the first place. |
| `SKULK_LLAMA_CPP_LOGITS_ALL` | Whether the llama.cpp runner loads each GGUF with `logits_all=True`, enabling per-request logprobs (`src/skulk/worker/runner/llama_cpp/runner.py`, `_logits_all_enabled`). Defaults **off**: `logits_all` makes llama.cpp pre-allocate an `n_ctx * vocab * 4` logits buffer up front, which at the model's full trained context is enormous (e.g. `131072 * 152064 * 4` = 74 GiB for a Qwen2.5 vocab) and OOMs the node on load. So logprobs is opt-in (`=1`); the runner is loaded once and the flag cannot be toggled per request. With it off a logprobs request degrades to a clear error chunk. Regardless of this flag the served context window is bounded by the KV budget placement reserved (`_serving_n_ctx`, `KV_CONTEXT_BUDGET_TOKENS`), never the model's full trained context (`n_ctx=0`) nor the larger request-admission ceiling, either of which would size the KV cache beyond reserved memory and exhaust the node on load. |
| `SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX` | Context-length cap (tokens, default `8192`) applied **only when `SKULK_LLAMA_CPP_LOGITS_ALL=1`** (`_logits_all_n_ctx`). Bounds the `logits_all` buffer (`n_ctx * vocab * 4`) so opting into logprobs does not blow up memory: at an ~150k vocab, 8192 is ~5 GiB. It is operator policy, so raising it far above the default reintroduces the large allocation it guards against. When logits_all is off the served context is the instance's admission ceiling (`_serving_n_ctx`), not the model's full trained context. |
| `SKULK_ZENOH_DATA_PLANE` | Soft default-on (#315). Resolved by `_resolve_zenoh_enabled` in `Node.create` (`src/skulk/main.py`): truthy (`1`/`true`/`yes`/`on`) forces the Zenoh DATA plane on (requires `SKULK_ZENOH_LISTEN`, #308), falsy (`0`/`false`/`no`/`off`) forces gossipsub, and **unset** uses Zenoh only when `SKULK_ZENOH_LISTEN` is set (else gossipsub, so a bare node never crashes on the listen requirement). When on, the `DATA` topic (per-token output) rides an Eclipse Zenoh peer session instead of gossipsub; all other planes stay on libp2p. Wired in `Router` (`uses_zenoh`). **Security (#308):** the session is **namespace-isolated** (keys prefixed by a segment that is a collision-resistant SHA-256 hash of the exact token libp2p isolates on (`SKULK_LIBP2P_NAMESPACE` when set, else the `NETWORK_VERSION` default `v0.0.1`; mirrors `swarm.rs`, not the legacy `EXO_LIBP2P_NAMESPACE`). Neither the raw token nor the derived namespace is logged (with no TLS the namespace is the only isolation value); startup logs only a short non-routing fingerprint), so a peer on a different namespace does not receive this fleet's `data` (parity with the libp2p private namespace). This is isolation between distinct clusters, NOT confidentiality against an adversary already on the same Zenoh network: the seed is non-secret operator config (also surfaced in `/v1/diagnostics/node`) and there is **no transport auth/TLS** by default, so on an untrusted network either enable Zenoh TLS (operator-configured) or keep it firewalled; a loud startup warning fires when on. |
| `SKULK_ZENOH_LISTEN` | Zenoh listen endpoint when the data plane is on. **Required explicitly** (#308 bind restriction): Skulk refuses to start the plane with this unset rather than silently binding `tcp/0.0.0.0:7447` (all interfaces). Set a specific private IP, e.g. `tcp/192.168.0.115:7447`; an explicit `0.0.0.0` is allowed but warns. |
| `SKULK_ZENOH_CONNECT` | Comma-separated explicit Zenoh peer endpoints (multicast scouting is off, so peers are explicit), e.g. `tcp/192.168.0.117:7447,tcp/192.168.0.122:7447`. Per-node. |
| `SKULK_DATA_REORDER_BUFFER` | Explicit override for the data-plane reorder buffer (#279 Phase 3). Unset (default): the buffer follows the DATA transport - ON for gossipsub (it reorders; the #301 fix), OFF for Zenoh (per-publisher FIFO, so arrival order is generation order; validated 20/20 on a 3-node sampled-MTP matrix). Set `1`/`0` to force it on/off regardless of transport (testing / belt-and-suspenders). Read in `API.__init__` (`_reorder_buffer_enabled`), with the transport signalled by `data_plane_zenoh` from `Node.create`. |
| `SKULK_SKIP_LLM_WARMUP` | Skip warmup synthesis (single-node debug only) |
| `SKULK_IMAGE_TRANSPORT_DEBUG` | Verbose logging in image-transport pipeline |
| `SKULK_VISION_DEBUG_SAVE_DIR` | Save debug image artifacts |
| `SKULK_NATIVE_VISION_REFERENCE_PATH` | Force native-vision reference path (Gemma 4) |
| `SKULK_OFFLINE` | Run without internet checks (no model fetching) |
| `SKULK_HEADLESS` | Deploy knob read by `deployment/install/skulk-startup.sh` (the LaunchAgent/systemd entrypoint). `1` on a node that serves the API without the web UI (e.g. a non-Mac worker like a Strix Halo/ROCm box with no Node/npm): boot-time prep skips the dashboard build and its otherwise-fatal `dashboard-react/dist` missing check, and the node runs with `DASHBOARD_DIR` unset (#333). Default `0` keeps the fail-loud behavior so a Mac with an accidentally-absent build is caught. |
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
