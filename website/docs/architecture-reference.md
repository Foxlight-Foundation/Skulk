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
- **Lives in:** `src/exo/master/main.py`
- **Owns:** the authoritative event log (via `DiskEventLog`); the indexer that assigns monotonic indices to events; the placement planner. Master identity itself lives outside the master process — each node tracks the current master independently via the election protocol (`src/exo/shared/election.py`); the `_master_node_id` cache is held on the API side at `src/exo/api/main.py:461`.
- **Communicates via:** `LOCAL_EVENTS` (consumes), `GLOBAL_EVENTS` (publishes indexed events), `COMMANDS` (consumes), `STATE_SYNC_MESSAGES` (publishes snapshots)
- **Election:** `src/exo/shared/election.py` — bully algorithm; a single master at a time
- **Failover:** transparent via re-election; new master picks up from the disk event log

### Worker

- **Role:** receives indexed events, applies them locally, downloads model weights, spawns + supervises runner subprocesses, dispatches tasks
- **Lives in:** `src/exo/worker/main.py`; planning at `src/exo/worker/plan.py`
- **Owns:** local view of `State` (derived); per-model `RunnerSupervisor` instances
- **Communicates via:** `GLOBAL_EVENTS` (consumes), `LOCAL_EVENTS` (publishes), `COMMANDS` (publishes for placement requests)

### RunnerSupervisor

- **Role:** parent-side lifecycle for one runner subprocess; signal handling; flight recorder buffer; SIGTERM/SIGKILL cleanup chain
- **Lives in:** `src/exo/worker/runner/runner_supervisor.py`
- **Spawns:** `mp.Process(target=entrypoint, daemon=True)` with the runner subtype's main loop
- **Cleanup chain:** `join(5s)` → `terminate()` (SIGTERM) → `join(5s)` → `kill()` (SIGKILL); plus parent-pid watchdog inside the subprocess for reparenting (SIGKILL of agent)

### Runner subprocess

- **Role:** owns one MLX model; serves inference tasks for it; participates in distributed collectives with peer runners across ranks
- **Entrypoint:** `src/exo/worker/runner/bootstrap.py::entrypoint`
- **Subtypes:**
  - `src/exo/worker/runner/llm_inference/runner.py` — text generation
  - `src/exo/worker/runner/embeddings/runner.py` — embeddings
  - `src/exo/worker/runner/image_models/runner.py` — image generation
- **Communicates via:** `mp.Queue` from worker (incoming tasks); `mp.Queue` to worker (outgoing events); `mlx.distributed` collectives with peer runners

### Router (libp2p)

- **Role:** transport for all inter-node communication
- **Lives in:** `src/exo/routing/` (Python wrapper); `rust/networking/` + `rust/exo_pyo3_bindings/` (Rust libp2p impl + PyO3 bindings)
- **Topics:** see "Pubsub topics" below
- **Discovery:** mDNS by default; `--bootstrap-peers` multiaddrs for explicit static peers

### Election

- **Role:** picks the cluster master via the bully algorithm
- **Lives in:** `src/exo/shared/election.py`
- **Communicates via:** `ELECTION_MESSAGES` topic
- **Triggers:** node startup, lost master heartbeat, explicit master abdication

### API

- **Role:** HTTP entry point; FastAPI app; OpenAI / Ollama / Claude / Responses / Skulk-native adapters; serves dashboard
- **Lives in:** `src/exo/api/main.py`; adapters at `src/exo/api/adapters/`
- **Default port:** 52415
- **Mounts:** dashboard at `/`; OpenAPI at `/api/openapi.json`

### Dashboard

- **Role:** operator UI for the same Skulk runtime
- **Lives in:** `dashboard-react/` (source); served by API at `/`
- **Stack:** React + TypeScript + styled-components + Vite
- **State:** Zustand (`dashboard-react/src/stores/uiStore.ts`, `dashboard-react/src/stores/chatStore.ts`)
- **Routing:** activity-style enum (`activeRoute` in `uiStore`); no react-router
- **Persistence:** sessionStorage for in-session UI; localStorage for cross-session preferences (theme, panel widths)

### Storage

- **Event log:** `src/exo/utils/disk_event_log.py` — append-only zstd-compressed msgpack
- **Model cache:** `~/.skulk/models/` (`SKULK_HOME` overrides)
- **Custom cards:** `~/.skulk/custom_model_cards/` as TOML
- **Built-in cards:** `resources/inference_model_cards/` as TOML
- **Optional model store:** shared host with rsync-style staging — `src/exo/store/`

## Pubsub topics

Defined in `src/exo/routing/topics.py`.

| Topic | Wire payload type | Inner payload | Publisher | Consumer |
|---|---|---|---|---|
| `GLOBAL_EVENTS` | `GlobalForwarderEvent` | indexed `Event` (post-master indexing) | Master | All nodes |
| `LOCAL_EVENTS` | `LocalForwarderEvent` | un-indexed `Event` | Workers (via `event_router.py`) | Master |
| `COMMANDS` | `ForwarderCommand` | `Command` (`PlaceInstance`, `DeleteInstance`, `TaskFinished`, `SetTracingEnabled`, etc.) | Workers, API | Master |
| `DOWNLOAD_COMMANDS` | `ForwarderDownloadCommand` | `DownloadCommand` (`StartDownload`, `DeleteDownload`, `CancelDownload`, `SyncConfig`, `PurgeStagingCache`, `RestartNode`) | API (download/restart/sync admin ops), Master, Workers | All nodes |
| `STATE_SYNC_MESSAGES` | `StateSyncMessage` | bidirectional: followers publish `kind="request"` for snapshot/config bootstrap; master publishes `kind="response"` with the requested payload (`StateSnapshotHydrated` etc.) | All nodes (request: followers; response: master) | All nodes |
| `ELECTION_MESSAGES` | `ElectionMessage` | bully election rounds | All nodes | All nodes |
| `CONNECTION_MESSAGES` | libp2p connection updates | peer arrivals / departures | Router | All nodes |

## Events

Discriminated union at `src/exo/shared/types/events.py`. Selected events:

| Event | Emitted when | Applied by |
|---|---|---|
| `InstanceCreated` | Master places a model | All nodes (update `State.instances`) |
| `InstanceDeleted` | Master deletes a placement | All nodes |
| `RunnerStatusUpdated` | Runner subprocess transitions state | All nodes |
| `RunnerFailed` | Runner crashes or exits unexpectedly | All nodes |
| `TaskAcknowledged` | Worker accepts a task | All nodes |
| `TaskStatusUpdated` | Task transitions state (`Running`, `Failed`, `Cancelled`, `Complete`); natural completion is the `Complete` variant, driven by the `TaskFinished` command | All nodes |
| `TaskDeleted` | Task is purged from cluster state | All nodes |
| `ChunkGenerated` | Runner emits an output chunk (token, tool call, error) | API queue subscribers |
| `TracesCollected` | Runner emits trace events for one rank | Master (merges across ranks) |
| `TracesMerged` | Master merges + persists a complete trace | API (writes to disk) |
| `TracingStateChanged` | Cluster tracing toggle changes | All nodes |

Apply function: `src/exo/shared/apply.py::apply` — pure `(State, IndexedEvent) -> State`.

## Commands

Two distinct command unions on two distinct topics:

### COMMANDS topic — `Command` union

Discriminated union at `src/exo/shared/types/commands.py`. Carried as `ForwarderCommand` over the `COMMANDS` pubsub topic.

| Command | What it requests | Master action |
|---|---|---|
| `PlaceInstance` | Spin up a model on the cluster | Pick ranks based on memory + topology; emit `InstanceCreated` |
| `DeleteInstance` | Tear down a placed model | Emit `InstanceDeleted`; workers tear down runners |
| `TaskFinished` | Mark a streaming task complete (sent by API on stream end) | Emit `TaskDeleted` (`TaskStatusUpdated(Complete)` is emitted earlier on the chunk path, not from `TaskFinished` directly) |
| `TaskCancelled` | Cancel an in-flight command (sent by API on `/v1/cancel`) | Emit `TaskStatusUpdated(Cancelled)` |
| `SetTracingEnabled` | Cluster-wide tracing toggle | Emit `TracingStateChanged` |
| `AddCustomModelCard` | User-added model card | Emit `CustomModelCardAdded`; nodes persist locally |
| `DeleteCustomModelCard` | Remove user card | Emit `CustomModelCardDeleted` |

### DOWNLOAD_COMMANDS topic — `DownloadCommand` union

Discriminated union at `src/exo/shared/types/commands.py`. Carried as `ForwarderDownloadCommand` over the `DOWNLOAD_COMMANDS` pubsub topic. Used for cluster-wide config sync and model-store coordination — separated from the main command channel because these are typically larger payloads and have different retry semantics.

| Command | What it requests |
|---|---|
| `SyncConfig` | Broadcast cluster config (`auth.api_keys` and `hf_token` stripped); followers observe and persist locally |
| Model store ops | Download / staging coordination commands (see `src/exo/store/`) |

### Tasks (not commands)

Note `CancelTask` is a **task** (`src/exo/shared/types/tasks.py`), not a command. Tasks are work units the runner executes; commands are imperative requests to the master. Cooperative task cancellation is implemented as a `CancelTask` task delivered to the runner over the `mp.Queue`.

## API endpoints

Lives in `src/exo/api/main.py` (route registration in `API.__init__`).

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
| `/place_instance` | POST | Place a model: master picks ranks. Takes `PlaceInstanceParams` (model id + placement preferences); not interchangeable with `/instance`, which takes a fully-specified `CreateInstanceParams`. |
| `/instance/{instance_id}` | GET / DELETE | Fetch / delete an instance |
| `/instance/placement` | GET | Compute placement preview |
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

`src/exo/shared/types/tasks.py`. Discriminated union of:

- `TextGeneration` — chat / responses / messages / ollama-chat
- `TextEmbedding` — embeddings
- `ImageGeneration` — images.generations
- `ImageEdits` — images.edits
- Sentinel: `Shutdown`, `CANCEL_ALL_TASKS`

### Chunks

`src/exo/shared/types/chunks.py`. Per-token output:

- `TokenChunk` — text / tool / token-level metadata
- `ToolCallChunk` — tool calls
- `ErrorChunk` — error result; terminal
- `PrefillProgressChunk` — distributed prefill progress
- `ImageChunk` — image generation output
- `EmbeddingChunk` — embedding output

### State

`src/exo/shared/types/state.py`. Treated as immutable by convention (replaced wholesale by `apply()` rather than mutated in place); the model itself is not declared `frozen=True` on `model_config`, so direct mutation is technically possible but considered a bug at every call site.

- `instances: Mapping[InstanceId, Instance]` — placed model instances (each carries shard assignments + per-runner state)
- `runners: Mapping[RunnerId, RunnerStatus]` — per-runner status union
- `downloads: Mapping[NodeId, Sequence[DownloadProgress]]` — in-flight model downloads per node
- `tasks: Mapping[TaskId, Task]` — in-flight or recently-completed tasks
- `last_seen: Mapping[NodeId, datetime]` — peer liveness timestamps
- `topology: Topology` — cluster-wide node graph + capabilities (encoded/decoded via `TopologySnapshot` for JSON round-tripping)
- `tracing_enabled: bool` — cluster-wide tracing flag
- `last_event_applied_idx: int` — water mark for the local apply
- `node_identities`, `node_memory`, `node_disk`, `node_system`, `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge`, `node_rdma_ctl: Mapping[NodeId, *]` — granular per-node telemetry that updates at independent frequencies
- `thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]]` — detected Thunderbolt-bridge cycles where every node has it enabled (>2 nodes)

Note: there is no `master_node_id` field on `State`. Master identity lives outside the event-sourced state — each node tracks the current master independently via the election protocol (`src/exo/shared/election.py`). `placements` is also not a field; placement information is derived from `instances` (each `Instance` has its own shard assignments).

### Diagnostics

`src/exo/shared/types/diagnostics.py`. Major models:

- `NodeDiagnostics` — runtime + identity + resources + processes + supervisor_runners + placements + warnings
- `RunnerSupervisorDiagnostics` — flight_recorder, status, phase, MLX memory, in_progress_tasks, milestones
- `RunnerFlightRecorderEntry` — at, phase, event, detail, attrs, context, mlxMemory
- `MlxMemorySnapshot` — active, cache, peak, wired_limit
- `ClusterDiagnostics` — fan-out wrapper
- `ClusterTimeline` — cross-rank merged: runners (synopsis) + timeline (entries sorted by `at`) + unreachableNodes
- `DiagnosticCaptureResponse` — capture bundle (process samples, flight recorder, MLX memory)

### Model card

`src/exo/shared/models/model_cards.py::ModelCard`. Fields:

- `model_id`, `family`, `quantization`, `base_model`, `n_layers`, `hidden_size`, `num_key_value_heads`
- `tasks: list[ModelTask]` — what task types this model serves
- `capabilities: list[str]` — text / vision / thinking / thinking_toggle / embedding
- `context_length`, `storage_size`, `supports_tensor`, `trust_remote_code`, `is_custom`
- `vision: VisionCardConfig | None` — image_token_id, model_type, BOI/EOI tokens
- `reasoning: ReasoningCardConfig | None` — supports_toggle, supports_budget, format, default_effort
- `modalities: ModalitiesCardConfig | None` — supports_native_multimodal, supports_audio_input
- `tooling: ToolingCardConfig | None` — tool_call_format, supports_tool_calling, builtin_tools
- `runtime: RuntimeCapabilityCardConfig | None` — prompt_renderer, output_parser, metal_fast_synch

### Capability profile

`src/exo/shared/models/capabilities.py::ResolvedCapabilityProfile`. Computed at request time from card + tokenizer + task params:

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

Family-specific in `src/exo/worker/engines/mlx/auto_parallel.py`. Each is a class implementing `TensorParallelShardingStrategy`. Dispatched at lines 830-905 via `isinstance(model, X)` chain (consolidation tracked under #130):

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

RotorQuant (block rotations + deferred quant) is research and lives in PR #103; it is not yet in the merged backend set. Verify the current valid values against `src/exo/worker/engines/mlx/constants.py`.

Selection logic: `src/exo/worker/engines/mlx/cache.py::make_kv_cache`. Some backends fall back to `default` for incompatible models (e.g., `optiq` for non-divisible head_dim).

## Configuration knobs

### `skulk.yaml`

| Section | Field | What |
|---|---|---|
| `model_store` | `enabled`, `host`, `port`, `path` | Shared model store config |
| `model_store.staging` | `enabled`, `node_cache_path`, `cleanup_on_deactivate` | Staging behavior |
| `inference` | `kv_cache_backend` | KV cache selection |
| `logging` | `enabled`, `ingest_url` | Centralized logging opt-in |
| `hf_token` | (string) | Local-only Hugging Face token (stripped from cluster broadcast) |

### Environment variables

`SKULK_*` is preferred; `EXO_*` accepted as legacy. Migration tracked under #110.

| Var | What |
|---|---|
| `SKULK_HOME` / `EXO_HOME` | Override `~/.skulk/` for cache + custom cards |
| `SKULK_FAST_SYNCH` / `EXO_FAST_SYNCH` | Force `MLX_METAL_FAST_SYNCH` on (`"on"`) or off (`"off"`); overrides per-model card |
| `SKULK_PIPELINE_EVAL_TIMEOUT_SECONDS` | Per-eval timeout in pipeline collectives (default 60s) |
| `SKULK_MLX_HANG_DEBUG` / `EXO_MLX_HANG_DEBUG` | Emit periodic stack traces from stuck phases |
| `SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS` | Interval for above (default 30s) |
| `SKULK_MAX_OUTPUT_TOKENS` / `EXO_MAX_TOKENS` | Default `max_tokens` (cluster default 4096; `DEFAULT_MAX_OUTPUT_TOKENS` constant) |
| `SKULK_NO_BATCH` / `EXO_NO_BATCH` | Disable continuous batching |
| `SKULK_KV_CACHE_BACKEND` / `EXO_KV_CACHE_BACKEND` | KV cache backend selection (overrides config) |
| `SKULK_LIBP2P_NAMESPACE` / `EXO_LIBP2P_NAMESPACE` | libp2p namespace for cluster isolation |
| `SKULK_SKIP_LLM_WARMUP` | Skip warmup synthesis (single-node debug only) |
| `SKULK_IMAGE_TRANSPORT_DEBUG` | Verbose logging in image-transport pipeline |
| `EXO_VISION_DEBUG_SAVE_DIR` | Save debug image artifacts |
| `EXO_NATIVE_VISION_REFERENCE_PATH` | Force native-vision reference path (Gemma 4) |
| `EXO_OFFLINE` | Run without internet checks (no model fetching) |
| `MLX_METAL_FAST_SYNCH` | Set by Skulk based on resolved card preference; not for direct operator use |
| `MLX_HOSTFILE`, `MLX_RANK`, `MLX_RING_VERBOSE`, `MLX_IBV_DEVICES`, `MLX_JACCL_COORDINATOR` | MLX upstream env vars; auto-set by Skulk during distributed init |

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

- **Lives at:** `src/exo/worker/runner/runner_supervisor.py` (the bounded buffer); emit helpers at `src/exo/worker/runner/diagnostics.py`
- **Capacity:** last 128 entries per runner
- **Always-on; local-only.** Not gossiped, but exposed via `/v1/diagnostics/*`
- **Emission helpers:**
  - `record_runner_phase(phase, event=..., detail=..., attrs=..., include_memory=False)` — fire one entry
  - `runner_phase(phase, detail=...)` — context manager: enter / exit pair

### Trace sessions

- **Lives at:** `src/exo/shared/tracing.py`
- **API:**
  - `begin_trace_session(task_id, rank, node_id, model_id, task_kind, tags)` — create
  - `record_trace_marker(name, rank, task_id, attrs)` — emit one event
  - `trace(category, name, ...)` — context manager / decorator
  - `pop_trace_session(task_id)` — collect + remove
  - `clear_trace_session(task_id)` — remove without collecting
- **Storage:** module-level dict `_trace_sessions: dict[str, TraceSession]`
- **Cluster path:** runner emits `TracesCollected` per rank → master merges to `TracesMerged` → API persists Chrome-trace JSON to disk

### MLX memory snapshot

- **Lives at:** `src/exo/worker/runner/diagnostics.py::capture_mlx_memory_snapshot`
- **Returns:** `MlxMemorySnapshot { active, cache, peak, wiredLimit, source }`
- **Best-effort:** returns None if MLX isn't loaded or the snapshot fails

### Process sampling (macOS only)

- **Lives at:** `src/exo/api/main.py::_collect_process_samples`
- **Wraps:** `sample <pid> <duration>`, `vmmap -summary <pid>`, `footprint -p <pid>`
- **Per-command timeout:** ~5-8s
- **Returns:** `list[DiagnosticProcessSample]` with `ok`, `stdout`, `stderr`, `error`

### Per-eval timeout

- **Lives at:** `src/exo/worker/engines/mlx/auto_parallel.py::eval_with_timeout`
- **Wraps:** any `mx.eval(...)` call with a daemon-thread watchdog
- **Default timeout:** 60s (`pipeline_eval_timeout_seconds()`, configurable via `SKULK_PIPELINE_EVAL_TIMEOUT_SECONDS`)
- **On timeout:** emits `pipeline_eval_timeout` flight-recorder event, then `os._exit(1)`
- **Used at:** every `mx.eval` in `PipelineFirstLayer`, `PipelineLastLayer`, `mx_barrier`

### Parent-pid watchdog

- **Lives at:** `src/exo/worker/runner/bootstrap.py::_install_parent_death_watchdog`
- **Mechanism:** daemon thread inside runner that polls `os.getppid()`; on reparenting, calls `mx.clear_cache()` + `gc.collect()` + `os._exit(1)`
- **Why:** SIGKILL of the agent leaves daemon `mp.Process` runners orphaned holding GPU memory. The watchdog detects the reparent and self-exits

## Centralized observability stack

Local Vector → VictoriaLogs → Grafana. Configuration:

- `src/exo/shared/logging.py` — loguru JSON sink to stdout
- `deployment/logging/vector.yaml` — Vector pipeline (stdin → VictoriaLogs)
- `deployment/logging/docker-compose.yml` — VictoriaLogs + Grafana stack
- `skulk.yaml` `logging.enabled` + `logging.ingest_url` — opt-in; cluster-synced

## File map quick reference

```
src/exo/
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
