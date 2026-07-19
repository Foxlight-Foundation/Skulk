---
id: architecture-reference
title: Architecture Reference
sidebar_position: 6
---

<!-- Copyright 2025 Foxlight Foundation -->

Dense per-symbol fact-sheet for AI assistants and operators who prefer reference style. For narrative and design rationale, see [Architecture](architecture). This document is intentionally terse; every entry has a file:line so you can jump to the code.

This file is intentionally dense. If you find a stale fact, fix it inline rather than working around it. The AGENTS.md "Documentation" section requires updates here when architectural shape changes.

## Components

### Master

- **Role:** elects + acts as cluster coordinator; indexes events; plans instance placements; publishes snapshots
- **Lives in:** `src/skulk/master/main.py`
- **Owns:** the authoritative event log (via `DiskEventLog`); the indexer that assigns monotonic indices to events; the placement planner. Master identity itself lives outside the master process: each node tracks the current master independently via the election protocol (`src/skulk/shared/election.py`); the `_master_node_id` cache is held on the API side at `src/skulk/api/main.py:461`.
- **Communicates via:** `LOCAL_EVENTS` (consumes), `GLOBAL_EVENTS` (publishes indexed events), `COMMANDS` (consumes), `STATE_SYNC_MESSAGES` (publishes snapshots)
- **Election:** `src/skulk/shared/election.py`; bully algorithm; a single master at a time
- **Formation robustness (#400):** a campaign is (re)started by a connection update only when the set of connected peers actually changes (`_apply_connection_updates` tracks `_connected_peers`), and an in-flight campaign is allowed to finish before the next one starts. Peers are multi-homed and libp2p pings/re-dials every few seconds (`PING_INTERVAL` in `rust/networking/src/discovery.rs`), so raw connection updates can arrive faster than `DEFAULT_ELECTION_TIMEOUT`; without this gating each update cancelled and restarted the campaign before it could elect a master, livelocking formation (worst at simultaneous multi-node cold start). Reducing the churn at its source (skip unreachable link-local dials, ping tuning) is a separate follow-up (#401).
- **Failover:** re-election picks a new master, which seeds its session from the node's prior replicated state (#273, `seed_state_for_new_session` in `src/skulk/shared/session_carryover.py`): **instances, downloads, node info maps, and tracing carry over**; in-flight tasks, runner statuses, topology, and liveness timestamps are deliberately dropped (tasks died with the old session's plumbing; runner processes are torn down by the worker re-creation; topology/liveness must come from live gossip, since a carried topology would keep a dead node's out-edges forever). Workers re-create runners for the carried instances through the ordinary plan loop, so placements survive a master restart with a model-reload-sized gap instead of a silent permanent 404. The election winner tears its own worker down and rebuilds it, which cancels its `RunnerSupervisor.run()` tasks; that teardown is **shielded from cancellation** (`runner_supervisor.py`) so each runner process is fully reaped (Metal reclaims its wired GPU memory on exit) before `worker.shutdown()` returns. Without the shield the join was cancelled, the old runner lingered holding its memory, and the rebuilt worker's pre-load memory guard saw the not-yet-reclaimed memory, falsely refused the re-creation, and the #290 re-place-wider path deleted the carried instance (the silent 404 this design exists to prevent). It only bit when the winner also hosted a rank of a carried instance and was memory-tight. The plan loop suppresses liveness-based instance pruning for `TOPOLOGY_SETTLE_GRACE_SECONDS` (60s) after master start so carried instances aren't deleted while topology is still rebuilding; instances whose ranks lived on the dead master are pruned after the grace. A freshly-booted node that wins election seeds empty (it has no prior view): identical to the pre-#273 behavior. The seed is indexed as **event 0 of the new session** (a logged `StateSnapshotHydrated`, `Master._index_seed_event`): late bootstrappers receive it inside the snapshot, early bootstrappers (including the promoted node's own worker, whose bootstrap races the promotion) receive it as the live first event: one delivery path, no idx-(-1) hydration skip.

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
  - `src/skulk/worker/runner/llm_inference/runner.py`: text generation
  - `src/skulk/worker/runner/embeddings/runner.py`: embeddings
  - `src/skulk/worker/runner/image_models/runner.py`: image generation
- **Communicates via:** `mp.Queue` from worker (incoming tasks); `mp.Queue` to worker (outgoing events); `mlx.distributed` collectives with peer runners

### Drafters (speculative decoding)

`src/skulk/worker/engines/mlx/drafters/`. The loop runs on single-node, tensor-parallel, AND pipeline placements. Multi-node PIPELINE placements use an EXPLICIT decider protocol (#254): exactly one rank (the decider, the last rank) holds the drafter, makes every speculative decision, and fans the outcomes out via fixed-shape per-round collectives: one `all_sum` lands the draft tokens (`_exchange_drafts`; the drafter's effective distribution rides along under sampling), and after the verify forward a second tiny `all_sum` lands the accept length and the next bonus token (`mtp_accept_decision`). The first sampled token of the request is broadcast the same way. Receiving ranks never draft, sample, or compare logits (they apply broadcast decisions to their own cache slices), so correctness never depends on cross-rank numerical determinism (heterogeneous chips, e.g. M5 vs M4 GEMM kernels and NAX reduced-precision B≥2 matmuls, produce divergent per-rank logits; the previous rank-symmetric design desynced on exactly that and SIGABRT'd in the Metal completion block, #252). A per-request `all_sum` agreement settles that exactly one rank holds a working drafter (speculation disables symmetrically otherwise); mid-request drafter failures abort loudly on multi-rank placements instead of silently forking the collective schedule. Assistant drafters (gemma4) cross-attend the target's KV, which the decider seat owns by construction (#201 Track 2b); sidecar drafters draft from the all-gathered trunk hidden from the same seat, and only the decider rank loads drafter weights. Multi-node TENSOR placements do NOT use the decider protocol (#263): draft logits go through the TP-sharded lm_head, an all-rank collective idle receivers would never join, so a lone TP decider GPU-times-out mid-draft. Instead every TP rank loads the sidecar (`sidecar_load_eligible` in `src/skulk/worker/engines/mlx/utils_mlx.py`, the same envelope assistants use) and drafts rank-symmetrically; the drafter agreement requires ready_count == group.size() on that path and disables speculation symmetrically on partial loads. Cross-attending drafters declare `reads_target_cache = True` and the loop keeps the target cache fully committed before every draft (no deferred replay); and they must hold the LIVE cache sequence, since reject-restores replace rotating entries in the loop's list. Forces `SequentialGenerator`. Greedy requests use argmax-prefix acceptance; temperature > 0 uses Leviathan-Chen probability-ratio acceptance over the effective sampler distributions (`src/skulk/worker/engines/mlx/generator/speculative_sampling.py`, depth forced to 1). Draft depth comes from the card's `mtp_max_depth` (default 1). Rounds are *bonus-driven*: the loop carries an emitted-but-unforwarded bonus token, verifies `[bonus, drafts]` in one K+1-token forward (the round's only target forward), commits the longest matching prefix, samples the next bonus from the first non-matching row, and drafts the very next round from the correction position.

- **Protocol:** `protocol.py::Drafter`: `begin_request(prompt_cache)` / `observe(hiddens, next_tokens)` / `draft(hidden, next_token, depth=1) -> (K, vocab) logits`. The generation loop owns verify/accept/reject and cache reconciliation, preferring the model's native `rollback_speculative_cache` (gemma4), else SSM snapshot/restore with *deferred replay* (restored-but-committed tokens ride at the front of the next verify forward; capped, flushed at stream end), else plain KV trim. Drafters own only their private state. The loop feeds every committed position's `(hidden, next token)` pair exactly once, in order (the pair-stream contract); the hidden convention is per-family (pre-final-norm for qwen-shaped trunks, post-norm for gemma4).
- **Builder:** `builder.py::build_drafter(model, mtp_weights, runtime)`: detects sidecar key layout, resolves family facts (norm convention, fc concat order) from layout-keyed defaults with model-card `runtime` overrides, and quantizes the sidecar block + fc to the target's `(group_size, bits)` on load (bf16 targets keep bf16 sidecars).
- **Implementations:**
  - `qwen_sidecar.py::QwenSidecarDrafter`: Phase 2: +1.0 zero-centered norm shift, `embed_first` concat, sidecar `mtp.layers.0` block instantiated from the target family's own decoder-layer class (strict-loaded), private `KVCache`. Validated 79–85% acceptance / 1.38–1.90x on Qwen3.5 9B–27B (issue #192, bonus-driven cadence).
  - `deepseek_sidecar.py::DeepseekSidecarDrafter`: legacy projection-only head; conventions unverified against real weights.
  - `gemma4_assistant.py::Gemma4AssistantDrafter`: wraps mlx-vlm's chain-trained assistant model: cross-attends over the target's KV (shared-KV extraction incl. RotatingKVCache temporal restore), consumes post-norm hiddens, loads via `assistant_model_repo` (bf16-enforced). Validated 84% acceptance / 35.1 tok/s on gemma-4-26B-A4B-4bit (depth 1) and 1.86x on E4B-8bit (depth 3).
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
- **Localization:** Tolgee provider in `dashboard-react/src/i18n/tolgee.ts`; app wrapper in `dashboard-react/src/main.tsx`; English namespace data in `dashboard-react/src/i18n/en/skulk.json`. All dashboard keys use the `skulk` namespace and are called through `t(key, englishFallback, params?)`, not `<T>`.
- **Translation loading:** `BackendFetch` reads CDN/static JSON from `VITE_TOLGEE_CDN_PREFIX` (default `/i18n`) with bundled English fallback; `VITE_TOLGEE_AVAILABLE_LANGUAGES` controls the comma-separated language allow/preload list and always includes `en`.

### Extensions (plugins)

- **Role:** load separately installed packages and call them at serving-path hooks; deployment-specific behavior without forking Skulk
- **Lives in:** `src/skulk/extensions/` (`types.py` contract, `loader.py` discovery + guarded dispatch); call sites in `API.chat_completions`
- **Discovery:** `skulk.extensions` entry-point group, scanned once at node startup (`load_extensions()` in `src/skulk/main.py`, API-spawning nodes only); entry point value = zero-arg factory returning a `SkulkExtension`
- **Contract:** `SkulkExtension` (name, `skulk_requires` PEP 440 specifier, `chat_middleware()`); `ChatMiddleware.transform_chat_request(context, task_params)` pre-dispatch + `ChatMiddleware.observe_chat_response(context, task_params, summary)` post-completion (background task, immutable `ChatResponseSummary`)
- **Context:** `ExtensionContext` = node_id + skulk_version + `embed_texts` (in-process equivalent of `POST /v1/embeddings`, backed by `API.embed_texts`; returns `None` when no embedding instance is placed) + telemetry-plane access (fabric-citizenship Phase 1, read + advertise):
  - `read_cluster()` (telemetry-plane READ surface): returns an immutable `tuple[ClusterNodeView, ...]` snapshotting `TelemetryView` per node: node_id, friendly_name, backends, participation, skulk_version, accelerator_vendor, ram_total_bytes, last_telemetry, capabilities; pure/in-memory, no mutation, fields `None`/empty until each reading arrives; `extensions/telemetry.py:snapshot_cluster`.
  - `advertise_capability(tag)` (telemetry-plane ADVERTISE surface, `API._advertise_capability`): records an opaque capability tag on `TelemetryView.local_advertised_capabilities` (the node's outbound set on the shared view). The worker's `InfoGatherer._monitor_capabilities` polls that set and gossips it as a `NodeCapabilities` (a `TaggedModel` in `TELEMETRY_PLANE_INFO`, `capabilities: frozenset[str]`); every node's `TelemetryView.apply()` coalesces it into `node_capabilities[node_id]`, surfacing in peers' `read_cluster().capabilities`. Additive + idempotent (blank tags ignored); published only while the set is non-empty (no gossip volume for the common no-capability node), EXCEPT one empty reading on the non-empty -> empty transition so peers clear the LWW entry after a withdrawal (a behavioral change only; the `NodeCapabilities` wire shape is unchanged). The `NodeCapabilities` variant itself is a wire type, so the same-version-fleet rule applies to it. A `--no-worker` node records the tag but does not gossip it (no gatherer).
  - `withdraw_capability(tag)` (`API._withdraw_capability`): liveness counterpart of advertise; discards the tag from the outbound set. Peers see the shrunken set on the next poll (or the single empty reading when the last tag goes). No-op for unknown tags.
  - `describe_node(node_id)` (heavy discovery half, `API._describe_node_capabilities`): local node reads `LoadedExtensions.capability_descriptors`; a peer is proxied via `GET /v1/capabilities` over `_reachable_peer_api_urls()` (the same peer-API browse the trace cluster endpoints use; the extension contract is transport-abstract so this can later ride the provider call plane). Returns `()` on unreachable/invalid payloads, never raises.
- **Provider facet (fabric-citizenship Phase 2a):** an extension implementing `capabilities() -> Sequence[CapabilityDescriptor]` is a provider (structural `@runtime_checkable` check `CapabilityProvider` in `extensions/types.py`; collected guarded in `LoadedExtensions.__init__`; duplicate `id@version` per node rejected loudly). `CapabilityDescriptor` (`extensions/capabilities.py`, frozen/strict, MCP-tool-aligned shape without MCP's JSON-RPC): `id` (lowercase pattern, doubles as the telemetry tag and is auto-advertised at API construction), `version` (semver; `qualified_id` = `id@version`), `title`, `description` (LLM-readable), `input_schema`/`output_schema` (JSON Schema dicts), `io_mode` (`unary`/`server_streaming`/`client_streaming`/`bidirectional`; chunk schemas required for the streaming modes, forbidden for unary), `annotations`. `descriptor_revision(d)` = 16-hex sha256 of the canonical JSON dump; calls will carry it so discovery and invocation cannot silently disagree. `on_start(context)` (`SupportsExtensionStartup`, dispatched guarded by `LoadedExtensions.run_startup_hooks` at API construction) is the startup hook: a pure provider has no chat hook through which to reach the context, so registration must not depend on a chat request. Endpoint: `GET /v1/capabilities` (`API.list_node_capabilities`) returns `{node_id, capabilities, revisions}`. Reference provider: `examples/extensions/echo-provider/` (not installed by default; serves `echo@1.0.0` calls).
- **Capability call (Phase 2b):** `ExtensionContext.call_capability(node, id, version, revision, payload, timeout_seconds=None)` -> `CapabilityResult` (typed; never raises). Envelope types in `extensions/calls.py`: `CapabilityCall` (call_id, capability_id, version, descriptor_revision, caller/target node, timeout_seconds capped at 300, payload), `CapabilityResult` (ok + result | typed `CapabilityError`), error codes `not_found / version_mismatch / revision_mismatch / invalid_payload / invalid_result / payload_too_large / overloaded / timeout / provider_error / unreachable`. Provider side: a provider implementing `CapabilityCallHandler.handle_call` is callable; registry `LoadedExtensions.call_handler(qualified_id)`; a descriptor without a handler is discovery-only. Dispatch chokepoint `API._dispatch_capability_call` guards in order: target-node addressing check -> handler lookup -> revision pin (`descriptor_revision`) -> then INSIDE the per-node concurrency bound (`_MAX_CONCURRENT_CAPABILITY_CALLS` = 8, excess -> `overloaded`) and the caller deadline (`anyio.fail_after` -> `timeout`): 1 MiB payload cap, input-schema validation (`extensions/validation.py`, JSON Schema 2020-12, NO remote $ref fetch), the handler (exception -> `provider_error`); then result-shape checks + result cap + output-schema validation -> `invalid_result`. Serialization/validation work counts against the bound and deadline (#513) so a storm of large-but-invalid payloads cannot drive unbounded concurrent validation; cheap guards stay outside so trivially-rejectable calls never consume a slot. Endpoint `POST /v1/capabilities/call` (always HTTP 200 with a typed result). Caller side `API._call_capability`: local target = in-process fast path (same guards); peer = direct peer-API POST with ONE budget clock spanning target resolution + the HTTP hop (#513): the lookup (`_peer_api_url_for`, first-hit, deadline-bounded) consumes from the same budget the provider gets, the envelope carries the REMAINING budget, and the HTTP deadline is remaining + 5s so the provider's typed timeout wins; a lookup that exhausts the budget returns typed `timeout`. Master never in the hot path; nothing event-sourced. Transport-abstract: the peer hop can move to a Zenoh queryable (needs queryable support in `rust/networking/src/zenoh_session.rs`, currently pub/sub-only) without changing the contract (#510 transport note). Handlers are async; blocking/CPU-heavy work must be moved off the event loop by the extension. New dependency: `jsonschema`.
- **Provider streaming (Phase 3):** `extensions/streams.py` is the transport-independent contract. `CapabilityStreamFrame` keys one logical stream by `(call_id, direction)`, uses shared `started/chunk/completed/failed/cancelled` kinds, and carries structured payload, optional raw `InlineMediaAttachment` (1 MiB/frame cap), optional staged `BlobMediaAttachment`, or typed `CapabilityStreamError`. `CapabilityStreamReceiver` enforces bounded ordering, one terminal, duplicate idempotency, late-frame rejection, five-second gap expiry, call/idle deadlines, and synthesized failures independently per direction. `CapabilityStreamHandler` serves `server_streaming`; `CapabilityInputStreamHandler.handle_input_stream(context, call, input_frames)` serves `client_streaming` and `bidirectional`. Skulk emits output sequence-zero `started`, validates exact identity/sequence and `output_chunk_schema` (or `output_schema` on a client-streaming completed result), and requires one terminal followed by iterator exhaustion. The terminal is withheld until the handler returns so its `finally` cleanup completes before dependent work can observe success; malformed or trailing output closes a closable iterator before the synthetic failure terminal is published. `ExtensionContext.stream_capability(...)` returns `CapabilityStreamSession(open_result, frames, input)`; `input` is `None` for server streaming, otherwise a `CapabilityStreamInput` sink whose `send_chunk` owns caller sequencing and whose `complete` is input half-close, not cancellation. Local/remote opening uses `POST /v1/capabilities/stream` as a control-sized admission request; both media directions use the distinct `PROVIDER_DATA` topic (not the `DataChunk` union), encoded by `routing/provider_streams.py` as a four-byte JSON-header length + canonical header + optional raw bytes. The receiving node is the owner key: caller for output, provider for input. The Zenoh DATA scheduler uses independent bounded queues keyed by `(owner, call_id, direction)`, with same-node short circuit, owner/process admission caps, publish deadline, typed saturation rejection, and a renewed-on-frame 30-minute idle resource lease. Lease expiry tombstones the stream, releases admission, and best-effort publishes a next-sequence transport failure. Early output-iterator close calls `POST /v1/capabilities/stream/cancel`; caller input `cancel` rides DATA and terminates the same logical call. No final acknowledgement in v1; master/State/event log are absent from the path.
- **Dynamic stream admission:** `CapabilityStreamAdmissionHandler.admit_stream(context, call)` runs after static input-schema validation, inside the stream concurrency/deadline budget, and before lifecycle creation. It returns `CapabilityError | None`; a typed rejection emits no `started` frame. Use it for live model/backend availability, not static schema rules.
- **Built-in TTS provider (Phase 4 first consumer):** production `API` construction prepends `BuiltinSpeechProvider` (`extensions/speech.py`) to the guarded registry, reserving `tts@1.0.0` ahead of external duplicates. Its descriptor is `server_streaming`, MP3-only in v1, and maps generic `model` + `text` input to the existing core `SpeechSynthesisTaskParams`; core model cards/store/mounting/placement/runner inference stay authoritative. `AudioChunk` base64 is decoded at the facade and emitted as raw `InlineMediaAttachment` bytes plus schema-validated model/format/index/partial/sample-rate metadata. The descriptor is always described, while `API._sync_builtin_speech_capability` advertises the `tts` telemetry tag whenever an MP3-capable mounted TTS card with `supports_streaming=true` is ready, re-evaluating on instance create/delete/snapshot hydration/config update. Admission revalidates the requested model before `started`; provider cancellation/deadline/transport failure cancels the core command and finalizes its DATA queue. The underlying core command still uses normal master placement/task routing; the generic provider opening and output lifecycle remain node-addressed/off-State.
- **Reference-audio TTS:** `POST /v1/audio/speech` accepts either its JSON form or a multipart form whose optional `reference_audio` upload is capped at 25 MiB. A supporting card must declare `audio.supports_reference_audio=true`. The API selects a ready single-host instance, pins `SpeechSynthesis.target_instance_id`, and sends only filename/content-type/digest metadata through the command path. Raw bytes travel as ordered `SpeechMediaPacket` chunks plus a terminal SHA-256 over `SPEECH_MEDIA`; reference uploads require Zenoh because gossipsub broadcast is forbidden for private media. The target worker keeps bounded, expiring command-keyed buffers outside State, verifies contiguous sequence and digest, and injects bytes only into its local runner task. The runner creates a temporary file only while upstream generation executes and removes it in `finally`. Dispatch, cancellation, malformed input, transport failure, task completion, and expiry clear retained buffers.
- **Built-in batch STT provider:** `BuiltinSpeechProvider` reserves `stt@1.0.0`, a `client_streaming` binary-input/batch-output contract over the authoritative `AudioTranscription` command and speech runner. The transport mode is intentionally not unary: encoded audio travels as ordered raw `InlineMediaAttachment` frames (1 MiB each, 25 MiB aggregate), caller `complete()` half-closes input, and inference then returns one schema-validated completed payload with model/text plus optional language/segments. The `stt` telemetry tag requires a ready single-host mounted STT runner; admission revalidates the requested model and metadata. The provider does not advertise progressive output or managed blobs. Provider ingress stays off State, and the shared core batch path retains raw audio only until authoritative placement before sending bounded `SPEECH_MEDIA` frames directly to the selected worker. The worker verifies owner, count, and digest before runner dispatch; no audio payload enters the ordered event log.
- **Built-in realtime STT provider (Phase 4 second consumer):** `BuiltinSpeechProvider` reserves `stt.realtime@1.0.0`, a `bidirectional` mono PCM16-to-transcript contract. It is described unconditionally but advertised/admitted only with reachable mounted STT capacity declaring both `supports_streaming=true` and `supports_realtime=true`. Admission pins `RealtimeAudioTranscription` to one ready single-host instance; the master creates the task without re-placement and reserves the instance against cross-API admission races. Transcript output follows normal owner-addressed DATA lifecycle, while caller PCM travels as bounded binary `RealtimeAudioPacket` values on `REALTIME_AUDIO` and never enters State or the event log. Same-node ingress short-circuits in the router; remote ingress requires Zenoh and is not advertised on the gossipsub fallback. Transport rejection is routed back to the source API and fails only the affected command. A bounded multiprocessing channel carries worker input to the speech runner. The worker permits bounded pre-dispatch buffering, cancels overflow, and retains bounded finished-command tombstones so late frames cannot accumulate. The runner requires upstream `create_streaming_session`, linearly resamples mono PCM16 to the session input rate, emits progressive `TranscriptionChunk` deltas, half-closes and drains on caller completion, and cancels promptly. After core output terminates, the provider sends `TaskFinished` and withholds its terminal until replicated task state is terminal or deleted; loss of that control-plane acknowledgement fails within a bounded deadline instead of releasing a next turn into stale busy admission. `api/realtime.py` adds `WS /v1/realtime`: same-origin browser or origin-less SDK clients send bounded OpenAI-style base64 PCM16 append/commit events; decoded 24 kHz mono bytes become raw provider frames, and provider output becomes delta/completed/failed events. The edge caps transcript text at 1 MiB per event and in its pre-commit buffer, emitting a typed failure and `1011` close on overflow. Optional typed server VAD incrementally resamples input to 16 kHz WebRTC frames, emits speech-start/speech-stop events, and auto-commits on silence or maximum duration. One socket serializes multiple utterances as distinct provider calls, rotates linked item IDs, resets VAD per turn, and rejects overlap while a committed turn drains. Optional response configuration sends final transcripts through a mounted chat model under a strict 1-4096 output-token ceiling (256 by default), with hidden reasoning disabled by default for speech-ready output, and optional mounted `tts@1.0.0` provider, exposing visible text and MP3 audio events; explicit cancellation and VAD barge-in cancel underlying commands. It has no noise reduction or tool execution. Cancellation on disconnect closes active Fabric/model work. Hypercorn caps client messages at 2 MiB and pings every 20 seconds. Dashboard chat requires both card-level realtime truth and the local API's live `stt.realtime` tag, captures microphone Float32 through an AudioWorklet, continuously resamples it to 24 kHz PCM16, bounds browser socket buffering, and retains the batch MediaRecorder path as fallback.
- **Fabric speech composition surface:** `WS /v1/fabric/chains/speech?stt_model=...` reuses the hardened `RealtimeTranscriptionBridge` rather than creating a parallel orchestration runtime. Its typed `session.update` selects server VAD plus optional mounted chat, TTS, and voice participants and optional bounded output-token and thinking controls. It therefore inherits realtime provider admission, remote data-plane routing, bounded conversation text, off-State media, cancellation, barge-in, and terminal guarantees from `/v1/realtime` while exposing the composition under an explicit Fabric endpoint.
- **Built-in VAD provider:** production `API` construction also prepends `BuiltinVadProvider` (`extensions/vad.py`) and always advertises stable `vad@1.0.0`. The bidirectional contract accepts bounded inline mono PCM16 at WebRTC-supported 8/16/32/48 kHz, re-frames arbitrary input chunks into exact 10/20/30 ms classifier windows, and emits typed `speech_started`/`speech_stopped` chunks. `VoiceActivityDetector` owns per-call minimum-speech, silence-hangover, preroll, and maximum-utterance state; callers may manually half-close, partial terminal PCM is rejected, no model is mounted, and media bytes are never retained.
- **Invariants:** guarded dispatch (a raising extension is logged and skipped, inference never degrades); extensions never own the chunk stream (Skulk accumulates, observers get a summary); no external extension installed = external hooks inert (first-party facades are explicitly registered core adapters)
- **Version gating:** extension refused at load when `skulk_requires` does not match the running version (mixed plugin/fabric versions = mixed-version-cluster anti-pattern)
- **Kill switch:** `SKULK_EXTENSIONS_DISABLE=1` skips discovery (node-local)

### Node Facts (capability derivation, #614)

- **Role:** one probe pass per process gathers what the node observed about itself; a pure derivation turns it into the advertised backend tags plus loud conflicts. Principle inversion: detection creates capability, configuration overrides it, disagreement is always loud (#609/#612/#462 made structural).
- **Lives in:** `src/skulk/facts/` (`probe.py` observation, `derive.py` derivation, `testing.py` synthetic-facts helpers); types in `src/skulk/shared/types/node_facts.py`
- **Probe:** `NodeFacts` records ALL observed GPUs across the supported vendor vocabulary (`GpuVendor = nvidia | amd | apple`) with `detection_source` (`nvml` full NVIDIA path / `nvidia_device_node` device visible but NVML unusable, the #612 degraded state / `amdgpu_sysfs` / `apple_platform`), importable deps (`pynvml`, `llama_cpp` + `llama_supports_gpu_offload()`, `mlx_audio`), engine binary states for `SKULK_LLAMA_SERVER_BIN` / `SKULK_VLLM_BIN` / `SKULK_RPC_SERVER_BIN` (`not_configured`/`ok`/`missing`/`not_executable`), the `llama-server --list-devices` report, and the raw `SKULK_*_BACKENDS` declarations verbatim. Cached once per process (`current_node_facts` / `refresh_node_facts`).
- **Derivation:** `derive_node_backends(facts) -> BackendDerivation` (pure). Precedence per engine: operator declaration > the binary's own `--list-devices` device list > hardware vendor inference (NVIDIA implies `cuda`; AMD implies `vulkan` for served engines, `rocm` for vLLM) > CPU floor. One exception to declaration-wins: a declared GPU backend for an in-process `llama_cpp` build that positively reports no GPU offload is dropped (advertises `llama_cpp-cpu`) so GPU GGUF work is not routed to a degraded build.
- **Conflicts:** `CapabilityConflict` codes: `gpu_serving_disabled` (GPU visible, all serving would run on CPU; error severity, the #609 silent `-ngl 0` class), `gpu_detection_degraded` (NVIDIA device visible but not fully readable; warn, #612), `invalid_engine_binary` (binary override unusable; warn, #462), `backend_override_conflict` (declaration claims unobserved hardware; honored but loud; warn). Severity source of truth: `CONFLICT_ERROR_CODES` in `node_facts.py`, shared with `node_health`.
- **Integration:** `probe_node_backends()` in `src/skulk/shared/backends.py` is now a thin cached delegate into this package. `NodeResources.capability_conflicts` carries the conflicts on existing `TELEMETRY` into `compute_node_health` (four matching `HealthCode` values) and the dashboard topology badges.

### Doctor (`skulk doctor`)

- **Role:** the node environment contract, executable: on-demand audit plus safe idempotent remediation over the same `NodeFacts` snapshot the capability pipeline uses
- **Lives in:** `src/skulk/doctor/` (`checks.py` registry, `cli.py`); subcommand dispatch in `src/skulk/main.py` (before node launch wiring)
- **Checks:** `engine-available`, `capability-conflicts`, `models-storage` (mirrors node_health's 10/2 GB disk thresholds), `dashboard-assets`. Verdicts `ok`/`degraded`/`fail`; every non-OK `CheckResult` states consequence + remediation. A crashing check degrades into a `fail` verdict rather than killing the audit.
- **`--fix`:** provisions the pinned llama-server build (Linux, no engine present), installs `nvidia-ml-py` when NVIDIA detection is degraded by its absence, creates the models directory. `--json` for machine output. Exit codes: 0 ok, 2 degraded-only, 1 any fail.
- **Docs generation:** the check list in `website/docs/node-doctor.md` is GENERATED from `REGISTRY` by `scripts/generate_doctor_docs.py`; edit the registry, not the page.

### Engine provisioning (managed llama-server, #614 Phase 3)

- **Role:** pinned, checksum-verified upstream llama-server builds so a fresh Linux node serves GGUF without building llama.cpp
- **Lives in:** `src/skulk/provisioning/` (`manifest.py` pins upstream llama.cpp release `b10068` with a sha256 per `(arch, variant)`; `llama_server.py` downloads, verifies, extracts, and wires)
- **Behavior:** auto-runs at Linux node startup when `SKULK_LLAMA_SERVER_BIN` is unset; installs under `SKULK_ENGINES_DIR` (default `~/.local/share/skulk/engines`) and exports `SKULK_LLAMA_SERVER_BIN` for the process (runner subprocesses inherit it; the facts probe then validates the managed binary like any other). Managed-source priority: pip-installed engine wheels outrank tarball provisioning and work offline once installed. NVIDIA tries `skulk-llama-server-cuda` (Foxlight-built binary + NVIDIA's official runtime wheels as deps) then `skulk-llama-server-vulkan`; AMD uses `skulk-llama-server-vulkan` (bundles the Apache-2.0 Khronos loader; the driver ICD, e.g. mesa-vulkan-drivers, remains the OS prerequisite). Each wheel's `llama-server-*` shim wires the loader path and execs, its `ggml-rpc-server-*` sibling shim is exported as `SKULK_RPC_SERVER_BIN` for multi-node donors, and a wheel whose version does not package `LLAMA_SERVER_PIN` (scheme `0.<build>.<rev>`) is ignored with a warning. Both wheels build from pinned upstream SOURCE in `.github/workflows/engine-wheel.yml` with sigstore build-provenance attestations (`gh attestation verify <wheel> --owner Foxlight-Foundation`). Source of truth is the Foxlight PEP 503 index on Cloudflare R2 (`wheels.foxlight.foundation`, published by `scripts/publish_wheel_index.py`; the CUDA wheel's ~450MB exceeds PyPI's per-file limit); the Vulkan wheel is additionally mirrored to PyPI (trusted publishing, skip-existing). The guard step keeps manifest/packages/installer in lockstep. Variant selection for tarballs (`select_variant_chain`): NVIDIA tries `cuda` then `vulkan`. The `cuda` tarball manifest slot is empty because upstream publishes NO Linux CUDA prebuilt; the CUDA channel is the Foxlight wheel (above), so the Vulkan tarball is only the fallback when no wheel is installed (works on bare metal but not on container clouds with compute-only driver stacks, where vLLM remains the first-class CUDA serving path). AMD uses `vulkan` (fleet-proven RADV); no GPU uses `cpu`. Explicit overrides always win; an INVALID override is never masked by a managed binary (stays a loud `invalid_engine_binary` conflict). Opt-out: `SKULK_NO_ENGINE_AUTOPROVISION=1`. Provisioning failure logs a warning and the node starts anyway (a node must start without network). macOS provisions nothing (in-process MLX owns Apple GPUs).
- **Installer:** `install.sh` at the repo root (#614 Phase 4): prerequisites (git/toolchain/rustup/uv) -> clone -> `uv sync` -> dashboard build when npm exists -> `skulk doctor --fix`. `--with-vllm` (NVIDIA Linux) creates a dedicated venv with `vllm==0.11.0` + `transformers<5` (`--torch-backend=cu128`; Skulk itself pins `transformers>=5`) and records `SKULK_VLLM_BIN` in `~/.skulk/skulk.env`. Idempotent on re-run. Related hard-dep change: `nvidia-ml-py` is a required dependency on `sys_platform == 'linux'` in `pyproject.toml` (nothing optional may be load-bearing for detection).

### Storage

- **Event log:** `src/skulk/utils/disk_event_log.py`: append-only length-prefixed msgpack records (`events.bin`, uncompressed live); rotated archives are zstd-compressed (`events.*.bin.zst`) on rotation/close. Disk is treated as bounded: archives are capped by count (5) AND total bytes (1 GiB); any persistence failure (ENOSPC at init, append, or compaction) drops the log into a degraded counting-only mode with one CRITICAL line (indices keep advancing so follower replay coherence survives), and a proactive free-space floor (2 GiB, checked every 1024 appends) degrades BEFORE the disk hits zero. The API-side log (`event_log/api/`, backs `GET /events` diagnostics only and records the replicated durable control stream) additionally ring-compacts: past 256 MiB of active file it keeps only the most recent 20k events.
- **Model cache:** `SKULK_MODELS_DIR` (default `SKULK_DATA_HOME/models`; on Linux that's `~/.local/share/skulk/models` via XDG, on macOS/Windows it's `~/.skulk/models`); `SKULK_HOME` and `SKULK_MODELS_DIR` env overrides apply
- **Custom cards:** `SKULK_CUSTOM_MODEL_CARDS_DIR` (default `SKULK_DATA_HOME/custom_model_cards`) as TOML
- **Built-in cards:** `resources/inference_model_cards/` as TOML
- **Optional model store:** shared host with rsync-style staging (`src/skulk/store/`)

## Pubsub topics

Defined in `src/skulk/routing/topics.py`.

| Topic | Wire payload type | Inner payload | Publisher | Consumer |
|---|---|---|---|---|
| `GLOBAL_EVENTS` | `GlobalForwarderEvent` | indexed `Event` (post-master indexing) | Master | All nodes |
| `LOCAL_EVENTS` | `LocalForwarderEvent` | un-indexed `Event` | Workers (via `event_router.py`) | Master |
| `COMMANDS` | `ForwarderCommand` | `Command` (`PlaceInstance`, `DeleteInstance`, `RefuseInstancePlacement`, `TaskFinished`, `SetTracingEnabled`, etc.) | API, Worker (`RefuseInstancePlacement`) | Master (command processor); Election (every node, observing commands to inform leader-changeover decisions) |
| `DOWNLOAD_COMMANDS` | `ForwarderDownloadCommand` | `DownloadCommand` (`StartDownload`, `DeleteDownload`, `CancelDownload`, `SyncConfig`, `PurgeStagingCache`, `RestartNode`) | API (download/restart/sync admin ops), Master, Workers | All nodes |
| `STATE_SYNC_MESSAGES` | `StateSyncMessage` | bidirectional: followers publish `kind="request"` for snapshot/config bootstrap; master publishes `kind="response"` with the requested payload (`StateSnapshotHydrated` etc.) | All nodes (request: followers; response: master) | All nodes |
| `ELECTION_MESSAGES` | `ElectionMessage` | bully election rounds on dedicated Python egress and a dedicated gossipsub behavior/protocol and handler queue; deduplicated legacy copy during migration | All nodes | All nodes |
| `CONNECTION_MESSAGES` | libp2p connection updates | peer arrivals / departures | Router | All nodes |
| `TELEMETRY` | `NodeTelemetry` | `GatheredInfo` plus non-terminal `DownloadPending` / `DownloadOngoing`; bounded latest-value admission and an isolated gossipsub protocol | Workers | All nodes (applied into `TelemetryView`) |
| `DATA` | `DataChunk` | `{command_id, kind, chunk?, sequence, owner_node}`: explicit `started/chunk/completed/failed/cancelled` lifecycle for token, image, embedding, transcription, and audio output | One serving output worker: rank 0 for text/embedding/speech, primary terminal stage for image generation | Owning API node only on Zenoh; API nodes on gossipsub; master does NOT consume it |

### Telemetry plane (#279)

`TELEMETRY` carries readings that are **last-write-wins and not decisions** instead of event-sourcing them into `State`. Local producers offer without blocking into a 256-key `OrderedDict`: an existing node/reading key is replaced, and only a full map of distinct keys evicts its oldest entry. Download progress keys also include model ID. One serialized packet can wait beyond the in-flight publish. A dedicated Python egress loop feeds a second Rust gossipsub behavior negotiated under `/skulk/telemetry/meshsub`, with independent per-peer handler queues from ordinary control and election traffic. `GET /v1/diagnostics/telemetry` exposes aggregate capacities, depths, coalescing, drops, publish failures/bytes, and pending age without retaining payloads or completed identifiers.

**Performance envelopes (observe-only; adaptive concurrency, Phase 0).** The API node records one observation per completed generation into an in-memory `PerformanceEnvelopeRegistry` (`src/skulk/api/performance_envelope.py`), keyed by `(hardware class, model, engine+backend, quantization)` and bucketed by the in-flight concurrency the serving instance was handling when the generation began. Each bucket keeps a bounded reservoir of time-to-first-token and steady-state decode-rate samples; the read side computes per-bucket p50/p90 TTFT, mean/p50 decode tokens/second, aggregate decode throughput, and a simple knee (the concurrency where aggregate throughput peaks). A stream tap (`API._tap_performance_envelope`, a guarded passthrough alongside the field-telemetry tap) feeds it, and it wraps **every** text-generation surface through one shared helper (`API._tapped_text_stream`: chat completions, the Claude/Responses adapters, the Ollama endpoints, realtime assistant turns), not only `/v1/chat/completions`. Concurrency attribution is per serving instance and comes from the runner itself: the **served engines** (`llama_server`, vLLM) stamp their own serving node, resolved backend, in-flight-at-admission count, and batching mode onto the terminal `GenerationStats` (`ServedConcurrentDispatch.stamp_runner_stats`), and the **in-process MLX runner** does the same (`Runner._stamp_stats` in `worker/runner/llm_inference/`), reporting `serving_batches=True` and its batch position at admission when running the batch generator (`BatchGenerator`), serial when running the `SequentialGenerator`. So the observation is correct across replicas and when several API nodes drive one instance, and reflects the true batch width regardless of engine. MLX is a batching engine on the batch path, not serial, which is why it must report rather than let the API guess. Only when a runner reports no serving node (a stats-less terminal, or the pre-gossip window before the shard's `resolved_backend` is stamped) does the tap fall back to this API node's outstanding-request count (`_resolve_envelope_context`, which records only when exactly one instance serves the model and skips served backends it cannot classify without the stamp). Hardware class comes from the serving node's `AcceleratorMetrics` + RAM tier via `hardware_class()`; quantization from the model card. It is exposed only through `GET /v1/diagnostics/performance-envelopes` (local) and `/v1/diagnostics/performance-envelopes/cluster` (fan-out), rendered by the dashboard Performance tab. It changes no serving behavior and never touches `State`, the event log, or the telemetry gossip plane; the schema mirrors the harness `concurrent` suite fields so offline and online curves compose. The runner-reported serving fields on `GenerationStats` ride the DATA plane, so this is a same-version-fleet wire addition.

Slice 1 moved `node_resources` (a node's `participation` role, `backends`, and resolved `data_transport`); slice 2 moved `node_memory` and `node_system` (the highest-volume readings, carried together by `MactopMetrics`/`MacmonMetrics`); slice 3 moved the observational readings `node_identities`, `node_disk`, and `node_rdma_ctl`. The set that rides telemetry is `TELEMETRY_PLANE_INFO` in `shared/types/telemetry.py`; the worker forwards those `GatheredInfo` variants to the telemetry sender (`worker/main.py`), and `apply_node_gathered_info` treats them as no-ops (`shared/apply.py`). The set includes the payload-free `NodeHeartbeat`, published every two seconds as the named primary liveness reading. `TelemetryView.node_last_heartbeat` stores its local receipt time separately from `node_last_telemetry`, which stores ordinary telemetry receipt as a fallback. The `TelemetryView` survives master re-election (Node-owned), so a freshly promoted master does not start blind, and these readings are no longer carried in the failover seed (`session_carryover.py`). `GET /state` merges them back in from the view (`API.get_cluster_state`) so the dashboard's wire shape is unchanged. `nodeResources.dataTransport` is the router's startup-resolved `gossipsub`/`zenoh` choice, injected into both initial and election-recreated workers rather than re-derived from environment state. A `--no-worker` node has no gatherer but still owns DATA receivers, so the node lifecycle publishes an equivalent reading every two seconds with no backends and effective management-only participation; that cadence also supplies fallback liveness below the health-warning threshold. API health/resource filtering includes fresh telemetry-only participants, local or remote, because replicated worker membership can omit management nodes; the shared 30-second liveness timeout keeps stopped telemetry-only participants from becoming ghosts. `compute_node_health` emits error-level `data_transport_mismatch` reasons for every live node when both values are positively advertised; missing startup telemetry is unknown, not a mismatch. Node identity is assembled from two readings (`MiscData` friendly-name + `StaticNodeInformation` static fields), so `TelemetryView.apply` **merges** them into one `NodeIdentity` rather than overwriting. Fabric-citizenship adds `node_capabilities` to the view: extension-advertised capability tags carried by the `NodeCapabilities` reading (the write half of the plane; see the extension `advertise_capability` fact above). The view also holds the node's OWN outbound `local_advertised_capabilities` set, which is NOT a peer reading and so is deliberately not pruned when a peer times out.

**Election transport isolation.** Python's `Router` gives `ELECTION_MESSAGES` its own bounded egress channel and publish loop instead of placing election behind the ordinary control/telemetry FIFO. `rust/networking/src/swarm.rs` composes two gossipsub behaviors into one libp2p swarm; the isolated behavior negotiates `/skulk/election/meshsub/{1.1.0,1.0.0}` and owns a separate per-peer connection-handler queue. During the migration window, election subscriptions and publications also use the default behavior so old and new binaries remain interoperable through a staggered upgrade. `Election` deduplicates identical candidates received over both protocols. Queue saturation on the shared path can therefore delay or drop ordinary fan-out without removing the isolated election path, while rolling compatibility does not double-count votes. A two-peer Rust integration test verifies custom-protocol negotiation and message delivery.

**Connection stability.** `rust/networking/src/discovery.rs` still dials every address once when mDNS first advertises it, preserving reachable Thunderbolt and other multi-homed paths. After a peer is connected on any path, failed link-local addresses are retried every 12 discovery rounds (60 seconds) rather than every five seconds; routable paths keep the normal retry cadence. Ping uses a five-second interval and timeout, and one socket closes only after three consecutive failures, with success resetting that socket's counter. The independent Python API reachability sweep continues probing advertised addresses so a working direct path can become a placement and ring-transport candidate. Together these rules retain fast links while keeping repeated link-local transport dials and transient scheduler stalls from manufacturing connection churn.

**Download progress admission.** Raw repository callbacks are first coalesced per file and normalized against the model store's canonical byte total. `DownloadCoordinator` then applies a monotonic per-attempt fraction high-water mark (5% steps, one-second floor, 30-second heartbeat). `DownloadPending` is an ordered lifecycle boundary; `DownloadCompleted` and `DownloadFailed` are the only download outcomes stored in event-sourced `State`; new `DownloadOngoing` values use bounded latest-value telemetry. `TelemetryView.effective_downloads()` overlays those live values for placement, workers, node health, and `/state`. Every new attempt/reset receives a `DownloadAttemptId`; terminal control events record it, and the view rejects mismatched or post-terminal telemetry, making cross-protocol reordering harmless. Legacy ongoing events still decode but `apply()` treats them as no-ops. A card's optional `source_revision` is also part of artifact identity: repository metadata and bytes are read at that full Hugging Face commit, store entries persist it, and direct/store staging caches carry a revision marker. A mismatch invalidates completed download shortcuts and replaces the staged or canonical copy instead of reusing mutable bytes.

`GET /state` also attaches a derived `nodeHealth` map (#388, `src/skulk/api/node_health.py`, `compute_node_health`): per live node, a `level` (`ok`/`warn`/`error`) plus `reasons` (each `code`/`message`/`remediation`). It is a pure read-only derivation over the same response data: terminal `DownloadFailed` entries in `State.downloads` (error, pairs with the master's download-failure recovery #381), low/full models-volume disk from `TelemetryView.node_disk` (warn/error), and liveness staleness approaching the timeout prune (warn). Capability conflicts from backend derivation (#614) add four `HealthCode` values mapped one-to-one from `NodeResources.capability_conflicts`: `gpu_serving_disabled` (error), `gpu_detection_degraded`, `invalid_engine_binary`, and `backend_override_conflict` (all warn); the message/remediation were composed on the owning node, and severity comes from the shared `CONFLICT_ERROR_CODES` source of truth. The full `HealthCode` vocabulary also includes `data_transport_mismatch`, `version_mismatch`, and `unreachable` (see below). Liveness is the freshest of the dedicated heartbeat receipt (`TelemetryView.node_last_heartbeat`), ordinary telemetry fallback receipt (`node_last_telemetry`), and `State.last_seen` (the last indexed control event, not a heartbeat), so the connectivity change-gate/de-dup does not false-warn on a healthy node. The dashboard renders it as an amber/red badge on the topology node whose hover names the problem and its fix.

**Normalized accelerator metrics (collector-agnostic GPU telemetry).** `SystemPerformanceProfile` carries an optional `accelerator: AcceleratorMetrics` block (`shared/types/profiling.py`): `vendor`/`name`/`utilization_ratio` (0..1)/`vram_total_bytes`/`vram_used_bytes`/`gtt_total_bytes`/`power_watts`/`temperature_celsius`/`clock_mhz`, each `None` when a collector cannot measure it (distinct from a real `0`). `gtt_total_bytes` is the GPU's GTT (graphics translation table) size, the amount of system RAM the GPU can map; on a unified-memory APU (AMD Strix Halo) it spans system memory, so placement counts it toward usable GPU memory (see the `PlaceInstance` row). The expression is the same regardless of collector; normalization happens at the collector boundary, never downstream. macOS fills it from `mactop` (`vendor="apple"`, `utilization_ratio = gpu_usage/100`; unified memory so `vram_*` stay `None`). AMD/Linux fills it from a new `InfoGatherer._monitor_gpu_linux` that reads passive amdgpu sysfs (`gpu_busy_percent`, `mem_info_vram_*`, `hwmon/power1_average`, `temp1_input`, `pp_dpm_sclk` via `utils/info_gatherer/linux_gpu.py`) and publishes a `LinuxGpuMetrics` telemetry variant carrying only the system profile (memory rides the separate `MemoryUsage` reading). Passive sysfs reads, never a GPU-colliding poll (the macmon/#249 lesson). NVIDIA/Linux fills it from the same `_monitor_gpu_linux` loop as an NVML fallthrough (AMD sysfs first, then NVML device 0) via `utils/info_gatherer/nvidia_gpu.py`: passive NVML queries through a small `NvmlLike` protocol (`pynvml` is a guarded import; a HARD dependency on Linux since #614 (nothing optional may be load-bearing), inert without an NVIDIA driver; the CUDA install recipe `deployment/cuda/install-deps.sh` provides it on CUDA nodes), `vendor="nvidia"`, per-field degradation so one unsupported query never blanks the rest, legacy scalars (`gpu_usage` percent / `temp` / `sys_power`) filled like the AMD collector. CUDA engine advertisement derives from Node Facts (#614): an NVML-visible device implies `cuda` with `SKULK_LLAMA_CPP_BACKENDS` as an override, not a prerequisite. The AMD `vram_total_bytes` exposes a Strix Halo's BIOS-carved GPU VRAM pool, which node memory does not report; placement admits GPU-offload nodes against it (see the `PlaceInstance` row).

**Connectivity readings stay on the control plane.** `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge`, and the derived `thunderbolt_bridge_cycles` are NOT telemetry: `apply()` builds the RDMA topology graph from `node_thunderbolt` (`MacThunderboltConnections` to `replace_all_out_rdma_connections`) and recomputes TB-bridge cycles from `node_network` + `node_thunderbolt_bridge`, and the planner reads `node_network` for host selection. Those define the graph placement runs on, so they must be ordered event-sourced state, not an unordered last-write-wins plane (#279 slice 3 scoping; refines the original "all of `NodeGatheredInfo` to telemetry" target). Because they stay on the ordered log but change rarely, the worker forwards a connectivity reading **only when its payload differs from the last value the master confirmed indexed** (`worker/main.py:_forward_info` gates on `_confirmed_forwarded_info`, which `_event_applier` populates from the master's indexed-event echo; unsorted `psutil` interface order on Linux is normalized by sorting in `get_network_interfaces` so an unchanged topology is byte-stable). Gating on the echo rather than on "last sent" matters: the delivery retry (`event_router` `out_for_delivery`) is bounded (`retry_max_attempts` = 6), so a change sent during a long masterless window can be dropped permanently; an unconfirmed reading simply re-sends on the next poll until it lands, then goes quiet. No periodic keepalive is emitted. This keeps the event log flat except for genuine topology changes. Without it, connectivity churn (~2/s per fleet) filled the master's 10k-event `REPLAY_TAIL_RETENTION_EVENTS` in ~83 min; a joining/failing-over node then replayed that burst and saturated bounded gossipsub send queues into a flap livelock. **Liveness is decoupled from these events and carried primarily by the telemetry plane instead.** The payload-free `NodeHeartbeat` publishes every two seconds and is tracked separately from ordinary telemetry fallback. The master warns once at a ten-second heartbeat gap and prunes only when heartbeat, ordinary telemetry, and the last indexed event have all exceeded 30 seconds. The resulting `NodeTimedOut.evidence` persists each age plus the effective deciding age and timeout. A re-elected master retains the Node-owned telemetry view and sees fresh heartbeats without waiting on a connectivity change.

**Placement reads two views.** The memory-fit check and the context-admission ceiling read `node_memory` from the `TelemetryView`, not `State`. Because the ceiling must be identical across ranks (divergent verdicts deadlock the collectives) and telemetry is unordered last-write-wins, the master computes the ceiling **once at placement time** and stamps it onto the instance (`BaseInstance.context_token_limit`, event-sourced); every worker rank, and the API's admission pre-flight, then read that stamped value instead of recomputing it.

**Capability-aware placement (heterogeneous nodes).** Backend tags are `<engine>-<compute>` (`mlx-metal`, `mlx_audio-metal`, `llama_cpp-vulkan`, `llama_cpp-rocm`, `llama_cpp-cuda`, `llama_cpp-cpu`); the engine selects the worker runner class, the compute names the accelerator (`src/skulk/shared/backends.py`). A node's backends come from the Node Facts pipeline (#614; `probe_node_backends` in `src/skulk/shared/backends.py` is now a thin cached delegate into `skulk.facts`): macOS advertises `{mlx, mlx-metal}` (plus `{mlx_audio, mlx_audio-metal}` when `mlx_audio` imports); other engine tags are derived per engine with precedence declaration > binary device list > hardware vendor inference > CPU floor (see the "Node Facts" component entry), so an unset `SKULK_LLAMA_CPP_BACKENDS` no longer forces `llama_cpp-cpu` on a node whose GPU and build are positively observed. Tags gossip in `NodeResources.backends` on the telemetry plane, alongside `NodeResources.capability_conflicts` carrying every loud observation-vs-declaration disagreement. A model card's `PlacementCardConfig` carries two axes orthogonal to memory/topology: `compatible_backends` (a **hard filter**: the planner excludes a node when `resources.backends & compatible_backends` is empty, `src/skulk/master/placement.py`) and `backend_preference` (an ordered **soft score**, `_cycle_backend_preference_score`). GGUF cards stamp the llama.cpp tags as compatible; MLX text/vision cards keep MLX; speech cards use `mlx_audio`. The master resolves each node's winning backend tag at placement (`resolve_node_backend`: card `compatible_backends` ∩ that node's advertised backends, ordered by `backend_preference`) and **stamps it onto the node's shard as `resolved_backend`** (#330); the worker reads that persisted tag at spawn (`bootstrap._resolve_text_engine`) so dispatch is deterministic from replicated state and cannot disagree with placement, falling back to a node-local re-probe only when the field is absent (resources had not gossiped yet). This also lets a card resolve to different engines per node on a heterogeneous cycle. See `website/docs/amd-strix-halo-nodes.md` for a non-Mac node.

**Model truth vs platform truth (capability gating).** A card's `compatible_backends` declares MODEL truth: which engines the model's artifacts run on. Which of the card's declared capabilities our runner implementations can currently exploit is PLATFORM truth and lives in code, never on cards: `platform_compatible_backends` (`src/skulk/shared/backends.py`) subtracts engines whose runner cannot serve a capability the card declares, and every placement-side read of `compatible_backends` goes through it (`placement._card_platform_backends`), as does the worker's fallback probe (`bootstrap._resolve_text_engine`). Current gates: `_VISION_SERVING_ENGINES = {mlx, llama_cpp}` because the served `llama_server` runner does not stage or pass the mmproj projector yet (upstream `llama-server` supports `--mmproj`; the gap is ours), so a card with a `[vision]` section never lands on a served or pooled placement where its advertised vision capability would silently degrade; `_SPEECH_SERVING_ENGINES = {mlx_audio}` keeps TTS/STT cards off text/image/embedding runners. Mounted `TextToSpeech` cards serve `POST /v1/audio/speech` through `SpeechSynthesis` tasks and the single-node `mlx_audio` speech runner, which emits `AudioChunk` data-plane output; stable `stream=true` requires the card to declare `audio.supports_streaming = true`, while non-streaming requests collect chunks into one raw audio response. Mounted `SpeechToText` cards serve `POST /v1/audio/transcriptions`: the API accepts multipart audio and retains it until the master creates an `AudioTranscription` task, then sends bounded raw `SPEECH_MEDIA` frames directly to the selected worker. The worker verifies the authoritative owner, frame count, and digest before dispatching the speech runner. Batch requests collect terminal `TranscriptionChunk` output in `json`, `text`, `verbose_json`, `srt`, `vtt`, or `ndjson` format. Card-qualified `stream=true` preserves actual model delta boundaries as typed SSE events; explicit `response_format=ndjson` streams the legacy chunk shape, and disconnect cancellation reaches the core command. Translation-capable STT cards reuse that path through experimental `POST /v1/audio/translations`, with an English target mapped to family-specific generation arguments. TTS cards with static `audio.voices` and `audio.default_voice` expose them through `GET /v1/audio/voices`. Cards declaring `audio.supports_reference_audio=true` may receive a bounded multipart reference clip; raw media travels only over the node-addressed Zenoh speech-media plane and never enters State. A card may additionally declare true realtime support: the stable `stt.realtime@1.0.0` provider accepts mono PCM16 on any API node with reachable mounted capacity and streams partial transcripts from the selected single-host runner. Same-node PCM short-circuits locally; remote PCM uses bounded node-addressed `REALTIME_AUDIO` packets over Zenoh and never enters State or the event log. The transcription-only `WS /v1/realtime` edge adapts OpenAI-style 24 kHz PCM16 append/commit events onto that provider without changing capability truth. The bundled `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` card is the first truthful realtime STT card; batch Parakeet and Whisper cards remain non-realtime. The dashboard voice loop consumes readiness and capability metadata: TTS can play draft/final assistant text; batch STT uses `MediaRecorder`; and realtime STT is selected only when the card declares it and the local API advertises `stt.realtime`, then captures and resamples microphone PCM through the WebSocket edge. When a runner gains a capability, flipping the code table lights up every affected card at once with no card edits. Related platform default: `resolve_node_backend`'s no-preference fallback orders `-cpu` compute tags last, so a card without an explicit preference for an engine never resolves a CPU build over a GPU build on alphabetical accident.

**Dashboard TTS streaming:** a streaming-qualified TTS card may include `pcm` in `audio.response_formats`. The speech runner converts model floats to clipped mono signed-16-bit little-endian samples, `POST /v1/audio/speech` exposes sample rate/channel/sample-format headers, and the dashboard feeds those samples to a bounded AudioWorklet queue. Visible assistant text is segmented into ordered sentence requests; reasoning/tool channels are excluded by the existing content splitter. Queue pressure pauses HTTP reads, and stop propagates through queued requests, the active fetch, the core command, and runner cancellation. Batch-only cards retain encoded complete-response playback.

**Served-backend engine (`llama_server`).** A third engine class, distinct from the in-process `mlx` and `llama_cpp` runners: instead of loading the model in-process, the worker launches an external **`llama-server`** subprocess and proxies its OpenAI HTTP API (`src/skulk/worker/runner/llama_server/runner.py`). This is the only path to llama.cpp's **native multi-token-prediction speculative decoding** (`--spec-type draft-mtp`): for models that ship MTP heads (Qwen3.6, DeepSeek-V3/R1, GLM, Kimi, Nemotron) the MTP orchestration lives in the server application (`tools/server`), not in `libllama` or `llama-cpp-python`, so it cannot be driven in-process. The runner spawns the server on an ephemeral port (reaped on parent death via `PR_SET_PDEATHSIG`), health-checks `/health`, then proxies the streaming `/v1/chat/completions` SSE into the normal `ChunkGenerated` data plane (`reasoning_content` becomes `is_thinking` token chunks; structured `tool_calls` become a `ToolCallChunk`). Per-request cancellation aborts the proxied HTTP connection (stopping server-side generation); `SIGTERM` is for instance teardown of the whole server, not a single request. The server emits structured reasoning/tools itself, so the in-process harmony/think/tool text parsers are unused here. A node advertises `llama_server` (plus compound tags) when `SKULK_LLAMA_SERVER_BIN` points at a `llama-server` binary (`_probe_served_backends`); cards route to it via `compatible_backends = ["llama_server-…"]`. The spec mode is card-driven: `RuntimeCapabilityCardConfig.served_spec_type` (`draft_mtp` / `draft_eagle3` / `draft_simple` / `ngram`) plus `served_spec_n_max` map to the `--spec-type` / `--spec-draft-n-max` flags; only this engine reads them (MLX speculation stays the `mtp_*` / `assistant_model_repo` fields). Some spec modes need a **separate draft GGUF** rather than baked-in heads: `served_spec_draft_repo` + `served_spec_draft_file` name that file (downloaded as a companion alongside the base, present-on-disk-checked, and passed as `--model-draft`). `draft_simple` / `draft_eagle3` always require one; Gemma 4 is `draft_mtp` driven by its assistant as a separate draft GGUF (llama.cpp PR #23398); Qwen3.6 / DeepSeek / GLM `draft_mtp` leave it unset (heads baked into the base GGUF). When the draft repo IS the base repo (a bundle shipping base + draft, e.g. Gemma 4), the draft shares the base's store entry and staging directory, so it is **co-fetched with the base** through the model-store download protocol (the worker sends `extra_gguf_files` on the download request; `select_store_gguf_download_files` keeps the draft's shard group, a registration guard verifies it landed, and `ModelStore.entry_missing_files` re-downloads a stale base-only entry to recover it). The staging fast path is draft-aware (`_staged_same_repo_draft_missing`), so a dir staged before the draft existed re-stages rather than serving without it. A separate-repo draft has its own `model_id` / staging dir and rides the normal companion path. Single-node placements are group-less. Measured on a Strix Halo (Radeon/Vulkan, llama.cpp b9820): dense Qwen3.6-27B with `draft-mtp` ran 2.19x vs no MTP (72-77% acceptance) end-to-end through Skulk's API. The shape (managed inference server plus OpenAI proxy) is shared with the `vllm` engine below.

**Served-backend engine (`vllm`).** The second served engine, reusing the `llama_server` managed-subprocess-plus-OpenAI-proxy shape with the `vllm serve` process (`src/skulk/worker/runner/vllm/runner.py`). vLLM is the **GPU-serving fast path**: its continuous batching and paged attention hold latency flat and grow aggregate throughput under concurrent load where the single-stream engines collapse. A Phase 0 spike on an A100 (gpt-oss-120B, same MXFP4 weights on both engines, driven by `vllm bench serve`) measured llama.cpp's time-to-first-token hitting ~31s at 64-way concurrency versus vLLM's ~0.5s, with vLLM's aggregate throughput ~1.75x and climbing while llama.cpp plateaued; conversely llama.cpp *won* single-stream ~2.8x, because the A100 (Ampere) has no native FP4 and vLLM's MXFP4 runs through the bf16 Marlin kernel (this flips on Blackwell, which has native FP4). So vLLM **coexists** with the in-process engines rather than replacing them: the planner picks by hardware and expected concurrency. The runner spawns `vllm serve <model_dir> --served-model-name <id> --gpu-memory-utilization <f> --max-model-len <n> --tensor-parallel-size 1` on an ephemeral port (reaped on parent death via `PR_SET_PDEATHSIG`), health-checks `/health` (a bare 200, unlike llama-server's JSON status), then proxies the streaming `/v1/chat/completions` SSE into the normal `ChunkGenerated` data plane. vLLM auto-detects CUDA vs ROCm from its own install, so there is no per-compute-backend flag; a node advertises `vllm` (+ `vllm-cuda`/`vllm-rocm`) when `SKULK_VLLM_BIN` points at the `vllm` CLI and a GPU backend is declared (`_probe_vllm_backends`; GPU-only, no `cpu` fallback since a vLLM node with no GPU is not a useful candidate). `vllm-` joins the GPU-offload prefixes in `placement_utils`, so discrete-VRAM admission auto-applies. Dispatch is **concurrent**: `main()` keeps up to N `TextGeneration` requests in flight at once, each streaming its own `/v1/chat/completions` request on a worker thread (a `ThreadPoolExecutor` of N workers, N from `SKULK_VLLM_MAX_CONCURRENT_REQUESTS`, default 32), so the server sees concurrent load and its continuous batching actually engages — a serial runner would defeat the engine's whole purpose. Because a `ThreadPoolExecutor` caps active threads but has an unbounded submit queue, an N-permit semaphore (`_dispatch_permits`) is acquired before each submit and released when a generation finishes, so submitted jobs are capped at N too: excess load blocks the dispatch loop (backpressuring the task receiver / master) instead of accumulating an unbounded in-process backlog. While saturated the loop still polls server liveness. N is a client-side admission bound, NOT vLLM's batch width (the server batches up to its own `--max-num-seqs`). Runner status is `RunnerRunning` while `_inflight > 0` and `RunnerReady` when the last generation drains (a lock-guarded counter, since worker threads finish concurrently; `_generate` is status-neutral, the dispatch loop owns the transition); `MpSender` event sends are thread-safe and each `DataChunk` carries `command_id`+`sequence` so the API demultiplexes interleaved streams. Lifecycle tasks (`LoadModel`, `Shutdown`) run inline on the dispatch thread; shutdown sets `CANCEL_ALL`, drains the pool, then tears down the server. Single-node text generation in this first slice: tool calling and per-token logprobs are rejected loudly (follow-ups), `gpu-memory-utilization` is a node env knob (`SKULK_VLLM_GPU_MEMORY_UTILIZATION`, default 0.90); the live subprocess/streaming path validates on GPU hardware. vLLM's own tensor/pipeline parallelism (multi-node) and vLLM-aware admission are later tracks; the same concurrent-dispatch pattern applied to `llama_server` (which needs `--parallel N` + N× context and VRAM-admission changes) is the immediate follow-up.

**GPU compute-capability telemetry.** `AcceleratorMetrics` carries `compute_capability` (`"<major>.<minor>"`, the NVIDIA SM level) plus derived `native_fp4` / `native_fp8` flags, filled by the NVML collector (`nvidia_gpu.read_accelerator_metrics` via `nvmlDeviceGetCudaComputeCapability`; derived at the collector boundary: FP4 = Blackwell sm100+ i.e. >= 10.0, FP8 = Ada/Hopper sm89/sm90 i.e. >= 8.9). This is the signal capability-keyed placement needs: the same model+engine performs oppositely across GPU generations (see the vLLM spike above), so engine/quant/placement must key on compute capability, not vendor. `None` on collectors that do not report it (AMD sysfs, Apple).

**Multi-node GGUF placement (RPC driver + donors, #328).** `llama_server` is a multi-node-capable engine (`_MULTI_NODE_ENGINES` in `src/skulk/shared/backends.py`, alongside `mlx`; the in-process `llama_cpp` stays single-node). Placement models it as an **asymmetric served placement**, not a Skulk-sharded pipeline: the planner mints a `LlamaRpcInstance` (`src/skulk/shared/types/worker/instances.py`, `InstanceMeta.LlamaRpc`) whose **driver** node runs `llama-server --rpc donor:port,...` and holds the model file, while each **donor** node runs `ggml-rpc-server` and lends GPU memory. llama.cpp splits weights/KV across the pooled devices proportional to their free memory itself, so Skulk computes NO GGUF layer math: the driver's shard nominally spans all layers, donors carry the degenerate `RpcDonorShardMetadata` (no layer range; the distinct type is what worker dispatch keys on). Placement rules (`src/skulk/master/placement.py`): a multi-node cycle is admissible only under the **common-engine cycle rule** (some single engine advertised by every node in the cycle AND multi-node capable) (this also fixed #414's mixed-engine hole for hybrid cards); a cycle resolves to the RPC shape when `llama_server` is the common multi-node engine and `mlx` is not; the driver is the biggest-usable-VRAM node (tie: download presence); donor endpoints are chosen at placement from observed connections via the ring's transport prioritiser (Thunderbolt first, VPN last) and stamped on the instance (`donor_endpoints`, the donor binds exactly that `ip:port`, never 0.0.0.0 and never link-local: a two-TB-port node routes 169.254/16 out one port only, so link-local endpoints break asymmetrically). Pooled admission uses the **same UMA-aware usable-GPU figure** as single-node placements (`usable_vram_by_node`): on a unified-memory APU the BIOS VRAM carve is not an allocation boundary (the GPU maps system RAM through GTT), so a carve-only figure falsely refuses pooled placements that load and serve fine (proven live on the Strix pair: pooled gpt-oss-120b, refused under a carve-only figure, loaded 40.6G driver / 22.8G donor and served at 44 tok/s). An RPC cycle node missing its accelerator telemetry surfaces as info-pending rather than falling back to the system-RAM formula. Smallest-cycle ranking still prefers single-node whenever the model fits one node, so the RPC shape only fires for the pooled-only model class; existing placements are behavior-identical. RPC is memory pooling, not speedup (spike: 120B MoE pooled decode 42-45 tok/s vs 56 single-node, prefill unchanged; TB improves load time, not decode). Donor death kills the driver (llama-server SIGABRTs), which the standard crash cascade turns into instance teardown + re-placement. Worker side: dispatch keys on the SHARD TYPE (`bootstrap.entrypoint` routes `RpcDonorShardMetadata` to the donor runner, `src/skulk/worker/runner/rpc_donor/runner.py`, which spawns `ggml-rpc-server -H <stamped-ip> -p <port> -c` and reports RunnerReady with no LoadModel; binary via `SKULK_RPC_SERVER_BIN` or the `SKULK_LLAMA_SERVER_BIN` sibling); the driver is the served runner with `--rpc <endpoints>` appended (rank 0 of a `LlamaRpcInstance` is the only multi-node shape it accepts). The plan gates are role-aware (`worker/plan.py`): donors skip the download/load/warmup gates entirely (no model file ever lands on a donor), `_init_distributed_backend` skips RPC instances (no ConnectToGroup: llama-server dials the donors directly), the driver's LoadModel requires the download on ITS node only plus every donor RunnerReady (endpoints answer before llama-server dials), and `_pending_tasks` never forwards inference tasks to a donor. The worker's local pre-spawn fit guard is skipped for RPC instances (llama.cpp decides the split at load; pooled admission already sized every node against its usable GPU memory, and a genuine misfit fails at llama-server load into the crash cascade).

### Data plane (#279 Phase 2)

`DATA` carries per-token **generation output** off the event log. Exactly one serving output worker publishes `DataChunk` (`{command_id, GenerationChunk}`) on this topic: rank 0 for text, embedding, and speech families, or the primary terminal pipeline stage for image generation. `RunnerSupervisor._emit` diverts `ChunkGenerated` to the data sender and diverts completed per-rank `TracesCollected` payloads to `TRACE_DATA`; task status, acknowledgements, and runner status stay on the ordered control-plane event sender. The owning API node drains `DATA` in `API._apply_data` and demuxes by `command_id` into the per-command stream queues (`_dispatch_generation_chunk`), exactly as the event path did. The sibling `PROVIDER_DATA` topic carries generic provider `ProviderStreamPacket` values keyed by call id plus direction, preserving provider-specific schemas and binary attachments outside the closed `GenerationChunk` union. `REALTIME_AUDIO` carries built-in realtime STT input as a bounded JSON lifecycle header plus raw PCM bytes from the owning API to the selected worker. `SPEECH_MEDIA` carries bounded request-scoped TTS reference audio and batch STT uploads from the API owner to selected speech workers. `TRACE_DATA` carries one terminal task-trace payload per runner rank to the owning API for bounded assembly and local persistence. `VISION_MEDIA` carries VLM and image-edit input from the API owner directly to every worker rank in the instance chosen by the authoritative `TaskCreated` event. The master sees only the control-sized command and task lifecycle: it never indexes, persists, or application-relays data-plane payloads.

Output frames never mutate `State`, so removing them from the ordered log remains loss-free for *state* correctness. `RunnerSupervisor` now emits `started` at sequence 0, payload frames after it, and exactly one `completed`, `failed`, or `cancelled`; status-only completion/cancellation still emits a terminal before producer state is cleared. The API reorders whole frames by the per-command sequence on gossipsub and uses O(1) sequence dedupe on Zenoh. A duplicate is idempotent. A bounded reorder window repairs normal reordering, but an unresolved gap, queue drop, or arrival-mode sequence jump is a transport failure: the API sends a terminal `ErrorChunk`, cancels a still-active producer, closes the endpoint queue, and increments `transportFailures` instead of skipping ahead and returning incomplete output. Stream state exists only while the command queue is live and is dropped on finalization. Vision input uses `opened -> chunk* -> completed -> accepted`, with `cancelled` and source-routed `transport_failed` terminals. Duplicate identical sequences are idempotent, conflicting duplicates fail the command, and workers release assembled base64 input to runner planning only after exact sequence/count/metadata/SHA-256 and authoritative-task-owner validation plus successful acknowledgement admission. Timed-out acknowledgement admission is retried without exposing the task early. Every selected rank must return `accepted`; a five-minute source deadline fails and cancels transfers missing an acknowledgement. Incomplete streams are bounded by frame count, per-command and process bytes, active streams, and a five-minute age limit.

Vision hard bounds are 64 API-staged plus active commands, 32 MiB per command, and 512 MiB across staged plus active source transfers; 16 remote dispatcher streams total/per destination owner, 66 queued frames per stream (one open, at most 64 payload, and one completion), and 64 concurrent rejection tasks; and 64 worker streams, 64 media chunks/32 MiB per command, 512 MiB per worker process, and 64 retained pre-task failure reports. Frames are capped at 512 KiB. Network receive uses independent 66-frame payload and 1024-frame metadata-only terminal lanes; overflow is source-routed as a typed failure when terminal capacity remains. The dispatcher geometry therefore caps queued serialized image media independently of generated output, while API and worker accounting remain charged through verification or terminal cleanup. Same-process producer and consumer channels are rendezvous-only. Reverse acknowledgements key their egress queues by command and producing rank so one rank's terminal cannot close another rank's acknowledgement.

Mixed-version clusters remain unsupported. To prevent a stale or malformed participant from reintroducing payloads during an upgrade mistake, the master rejects every legacy `SendInputChunk` command and applies an explicit event-jurisdiction guard before ordering. Payload-bearing runner IPC events, observational telemetry, and transient download progress remain decodable but are skipped at their source sequence before indexing, retention, replay, state application, or global broadcast. Only the enumerated durable control decisions, download reset/terminal transitions, and ordered connectivity facts can enter the event log.

Because `DATA` has no replay, token and streaming-audio receivers retain the 120-second mid-stream idle deadline. It wraps producer receive only, arms after real output, and does not bound queueing/prefill time. A stall becomes a terminal error; a still-active producer is cancelled, while an already-terminal task follows normal cleanup. Independently, every producer-side remote command queue has a 30-minute no-frame resource lease. Every producer frame observed by egress renews it; expiry closes and tombstones the queue before any recovery publish, releases owner/process admission, and best-effort emits a correctly sequenced typed failure. This longer lease bounds orphaned egress state without applying the API's post-output deadline to model prefill. `DataPlaneObserver` reports lifecycle counts, first-byte and stream-span samples, duplicates, out-of-order frames, skipped sequences, late frames, missing starts/terminals, idle timeouts, and synthesized transport failures through `NodeDiagnostics.data_plane`. `DataPlaneEgressObserver` contributes current/peak queue depth, active command queues, local short circuits, remote enqueue/publish/drop/failure counts, idle stream reclamations, bytes, latency, and per-owner pressure. The dashboard Node tab renders the operational subset.

**Zenoh data-plane transport (soft default-on, #315/#316).** The `DATA`, `PROVIDER_DATA`, `REALTIME_AUDIO`, `SPEECH_MEDIA`, `TRACE_DATA`, and `VISION_MEDIA` topics can ride an Eclipse Zenoh peer session instead of gossipsub; control, telemetry, and election planes stay on libp2p. Transport selection is resolved by `_resolve_zenoh_enabled(SKULK_ZENOH_DATA_PLANE, SKULK_ZENOH_LISTEN)`: explicit `SKULK_ZENOH_DATA_PLANE` of `1`/`true`/`yes`/`on` forces Zenoh on (and still requires an explicit listen, #308), `0`/`false`/`no`/`off` forces gossipsub, and **unset** is soft default-on (Zenoh when `SKULK_ZENOH_LISTEN` is configured, else gossipsub) so a bare node with no Zenoh config stays on gossipsub instead of failing the #308 listen requirement. Mixed transports remain unsupported and are never bridged. Each node advertises its already-resolved choice in `NodeResources.data_transport`; `/state` exposes it and derives `data_transport_mismatch` health errors when live nodes disagree, while node diagnostics adds a matching warning. The `Router` holds an optional `ZenohHandle` (`skulk_pyo3_bindings`, backed by `rust/networking/src/zenoh_session.rs`). Vision ingress owns a separate bounded dispatcher and observer from generated/provider/realtime output, so upload pressure cannot consume their queue or stream admission. The session is a Zenoh `peer` with multicast scouting OFF and gossip + explicit `SKULK_ZENOH_CONNECT` endpoints (the macOS-Local-Network-Privacy-safe posture), publishing `Reliable` + `Block` on a single priority so one producer's frames are FIFO per key. Model DATA keeps its transport-conditional reorder policy; provider DATA always applies `CapabilityStreamReceiver` because the generic contract requires bounded duplicate/reorder/gap handling independent of the selected transport. Remote realtime and reference audio are disabled on the gossipsub fallback so private audio is never broadcast cluster-wide; batch STT, trace payloads, and vision media retain target-filtered gossipsub fallback delivery on the trusted fabric.

Key-addressed unicast uses `data/<owner_node>` for model output, provider-specific keys for `PROVIDER_DATA`, `realtime_audio/<target_node>` for PCM ingress, `speech_media/<target_node>` for reference or batch transcription audio, `trace_data/<owner_node>` for completed rank traces, and `vision_media/<target_node>` for image input and reverse acknowledgements; each node subscribes only to its own suffix. The in-process networking channel carries an `OutboundPacket` with topic, routing key, command stream key, terminal marker, and serialized bytes. Same-node frames are delivered before egress and short-circuit Zenoh entirely. Remote frames enter a bounded queue per logical stream; every stream has an independent publish task and five-second publish deadline. Queue saturation or publish failure drops only that stream's frame. Input transport rejection is converted to a source-routed `transport_failed` packet so the owning API terminates only the affected command. Trace transport is best-effort diagnostics: an undeliverable rank packet expires from the owner's bounded incomplete assembly without affecting inference. Terminal delivery closes the stream worker. `NodeDiagnostics.visionMediaEgress` reports the isolated upload queue depth, active streams, local short circuits, remote enqueue/publish/drop/failure counts, bytes, latency, lease reclamations, and per-target pressure; `visionMediaIngress` reports API-staged/active commands and bytes, pending worker acknowledgements, worker-side retained streams/frames/bytes, verification, rejection, and expiration outcomes. On gossipsub, target-tagged batch STT, trace, and vision packets are network broadcasts. Vision is filtered in the router; speech workers and trace-owning APIs discard non-target packets before assembly or persistence. Remote realtime ingress and all reference-audio ingress are unavailable.

**Media framing decision (#509).** OpenAI-compatible response models retain base64 strings in `AudioChunk`, and the speech runner still receives one verified base64 field across its local process boundary. Neither representation is the Fabric media contract or the cluster upload path. `extensions/streams.py` defines a typed JSON lifecycle header plus either raw `InlineMediaAttachment` bytes or a `BlobMediaAttachment` (staged id, byte size, SHA-256, media type); `encode_capability_stream_frame` and `SPEECH_MEDIA` keep cluster media bytes outside JSON. `scripts/benchmark_data_plane_framing.py` measures the two shapes. On the development machine (50 median iterations), 64 KiB/256 KiB/1 MiB payloads used 1.338/1.334/1.334x wire bytes under base64 DATA JSON versus 1.004/1.001/1.000x under binary framing; base64 encode+decode median cost reached about 4.0 ms at 1 MiB versus about 0.008 ms for header framing. These timings are machine-specific; the size ratio is structural. Therefore provider and batch speech media use inline binary on the cluster transport, large immutable results use staged blobs, and text/tool/transcription metadata stays schema-validated JSON.

**Version policy:** all cluster nodes must run the same Skulk version and source build; a mixed-build fleet is a degraded deployment window, not a supported workload mode (see "Deployment & versioning" in [Architecture](architecture)). Correctness-bearing events, commands, state, telemetry envelopes, and DATA types keep `extra="forbid"`; there is no legacy bridge or transition-hydration concession. Peer operational diagnostics are deliberately different: `parse_peer_node_diagnostics` validates with recursive `extra="ignore"`, additive counters carry defaults, `/v1/diagnostics/cluster` reports aggregate/per-node `versionStatus`, and `/state.nodeHealth` emits `version_mismatch` while live identity telemetry disagrees. This keeps a staged rollout observable without claiming cross-version inference or state compatibility.

## Events

Discriminated union at `src/skulk/shared/types/events.py`. Selected events:

| Event | Emitted when | Applied by |
|---|---|---|
| `InstanceCreated` | Master places a model | All nodes (update `State.instances`) |
| `InstanceDeleted` | Master deletes a placement | All nodes |
| `RunnerStatusUpdated` | Runner subprocess transitions state | All nodes |
| `RunnerFailed` | Runner crashes or exits unexpectedly | All nodes |
| `TaskAcknowledged` | Worker accepts a task | All nodes |
| `TaskStatusUpdated` | Task transitions state (`Running`, `Failed`, `Cancelled`, `Complete`, `TimedOut`, the last emitted by the worker on shutdown timeouts, `worker/main.py:474`). The `Complete` variant is emitted by the runner / worker on natural finish (e.g. `worker/main.py:362,388,450`, runner `send_task_status(..., TaskStatus.Complete)`). The `TaskFinished` command sent by API on stream end triggers `TaskDeleted` only (`master/main.py:444-450`), not this event. The `Cancelled` variant (operator instance deletion via `get_transition_events`) additionally makes the API terminate that command's open stream with an error chunk. | All nodes |
| `TaskFailed` | Master plan loop fails in-flight API tasks (TextGeneration / ImageGeneration / ImageEdits / TextEmbedding) whose instance is gone or dying (`orphaned_task_failure_events` in `master/main.py`, emitted BEFORE `InstanceDeleted`/`NodeTimedOut` so it indexes ahead of the applies that delete the task). `apply_task_failed` sets `task_status=Failed` (terminal, making re-emission idempotent) plus error fields. The API reacts by delivering a terminal `ErrorChunk` into the command's stream (`_terminate_command_stream`): streaming closes with an error event, non-streaming returns 500. On master failover the new session cannot carry old tasks, so the API's session `reset()` fails all open command streams directly instead (`_fail_open_command_streams_for_session_reset`). | All nodes |
| `TaskDeleted` | Task is purged from cluster state | All nodes |
| `ChunkGenerated` | Runner IPC emits an output chunk (token, tool call, error) | Runner supervisor diverts it to `DATA`; the master rejects any legacy event copy |
| `TracesCollected` | Runner IPC emits trace events for one rank | Runner supervisor diverts it to `TRACE_DATA`; the master rejects any legacy event copy |
| `TracesMerged` | Legacy decode-only trace envelope | Not emitted; the owning API merges `TRACE_DATA` packets directly |
| `TracingStateChanged` | Cluster tracing toggle changes | All nodes |
| `StagedModelEvicted` | A store-deleted model's locally-staged copies should be dropped fleet-wide (#427) | All nodes: `apply` removes the model's download entries from State (so the planner re-stages on a future placement instead of loading deleted files); each worker reacts by `rmtree`-ing the model's staged directory (`_evict_staged_model` → `ModelStoreClient.evict_shard`, which never touches the store's canonical copy) |

Apply function: `src/skulk/shared/apply.py::apply`, a pure `(State, IndexedEvent) -> State`.

Snapshot bootstrap is followed by a retained-tail request. The master serves at most `EVENT_LOG_REPLAY_BATCH_SIZE` (10,000) events per request, but does not enqueue that tail as one burst: a single background replay worker coalesces overlapping requests and emits `EVENT_LOG_REPLAY_CHUNK_SIZE` (32) events followed by `EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS` (250 ms) of pacing. Replay therefore cannot block the command processor, and repeated NACKs cannot create concurrent full-tail broadcasters. Separately, `EventLogGrowthMonitor` resets during active tasks/downloads and warns when otherwise-idle indexing remains at or above 60 events/min across a 60-second window, with a five-minute warning cooldown.

## Commands

Two distinct command unions on two distinct topics:

### COMMANDS topic: `Command` union

Discriminated union at `src/skulk/shared/types/commands.py`. Carried as `ForwarderCommand` over the `COMMANDS` pubsub topic.

| Command | What it requests | Master action |
|---|---|---|
| `PlaceInstance` | Spin up a model on the cluster. Optional `excluded_nodes: list[NodeId]`: planner treats those nodes as if absent for *this placement only*; already-running instances on them are not affected. | Pick ranks based on memory + topology (filtered by `excluded_nodes`); emit `InstanceCreated`. Memory admission is per-node (Tensor = even split, Pipeline = proportional to available): a node's weight share x an engine-aware overhead factor (`memory_overhead_factor`, 1.30 for MLX / 1.10 for GGUF; see below) + an explicit KV-cache reservation for `KV_CONTEXT_BUDGET_TOKENS` (8192, the admission **floor**, NOT the served size: the runner serves the larger memory-fit window from `instance_context_token_limit`, up to the card's max, which fits because it is derived from this same per-node working set; a card whose own advertised max context is below 8192 serves that smaller max, a gguf card whose fit is uncomputable clamps back to this floor to avoid a load-time OOM, and a gguf instance on a node WITHOUT discrete VRAM is also clamped to this floor since its fit is derived from static `ram_total` while load competes with live `ram_available` — so CPU / non-discrete-VRAM gguf nodes keep 8192, and the large served context is a discrete-VRAM/GPU behavior) + `MEMORY_OVERHEAD_FLOOR` (256 MB), each node capped at `GPU_WORKING_SET_FRACTION` (0.75) of `ram_total` (the Metal GPU working-set ceiling, since gossiped `ram_available` can exceed what the GPU may wire). On macOS the gossiped `ram_available` is itself the GPU-wireable figure `total − wired − anonymous − compressor` (vm_stat snapshot per telemetry sample; see `MemoryUsage` below) rather than the naive free-plus-inactive figure that counted reclaimable file cache as used. A node that reports **discrete GPU VRAM** (AMD/NVIDIA `vram_total_bytes` in `node_system`) is instead admitted against its usable VRAM (`min(vram_total − vram_used, GPU_VRAM_WORKING_SET_FRACTION (0.90) × vram_total)` via `usable_vram_by_node`), because a GPU-offload engine (llama.cpp/vLLM) allocates weights + KV from VRAM, not system RAM (e.g. a Strix Halo's 64 GB VRAM pool, separate from its 64 GB system RAM, which a 0.75×system-RAM cap would wrongly refuse). On a **unified-memory APU** node (the accelerator's GTT spans the whole system: `gtt_total_bytes > vram_total_bytes` AND `gtt_total_bytes ≥ ram_total`) usable GPU memory is the working-set-capped VRAM (`0.90 × vram_total`) plus the system RAM the GPU can map via GTT, minus `UMA_GPU_OS_HEADROOM` (16 GB), so a model larger than the BIOS VRAM carve-out runs through GTT (e.g. a 58.5 GB GGUF gpt-oss-120B on a 128 GB Strix Halo with a 64 GB carve-out). The dual gate matters because a discrete amdgpu card also reports a `gtt_total_bytes` (its default can equal VRAM); requiring GTT to cover all of system RAM keeps a dedicated card on the conservative VRAM-only path. Apple unified-memory nodes report no discrete VRAM and keep the system-RAM ceiling. The weight-overhead factor is engine-aware (`memory_overhead_factor`): GGUF/llama.cpp models use `LLAMA_CPP_MEMORY_OVERHEAD_FACTOR` (1.10), lighter than MLX's 1.30, because the C++ runtime carries no MLX buffer cache or Python interpreter overhead. Estimation lives in `skulk.shared.models.memory_estimate`, shared with the worker's local pre-spawn OOM guard so the two checks use the same estimator. The worker guard (`_local_shard_fit_error`, `footprint_exceeds_usable`) applies a `_LOAD_FIT_TOLERANCE` (10%): it refuses only when the footprint exceeds *live* usable beyond that margin, because the footprint is already padded (overhead factor + KV + floor) and the master admits on a *gossiped* figure that can sit a few hundred MB above the worker's live reading. Without the tolerance a sub-GB live-versus-gossip jitter flips a master-admitted borderline split to a refusal (#383: a 0.2GB / 2% miss refused a 24B model at the LoadModel re-check across a 3-node ring); a gross shortfall still refuses, preserving the leak-on-OOM guard. Failures raise typed `PlacementError`s, with `PlacementInfoPendingError` for the cluster-startup windows where cluster info has not finished gossiping (connection edges lag identities; memory info lags the edges). The API dry-runs placement before forwarding (400 on impossible, 503 after a 15s wait on pending info). Instance listener ports (ring `ephemeral_port`, JACCL coordinator, RPC donor ports) are allocated by `random_ephemeral_port` from the reserved band `_PLACEMENT_PORT_RANGE` (24000-31999), below BOTH the Linux (32768+) and macOS (49152+) OS ephemeral ranges, and exclude ports already held by live instances (`_listener_ports_in_use`), so a placement's listener can never collide with a short-lived OS connection or a sibling instance; band exhaustion raises `PlacementError` loudly instead of looping (#457). |
| `DeleteInstance` | Tear down a placed model | Emit `InstanceDeleted`; workers tear down runners |
| `RefuseInstancePlacement` | Worker → master: this node cannot fit its shard at load time (the live GPU-wireable reading sits below what the gossiped telemetry admitted). Carries `instance_id`, `node_id`, `reason`. | Delete the refused instance and **re-place the same model one node wider** (`min_nodes` = refused width + 1, via `replacement_command_for_refused_instance` + `place_instance`), so each node holds a smaller share. If no wider cycle exists (`PlacementError`), the master does not stop there on a heterogeneous cluster: it **falls back to single-node width excluding the refuser** (`fallback_command_for_refused_instance`, #455), because the wider-split assumption (every node can hold a smaller share) breaks when engines differ per node (a GGUF model refused by one GPU node may still fit alone on another GPU node but can never widen onto a Mac). Fallback instances are tracked in `_fallback_placed_instances`: a refusal against a fallback placement is **terminal** (tear down, cancel the model downloads the doomed placement started via `cancel_unnecessary_downloads`, give up; #456), bounding the refusal chain at two hops so it cannot oscillate between refusers. Idempotent: a refusal for an already-removed instance is a no-op. Fixes #290 (place-then-silently-vanish on tight multi-node splits). |
| `TaskFinished` | Mark a streaming task complete (sent by API on stream end) | Emit `TaskDeleted` (`TaskStatusUpdated(Complete)` is emitted earlier on the chunk path, not from `TaskFinished` directly) |
| `TaskCancelled` | Cancel an in-flight command (sent by API on `/v1/cancel`) | Emit `TaskStatusUpdated(Cancelled)` |
| `SetTracingEnabled` | Cluster-wide tracing toggle | Emit `TracingStateChanged` |
| `AddCustomModelCard` | User-added model card | Emit `CustomModelCardAdded`; nodes persist locally |
| `DeleteCustomModelCard` | Remove user card | Emit `CustomModelCardDeleted` |
| `EvictStagedModel` | Drop a store-deleted model's locally-staged copies fleet-wide. Sent by the API right after `DELETE /store/models/{id}` removes the store host's canonical copy, because workers cache their own staged shards independently of the store (#427). | Emit `StagedModelEvicted` |

**Download-failure recovery (#381, plan loop, not a command):** a multi-node instance whose ring forms but where one rank's model **download** fails terminally (disk full, transient HF/network error) sits at `RunnerConnected` forever: the failed rank never becomes load-ready and nothing fails or recovers it. The master's `_plan` reconcile (`_recover_download_failed_instances`, gated on the same `TOPOLOGY_SETTLE_GRACE_SECONDS` as the liveness passes) detects it via `instances_wedged_by_download_failure` (a not-all-ready instance whose any rank node carries a terminal `DownloadFailed` for the model), fails in-flight API tasks bound to it (cause surfaced), tears it down, and re-places at the **same width excluding the failed node(s)** (`replacement_command_for_download_failed_instance` + `place_instance(excluded_nodes=...)`). `PlacementError` (no healthy node set hosts the width, e.g. a cluster-wide failure) is terminal: stop at the teardown. Deduped via `_download_failure_recovered` (same rationale as `_refusal_replaced`: events are emitted by the plan pass but not applied until they round-trip). A ready/serving instance is never torn down by this path even if a stale `DownloadFailed` lingers. Recovery also **consumes the terminal failure record** (#454): it emits a `NodeDownloadProgress(DownloadPending)` for each failed node+model, which `apply()` replaces over the stale `DownloadFailed`; without the reset the stale record lingered in session state and the wedge scan condemned every future placement of that model touching that node long after the cause (e.g. a freed disk) was gone. RPC donor nodes are excluded from the wedge scan (`RpcDonorShardMetadata` ranks never download the model, #328), so a stale failure on a donor cannot condemn a pooled instance.

### DOWNLOAD_COMMANDS topic: `DownloadCommand` union

Discriminated union at `src/skulk/shared/types/commands.py`. Carried as `ForwarderDownloadCommand` over the `DOWNLOAD_COMMANDS` pubsub topic. Used for cluster-wide config sync and model-store coordination, separated from the main command channel because these are typically larger payloads and have different retry semantics.

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
| `/store/storage` | GET | Local node's storage breakdown: staged models (size, last-use, in-use incl. companions), event-log bytes, disk free. Staging eviction: `cleanup_on_deactivate` default true; not-in-use staged models kept newest-first up to `staging_keep_recent_gb` (40 GiB default), enforced at deactivate AND node startup (crash-orphan reconciliation); `src/skulk/store/staging_eviction.py`. Companion repos (MTP sidecar / assistant / split vision weights) resolve through `companion_download_specs()` (`src/skulk/download/download_utils.py`) on every resolution path: required companions (vision) fail the load loudly, best-effort companions (sidecar/assistant) log and degrade to plain decode. |
| `/instance/previews` | GET | List candidate placements |

### State / events

| Endpoint | Method | What |
|---|---|---|
| `/state` | GET | Cluster state snapshot |
| `/events` | GET | Stream stored events (debug) |
| `/node_id` | GET | Local node identity |
| `/config` | GET / PUT | Cluster config (sanitized) |

### Field telemetry (opt-in)

- Consent lives in `skulk.yaml` under `telemetry:` (tri-state `consent`:
  `unasked` / `enabled` / `disabled`; separate `diagnostics_consent`;
  `install_id` = random UUID, the anonymous rate-limit key AND deletion
  capability; `consented_at` / `consented_version` stamps; `ingest_url`).
  Persisted in the file so it survives restarts (State is rebuilt per
  session); edited via the dashboard (first-run consent modal, then
  Settings). Nothing is queued or sent while `unasked` or `disabled`.
- Collector: `src/skulk/api/field_telemetry.py` on the API node. Taps the
  chat-completions chunk stream (innermost wrap, before the extensions tap),
  records one sample per generation (model id, canonical hardware classes
  like `apple-m4-24gb`, TTFT, decode tok/s, token COUNTS, `error_class`
  enum), plus peer-observed `node-death` samples by diffing the visible node
  set between flushes. Bounded queue (1000, drop-on-overflow, drops
  counted), 60s fail-silent flush to `POST <ingest_url>/v1/telemetry`,
  batch kept on failure. Content-free by construction; the ingest service
  enforces the same allowlist independently.
- Dashboard: first-run consent modal (localStorage `skulk-telemetry-consent-seen`
  is the browser-local no-nag marker; dismissal leaves the fleet setting
  `unasked`), permanent toggles in Settings, and `GET /v1/telemetry/preview`
  shows the exact pending batch.

### Tracing

| Endpoint | Method | What |
|---|---|---|
| `/v1/tracing` | GET / PUT | Cluster tracing on/off |
| `/v1/telemetry/preview` | GET | Field-telemetry consent state + the exact pending sample batch |
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
| `/v1/diagnostics/telemetry` | GET | Aggregate local telemetry admission and isolated-egress metrics |
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

- `TextGeneration`: chat / responses / messages / ollama-chat
- `TextEmbedding`: embeddings
- `ImageGeneration`: images.generations
- `ImageEdits`: images.edits
- Sentinel: `Shutdown`, `CANCEL_ALL_TASKS`

### Chunks

`src/skulk/shared/types/chunks.py`. Per-token output:

- `TokenChunk`: text / tool / token-level metadata
- `ToolCallChunk`: tool calls
- `ErrorChunk`: error result; terminal
- `PrefillProgressChunk`: distributed prefill progress
- `ImageChunk`: image generation output
- `EmbeddingChunk`: embedding output

### State

`src/skulk/shared/types/state.py`. Treated as immutable by convention (replaced wholesale by `apply()` rather than mutated in place); the model itself is not declared `frozen=True` on `model_config`, so direct mutation is technically possible but considered a bug at every call site.

- `instances: Mapping[InstanceId, Instance]`: placed model instances (each carries shard assignments + per-runner state)
- `runners: Mapping[RunnerId, RunnerStatus]`: per-runner status union
- `downloads: Mapping[NodeId, Sequence[DownloadProgress]]`: durable completed/failed download outcomes per node; live pending/ongoing progress resides in `TelemetryView` and is overlaid for consumers
- `tasks: Mapping[TaskId, Task]`: in-flight or recently-completed tasks
- `last_seen: Mapping[NodeId, datetime]`: last indexed control-plane event per node; **not** a heartbeat or proof of current liveness, and stale by design for healthy nodes with no changing control state
- `topology: Topology`: cluster-wide node graph + capabilities (encoded/decoded via `TopologySnapshot` for JSON round-tripping)
- `tracing_enabled: bool`: cluster-wide tracing flag
- `last_event_applied_idx: int`: water mark for the local apply
- `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge: Mapping[NodeId, *]`: the **connectivity** per-node maps that stay on the event path because they define the topology graph (see "Connectivity readings stay on the control plane" under the Telemetry plane section). They update at independent frequencies via `NodeGatheredInfo`.
- `node_resources` (slice 1), `node_memory` + `node_system` (slice 2), and `node_identities` + `node_disk` + `node_rdma_ctl` (slice 3) are **not** `State` fields: they moved to the telemetry plane (`TelemetryView`, gossiped on `TELEMETRY`, see "Telemetry plane" above) as part of #279. `State` keeps `extra="forbid"`, so a pre-#279 snapshot carrying the old `nodeResources`/`nodeMemory`/`nodeSystem`/`nodeIdentities`/`nodeDisk`/`nodeRdmaCtl` keys is rejected, which is the intended behavior, since mixed-version clusters are unsupported and a node never reloads its own persisted `State` across restart anyway (identity is ephemeral; State is rebuilt from the event log / state-sync). An earlier before-validator that stripped those keys was removed in #294 because it broke state-sync (it forced strict Python-mode validation, rejecting ISO datetime strings like `lastSeen`).
- `thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]]`: detected Thunderbolt-bridge cycles where every node has it enabled (>2 nodes)

Note: there is no `master_node_id` field on `State`. Master identity lives outside the event-sourced state: each node tracks the current master independently via the election protocol (`src/skulk/shared/election.py`). `placements` is also not a field; placement information is derived from `instances` (each `Instance` has its own shard assignments).

### Diagnostics

`src/skulk/shared/types/diagnostics.py`. Major models:

- `NodeDiagnostics`: runtime + identity + resources + processes + supervisor_runners + placements + warnings. Cross-node reads use the tolerant operational decoder, which ignores unknown additive fields recursively; missing additive stream-reclamation counters default to zero. `warnings` includes mixed-build, mixed-DATA-transport, and **leaked-wired-memory** alerts (`_leaked_wired_warning` in `src/skulk/api/main.py`). The wired-memory warning is emitted when `resources.current_wired` exceeds ~5GB with zero `process_alive` runners (the signature of wired memory leaked by an abnormal Metal termination that only a reboot reclaims, #239). Server-side counterpart of `tests/preflight_mem.sh`. To stop a doomed runner from compounding such a leak, the worker circuit-breaks runner crash loops (`CrashWindow` in `src/skulk/utils/crash_window.py`, 3 failures within 60s) and gives up rather than relaunching it. The give-up action depends on *why*: a genuine crash or GPU wedge deletes the instance via `DeleteInstance`, but a **memory fit refusal** (the pre-spawn guard rejecting the shard) sends `RefuseInstancePlacement` instead, so the master re-places the model one node wider rather than letting it silently vanish (#290). A third trigger is a **first-status-report deadline** (#272, `_RUNNER_FIRST_REPORT_DEADLINE_SECONDS`, 120s, via `runners_never_reported`): a runner frozen between spawn and its first status report (SIGSTOP, a hang in early import) never trips the crash breaker (the process is alive) and stalls `ConnectToGroup` forever (the group-init gate waits for every rank to report), so on expiry the worker gives the instance up through the same edge-latched breaker. The supervisor distinguishes "reported idle" from "never reported" via `has_reported_status` because `status` defaults to `RunnerIdle` before the process ever speaks.
- `ProviderDiagnostics`: bounded process-local provider evidence included in `NodeDiagnostics.provider`: active unary/stream calls and concurrency limits, stream admissions and overload rejections, caller-input queue depth, frame and inline-media byte volume, first-output/lifetime timing, terminal outcomes, cancellation requests, and missing terminals. Values are aggregated and keyed only by qualified capability ID; completed call IDs and payloads are not retained.
- `NodeResourceDiagnostics`: gathered_memory, current_memory, **current_wired** (OS-level wired in use; macOS-only via `read_wired_memory_bytes`/psutil, since MLX's own accounting can't see leaked wired), disk, system, network. `current_wired` is read locally on the diagnostics path and deliberately kept OFF the gossiped `MemoryUsage` so the `NodeGatheredInfo` event wire format is unchanged across a mixed-version rollout.
- `MemoryUsage`: ram_total, ram_available, swap_total, swap_available. On macOS, `ram_available` is the GPU-wireable figure `total − wired − anonymous − compressor` from a `vm_stat` snapshot taken per telemetry sample (`MachMemoryCategories` / `parse_vm_stat_output` in `src/skulk/shared/types/profiling.py`), falling back to mactop's raw `available` (free+inactive+speculative, which counts reclaimable file cache as used) when `vm_stat` fails. Value-only change: the gossiped shape is unchanged, so mixed-version clusters interoperate.
- `RunnerSupervisorDiagnostics`: flight_recorder, status, phase, MLX memory, in_progress_tasks, milestones
- `RunnerFlightRecorderEntry`: at, phase, event, detail, attrs, context, mlxMemory
- `MlxMemorySnapshot`: active, cache, peak, wired_limit (MLX's configured limit, not OS wired usage)
- `ClusterDiagnostics`: fan-out wrapper with aggregate `versionStatus`; every `ClusterNodeDiagnostics` entry carries its own comparison status and remains present when reachable
- `ClusterTimeline`: cross-rank merged: runners (synopsis) + timeline (entries sorted by `at`) + unreachableNodes
- `DiagnosticCaptureResponse`: capture bundle (process samples, flight recorder, MLX memory)

### Node facts

`src/skulk/shared/types/node_facts.py` (+ derivation in `src/skulk/facts/derive.py`, doctor verdicts in `src/skulk/doctor/checks.py`); all frozen:

- `NodeFacts`: everything one probe pass observed/was-declared on this node; no derived capability. `platform`, `gpus: tuple[GpuDeviceFact, ...]` (all vendors, all devices; each with `vendor`, `name`, `index`, `detection_source` in `nvml`/`nvidia_device_node`/`amdgpu_sysfs`/`apple_platform`, `vram_total_bytes`, `gtt_total_bytes`, `compute_capability`), importable-dep flags (`pynvml_importable`, `llama_cpp_importable` + `llama_cpp_gpu_offload`, `mlx_audio_importable`), `EngineBinaryFact` per engine env var (state `not_configured`/`ok`/`missing`/`not_executable`), `llama_server_device_probe: LlamaServerDeviceProbe` (`--list-devices` outcome + computes), and the raw declared `SKULK_*_BACKENDS` strings verbatim.
- `CapabilityConflict`: one loud observation-vs-declaration disagreement: `code` (`gpu_serving_disabled` [error] / `gpu_detection_degraded` / `invalid_engine_binary` / `backend_override_conflict` [warn]), `message`, `remediation` (composed on the owning node). Advertised on `NodeResources.capability_conflicts`; mapped one-to-one onto `nodeHealth` reasons. Severity source of truth: `CONFLICT_ERROR_CODES`.
- `BackendDerivation` (`facts/derive.py`): the pure result of `derive_node_backends(facts)`: `backends: frozenset[str]` (the tags to advertise), `conflicts`, and informational `notes` (one warning-level log line each).
- `CheckResult` (`doctor/checks.py`): one doctor verdict, self-contained for rendering: `check_id`, `title`, `verdict` (`ok`/`degraded`/`fail`), `detail`, `consequence`, `remediation`, `fix_available` (whether `skulk doctor --fix` can remediate safely).

### Model card

`src/skulk/shared/models/model_cards.py::ModelCard`. Fields:

- `model_id`, `family`, `quantization`, `base_model`, `n_layers`, `hidden_size`, `num_key_value_heads`
- `tasks: list[ModelTask]`: what task types this model serves
- `capabilities: list[str]`: text / vision / thinking / thinking_toggle / embedding / tts / stt
- `context_length`, `storage_size`, `supports_tensor`, `trust_remote_code`, `is_custom`
- `source_revision: str | None`: full immutable Hugging Face commit used for qualified artifacts; `None` follows mutable `main`
- `vision: VisionCardConfig | None`: image_token_id, model_type, BOI/EOI tokens
- `reasoning: ReasoningCardConfig | None`: supports_toggle, supports_budget, format, default_effort
- `modalities: ModalitiesCardConfig | None`: supports_native_multimodal, supports_audio_input
- `audio: AudioCardConfig | None`: kind (`tts`/`stt`), default_response_format, response_formats, supports_streaming, supports_realtime, supports_voice_listing, voices, default_voice, supports_reference_audio, supports_translation, sample_rates. TTS `supports_streaming=true` enables the stable chunked HTTP/provider path without an experiment gate.
- `tooling: ToolingCardConfig | None`: tool_call_format, supports_tool_calling, builtin_tools
- `runtime: RuntimeCapabilityCardConfig | None`: prompt_renderer, output_parser, metal_fast_synch, mtp_heads, mtp_max_depth, mtp_sidecar_repo, mtp_norm_convention, mtp_concat_order, assistant_model_repo, speculative_multi_node (set `false` where multi-node speculation measures slower than plain sharded decode, e.g. gemma-4-26B-A4B MoE, 2026-06-06 matrix: 30.2 plain vs 28.2 MTP on 2 nodes; single-node speculation unaffected; card-driven so the agreement collective stays rank-symmetric)
- `placement: PlacementCardConfig` (the only section the planner reads directly, `master/placement.py`). `compatible_backends: frozenset[str]` is a **hard filter** (route only to nodes whose advertised `NodeResources.backends` intersect it; default `{"mlx"}`). `backend_preference: tuple[str, ...]` is a **soft, ordered** rank among those backends: the planner prefers a cycle that can serve an earlier-listed tag (`_cycle_backend_preference_score`) and the runner picks the earliest backend the node has. Backends use compound `<engine>-<compute>` tags (`mlx-metal`, `mlx_audio-metal`, `llama_cpp-vulkan`, `llama_cpp-rocm`, and so on; vocabulary + node probing in `src/skulk/shared/backends.py`); nodes also advertise the bare engine tag (`mlx`) for back-compat with original `{"mlx"}` cards, and `mlx_audio` for speech-capable macOS nodes when `mlx_audio` imports. The split is deliberate: filter answers "which nodes are allowed", preference answers "fastest for *this* model" (Vulkan vs ROCm performance is model-dependent), so a Vulkan-preferring model still degrades gracefully onto a ROCm-only node. Also `min_vram_gib` (hard) and `max_context_tokens` (soft KV-budget cap).

### Capability profile

`src/skulk/shared/models/capabilities.py::ResolvedCapabilityProfile`. Computed at request time from card + tokenizer + task params:

- `family` (string)
- `supports_thinking`, `supports_thinking_toggle`, `supports_thinking_budget`
- `supports_image_input`, `supports_audio_input`, `supports_native_multimodal`
- `supports_speech_synthesis`, `supports_transcription`, `supports_speech_translation`
- `supports_audio_output`, `supports_realtime_audio`, `default_audio_response_format`, `audio_response_formats`
- `supports_tool_calling`
- `thinking_format: ReasoningFormat`: None_ / TokenDelimited / ChannelDelimited
- `default_reasoning_effort`, `disabled_reasoning_effort`
- `prompt_renderer: PromptRendererType`: Tokenizer / Gemma4 / Dsml
- `output_parser: OutputParserType`: Generic / Gemma4 / GptOss / DeepseekV32
- `tool_call_format: ToolCallFormat`: Generic / Gemma4 / GptOss / Dsml
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

Inventory snapshot; see #130 for consolidation plan.

| Family | Total lines | Primary locations |
|---|---|---|
| Gemma 4 | ~600 | `gemma4_prompt.py`, vision tower wrapping in `utils_mlx.py:333-456`, `parse_gemma4_thinking_channels` in `model_output_parsers.py`, native-vision branches in `generate.py:1337-1900` |
| Qwen (5 variants) | ~350 | `QwenShardingStrategy` in `auto_parallel.py:1267-1567` |
| DeepSeek V3.2 | ~350 | `dsml_encoding.py`, `parse_deepseek_v32` in `model_output_parsers.py:374-516` |
| GLM-4 (Lite + MoE) | ~280 | Two strategies in `auto_parallel.py` |
| MiniMax | ~225 | `MiniMaxShardingStrategy` + custom attention wrapper in `auto_parallel.py:1148-1226` |
| NemotronH | ~210 | `NemotronHShardingStrategy` + Mamba2 hybrid cache |
| GPT-OSS | ~180 | MLX: `parse_gpt_oss` (token-level Harmony parser via `openai_harmony`) + `GptOssShardingStrategy`. llama.cpp: `HarmonyTextParser` in `harmony_text_parser.py` reparses the harmony channel markers from llama.cpp's detokenized *string* deltas (the engine exposes no token ids), splitting `analysis`→reasoning / `final`→content and stripping markers; wired in `llama_cpp/runner._generate`, gated on `OutputParserType.GptOss`, and dependency-free (no MLX/openai_harmony) so it runs on non-Mac GPU nodes. |
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
| `model_store.staging` | `enabled`, `node_cache_path`, `cleanup_on_deactivate` (default true; gates the recency pass, off = no auto-eviction), `staging_keep_recent_gb` (default 40, recency floor for idle copies; 0 = strict evict-on-deactivate) | Staging behavior. Recency pass is lifecycle-triggered (deactivate + startup), NOT disk-pressure-triggered. Store-delete (`EvictStagedModel`) and `POST /store/purge-staging` are separate unconditional paths that ignore both the toggle and the budget |
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
| `SKULK_EXTENSIONS_DISABLE` | `1` skips extension (plugin) discovery entirely on this node; see Extensions component section |
| `SKULK_GPU_TELEMETRY_VENDOR` | `nvidia` pins Linux GPU telemetry (and the worker VRAM fit guard) to the NVIDIA adapter on mixed-GPU hosts; default prefers AMD sysfs, then NVML |
| `SKULK_TELEMETRY_DISABLE` | `1` hard-disables field telemetry on this node regardless of the fleet consent setting |
| `SKULK_ENABLE_EXPERIMENTAL_MODE` | Node-local master gate for in-development features (off unless truthy: `1`/`true`/`yes`/`on`). Read by `experimental_mode_enabled` (`src/skulk/shared/experimental.py`); surfaced in `GET /config` as `effective.experimental_mode_enabled`. When on, the dashboard reveals the "Experiments" settings section; when off, experimental features stay inert. `speech_translation` is the active speech experiment; `tts_streaming` and `stt_realtime` are deprecated compatibility fields. |
| `SKULK_MLX_HANG_DEBUG` / `SKULK_MLX_HANG_DEBUG` | Emit periodic stack traces from stuck phases |
| `SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS` | Interval for above (default 30s) |
| `SKULK_MAX_OUTPUT_TOKENS` / `SKULK_MAX_TOKENS` | Default `max_tokens` (cluster default 4096; `DEFAULT_MAX_OUTPUT_TOKENS` constant) |
| `SKULK_NO_BATCH` / `SKULK_NO_BATCH` | Disable continuous batching |
| `SKULK_KV_CACHE_BACKEND` / `SKULK_KV_CACHE_BACKEND` | KV cache backend selection (overrides config) |
| `SKULK_LIBP2P_NAMESPACE` / `SKULK_LIBP2P_NAMESPACE` | libp2p namespace for cluster isolation |
| `SKULK_LLAMA_CPP_BACKENDS` | Comma-separated llama.cpp compute backends this node was built with, e.g. `vulkan` or `vulkan,rocm` (valid: `vulkan`/`rocm`/`cuda`/`cpu`; `metal` is MLX-only and ignored). **Semantics changed by #614: an OVERRIDE over facts-based derivation, no longer the sole source.** Declared, it wins (a GPU compute no observed hardware supports raises a `backend_override_conflict` but is still honored). Unset, derivation takes over: a binding that positively reports GPU offload (`llama_cpp.llama_supports_gpu_offload()`) on a node with a visible GPU derives its tags from the hardware (NVIDIA -> `cuda`; AMD -> `vulkan,rocm`); otherwise the CPU floor (`llama_cpp-cpu`). Inert until a node has `llama_cpp` importable. The GPU-offload cross-check still applies to declarations: a declared GPU backend on a wheel with no GPU offload compiled in (the classic case where `uv sync` restored the CPU-only PyPI wheel over a source-built GPU wheel) is dropped to `llama_cpp-cpu` with a loud conflict, so GPU GGUF work is not routed to a degraded build. The service entrypoint (`deployment/install/skulk-startup.sh`) runs `uv sync --inexact` when this declares a GPU backend, so a routine sync does not prune the source-built wheel in the first place. |
| `SKULK_LLAMA_CPP_LOGITS_ALL` | Whether the llama.cpp runner loads each GGUF with `logits_all=True`, enabling per-request logprobs (`src/skulk/worker/runner/llama_cpp/runner.py`, `_logits_all_enabled`). Defaults **off**: `logits_all` makes llama.cpp pre-allocate an `n_ctx * vocab * 4` logits buffer up front, which at the model's full trained context is enormous (e.g. `131072 * 152064 * 4` = 74 GiB for a Qwen2.5 vocab) and OOMs the node on load. So logprobs is opt-in (`=1`); the runner is loaded once and the flag cannot be toggled per request. With it off a logprobs request degrades to a clear error chunk. Regardless of this flag the served context window is the instance's memory-fit ceiling (`serving_n_ctx` returns `instance_context_token_limit`: the largest context that fits the hosting node's working set after weights and overhead, capped at the card's advertised max), never the model's full trained context (`n_ctx=0`), which would size the KV cache beyond available memory and exhaust the node on load. Placement admits the node against the SAME working set (VRAM on a GPU node) this ceiling is derived from, so the KV cache fits at that window by construction, and on an admitted node it is at least the `KV_CONTEXT_BUDGET_TOKENS` admission floor -- except when the model's own advertised max context is smaller (then it serves that smaller max), when the fit is uncomputable for a gguf card (no `num_key_value_heads`, missing memory, RPC-donor shard), or for a gguf instance on a node WITHOUT discrete VRAM (whose fit is derived from static `ram_total` while load competes with live `ram_available`); in those cases `instance_context_token_limit` clamps back to the floor so llama.cpp never preallocates a window that OOMs on load. The large served context is therefore a discrete-VRAM/GPU behavior; CPU / non-discrete-VRAM gguf nodes keep 8192. This replaces the previous fixed 8192 clamp, which made served models unusable for real-context work (a codebase does not fit in 8192 tokens). |
| `SKULK_LLAMA_CPP_FLASH_ATTN` | Whether the llama.cpp runner loads each GGUF with Flash Attention (`src/skulk/worker/runner/llama_cpp/runner.py`, `_flash_attn_enabled`). Defaults **on** (`=1`): FA is the modern llama.cpp default and matters most for models with per-layer-varying V embeddings (gemma's interleaved sliding-window attention), where without it llama.cpp pads the V cache and falls back to a full-size SWA cache, wasting VRAM and slowing attention. It is a load-time construction flag (cannot be toggled per request). Set `=0` to disable on a backend whose compiled build lacks Flash Attention kernels. |
| `SKULK_LLAMA_SERVER_BIN` | Path to the external `llama-server` binary the served-backend engine (`llama_server`) launches and proxies (`src/skulk/worker/runner/llama_server/runner.py`). When set to an existing executable, the node advertises the `llama_server` engine (+ compound tags) via `_probe_served_backends` and becomes a placement candidate for served-engine cards; unset means the node never advertises it. The binary must be recent enough to expose `--spec-type` (>= llama.cpp b9196 for `draft-mtp`). |
| `SKULK_LLAMA_SERVER_BACKENDS` | Compute backends the `llama-server` build was compiled with (comma-separated, e.g. `vulkan` or `vulkan,rocm`; same vocabulary as `SKULK_LLAMA_CPP_BACKENDS`). **Semantics changed by #614: an OVERRIDE over derivation, no longer the sole source.** Declared (or falling back to a `SKULK_LLAMA_CPP_BACKENDS` declaration; the GPU is the same whichever engine drives it), it wins. Unset, derivation asks the configured binary itself (`llama-server --list-devices`, ground truth for what the build can drive on this machine), then infers from observed GPU hardware (NVIDIA -> `cuda`, AMD -> `vulkan`), then floors at `cpu`; a CPU-only resolution on a node with a visible serving GPU raises the error-severity `gpu_serving_disabled` conflict instead of silently launching `-ngl 0`. |
| `SKULK_VLLM_BIN` | Path to the `vllm` CLI the served-backend `vllm` engine launches (`vllm serve`; `src/skulk/worker/runner/vllm/runner.py`). When set to an existing executable and a GPU backend is declared, the node advertises the `vllm` engine (+ `vllm-cuda`/`vllm-rocm`) via `_probe_vllm_backends` and becomes a placement candidate for vLLM cards; unset means the node never advertises it. |
| `SKULK_VLLM_BACKENDS` | Compute backends the vLLM install targets (comma-separated). vLLM is GPU-only in scope, so only `cuda`/`rocm` are honored (`metal`/`vulkan`/`cpu` are ignored). Unset falls back to the node's `SKULK_LLAMA_SERVER_BACKENDS` then `SKULK_LLAMA_CPP_BACKENDS` declaration; with no declaration anywhere in the chain, derivation (#614) infers from observed GPU hardware (NVIDIA -> `cuda`, AMD -> `rocm`). There is deliberately no `cpu` floor: a vLLM node with no GPU backend is not a useful placement target and advertises nothing (with a loud conflict when `SKULK_VLLM_BIN` is set anyway). |
| `SKULK_VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU VRAM `vllm serve` may use for weights + KV cache (`--gpu-memory-utilization`; `src/skulk/worker/runner/vllm/runner.py`). Default `0.90` (vLLM's own default); an unparseable or out-of-`(0, 1]` value falls back to the default. A node-local serving knob for now; a card-level override arrives with vLLM-aware admission. |
| `SKULK_VLLM_MAX_CONCURRENT_REQUESTS` | Upper bound on concurrent in-flight generations the vLLM runner streams to one `vllm serve` at once (the dispatch `ThreadPoolExecutor` width; `src/skulk/worker/runner/vllm/runner.py`). Default `32`; an unparseable or below-1 value falls back to the default. A client-side admission bound (an N-permit semaphore caps submitted jobs, so excess load backpressures the task receiver rather than piling up an unbounded in-process queue), NOT vLLM's batch width (the server batches up to its own `--max-num-seqs`, default 256). |
| `SKULK_RPC_SERVER_BIN` | Optional path to the `ggml-rpc-server` binary an RPC memory-donor runner launches (#328 multi-node GGUF pooling; `src/skulk/worker/runner/rpc_donor/runner.py`). Unset, the donor looks for `ggml-rpc-server` next to `SKULK_LLAMA_SERVER_BIN` (both build from one llama.cpp tree with `-DGGML_RPC=ON`; the upstream target was renamed from `rpc-server`). Neither present means a donor runner fails loudly at spawn. |
| `SKULK_NO_ENGINE_AUTOPROVISION` | `1` disables engine auto-provisioning on this node (`src/skulk/provisioning/llama_server.py`). By default a Linux node with no `SKULK_LLAMA_SERVER_BIN` override downloads the pinned, checksum-verified upstream llama-server build at startup (Vulkan variant when an NVIDIA/AMD GPU is visible, else CPU) and exports `SKULK_LLAMA_SERVER_BIN` for the process. Node-local launch policy; explicit binary overrides always win regardless of this flag, and an invalid override is never masked by a managed binary. Managed builds install under the `SKULK_ENGINES_DIR` constant (`SKULK_DATA_HOME/engines`, so `SKULK_HOME` relocates it; not an env var of its own), keyed as `engines/llama-server/<pin>/<variant>/`. |
| `SKULK_LLAMA_SERVER_FORCE_NO_SPEC` | When truthy (`1`/`true`/`yes`/`on`), the served runner (`_force_no_spec`) ignores a card's `served_spec_type` and launches `llama-server` without any `--spec-type` / `--model-draft` flags, so the same GGUF serves in plain decode. This is the apples-to-apples **MTP off** baseline for an on-vs-off served throughput comparison (identical weights, speculation disabled), and a debug lever for a misbehaving spec pairing. Node-level, read at runner launch; unset in normal operation. |
| `SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX` | Context-length cap (tokens, default `8192`) applied **only when `SKULK_LLAMA_CPP_LOGITS_ALL=1`** (`_logits_all_n_ctx`). Bounds the `logits_all` buffer (`n_ctx * vocab * 4`) so opting into logprobs does not blow up memory: at an ~150k vocab, 8192 is ~5 GiB. It is operator policy, so raising it far above the default reintroduces the large allocation it guards against. When logits_all is off the served context is the instance's admission ceiling (`_serving_n_ctx`), not the model's full trained context. |
| `SKULK_ZENOH_DATA_PLANE` | Soft default-on (#315). Resolved by `_resolve_zenoh_enabled` in `Node.create` (`src/skulk/main.py`): truthy (`1`/`true`/`yes`/`on`) forces the Zenoh DATA plane on (requires `SKULK_ZENOH_LISTEN`, #308), falsy (`0`/`false`/`no`/`off`) forces gossipsub, and **unset** uses Zenoh only when `SKULK_ZENOH_LISTEN` is set (else gossipsub, so a bare node never crashes on the listen requirement). When on, the `DATA` topic (per-token output) rides an Eclipse Zenoh peer session instead of gossipsub; all other planes stay on libp2p. Wired in `Router` (`uses_zenoh`). **Security (#308):** the session is **namespace-isolated** (keys prefixed by a segment that is a collision-resistant SHA-256 hash of the exact token libp2p isolates on (`SKULK_LIBP2P_NAMESPACE` when set, else the `NETWORK_VERSION` default `v0.0.1`; mirrors `swarm.rs`, not the legacy `EXO_LIBP2P_NAMESPACE`). Neither the raw token nor the derived namespace is logged (with no TLS the namespace is the only isolation value); startup logs only a short non-routing fingerprint), so a peer on a different namespace does not receive this fleet's `data` (parity with the libp2p private namespace). This is isolation between distinct clusters, NOT confidentiality against an adversary already on the same Zenoh network: the seed is non-secret operator config (also surfaced in `/v1/diagnostics/node`) and there is **no transport auth/TLS** by default, so on an untrusted network either enable Zenoh TLS (operator-configured) or keep it firewalled; a loud startup warning fires when on. |
| `SKULK_ZENOH_LISTEN` | Zenoh listen endpoint when the data plane is on. **Required explicitly** (#308 bind restriction): Skulk refuses to start the plane with this unset rather than silently binding `tcp/0.0.0.0:7447` (all interfaces). Set a specific private IP, e.g. `tcp/192.168.0.115:7447`; an explicit `0.0.0.0` is allowed but warns. |
| `SKULK_ZENOH_CONNECT` | Comma-separated explicit Zenoh peer endpoints (multicast scouting is off, so peers are explicit), e.g. `tcp/192.168.0.117:7447,tcp/192.168.0.122:7447`. Per-node. |
| `SKULK_DATA_REORDER_BUFFER` | Explicit override for the data-plane reorder buffer (#279 Phase 3). Unset (default): the buffer follows the DATA transport - ON for gossipsub (it reorders; the #301 fix), OFF for Zenoh (per-publisher FIFO, so arrival order is generation order; validated 20/20 on a 3-node sampled-MTP matrix). Set `1`/`0` to force it on/off regardless of transport (testing / belt-and-suspenders). Read in `API.__init__` (`_reorder_buffer_enabled`), with the transport signalled by `data_plane_zenoh` from `Node.create`. |
| `SKULK_SKIP_LLM_WARMUP` | Skip warmup synthesis (single-node debug only) |
| `SKULK_IMAGE_TRANSPORT_DEBUG` | Verbose logging in image-transport pipeline |
| `SKULK_VISION_DEBUG_SAVE_DIR` | Save debug image artifacts |
| `SKULK_NATIVE_VISION_REFERENCE_PATH` | Force native-vision reference path (Gemma 4) |
| `SKULK_NODE_NAME` | Node display-name override used ahead of the hostname/Computer Name fallback. Node-local launch config for machines whose hostname is meaningless: containers and rented GPU pods get runtime-random hostnames (`a21147cd1ae7`) and an unprivileged container cannot call sethostname (#555) |
| `SKULK_OFFLINE` | Run without internet checks (no model fetching) |
| `SKULK_HEADLESS` | Deploy knob read by `deployment/install/skulk-startup.sh` (the LaunchAgent/systemd entrypoint). `1` on a node that serves the API without the web UI (e.g. a non-Mac worker like a Strix Halo/ROCm box with no Node/npm): boot-time prep skips the dashboard build and its otherwise-fatal `dashboard-react/dist` missing check, and the node runs with `DASHBOARD_DIR` unset (#333). Default `0` keeps the fail-loud behavior so a Mac with an accidentally-absent build is caught. |
| `VITE_TOLGEE_CDN_PREFIX` | Dashboard build-time env var. CDN/static prefix for Tolgee JSON bundles, default `/i18n`; Tolgee fetches namespaced bundles as `{prefix}/{namespace}/{language}.json`, and the dashboard uses the `skulk` namespace. |
| `VITE_TOLGEE_AVAILABLE_LANGUAGES` | Dashboard build-time env var. Comma-separated language tags available from the Tolgee CDN/static prefix; `en` is always included and bundled as the fallback namespace. |
| `SKULK_TEST_DISTRIBUTED_MODEL` | Tests only: force the distributed/prefix-cache slow-test model (`gpt-oss-20b` or `llama-3.2-1b`); default auto-selects by Metal working-set size |
| `MLX_METAL_FAST_SYNCH` | Set by Skulk based on resolved card preference; not for direct operator use |
| `MLX_HOSTFILE`, `MLX_RANK`, `MLX_RING_VERBOSE`, `MLX_IBV_DEVICES`, `MLX_JACCL_COORDINATOR` | MLX upstream env vars; auto-set by Skulk during distributed init. Ring hostfile addresses are chosen per neighbor pair from OBSERVED libp2p connections, ranked thunderbolt > maybe_ethernet > ethernet > wifi > unknown > VPN/overlay. Tailscale CGNAT (100.64/10, fd7a:115c:a1e0::/48) addresses are detected by ADDRESS (utun types don't gossip) and rank strictly last: the overlay exists for external reachability and may be DERP-relayed, so it is only used when a pair has no local candidate (#265). Selection lives in `_find_ip_prioritised` / `get_mlx_ring_hosts_by_node` (`src/skulk/master/placement_utils.py`) |

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
  - `record_runner_phase(phase, event=..., detail=..., attrs=..., include_memory=False)`: fire one entry
  - `runner_phase(phase, detail=...)`: context manager: enter / exit pair

### Trace sessions

- **Lives at:** `src/skulk/shared/tracing.py`
- **API:**
  - `begin_trace_session(task_id, rank, node_id, model_id, task_kind, tags)`: create
  - `record_trace_marker(name, rank, task_id, attrs)`: emit one event
  - `trace(category, name, ...)`: context manager / decorator
  - `pop_trace_session(task_id)`: collect + remove
  - `clear_trace_session(task_id)`: remove without collecting
- **Storage:** module-level dict `_trace_sessions: dict[str, TraceSession]`
- **Cluster path:** runner emits `TracesCollected` IPC per rank → runner supervisor sends one owner-addressed `TRACE_DATA` packet → owning API assembles the expected ranks and persists Chrome-trace JSON to disk

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

- `src/skulk/shared/logging.py`: loguru JSON sink to stdout
- `deployment/logging/vector.yaml`: Vector pipeline (stdin → VictoriaLogs)
- `deployment/logging/docker-compose.yml`: VictoriaLogs + Grafana stack
- `skulk.yaml` `logging.enabled` + `logging.ingest_url`: opt-in; cluster-synced

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

### Engine pin advancement (deliberate, never automatic)

The managed llama-server engine is pinned (`LLAMA_SERVER_PIN` in
`src/skulk/provisioning/manifest.py`, currently `b10068`) so upstream churn is
a discrete validated event, not a continuous silent risk (the same
`source_revision` doctrine model cards apply to Hugging Face artifacts).
Advancing the pin is a checklist, and skipping steps is how a silent behavior
change ships:

1. Bump `LLAMA_SERVER_PIN` and re-record every artifact sha256 in
   `LLAMA_SERVER_ARTIFACTS` from the new release's asset digests
   (`gh api repos/ggml-org/llama.cpp/releases/tags/<tag>`).
2. Bump the wheel version in `packaging/skulk-llama-server-cuda/pyproject.toml`
   (scheme `0.<build>.<rev>`) and the pinned range in `install.sh`; the
   `engine-wheel` workflow guard fails if any of the three disagree.
3. Rebuild and publish the wheel (`engine-wheel` workflow, dispatch with
   `publish=true`).
4. Run the fresh-box gauntlet (RunPod runbook section 11 in foxlight-docs)
   against the new pin before it reaches dev: install path, doctor verdicts,
   GPU-speed serving.
5. Check upstream release notes for served-API/flag behavior changes (the
   harness #69 keepalive change is the canonical example of what a pin bump
   can smuggle in) and update runner/proxy assumptions if needed.

The AGENTS.md "Documentation" section requires updates here when architectural shape changes:

- New component → add to "Components"
- New pubsub topic → add to "Pubsub topics"
- New event / command type → add to "Events" / "Commands"
- New state field → update "State" Pydantic model section
- New major API endpoint → add to the right "API endpoints" sub-table
- New family adapter → update "Family-specific code locations"
- New environment variable → add to "Configuration knobs"

Keep entries terse. Narrative belongs in [Architecture](architecture).
