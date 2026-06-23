<!-- Copyright 2025 Foxlight Foundation -->

# Changelog

This project records release notes here and mirrors public-facing notes in
`website/docs/release-notes/`.

## [Unreleased]

### Fixed

- **Auto-imported Qwen3 reasoning models no longer return empty content (#384).**
  A Qwen3-family model with no built-in card (a fresh quant imported on demand,
  e.g. `Qwen3.6-35B-A3B-nvfp4`) arrived with empty capabilities, so its resolved
  profile reported no thinking and no thinking toggle. Thinking is on by default
  for Qwen3, so the model reasoned unconditionally and a normal chat request
  spent its whole token budget on the reasoning channel, returning empty
  `content`. Capability resolution now recognizes the Qwen3 / Qwen3.5 / Qwen3.6
  family (token-delimited `<think>` toggle), so an auto-imported variant is
  treated as toggle-capable and the dashboard/API off-by-default path can
  suppress thinking. Built-in cards keep their explicit declarations, an explicit
  `reasoning` section still overrides the family default, and Coder variants
  (instruct-only, no thinking) are excluded.

- **Context-length and other runner errors now surface as structured errors on
  the Claude, Responses, and Ollama wire formats (#276).** Those adapters
  previously raised on a runner error (a 500 on the non-streaming path) or broke
  out of the stream and then emitted an empty successful completion (a bogus
  success for what is actually a clean request rejection). Since the API streams
  every response (the HTTP status is committed to 200 before generation), each
  adapter now emits a structured error envelope in the body and stops, reusing
  the same `error_chunk_response` mapping the OpenAI chat-completions surface
  uses: a `context_length_exceeded` rejection becomes an `invalid_request_error`
  (400), everything else an internal error (500). No more empty-success-on-error
  for clients feeling out context limits.

- **A runner that never reports after spawn no longer stalls an instance forever
  (#272).** A runner frozen between spawn and its first status report (a
  SIGSTOP, a hang in early import or device init) left the instance stuck in
  pre-init coordination indefinitely: `ConnectToGroup` is only planned once
  every rank has reported, and the crash breaker never tripped because the
  process was alive. The worker now applies a first-status-report deadline
  (`_RUNNER_FIRST_REPORT_DEADLINE_SECONDS`, 120s); a runner silent past it gives
  the instance up through the same circuit breaker, so the placement fails and
  recovers instead of hanging. The deadline is generous enough for slow imports
  and weight mmaps.

- **A rank's failed download no longer wedges a multi-node instance forever
  (#381).** If one rank's model download failed terminally (disk full, a
  transient Hugging Face or network error), the ring still formed and every rank
  waited for all ranks to become load-ready; the failed rank never would, and
  nothing failed or recovered the instance, so it sat "loading" at
  `RunnerConnected` indefinitely until a manual restart. The master's plan loop
  now detects this from replicated state (a not-yet-ready instance whose any rank
  node carries a terminal `DownloadFailed` for the model), fails any in-flight
  request bound to it with the download error surfaced, tears the instance down,
  and re-places the model at the same width excluding the failed node(s). A
  transient or single-node failure self-heals onto healthy nodes; a cluster-wide
  shortfall fails cleanly with the reason (`PlacementError` is terminal, bounding
  recovery to the available nodes) instead of hanging.

- **Node logs are now bounded and cannot fill the disk (#382).** Two paths grew
  without limit: the durable `~/.skulk/logs/skulk.log` rotated only once at
  startup, so a long-lived node grew it forever; and the service-manager capture
  files (`skulk.stderr.log` / `skulk.stdout.log`) accumulated across restarts,
  reaching tens of GB on the fleet. Now `skulk.log` rotates at 100 MB with the
  last few runs kept as compressed archives; the capture files are truncated on
  each restart (keeping a 5 MB tail of the previous run as `*.log.1`, tunable via
  `SKULK_CAPTURE_KEEP_BYTES`); the console sink drops ANSI color when stderr is
  not a terminal so captured logs are plain and greppable; and the service now
  launches at info verbosity by default instead of `-v` debug (the libp2p
  transport firehose was the bulk of the volume). Set `SKULK_VERBOSITY=-v` to opt
  back into verbose logging while debugging.

- **The centralized store now honors a card's pinned GGUF quant on download
  (#344).** A store-routed download re-derived the quant from the model id alone
  (the default preference), so a custom card pinning a non-default quant (e.g.
  `Q8_0`, `Q3_K_M`) silently got the default instead. The store download request
  now carries the card's `gguf_file`: the worker's store client sends it in the
  `POST /models/{id}/download` body, the store fetches that quant's shard group,
  and a pin absent from the repo falls back to the default. Auto-built cards
  (whose pin matches the default) are unaffected.

- **A llama.cpp request between the KV budget and the model's context ceiling is
  now cleanly rejected instead of failing at the runner (#362).** The llama.cpp
  runner allocates its KV cache up front and caps the loaded context to
  `KV_CONTEXT_BUDGET_TOKENS` (8192; `_serving_n_ctx`), but the API's admission
  ceiling (`instance_context_token_limit`) was the memory/card value, often tens
  of thousands of tokens. A request above the budget was therefore admitted and
  then failed or truncated at generation. The admission ceiling for a
  GGUF/llama.cpp instance is now capped to the same budget, so the API returns a
  clear `context_length_exceeded` up front and admission matches what the runner
  serves. (Enabling logprobs lowers the runner window further; this was the
  originally reported case, now subsumed. A node that overrides
  `SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX` *below* the budget remains a narrow per-node
  residual, since the master cannot see node-local env at placement.)

### Added

- **Vision GGUF VLMs now run on the llama.cpp engine (#128).** A vision GGUF
  (LLaVA / Qwen-VL style, with a separate `mmproj` projector) can be served on a
  llama.cpp node: the runner loads the projector through llama-cpp-python's
  multimodal chat handler (the general `MTMDChatHandler` by default, or a
  family-specific handler selected from the card's vision `model_type`), and an
  image request's content is passed inline so the handler splices the image
  features itself. A GGUF repo is marked vision-capable from its `config.json`
  vision section when present, or, for the many GGUF VLM repos that ship no
  `config.json`, from the mere presence of an `mmproj` projector. Validated live
  on an AMD Strix Halo (Vulkan) node serving `Qwen2-VL-2B-Instruct-GGUF`
  (reads text in images and describes structured scenes). `image_token_id` on
  the vision card config is now optional: it is required only by the MLX vision
  path; the llama.cpp handler inserts image features without it.

### Fixed

- **Vision GGUF models now download their multimodal projector (#346).** The
  selective GGUF allow-list (`gguf_allow_patterns`, used by both the direct-
  HuggingFace download and the centralized store) only matched the selected LM
  quant's shard group, so a LLaVA-style vision GGUF fetched its weights but not
  its separate `mmproj-*.gguf` projector, and llama.cpp could not do image
  inference. The allow-list now always includes a `*mmproj*.gguf` glob: it
  matches nothing on a text-only repo (no cost) and pulls the projector on a
  vision repo. Foundation for vision GGUF VLMs on llama.cpp (#128).

### Changed

- **The placement single-node constraint is now a named engine capability
  (#328 groundwork).** The hard-coded "llama.cpp is single-node only" check in
  the planner became `engine_supports_multi_node` (MLX yes; llama.cpp not until
  its RPC backend is wired into the runner). Behavior is unchanged today: a
  model whose only compatible engine is single-node is still pinned to a
  one-node cycle, and a card that also allows a multi-node engine (MLX) still
  places across nodes. This is the single hinge to flip when multi-node
  llama.cpp (RPC) lands.

- **Placement now records the resolved backend on each shard (#330).** The master
  resolves which backend a node will use (the card's `compatible_backends`
  intersected with that node's advertised backends, ordered by
  `backend_preference`) at placement time and stamps the winning tag onto the
  node's shard as `resolved_backend`. The worker reads it at runner-spawn instead
  of re-probing its own backends, so engine dispatch is deterministic from
  replicated state and cannot disagree with the placement decision; it also lets
  a card resolve to different engines per node on a heterogeneous cycle. The
  worker falls back to its local probe when the field is absent (a node whose
  resources had not yet gossiped at placement). Foundation for pluggable engines
  (#284) and multi-node llama.cpp (#328).

### Removed

- **The `EXO_*` environment-variable deprecation runway is gone (#324).** Legacy
  `EXO_*` env vars from the pre-rename (exo to skulk) deployments are no longer
  honored: the package-import alias shim (`skulk/__init__.py`), every `EXO_*`
  fallback in `constants.py` and across the worker/API/store, and the
  `~/.exo` path fallbacks (home dir, model staging, download staging) were
  removed. Only the `SKULK_*` names and `~/.skulk/` paths are read now. The
  whole fleet must run the same Skulk version, so re-set any `EXO_*` vars to
  their `SKULK_*` names before upgrading. (The libp2p private-network pre-shared
  key still derives from the `exo_discovery_network` seed in `swarm.rs`; changing
  that is a wire-compatibility break handled separately as a coordinated
  fleet-wide upgrade.)

### Changed

- **The libp2p private-network key seed is now `skulk_discovery_network` (#324).**
  `swarm.rs` derived the PNET pre-shared key from the literal
  `exo_discovery_network` (an exo-rename residue). The seed is now
  `skulk_discovery_network`. **This is a wire-compatibility break**: a node built
  with the new seed cannot form a libp2p cluster with a node built on the old
  seed, so it must roll out as a single coordinated whole-fleet rebuild and
  restart (do not roll nodes one at a time). The Zenoh data-plane namespace is
  unaffected (it derives from `NETWORK_VERSION` / `SKULK_LIBP2P_NAMESPACE`, not
  this seed).

- **Zenoh data plane is now soft default-on (#315).** The `DATA` topic (per-token
  generation output) uses the Eclipse Zenoh transport by default when a node is
  configured for it. `SKULK_ZENOH_DATA_PLANE` is now tri-state
  (`_resolve_zenoh_enabled`): `1`/`true`/`yes`/`on` forces Zenoh on (still requires
  an explicit `SKULK_ZENOH_LISTEN`, #308), `0`/`false`/`no`/`off` forces gossipsub,
  any other non-empty value is rejected, and
  **unset** is the soft default (Zenoh when `SKULK_ZENOH_LISTEN` is set, else
  gossipsub). A bare node with no Zenoh config (e.g. a fresh `uv run skulk`) stays
  on gossipsub rather than failing the listen requirement, so the listen endpoint
  is the opt-in signal under the default. Control, telemetry, and election planes
  stay on libp2p. Validated by a full e2e suite over Zenoh (coherence across
  dense/MoE single- and multi-node, churn/soak/refusal, master-failover
  continuity).

### Added

- **AMD / Linux GPU nodes can join a cluster and serve GGUF models through
  llama.cpp (#325, #331).** A non-Mac box (validated on an AMD Ryzen AI Max+ 395
  "Strix Halo", `gfx1151`, via the Vulkan backend) joins as a worker that serves
  GGUF models on its GPU alongside Apple Silicon nodes serving MLX. Backends are
  self-describing `<engine>-<compute>` tags (`mlx-metal`, `llama_cpp-vulkan`,
  ...); a model card declares `compatible_backends` (a hard placement filter) and
  `backend_preference` (a soft, graceful-fallback ranking), so a GGUF model lands
  only on a llama.cpp node and an MLX model only on the Macs, automatically. The
  llama.cpp runner is single-node and streams tokens onto the existing data
  plane. See `website/docs/amd-strix-halo-nodes.md`.

- **The llama.cpp engine matches MLX on logprobs and tool calling (#356).** GGUF
  models served on an AMD node support per-token `logprobs` / `top_logprobs`
  (opt-in via `SKULK_LLAMA_CPP_LOGITS_ALL=1`, which loads the model retaining
  per-token logits and caps the served context so the logits buffer stays
  bounded; off by default) and tool calling (a request's `tools` are forwarded;
  a structured tool call is emitted when the model returns one, else its prose).
  Multi-token prediction / speculative decoding remains MLX-only: GGUF models
  advertise no MTP capability, so an AMD node serves plain autoregressive without
  promising a speedup it cannot deliver.

- **Collector-agnostic accelerator telemetry (#353, #354).** Node telemetry now
  carries a vendor-neutral `accelerator` block (vendor / utilization / VRAM /
  power / temperature / clock) filled at the collector boundary: mactop on Apple,
  and a passive-sysfs collector for AMD/Linux GPUs, so a non-Mac GPU node is not a
  telemetry blind spot. The dashboard renders it in a vendor-aware accelerator
  panel.

- **Heterogeneous-node identity in the topology (#355).** A Linux node reports a
  real model / chip / OS (DMI + `/proc/cpuinfo` + `os-release`) instead of
  "Unknown", and the dashboard labels non-Mac nodes correctly rather than
  prefixing "macOS".

- **The model store downloads only the selected GGUF quant (#339).** When the
  store host downloads a multi-quant GGUF repo from HuggingFace on a worker's
  behalf, it now fetches exactly what the direct-HuggingFace path fetches: the
  preferred quant's shard group plus `config.json`, and nothing else (not the
  other quantizations, not `original/*` full-precision weights, not `metal/*`
  artifacts). This matches the selective allow-patterns
  (`resolve_allow_patterns`) the direct path already applies, so a store-routed
  download is no larger than a direct one. Non-GGUF repos are unaffected.

- **GGUF cards can be built from the binary header when no `config.json` is
  present (#327).** A GGUF repo that ships only the `.gguf` weights (no
  `config.json`) now has its structural fields (layer count, hidden size,
  KV-head count, context length) read directly from the selected file's GGUF
  metadata header via a ranged read of the file start, instead of failing the
  card build. Repos that ship `config.json` (most community GGUF repos) still
  use it; the header read is the fallback. Completes the selective-quant GGUF
  download/load path so more llama.cpp repos work without a hand-written card.

- **Zenoh data-plane hardening toward default-on (#308 + #309).** Security
  (#308): the Zenoh session now sets a **namespace** (a collision-resistant
  SHA-256 hash of the exact token libp2p isolates on: `SKULK_LIBP2P_NAMESPACE`
  when set, else the `NETWORK_VERSION` default `v0.0.1`, mirroring `swarm.rs`) so
  foreign peers on a different namespace cannot subscribe to this fleet's `data`,
  restoring parity with the libp2p private namespace; and `SKULK_ZENOH_LISTEN` is
  now **required explicitly** when the plane is enabled rather than silently
  defaulting to `0.0.0.0` (an explicit `0.0.0.0` still works but warns). TLS/ACL
  stay operator-configurable for untrusted networks (documented; not built in).
  Robustness (#309): the DATA plane egresses on its **own outbound loop**, so its
  `CongestionControl::Block` backpressure can no longer stall the shared
  control-plane publish loop (commands/events); and `ZenohSession` publish/
  subscribe no longer hold the publishers/subscribers mutex across the
  `declare`/`put` await, so per-command concurrent publishes don't serialize.

- **Data-plane reorder buffer is now transport-conditional (#279 Phase 3).** The
  per-command `sequence` reorder buffer (the #301 fix for gossipsub reordering
  multi-node output) is now skipped when the DATA plane rides Zenoh, which
  delivers each command's chunks per-publisher FIFO, so output dispatches in
  arrival order, eliminating the per-token buffering/reordering hop. The buffer
  stays ON for the gossipsub default (which reorders). The API selects this from
  the transport (`data_plane_zenoh`, from `SKULK_ZENOH_DATA_PLANE`);
  `SKULK_DATA_REORDER_BUFFER` (`1`/`0`) overrides explicitly. Validated 20/20 on
  a 3-node sampled-MTP coherence matrix with the buffer off. The full removal of
  the `sequence` field and reorder machinery is deferred until Zenoh is the
  default DATA transport.

- **Optional Eclipse Zenoh transport for the data plane (experimental, default
  off).** When `SKULK_ZENOH_DATA_PLANE` is set, the `DATA` topic (per-token
  generation output) rides a Zenoh `peer` session instead of gossipsub; control,
  telemetry, and election planes stay on libp2p. Endpoints are per-node and
  explicit via `SKULK_ZENOH_LISTEN` / `SKULK_ZENOH_CONNECT` (multicast scouting
  off, gossip on, the macOS Local Network Privacy-safe posture). The swap is
  transparent above the transport: `DataChunk`, the per-command `sequence`, and
  the reorder buffer are unchanged, so the two transports are interchangeable
  behind the flag. Publishers use `Reliable` + `Block` on a single priority for
  per-key FIFO. With the flag unset, behavior is identical to before. Foundation
  for #279's data-plane evolution (later phase: removing the app-layer reorder
  buffer once Zenoh's per-publisher ordering is relied on).
- **Zenoh data plane is key-addressed per owner (#279 Phase 2), killing the
  cluster-wide fan-out.** The owning API node stamps its node id on the serving
  command (`owner_node`); the master carries it onto the worker task, and the
  rank-0 supervisor stamps it onto each `DataChunk`. On Zenoh the `DATA` topic
  now publishes to the key `data/<owner_node>` and each node subscribes only to
  `data/<own_node_id>`, so generation output reaches just the owning API node
  instead of every node in the cluster. On gossipsub (flag off) `owner_node` is
  ignored and the topic broadcasts as before, so the transports stay
  interchangeable behind the flag.

### Fixed

- **Placement now counts a unified-memory GPU node's GTT-mapped system RAM, not
  just its BIOS VRAM carve-out, and uses a lighter overhead factor for GGUF.** On
  an AMD APU (Strix Halo / Ryzen AI Max) the GPU addresses the BIOS VRAM
  carve-out plus system RAM through GTT, so a model larger than the carve-out
  runs there. `usable_vram_by_node` now detects a unified-memory node (its GTT
  aperture spans the whole system: `gtt_total_bytes > vram_total_bytes` AND
  `gtt_total_bytes ≥ ram_total`, which a discrete card whose GTT default merely
  equals VRAM does not satisfy) and counts working-set-capped VRAM plus
  GTT-mappable system RAM (minus a 16 GB OS headroom) toward the usable pool. The
  weight-overhead factor is now engine-aware: GGUF/llama.cpp models use
  1.10 (lighter C++ runtime) instead of MLX's 1.30. Together these let large GGUF
  MoEs place on a 128 GB Strix Halo node (e.g. a 58.5 GiB gpt-oss-120B on a node
  with a 64 GiB VRAM carve-out). The worker's local pre-spawn guard mirrors the
  same unified-memory math so it never refuses a placement the master admitted.

- **Placement now admits GPU-offload nodes against their discrete VRAM, not
  system RAM.** The memory fit check capped every node at
  `GPU_WORKING_SET_FRACTION` (0.75) of *system* RAM, a Metal/Apple-unified-memory
  assumption. On a discrete-VRAM node (a Strix Halo box whose BIOS carves 128 GB
  into ~64 GB system + 64 GB GPU VRAM) that refused models that fit fine in the
  64 GB VRAM the llama.cpp/Vulkan engine actually allocates from (e.g.
  `Llama-3.3-70B` at ~40 GB: "needs 54.2 GB but can use 46.1 GB"). Placement now
  detects discrete VRAM from the node's accelerator telemetry
  (`usable_vram_by_node`: AMD/NVIDIA `vram_total_bytes`) and admits against
  `min(vram_total − vram_used, GPU_VRAM_WORKING_SET_FRACTION (0.90) × vram_total)`.
  Apple unified-memory nodes are unchanged (they report no discrete VRAM). This
  is engine-agnostic, so it carries forward to vLLM/CUDA nodes.

- **A large-context GGUF no longer OOM-kills the node on load.** The llama.cpp
  runner loaded models with `n_ctx=0`, which sizes the KV cache for the model's
  full trained context (e.g. gemma-4's 128k) instead of the per-instance context
  budget placement actually reserved memory for. On a memory-tight node (observed
  loading gemma-4-31B on a Strix Halo Vulkan node) the kernel OOM-killed the whole
  worker process, so the instance vanished instead of failing cleanly. The runner
  now bounds `n_ctx` to the KV budget placement actually reserved memory for
  (`KV_CONTEXT_BUDGET_TOKENS`, 8192 tokens), clamped down by the instance's
  admission ceiling (#145) on a smaller node, so the up-front KV cache never
  exceeds what the cluster sized for the placement. (Serving llama.cpp beyond that
  budget needs placement to reserve the larger KV footprint, tracked separately
  with VRAM-aware admission.)

- **llama.cpp logprobs no longer OOM a node on load.** Defaulting the runner to
  `logits_all=True` for logprobs parity made llama.cpp pre-allocate an
  `n_ctx * vocab * 4` logits buffer at the model's full trained context, e.g.
  `131072 * 152064 * 4` = 74 GiB for a Qwen2.5-7B GGUF, failing the load with an
  allocation error. logprobs is now opt-in (`SKULK_LLAMA_CPP_LOGITS_ALL=1`) and,
  when enabled, caps the served context (`SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX`,
  default 8192) so the buffer stays bounded; with it off the served context is
  the instance's admission ceiling, not the model's full trained context.

- **The source-built GPU llama.cpp wheel survives `uv sync` (#358).** On a node
  that declares a GPU llama.cpp backend, the service entrypoint now runs
  `uv sync --inexact`, so a routine sync no longer prunes the out-of-resolution
  source-built wheel (which previously dropped the node to CPU-only until a manual
  rebuild). As a safety net, the node cross-checks a declared GPU backend against
  the actual build (`llama_cpp.llama_supports_gpu_offload()`): if the wheel has no
  GPU offload compiled in, it advertises only `llama_cpp-cpu` so GPU work is never
  routed to a degraded build. `SKULK_AUTO_UPDATE=0` is no longer required as a
  workaround on GPU nodes.

- **The llama.cpp tool path honors cancellation (#357).** A tool-enabled request
  runs one blocking, uninterruptible `create_chat_completion`; it now checks
  cancellation at the boundaries around that call (skip if already cancelled,
  suppress the result if a cancel landed while it ran), so a cancelled tool
  request neither delivers output nor is marked complete, matching the streaming
  path's cancellation semantics.

- **Headless/non-Mac nodes boot without the built dashboard (#333).** A worker
  node with no `dashboard-react/dist` (for example a Linux node with no node/npm
  to build the UI) previously failed to start: `constants.py` resolved
  `DASHBOARD_DIR` at import and raised `FileNotFoundError`, and the API's
  `StaticFiles` mount raised when the directory was absent. `DASHBOARD_DIR` is
  now `None` when the assets are absent and no `SKULK_DASHBOARD_DIR` override is
  set, and the API skips serving the dashboard (logging a notice) while serving
  the full API. Nodes that have the assets, or set `SKULK_DASHBOARD_DIR`, serve
  the UI unchanged.

- **Embedding tasks reach a clean terminal state (#326).** The embedding runner
  held `RunnerReady` across the `TextEmbedding` forward pass, but the supervisor
  asserts the runner is in an active state when it forwards a task's terminal
  status, so a completing embedding task tripped the assertion and aborted the
  event forwarder. The runner now holds `RunnerRunning` across the forward pass
  and returns to `RunnerReady` only after the terminal status is emitted,
  matching the MLX and llama.cpp text runners.

- **The failover-seed event round-trips through the disk event log again.** A
  `StateSnapshotHydrated` (the failover seed, indexed as event 0) is read back
  through the `Event` TypeAdapter, whose `TaggedModel` wrap validator unwraps the
  `{ClassName: inner}` envelope by re-validating the inner payload as a *python*
  object. Under `State`'s `strict=True` that path skips JSON-mode coercion, so
  the ISO datetime strings JSON produced for `last_seen` were rejected
  (`datetime_type`) and `DiskEventLog.read_range` halted at the seed. The
  phantom-node fix (#291) had started re-stamping `last_seen` on the seed, so
  every carried seed now hit this. `State` now coerces `last_seen` strings back to
  `datetime` in a field-scoped `before` validator (it does not force the whole
  model into python-mode validation, unlike a model-level validator). This
  unblocks event-log replay across a failover and is a prerequisite for #279
  Phase 3 snapshot/truncate (snapshots persist a full `State`, `last_seen`
  included).

- **Multi-node generation output is no longer silently reordered (#279 Phase 2b
  sequencing).** #279 Phase 2a moved per-token output (`ChunkGenerated`) off the
  master-indexed control plane (where the monotonic event `idx` gave every chunk
  a total order) onto the best-effort `DATA` gossip topic, which has no ordering
  key. When the producing rank-0 worker and the owning API node are different
  nodes, the gossip mesh can deliver a command's chunks out of order, and the API
  consumed them in arrival order, silently transposing tokens/sub-words in the
  response (`"Question"` -> `"Qesution"`). It was specific to multi-node *sampled*
  speculative decoding (single-node is local/in-order; greedy emits steadily) and
  hit ~90% of responses at temperature 0.2; the model battery never caught it
  because it only checked `finish_reason` and token count, never output
  coherence. `DataChunk` now carries a per-command monotonic `sequence` stamped
  by the producing supervisor, and the API reorders by it in a small per-command
  buffer before dispatch (releasing strictly in order, dropping duplicates). A
  genuinely dropped sequence on the best-effort topic is bounded two ways so it
  can never stall a stream: a size cap skips the gap if chunks pile up behind it,
  and a periodic sweep releases a gap left unfilled for `_REORDER_GAP_FLUSH_SECONDS`
  even when no later chunk arrives to trigger the cap (the dropped-seq-0 case,
  where the stream's own idle backstop never arms because nothing was yielded
  yet). The buffer is created only while a command has a live stream and cleared
  with it, so late chunks after finalize don't leak; the producer drops its
  per-command sequence counter on the terminal chunk for the same reason.

- **Deleting an instance no longer leaks its runner records (unbounded
  `State.runners` growth).** Runner status records were only removed by a
  terminal `RunnerStatusUpdated(RunnerShutdown)`, but that final status is
  unreliably delivered: the worker's Shutdown handler cancels the supervisor's
  event forwarder (`runner.shutdown()`) as soon as the Shutdown task
  completes/times out, usually before the runner process's `RunnerShutdown` is
  forwarded, and on a master-failover teardown the forwarder is torn down
  outright. Every instance delete therefore leaked one `RunnerShuttingDown`
  record per rank (one per node for a multi-node instance), so `State.runners`
  grew without bound over the cluster's lifetime, bloating state-sync snapshots.
  Two changes close it: `apply_instance_deleted` now prunes the deleted
  instance's runner records directly (mirroring `apply_node_timed_out`), and
  `apply_runner_status_updated` ignores updates for a runner that belongs to no
  instance, so the late `RunnerShuttingDown` that races behind `InstanceDeleted`
  can no longer resurrect the record. Deletion is now atomic and independent of
  the shutdown handshake. The actual runner-process teardown is driven
  separately by the Shutdown task, so dropping the status record early is safe.

- **Master failover no longer silently kills a healthy serving instance on a
  memory-tight node.** On a master-election transition the winning node tears
  its worker down (`worker.shutdown()`) and rebuilds it; that cancels each
  `RunnerSupervisor.run()`, whose teardown `finally` reaps the runner process so
  Metal reclaims its wired GPU memory on exit. The teardown was not shielded
  from cancellation, so the first `await` in it (the process join) re-raised
  immediately and the runner process was never reaped, so it lingered holding
  its GPU memory. The replacement worker then planned `CreateRunner` for the same
  carried shard, the pre-load memory guard saw the not-yet-reclaimed memory,
  falsely refused, and the #290 re-place-wider path deleted the carried instance
  (every subsequent request 404'd until a manual re-place). The teardown is now
  wrapped in a shielded `CancelScope`, so the runner process is fully joined
  (memory reclaimed) before `worker.shutdown()` returns and the replacement
  worker admits against true post-reclaim availability. Only bites when the
  election winner also hosts a rank of a carried instance and is memory-tight
  (common on small clusters); restores the documented "survives master failover"
  guarantee. The terminate/kill joins are now also off-thread (`to_thread`)
  instead of blocking the event loop.

- **Data-plane streams can't hang on a dropped final chunk (#279 Phase 2b).**
  Output chunks ride the best-effort `DATA` topic (no replay), so a dropped
  final chunk would leave a streaming response blocked on `receive()` forever.
  `_token_chunk_stream` now applies a per-receive idle timeout
  (`_STREAM_IDLE_TIMEOUT_SECONDS`, 120s): once the first real output token has
  arrived, a gap longer than the timeout closes the stream with a terminal error
  instead of hanging. The timeout wraps only the receive (not the yield), so it
  measures producer silence, never a slow client. Time-to-first-token is left
  unbounded, so a request queued behind a long decode or in a slow prefill never
  trips it (prefill-progress chunks are not treated as output and do not arm the
  timer). A stall whose task has already reached a terminal status is a dropped
  *final* chunk, so it cleans up via the normal `TaskFinished` path; a stall on a
  still-active task sends `TaskCancelled` to tear the stuck runner down (avoiding
  both an orphaned runner and a leaked master task/command mapping).

### Changed

- **Plane separation #279 Phase 2a: generation output chunks move to a data
  plane, off the master.** Per-token output (`ChunkGenerated`) used to flow
  worker → master (index + disk write + cluster-wide rebroadcast) → owning API,
  for data that never mutates `State` and is only ever read by that one API
  node. It now travels a new `DATA` topic as `DataChunk` (`{command_id, chunk}`)
  directly from the serving rank-0 worker to the owning API node, which demuxes
  by `command_id` into the per-command stream queues. The master no longer
  indexes, persists, or rebroadcasts output chunks; the API event log no longer
  records the per-token firehose (it had grown ~54MB in 9 idle hours). This
  removes the per-token master hop + disk write that dominated event-log volume
  and was the #278 storm vector. Inbound vision chunks (`InputChunkReceived`)
  stay on the control plane for now. Producer split lives in
  `RunnerSupervisor._emit`; the API consumes in `API._apply_data`.

- **Plane separation #279 slice 3: observational node readings move to the
  telemetry plane.** `node_identities`, `node_disk`, and `node_rdma_ctl` now ride
  the last-write-wins `TELEMETRY` topic into the node-owned `TelemetryView`
  instead of being event-sourced into `State` (joining `node_resources` from
  slice 1 and `node_memory`/`node_system` from slice 2). They are no longer
  persisted in the event log or carried in the failover seed; `GET /state`
  merges them back in so the dashboard wire shape is unchanged. The
  **connectivity** readings (`node_network`, `node_thunderbolt`,
  `node_thunderbolt_bridge`, and the derived `thunderbolt_bridge_cycles`)
  deliberately stay on the control plane — they define the topology graph
  (`apply()` builds RDMA edges and TB-bridge cycles from them, and the planner
  reads `node_network` for host selection), so they remain ordered rather than
  unordered telemetry.

### Fixed

- **Tight multi-node placements no longer silently vanish (#290).** The
  master admits placements on the gossiped (telemetry-plane, last-write-wins)
  available memory, while each worker's pre-spawn OOM guard reads a fresh live
  GPU-wireable figure at load time. On a borderline split the live reading can
  sit just below what the master admitted, so the master placed a cycle the
  worker then refused, and the instance was torn down ("instance vanished")
  with no recovery. The worker now sends a new `RefuseInstancePlacement`
  command for the memory-refusal case (distinct from a crash or GPU wedge,
  which still `DeleteInstance`), and the master re-places the same model one
  node wider (`min_nodes` = refused width + 1) so each node holds a smaller
  share. The loop is bounded: once even a full-width split raises
  `PlacementError` the master stops at the deletion. Refusals for
  already-removed instances are no-ops, so redelivery and operator deletes are
  safe.

## [1.2.0] - 2026-06-11

### Fixed

- **Abandoned requests can no longer storm the event log into election
  churn (#278).** An idle SequentialGenerator re-reported every
  ever-cancelled task id on every step without pruning the set, and the
  runner supervisor converted each re-report into a fresh
  `TaskStatusUpdated(Cancelled)` + `TaskDeleted` pair — observed live at
  ~800 events/s with 12,000+ events minted for a single dead task. The
  flood drowned replica apply loops, starved liveness into cascading
  elections, and silently lost placements. Five-layer fix: the idle
  generator now reports each cancellation exactly once (preserving the
  forward-looking CANCEL_ALL marker); the supervisor forwards a terminal
  status at most once per task; the master refuses to index task-lifecycle
  events for tasks absent from state (capping any future emitter at zero
  amplification); the event router's delivery retry gains exponential
  backoff and a max-attempts cap instead of unbounded fixed-interval
  resend; and the disk event log refreshes its diagnostic metadata file on
  a coarse cadence instead of one open/truncate/write/close per appended
  event (previously the dominant physical-write term of every indexed
  event, cluster-wide).

- **Long-context requests are rejected cleanly instead of OOM-crashing the
  runner (#145, phase 1).** The within-request KV cache grew one entry per
  token with no bound and no preflight check, so a request whose prompt plus
  output exceeded what the hosting node(s) could hold killed the runner
  mid-generation with an unhandled Metal OOM (SIGABRT, broken stream or 500
  for the client, wired GPU memory leaked). Each placed instance now carries
  a static context-token ceiling — the smaller of the card's advertised
  context length and the KV tokens that fit beside the weight share on every
  hosting node — computed deterministically from gossiped node memory so all
  ranks of a multi-node instance enforce the identical limit. Requests are
  admitted against it before prefill: explicit `max_tokens` overflow and
  window-filling prompts get an OpenAI-style `context_length_exceeded`
  invalid-request error (400 at the API when detectable pre-dispatch), and an
  omitted `max_tokens` is clamped to the remaining window so generation ends
  with `finish_reason: "length"`. Unquantized KV only; quantized-KV budget
  math is phase 2.

- **Instance placements survive master failover (#273).** A newly-elected
  master previously always started its session from an empty state: the
  empty snapshot propagated to every follower, each worker's plan loop saw
  no instances and shut down its healthy runners, and every placed model
  silently became a 404 until an operator re-placed it — a full serving
  outage from a single master restart (found live when a churn test
  happened to bounce the master). The promoted node now seeds the new
  session from its prior replicated state: instances, downloads, node info,
  and the tracing flag carry over, while in-flight tasks, runner statuses,
  topology, and liveness timestamps are deliberately dropped (they are
  session-scoped or must come from live gossip — a carried topology would
  keep a dead node's edges forever). Workers re-create runners for the
  carried instances through the ordinary plan loop, so serving resumes
  after a model-reload-sized gap with no operator action. The master's
  liveness-based instance pruning is suppressed for a 60-second
  topology-settle grace after promotion so carried instances aren't deleted
  while connection gossip is still rebuilding the topology; instances whose
  ranks lived on the dead master are pruned normally after the grace. A
  freshly-booted election winner seeds empty, exactly as before.

- **A stalled distributed group can no longer hang an instance forever, and
  ring transport selection follows operator intent (#265).** Two changes:
  (1) `mx.distributed.init` now runs under a hard deadline (default 120s,
  `SKULK_GROUP_CONNECT_DEADLINE_SECONDS`) — the ring backend with
  `strict=True` blocks indefinitely when a neighbor socket fails its
  post-TCP rank handshake, which left a 4-node placement looping request
  timeouts and cancels for 30+ minutes with no recovery; expiry now exits
  the runner via the wedge path, the worker gives the instance up on the
  first failure, and the fresh placement mints a new ring port (also
  clearing stale-socket handshake collisions from same-port retries).
  (2) VPN/overlay addresses (Tailscale CGNAT `100.64/10` and
  `fd7a:115c:a1e0::/48`, detected by address since utun interfaces gossip
  as "unknown") now rank strictly last in ring transport selection —
  Tailscale exists for external reachability and may be DERP-relayed (a
  ring link between two machines on the same switch was observed riding
  the Dallas relay); a pair with any Thunderbolt/LAN candidate never
  selects the overlay, while genuinely cross-network pairs still work.
  First test coverage for `get_mlx_ring_hosts_by_node` and the transport
  ranking.
- **The Thunderbolt interface label survives classification (#222).** The
  hardware-port parser set "thunderbolt" from the port header, then the
  device-line branch unconditionally rewrote every en2+ device to
  `maybe_ethernet` — and Mac Thunderbolt ports are always en2+, so the
  thunderbolt label could never exist on macOS and the ring's TB-first
  transport priority was dead code (it worked only because maybe_ethernet
  happened to outrank ethernet). The downgrade now applies only to the
  genuinely ambiguous case (a generic "Ethernet Adapter" port on en2+, which
  may be a USB dongle); specifically-classified ports keep their labels, and
  unclassified ports (e.g. an iPhone tether) stay at lowest priority instead
  of being promoted.
- **Peer churn can no longer crash healthy bystander nodes (#266).** When a
  master transition replaced the worker, the telemetry forwarder exited first
  (its event stream closes), and the InfoGatherer's next send raced into the
  closed channel — the unhandled `BrokenResourceError` took the entire
  process down (observed twice in one night, once on the cluster hub). A
  closed/broken telemetry channel is now treated as the stop signal it is:
  the gatherer exits cleanly (the replacement worker brings a fresh one), and
  the per-monitor `except Exception` blocks — which exist to survive flaky
  *gathering* — explicitly re-raise channel closure instead of swallowing it
  and spinning on a dead channel.

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
