# exo Model Store

> **Status:** Phase 1 — centralized LAN model distribution
> **Branch:** `claude/review-artifact-gRhSf`
> **Config file:** `exo.yaml` (copy from `exo.yaml.example`, optional, zero-config compatible)

---

## What problem does this solve?

In a standard exo cluster every node independently downloads its assigned
model shard from HuggingFace at inference time.  For a small home cluster
(2–8 machines) this creates four pain points:

| Pain point | Impact |
|---|---|
| Redundant downloads | Every node pulls the same model files; a 30B model may mean 20 GB × N nodes of internet traffic |
| Slow cold starts | A node that hasn't seen a model yet blocks inference for the whole cluster |
| Version drift | Each node downloads on its own schedule; model files can diverge |
| No offline mode | The cluster is dead without an internet connection after the first run |

The model store solves all four by introducing a single designated
**store host** node — typically the machine with the most attached storage —
that holds all model files.  Every other node pulls its assigned shard from
the store over the local network (Thunderbolt, 10 GbE, or even fast Wi-Fi)
instead of from HuggingFace.

---

## How it extends EXO

This is a **non-breaking opt-in extension** to the existing exo architecture.
If `exo.yaml` is absent (the default for any existing deployment), the model
store is not active and exo behaves identically to the upstream default.

### What changes when the model store is enabled

| Component | Without store | With store |
|---|---|---|
| `DownloadCoordinator` | Uses `ResumableShardDownloader` → HuggingFace | Uses `ModelStoreDownloader` wrapping `ResumableShardDownloader` |
| `ModelStoreDownloader` | n/a | Intercepts `ensure_shard()`, stages from store; falls back to inner downloader if model not present |
| Store host startup | n/a | `ModelStore` (registry) + `ModelStoreServer` (HTTP) started alongside other node components |
| Worker shutdown | n/a | `ModelStoreClient.evict_shard()` removes staged files when `cleanup_on_deactivate: true` |
| MLX / inference | Loads from `~/.exo/models/` | Loads from `~/.exo/staging/` (or `node_cache_path`) — **no change to the inference stack** |

### What does NOT change

- The libp2p routing, election, and master/worker architecture is unchanged.
- The OpenAI-compatible REST API is unchanged.
- The MLX inference backend is unchanged.
- Nodes without `exo.yaml` continue to work normally and can coexist with
  store-enabled nodes (useful during a phased rollout).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Cluster                                                            │
│                                                                     │
│  ┌───────────────────┐          HTTP :58080            ┌─────────┐ │
│  │  mac-studio-1     │◄────────────────────────────────│ kite2   │ │
│  │  (store host)     │  GET /models/{id}/{file}        │         │ │
│  │                   │                                 │ staging │ │
│  │  ModelStore       │◄────────────────────────────────│ ~/.exo/ │ │
│  │  (registry.json)  │                                 │ staging/│ │
│  │                   │                                 └─────────┘ │
│  │  ModelStoreServer │                                             │
│  │  (aiohttp)        │◄────────────────────────────────┌─────────┐ │
│  │                   │                                 │ kite3   │ │
│  │  /Volumes/        │                                 │         │ │
│  │  ModelStore/      │                                 │ staging │ │
│  │  models/          │                                 │ /Vols/  │ │
│  │    mlx-comm--.../ │                                 │ FastSSD/│ │
│  │    ...            │                                 └─────────┘ │
│  └───────────────────┘                                             │
│                                                                     │
│  All nodes: libp2p mesh, election, master/worker unchanged         │
└─────────────────────────────────────────────────────────────────────┘
```

### Module layout

```
src/exo/store/
  __init__.py              (empty — package marker)
  config.py                ExoConfig, ModelStoreConfig, StagingNodeConfig
                           load_exo_config(), resolve_node_staging()
  model_store.py           ModelStore — registry + path resolution (store host only)
  model_store_server.py    ModelStoreServer — aiohttp HTTP file server (store host only)
  model_store_client.py    ModelStoreClient — HTTP staging client (all nodes)
                           ModelStoreDownloader — ShardDownloader wrapper (all nodes)
```

### Integration points in the existing codebase

| File | What was added |
|---|---|
| `src/exo/main.py` | Load `exo.yaml`; build `ModelStore`, `ModelStoreServer`, `ModelStoreClient`; wrap `exo_shard_downloader()` with `ModelStoreDownloader`; start server on store host; re-wrap on election transitions |
| `src/exo/worker/main.py` | Accept `store_client` and `staging_config` in `Worker.__init__`; call `evict_shard()` in the `Shutdown` task handler |
| `pyproject.toml` | Added `pyyaml` and `types-PyYAML` dependencies |

---

## Prerequisites

- All nodes running the same exo build (this branch or a merged version)
- The store host node has the storage device mounted and accessible at `store_path`
- Port `58080` (or your configured `store_port`) is reachable from all worker
  nodes — no firewall blocking
- `exo.yaml` present at the project root on **every node** in the cluster
  (or absent to fall back to standard behaviour on that node)

---

## Quick start

### 1 — Create `exo.yaml` in the project root

The minimum viable config:

```yaml
model_store:
  enabled: true
  store_host: mac-studio-1    # change to your store host's hostname
  store_path: /Volumes/ModelStore/models
```

Copy this to the same path on every node.  Use `node_overrides` to tune
per-node staging (see [Configuration reference](#configuration-reference)).

### 2 — Run exo normally

```bash
uv run exo
```

No new flags or environment variables needed.  On startup:

- **Store host** (`mac-studio-1`): detects that its hostname matches
  `store_host`, starts `ModelStoreServer` on port `58080`.
- **Worker nodes**: detect they are not the store host, start a
  `ModelStoreClient` pointed at `mac-studio-1:58080`.

Watch for these log lines to confirm the store is active:

```
# Store host
ModelStore: this node is the store host — store at /Volumes/ModelStore/models, server on port 58080
ModelStoreServer listening on 0.0.0.0:58080 (store: /Volumes/ModelStore/models)

# Worker nodes
ModelStoreDownloader: staging mlx-community/Qwen3-30B-A3B-4bit from store → ~/.exo/staging/mlx-community--Qwen3-30B-A3B-4bit
ModelStoreClient: staged mlx-community/Qwen3-30B-A3B-4bit to ~/.exo/staging/... (21474836480 bytes)
```

### 3 — Request a model

```bash
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-30B-A3B-4bit",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

**First request (model not yet in store):** If `allow_hf_fallback: true`,
nodes download from HuggingFace as normal.  The model lands in `~/.exo/models/`
on each node.

**Subsequent requests:** Worker nodes stage from `mac-studio-1:58080`.
HuggingFace is not contacted.

---

## Configuration reference

```yaml
model_store:
  enabled: true                        # false → store disabled (same as no file)

  store_host: mac-studio-1             # hostname or libp2p node_id of store node
  store_port: 58080                    # HTTP port — must be open on store host
  store_path: /Volumes/ModelStore/models  # absolute path on store host

  download:
    allow_hf_fallback: true            # false → error if model not in store

  staging:
    # Default applied to all nodes without a matching node_override
    enabled: true
    node_cache_path: ~/.exo/staging    # ~ expanded per-node
    cleanup_on_deactivate: true        # delete staged files on instance shutdown

  node_overrides:
    # Keyed by hostname (socket.gethostname()) or libp2p node_id.
    # First match wins; unspecified fields inherit from the base config.
    mac-studio-1:
      staging:
        node_cache_path: /Volumes/ModelStore/models  # load directly from store
        cleanup_on_deactivate: false                 # store IS the cache
```

### Field reference

#### `model_store.store_host`

Compared against (in order):
1. The node's libp2p peer ID (`node_id`)
2. The node's hostname (`socket.gethostname()`)

Set it to the **hostname** of your store node for simplest configuration.

#### `model_store.staging.node_cache_path`

MLX loads the model from this path.  Default is `~/.exo/staging/`, which
sits alongside the existing `~/.exo/models/` cache exo already uses.

For the store host, point this at the same directory as `store_path` so that
no local copy is made — MLX loads directly from the store device:

```yaml
node_overrides:
  mac-studio-1:
    staging:
      node_cache_path: /Volumes/ModelStore/models
      cleanup_on_deactivate: false
```

#### `model_store.download.allow_hf_fallback`

| Value | Behaviour |
|---|---|
| `true` (default) | Model not in store → download from HuggingFace |
| `false` | Model not in store → raise `ModelNotInStoreError` |

Use `false` for air-gapped clusters to guarantee no unexpected internet traffic.

---

## Store HTTP API

The `ModelStoreServer` exposes a small read-only HTTP API on the store host.
All responses are JSON unless noted.  The API is **not authenticated** — it is
designed for trusted LAN use.  Do not expose `store_port` to untrusted networks.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check; returns `store_path`, `free_bytes`, `total_bytes`, `used_bytes` |
| `/registry` | GET | Full store index — array of model entries with paths, file lists, sizes, timestamps |
| `/models` | GET | Array of model ID strings in the store |
| `/models/{id}/files` | GET | File list for one model (`%2F` for `/` in model ID) |
| `/models/{id}/{path}` | GET | File content; supports `Range` header (HTTP 206) |

```bash
# Liveness check
curl http://mac-studio-1:58080/health

# List models
curl http://mac-studio-1:58080/models

# File list for a model
curl http://mac-studio-1:58080/models/mlx-community%2FQwen3-30B-A3B-4bit/files

# Manual resumable download of one file
curl -H "Range: bytes=1073741824-" \
  http://mac-studio-1:58080/models/mlx-community%2FQwen3-30B-A3B-4bit/model-00001-of-00008.safetensors \
  -o model-00001.partial
```

---

## Data flow

### First run — model not in store, HF fallback enabled

```
DownloadCoordinator
  → ModelStoreDownloader.ensure_shard()
      → ModelStoreClient.is_model_available()   # HTTP 404
      → allow_hf_fallback=true
        → inner.ensure_shard()                  # ResumableShardDownloader
            → HuggingFace → ~/.exo/models/...
              → return ~/.exo/models/...
                → MLX loads from ~/.exo/models/...
```

### Subsequent runs — model in store

```
DownloadCoordinator
  → ModelStoreDownloader.ensure_shard()
      → ModelStoreClient.is_model_available()   # HTTP 200
      → ModelStoreClient.stage_shard()
          → GET mac-studio-1:58080/models/{id}/files
          → GET mac-studio-1:58080/models/{id}/{file}  (per file, Range resume)
            → ~/.exo/staging/mlx-community--Qwen3-30B-A3B-4bit/
              → return staging path
                → MLX loads from ~/.exo/staging/...

Instance shutdown (Shutdown task in Worker)
  → ModelStoreClient.evict_shard()
      → rm -rf ~/.exo/staging/mlx-community--Qwen3-30B-A3B-4bit/
        (store copy on mac-studio-1 untouched)
```

### Store host — no network round-trip

```
ModelStoreDownloader.ensure_shard()
  → ModelStoreClient.stage_shard()   (local_store_path set)
      → _stage_local()
          → shutil.copy2() per file  (or no-op if node_cache_path == store_path)
            → MLX loads from node_cache_path
```

---

## Registry and store population

### Registry format

`{store_path}/registry.json` is written and read by `ModelStore`:

```json
{
  "mlx-community/Qwen3-30B-A3B-4bit": {
    "model_id": "mlx-community/Qwen3-30B-A3B-4bit",
    "store_path": "mlx-community--Qwen3-30B-A3B-4bit",
    "files": ["config.json", "tokenizer.json", "model-00001-of-00008.safetensors"],
    "downloaded_at": "2026-03-20T14:32:00+00:00",
    "total_bytes": 21474836480
  }
}
```

### Staging directory layout (worker nodes)

```
~/.exo/staging/
  mlx-community--Qwen3-30B-A3B-4bit/
    config.json
    tokenizer.json
    model-00003-of-00008.safetensors   ← only this node's assigned shard
  mlx-community--Llama-3.1-8B-Instruct-4bit/
    ...
```

### Pre-populating the store (Phase 1)

In Phase 1, the registry must be populated manually.  The recommended workflow
is to download models directly to the store on the store host using
`huggingface-cli`, then register them:

```bash
# 1. Download to the store directory on mac-studio-1
huggingface-cli download mlx-community/Qwen3-30B-A3B-4bit \
  --local-dir /Volumes/ModelStore/models/mlx-community--Qwen3-30B-A3B-4bit

# 2. Register in the store index
python - <<'EOF'
from pathlib import Path
from exo.store.model_store import ModelStore

store = ModelStore(Path("/Volumes/ModelStore/models"))
p = Path("/Volumes/ModelStore/models/mlx-community--Qwen3-30B-A3B-4bit")
files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
store.register_model("mlx-community/Qwen3-30B-A3B-4bit", p, files, total)
print(f"Registered {len(files)} files, {total:,} bytes")
EOF
```

Phase 2 will automate this (see [Roadmap](#roadmap)).

---

## Troubleshooting

### Worker can't reach the store host

```bash
curl http://mac-studio-1:58080/health
```

If this fails:
- Check that `ModelStoreServer` started on `mac-studio-1` (look for
  `ModelStoreServer listening on 0.0.0.0:58080` in the store host's logs)
- Check that port `58080` is open in any firewall on `mac-studio-1`
- Confirm `store_host: mac-studio-1` matches the hostname exactly
  (`hostname` command on the store host)

### Model shows as unavailable despite being in the store directory

```bash
curl http://mac-studio-1:58080/registry
```

Check that:
- A registry entry exists for the model ID
- The `store_path` in the entry points to the correct subdirectory
- The subdirectory exists on disk

If the entry is missing, re-run the registration script above.

### Staged files not cleaned up

Check that `cleanup_on_deactivate: true` is set for the node (and not
overridden to `false` in `node_overrides`).  Eviction is logged at INFO level:

```
ModelStoreClient: evicted staged shard for mlx-community/Qwen3-30B-A3B-4bit from ...
```

### Transfer stalls partway through

Staging is automatically resumable.  Re-request the model — `ModelStoreClient`
detects `.partial` files and sends a `Range` header to continue from the last
byte received.  No manual cleanup needed.

### HuggingFace downloads are being used despite the store being configured

1. Check the store host logs — is `ModelStoreServer` running?
2. `curl http://<store_host>:58080/models` — is the model listed?
3. If the model isn't listed, it hasn't been registered.  Register it (see above).
4. Check worker logs for `ModelStoreDownloader: ... not in store, falling back to HuggingFace`.

---

## Roadmap

This is a phased feature.  Each phase is a self-contained improvement that
can be reviewed and merged independently.

### Phase 1 — Centralized LAN distribution *(this branch)*

- `ModelStore`: registry + path resolution on the store host
- `ModelStoreServer`: HTTP file server with Range/resume support
- `ModelStoreClient`: HTTP staging client with `.partial` resume
- `ModelStoreDownloader`: `ShardDownloader` wrapper; transparent HF fallback
- `exo.yaml`: optional cluster config file; zero-config when absent
- Manual store population via `huggingface-cli` + `register_model()`

**What Phase 1 does not do:**
- Automatic population when a model is first downloaded from HuggingFace
  (models downloaded via HF fallback land in `~/.exo/models/`, not the store)
- Store host failover
- Dashboard integration
- Parallel file downloads within a single model

---

### Phase 2 — Automatic store population

When a model is downloaded from HuggingFace via the fallback path, the
`ModelStoreDownloader` (or the `DownloadCoordinator`) automatically copies the
completed download into the store and registers it.  After Phase 2, the store
self-populates as models are first requested — no manual registration needed.

Scope:
- Hook `DownloadCoordinator`'s `DownloadCompleted` event to trigger a
  background copy-and-register on the store host.
- Or: add a post-download hook in `ResumableShardDownloader` that notifies
  the store client when a file is complete.
- Registry update is idempotent so concurrent completions are safe.

---

### Phase 3 — Dashboard integration

Surface store status in the Svelte dashboard so users can see at a glance
what models are in the store, which nodes have staged which models, and
how much space remains on the store device.

Scope:
- Add store health and registry endpoints to the FastAPI layer (proxying
  `ModelStoreServer`'s `/health` and `/registry`).
- Extend the existing state or add a new dashboard-specific event type
  for store status.
- Dashboard: "Store" panel showing:
  - Store host, connection status, disk usage
  - Model list with file counts and sizes
  - Per-node staging status (which models are staged, how much local space used)
- Provide `curl`-friendly API so the panel can be built incrementally.

---

### Phase 4 — Store host failover

If the designated store host goes offline, worker nodes currently cannot stage
new models (though already-staged models continue to work until evicted).

Scope:
- Track store host liveness via the existing peer connection events.
- On store host failure: emit a `StoreHostUnavailable` event; nodes fall back
  to HuggingFace if `allow_hf_fallback: true`.
- Optional: allow a secondary store host to be configured.
- Stretch goal: replicate the store registry to a secondary node and promote
  it automatically.

---

### Phase 5 — Shard-aware staging and heterogeneous clusters

Currently, `ModelStoreClient.stage_shard()` downloads **all** model files to
every worker, regardless of which layers that node is responsible for.  MLX
then loads only the relevant shard.  This wastes local staging space.

Scope:
- Accept a `ShardMetadata` in `stage_shard()` and filter the file list to
  only the safetensors files that correspond to the assigned layer range.
- Requires understanding the safetensors index to map layers → files.
- Unlocks running models on nodes with less RAM than the full shard.
- Directly enables "Manual shard control" as a user-facing feature.

---

### Phase 6 — Offline / air-gapped mode improvements

`allow_hf_fallback: false` already prevents HuggingFace access.  Phase 6
hardens the offline story:

- CLI command to list models available in the store (`exo store list`).
- CLI command to pre-fetch a model into the store without running inference
  (`exo store pull mlx-community/Qwen3-30B-A3B-4bit`).
- Store integrity check: verify file hashes on startup and surface
  corruption via the dashboard.
- Cleaner error messaging when a model is requested in offline mode and
  is not in the store.

---

## Contributing / upstream considerations

This feature is designed to be proposable to the upstream exo project.
Key design choices that support that goal:

- **Zero-config compatible**: the feature is entirely opt-in via `exo.yaml`.
  Existing deployments are unaffected.
- **No new required dependencies**: `aiohttp` and `aiofiles` are already in
  the dependency graph; only `pyyaml` was added.
- **Wrapping, not forking**: `ModelStoreDownloader` wraps the existing
  `ShardDownloader` interface rather than replacing it, so the existing
  download stack is preserved intact.
- **Minimal integration surface**: only two existing files were modified
  (`main.py` and `worker/main.py`); everything else is new code in
  `src/exo/store/`.
- **Typed and strict**: all new code follows the project's `strict=True`
  Pydantic and `basedpyright` conventions.
- **`exo.yaml` is extensible**: future cluster-level features (networking
  tuning, custom backends, resource policies) can add top-level sections
  without breaking existing configs.
