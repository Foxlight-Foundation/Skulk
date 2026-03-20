# foxcluster Model Store

> **Branch:** `claude/review-artifact-gRhSf`
> **Status:** v1 implementation — see [Open questions](#open-questions--pass-2) for deferred features

---

## What it does

By default, every EXO node downloads its model shard independently from HuggingFace at inference time. In a small home cluster this means:

- redundant external bandwidth (every node pulls the same files)
- no offline capability after the first download
- model versions can drift between nodes
- slow cold starts on 30B+ models

foxcluster replaces this with a single designated **store host** node (e.g. `kite1`) that holds all model files on fast Thunderbolt-attached storage. Worker nodes pull their assigned shard from the store over HTTP at Thunderbolt speeds. HuggingFace is only contacted once per model, ever.

MLX always receives a **local filesystem path** — the inference stack is completely unaware the store exists.

---

## Prerequisites

- All nodes running the same foxcluster build (this branch)
- The store host node has sufficient storage mounted and accessible at the configured path
- The store host port (default `58080`) is reachable from all worker nodes
- `foxcluster.yaml` present in the project root on **every node** (or omitted to fall back to standard EXO behaviour)

---

## Quick start

### 1. Create `foxcluster.yaml` in the project root

```yaml
model_store:
  enabled: true
  store_host: kite1          # hostname of the node with the storage device
  store_port: 58080
  store_path: /Volumes/FoxStore/models

  download:
    allow_hf_fallback: true  # fall back to HuggingFace if model not yet in store

  staging:
    enabled: true
    node_cache_path: ~/.foxcluster/stage
    cleanup_on_deactivate: true
```

The file must be at the **same path on every node** (it sits alongside the `exo` project root). If the file is absent on a node, that node behaves identically to upstream EXO.

### 2. Run EXO normally

```bash
uv run exo
```

No new flags or commands are needed. On startup:

- **Store host node** (`kite1`): detects it matches `store_host`, starts `FoxStoreServer` on port `58080`.
- **Worker nodes** (`kite2`, `kite3`): detect they are not the store host, start `FoxStoreClient` pointed at `kite1:58080`.

### 3. Request a model via the API

```bash
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3-30B-A3B-4bit", "messages": [{"role": "user", "content": "hi"}]}'
```

**First run (model not in store):** If `allow_hf_fallback: true`, nodes download from HuggingFace as normal.

**Subsequent runs:** Worker nodes pull from `kite1:58080` instead of HuggingFace. Transfers resume automatically if interrupted.

---

## Configuration reference

```yaml
model_store:
  enabled: true                   # set false to disable entirely (same as no file)

  store_host: kite1               # hostname or node_id of the store host
  store_port: 58080               # port FoxStoreServer listens on (store host only)
  store_path: /Volumes/FoxStore/models  # absolute path on the store host

  download:
    allow_hf_fallback: true       # if false, raise an error when model not in store

  staging:
    enabled: true                 # if false, skip staging (useful for testing)
    node_cache_path: ~/.foxcluster/stage  # local staging dir (~ expanded)
    cleanup_on_deactivate: true   # delete staged files when instance deactivates

  node_overrides:                 # per-node config (matched by hostname or node_id)
    kite1:
      staging:
        # Store host: stage path IS the store — no network hop
        node_cache_path: /Volumes/FoxStore/models
        cleanup_on_deactivate: false   # keep files; store is the canonical copy
    kite3:
      staging:
        node_cache_path: /Volumes/FastSSD/foxstage   # use fast local SSD
        cleanup_on_deactivate: true
```

### `store_host` matching

`store_host` is compared against:
1. The node's `node_id` (the libp2p peer ID)
2. The node's hostname (`socket.gethostname()`)

Set it to the **hostname** of the store node (e.g. `kite1`) for simplest config.

### Store host `node_overrides`

On the store host, staging copies files from the store path to `node_cache_path`. If you set `node_cache_path` to the same directory as `store_path`, no copy happens — MLX loads directly from the store device. This is the recommended config for the store host:

```yaml
node_overrides:
  kite1:
    staging:
      node_cache_path: /Volumes/FoxStore/models
      cleanup_on_deactivate: false
```

### `allow_hf_fallback`

| Value | Behaviour |
|---|---|
| `true` (default) | If model not in store, download from HuggingFace normally |
| `false` | If model not in store, raise `StoreModelNotFoundError` — use for air-gapped clusters |

---

## Store host HTTP API

The store server exposes a small HTTP API on port `58080`. This can be used for monitoring or pre-population scripts.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check. Returns `store_path`, `free_bytes`, `total_bytes`, `used_bytes`. |
| `/registry` | GET | Full store index: all models, file lists, sizes, timestamps. |
| `/models` | GET | List of model IDs currently in the store. |
| `/models/{id}/files` | GET | File list for a specific model. |
| `/models/{id}/{path}` | GET | Serve a file. Supports `Range` header for resumable transfers (`206 Partial Content`). |

Examples:

```bash
# Check store health
curl http://kite1:58080/health

# List models in store
curl http://kite1:58080/models

# List files for a model
curl http://kite1:58080/models/mlx-community%2FQwen3-30B-A3B-4bit/files

# Resume a partial download manually
curl -H "Range: bytes=1073741824-" \
  http://kite1:58080/models/mlx-community%2FQwen3-30B-A3B-4bit/model-00001-of-00008.safetensors \
  -o model-00001.partial
```

---

## Data flow

### First run — model not in store, HF fallback enabled

```
Worker node (kite2)
  FoxShardDownloader.ensure_shard()
    → is_model_available() → store returns 404
    → allow_hf_fallback=true → delegate to ResumableShardDownloader
      → download_shard() → HuggingFace → ~/.exo/models/...
        → MLX loads from ~/.exo/models/...
```

### Subsequent runs — model in store

```
Worker node (kite2)
  FoxShardDownloader.ensure_shard()
    → is_model_available() → store returns 200
    → FoxStoreClient.stage_shard()
        → GET kite1:58080/models/{id}/files  (file list)
        → GET kite1:58080/models/{id}/{file} (per file, with Range resume)
          → ~/.foxcluster/stage/{model}/
            → MLX loads from ~/.foxcluster/stage/{model}/

Instance deactivated
  → Worker._maybe_evict_shard()
    → FoxStoreClient.evict_shard()
      → rm -rf ~/.foxcluster/stage/{model}/
        (store copy on kite1 untouched)
```

### Store host (kite1)

```
FoxShardDownloader.ensure_shard()
  → is_model_available() → local path check (no HTTP)
  → FoxStoreClient.stage_shard() with local_store_path set
    → shutil.copy2() from store path to node_cache_path
      (no-op if node_cache_path == store_path)
```

---

## Staging directory layout

```
~/.foxcluster/stage/
  mlx-community--Qwen3-30B-A3B-4bit/
    config.json
    tokenizer.json
    model-00003-of-00008.safetensors
    ...
  mlx-community--Llama-3.1-8B-Instruct-4bit/
    ...
```

Model IDs are sanitized with `/` → `--` (same convention EXO uses for `~/.exo/models/`).

---

## Store registry

The store host maintains a registry at `{store_path}/registry.json`:

```json
{
  "mlx-community/Qwen3-30B-A3B-4bit": {
    "model_id": "mlx-community/Qwen3-30B-A3B-4bit",
    "store_path": "mlx-community--Qwen3-30B-A3B-4bit",
    "files": ["config.json", "tokenizer.json", "model-00001-of-00008.safetensors", "..."],
    "downloaded_at": "2026-03-20T14:32:00+00:00",
    "total_bytes": 21474836480
  }
}
```

The registry is written when a model is explicitly registered via `FoxModelStore.register_model()`. In v1, this must be done manually after copying model files to the store (automatic population from HF downloads is a v2 feature — see [Step 7 of the design doc](../foxcluster-model-store-plan.md)).

### Pre-populating the store manually

Copy model files to the store directory following EXO's naming convention, then register them:

```python
from pathlib import Path
from exo.store.fox_model_store import FoxModelStore

store = FoxModelStore(Path("/Volumes/FoxStore/models"))
model_path = Path("/Volumes/FoxStore/models/mlx-community--Qwen3-30B-A3B-4bit")

files = [str(p.relative_to(model_path)) for p in model_path.rglob("*") if p.is_file()]
total_bytes = sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())

store.register_model(
    model_id="mlx-community/Qwen3-30B-A3B-4bit",
    model_path=model_path,
    files=files,
    total_bytes=total_bytes,
)
```

Or simply download from HuggingFace directly to the store path using `huggingface-cli`:

```bash
# On kite1 (store host)
huggingface-cli download mlx-community/Qwen3-30B-A3B-4bit \
  --local-dir /Volumes/FoxStore/models/mlx-community--Qwen3-30B-A3B-4bit

# Then register it
python -c "
from pathlib import Path
from exo.store.fox_model_store import FoxModelStore
store = FoxModelStore(Path('/Volumes/FoxStore/models'))
p = Path('/Volumes/FoxStore/models/mlx-community--Qwen3-30B-A3B-4bit')
files = [str(f.relative_to(p)) for f in p.rglob('*') if f.is_file()]
store.register_model('mlx-community/Qwen3-30B-A3B-4bit', p, files, sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))
print('Registered.')
"
```

---

## Troubleshooting

**Worker node can't reach store host**

```bash
curl http://kite1:58080/health
```

If this fails, check:
- `FoxStoreServer` started on kite1 (look for `FoxStoreServer listening on 0.0.0.0:58080` in logs)
- Firewall allows TCP `58080` from worker nodes
- `store_host: kite1` in `foxcluster.yaml` matches kite1's hostname exactly

**Model not found in store despite being registered**

```bash
curl http://kite1:58080/models
curl http://kite1:58080/registry
```

Check that:
- The directory at `store_path/{model_id_sanitized}/` exists on kite1
- The registry entry's `store_path` points to that directory

**Staged files not cleaned up after deactivation**

Check that `cleanup_on_deactivate: true` is set for the node (not overridden to `false` in `node_overrides`). Eviction is logged at INFO level: `Worker: evict_shard ...`

**Resuming a stalled transfer**

Staging is automatically resumable. If a transfer stalls, simply re-request the model — the client detects `.partial` files and issues a `Range` request to continue from where it left off.

---

## Code layout

```
src/exo/store/
  __init__.py
  config.py              — Pydantic schema for foxcluster.yaml
  fox_model_store.py     — FoxModelStore: registry + path resolution (store host)
  fox_store_server.py    — FoxStoreServer: aiohttp HTTP file server
  fox_store_client.py    — FoxStoreClient: HTTP staging client
                           FoxShardDownloader: ShardDownloader wrapper

foxcluster.yaml          — Example cluster config (edit for your cluster)
```

Key integration points in existing EXO code:

| File | Change |
|---|---|
| `src/exo/main.py` | Loads `foxcluster.yaml`, builds store components, wraps `exo_shard_downloader()` with `FoxShardDownloader`, starts `FoxStoreServer` on store host |
| `src/exo/worker/main.py` | `Worker.__init__` accepts `store_client` + `staging_config`; eviction fires in `Shutdown` handler |
| `pyproject.toml` | Added `pyyaml` + `types-PyYAML` |

---

## Open questions / Pass 2

These are explicitly deferred from v1. Do not let them influence the current implementation.

- **Store host failover.** If kite1 goes down, worker nodes cannot stage new models. v2 could elect a secondary store host or fall back to HF.
- **Automatic store population from HF.** Currently HF downloads land in `~/.exo/models/`, not in the store. v2 will hook the download completion path to copy into the store and register automatically.
- **Model version management.** Registry tracks latest download only. Multiple revisions not supported in v1.
- **Store-side shard pre-slicing.** Currently nodes stage all model files, not just the layers they need. v2 may pre-slice safetensors at store time.
- **Manual shard control.** Primary v2 target — resolves heterogeneous memory profile load failures on nodes with less RAM.
- **Dashboard integration.** Store health, model inventory, and per-node staging status should surface in the Svelte frontend.
