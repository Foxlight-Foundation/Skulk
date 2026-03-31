<!-- Copyright 2025 Foxlight Foundation -->

# Skulk Model Store

The model store is one of the biggest additions Skulk makes on top of upstream EXO.

In a normal cluster without a model store, each node may need to download model data for itself.
With the model store enabled, one node becomes the shared store host and other nodes stage from it over the LAN.

## Why You Would Use It

Use the model store when:

- you have more than one node
- your models are large
- you want fewer repeated downloads
- you want a cleaner offline story after the first download
- you want model files to live on a dedicated large disk or volume

## What Changes When It Is Enabled

Without the model store:

- nodes download model data independently
- cold starts can be slower across the cluster
- repeated downloads are more common

With the model store:

- one node hosts the shared model store
- other nodes stage needed files from that host
- Skulk keeps the same cluster and inference architecture, but changes where model artifacts come from

## What Does Not Change

- the libp2p mesh, election, master, and worker model stay the same
- the main Skulk API stays the same
- the dashboard remains your main control surface
- single-node Skulk still works fine without the model store

## Before You Start

Make sure:

- all nodes are running the same Skulk build
- you know which machine should be the store host
- that machine has enough storage for the models you want to share
- the chosen `store_path` is mounted and writable

The store server uses port `58080` by default.

## Recommended Setup: Dashboard First

This is the simplest path for most people.

1. Start Skulk on all nodes with `uv run exo`.
2. Open the dashboard on the node you want to become the store host.
3. Go to **Settings**.
4. Enable the store host toggle.
5. Choose the store path.
6. Save the config.
7. Restart Skulk on all nodes if the dashboard tells you a restart is required.

After that, use the dashboard or API normally. When models are available in the store, worker nodes stage from the store host instead of downloading independently.

## Manual Setup with `exo.yaml`

If you prefer to configure the model store manually, put the same `exo.yaml` file on each node.

Minimal example:

```yaml
model_store:
  enabled: true
  store_host: mac-studio-1
  store_path: /Volumes/ModelStore/models
```

For most users:

- `store_host` should be the hostname of the store machine
- `store_path` should be an absolute path on that host

## Example Full Configuration

```yaml
model_store:
  enabled: true
  store_host: mac-studio-1
  store_port: 58080
  store_path: /Volumes/ModelStore/models

  download:
    allow_hf_fallback: true

  staging:
    enabled: true
    node_cache_path: ~/.exo/staging
    cleanup_on_deactivate: true

  node_overrides:
    mac-studio-1:
      staging:
        node_cache_path: /Volumes/ModelStore/models
        cleanup_on_deactivate: false
```

## How to Think About It

There are two important paths:

- `store_path`: the shared source of truth on the store host
- `node_cache_path`: the local staging area where a node prepares files before loading them

For worker nodes, `node_cache_path` is usually a fast local path such as `~/.exo/staging`.

For the store host, you often point `node_cache_path` at the same directory as `store_path` so the store host can load directly from the shared volume without making another copy.

## Important Fields

### `model_store.enabled`

Turns the model store on or off without deleting the config file.

### `model_store.store_host`

The hostname or node ID of the store host.

For most users, hostname is the easiest and most reliable choice.

### `model_store.store_port`

HTTP port used for store transfers.

Default: `58080`

### `model_store.store_path`

Absolute path on the store host where shared models live.

### `model_store.download.allow_hf_fallback`

Controls what happens if a requested model is not already in the store.

| Value | Behavior |
|-------|----------|
| `true` | Fall back to Hugging Face download when needed |
| `false` | Fail instead of downloading from Hugging Face |

Use `false` if you want stricter offline or air-gapped behavior.

### `model_store.staging.node_cache_path`

Where a node stages files before loading them.

### `model_store.staging.cleanup_on_deactivate`

If `true`, staged files are cleaned up when instances are shut down.

## Typical Flow

### First time a model is needed

If the model is not already in the store and fallback is enabled:

1. Skulk requests the model.
2. The store-aware download path checks the store.
3. If the model is missing, Skulk falls back to Hugging Face.
4. The model lands in the appropriate local or store-managed path.

### Later requests

Once the model exists in the store:

1. worker nodes ask the store host for the needed files
2. files are staged locally
3. inference loads from the staged path

## Useful Store Endpoints

These are exposed through the main Skulk API:

- `GET /store/health`
- `GET /store/registry`
- `GET /store/downloads`
- `POST /store/models/{model_id}/download`
- `GET /store/models/{model_id}/download/status`
- `DELETE /store/models/{model_id}`
- `POST /store/purge-staging`
- `POST /store/models/{model_id}/optimize`

Common meanings:

- `503 Store not configured`: the cluster is not configured to use a model store
- `503 Store unreachable`: the store is configured, but the API cannot reach it
- `404`: the model or job does not exist
- `409`: a conflicting operation is already in progress

## Troubleshooting

### The store host seems unreachable

Check:

- that the store host is running
- that `store_host` matches the real hostname
- that port `58080` is reachable on your LAN

Useful check:

```bash
curl http://STORE_HOST:58080/health
```

### The model is on disk but does not appear in the store registry

Check:

- that the model is in the configured `store_path`
- that the registry knows about it
- that the dashboard Store Registry view shows it

Useful check:

```bash
curl http://localhost:52415/store/registry
```

### Nodes still download from Hugging Face

Check:

- whether the model is already present in the store
- whether `allow_hf_fallback` is still `true`
- whether the store host is reachable from worker nodes

### Staged files are not being cleaned up

Check `cleanup_on_deactivate` and any `node_overrides` that may disable cleanup for a specific machine.

## Good Defaults for Most Clusters

- use the dashboard to manage the store config
- choose one machine with the most storage as the store host
- keep `allow_hf_fallback: true` while you are getting started
- use a fast local staging path on worker nodes
- point the store host's `node_cache_path` at the store itself

## Related Docs

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- [API guide](api.md)
- [Architecture overview](architecture.md)
- [exo.yaml example](https://github.com/Foxlight-Foundation/Skulk/blob/main/exo.yaml.example)
