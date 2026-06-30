---
id: model-store
title: Skulk Model Store
sidebar_position: 3
---

<!-- Copyright 2025 Foxlight Foundation -->

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

### GGUF repositories download only the pinned quantization

A GGUF repository often ships several quantizations of the same model (for
example `Q4_K_M`, `Q5_K_M`, `Q8_0`, `bf16`). The store downloads only the
quantization a model card pins (its `gguf_file`), plus the multimodal projector
for a vision model, rather than every quant in the repository. This keeps a
single-quant download to roughly the size of that one file instead of the whole
repo.

### The store host advertises a routable address

The store host broadcasts the address other nodes use to reach it. Even when you
configure `store_host` as a hostname, the store host resolves and advertises its
own best routable IPv4 (a private LAN address is preferred). This avoids a
failure mode on a Thunderbolt-meshed fleet, where a bare hostname could resolve
through mDNS to a link-local Thunderbolt address (`169.254.x`) that a peer
without a direct Thunderbolt link cannot reach, even though the LAN path works.
An operator-supplied routable IP in `store_http_host` is still honored as-is.

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

1. Start Skulk on all nodes with `uv run skulk`.
2. Open the dashboard on the node you want to become the store host.
3. Go to **Settings**.
4. Enable the store host toggle.
5. Choose the store path.
6. Save the config.
7. Restart Skulk on all nodes if the dashboard tells you a restart is required.

After that, use the dashboard or API normally. When models are available in the store, worker nodes stage from the store host instead of downloading independently.

## Manual Setup with `skulk.yaml`

If you prefer to configure the model store manually, put the same `skulk.yaml` file on each node.

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
    node_cache_path: ~/.skulk/staging
    # Keep the newest ~40 GiB of idle staged copies warm and evict older ones
    # on deactivation/startup. Warm and bounded. Set false to keep everything
    # (unbounded) and reclaim disk only via POST /store/purge-staging.
    cleanup_on_deactivate: true
    staging_keep_recent_gb: 40

  node_overrides:
    mac-studio-1:
      # The store host loads directly from the store path, so it makes no
      # second copy and the recency budget is skipped here regardless.
      staging:
        node_cache_path: /Volumes/ModelStore/models
```

## How to Think About It

There are two important paths:

- `store_path`: the shared source of truth on the store host
- `node_cache_path`: the local staging area where a node prepares files before loading them

For worker nodes, `node_cache_path` is usually a fast local path such as `~/.skulk/staging`.

For the store host, you often point `node_cache_path` at the same directory as `store_path` so the store host can load directly from the shared volume without making another copy.

## Staging Cache and Disk Management

When a worker needs a model it does not host, it copies that model's files from
the store host into its local staging directory (`node_cache_path`, default
`~/.skulk/staging`) and loads from there. Staged copies are independent per
node: the store host keeps the canonical copy, and each worker keeps its own
staged copy of whatever it has run. Left unmanaged, staging grows without bound,
so Skulk keeps it in check with a single recency policy plus an explicit delete
path.

### What counts as "in use"

A staged copy is **in use** whenever a live runner depends on it, including
companion repositories that no instance names directly (a speculative-decoding
draft model, an assistant model, or separate vision weights). In-use copies are
never evicted automatically.

### The recency budget

Idle (not-in-use) staged copies are kept newest-first up to
`staging_keep_recent_gb` (default 40 GiB); anything beyond that budget is
deleted. The budget is a floor for recently used data, not a disk-pressure
threshold. Nothing watches free space and triggers eviction when the disk fills:
the check runs at two specific moments, and only when `cleanup_on_deactivate` is
`true`:

- when a model instance is shut down, and
- at node startup, which reconciles copies orphaned by a crash or kill.

`cleanup_on_deactivate` is the on/off switch for that check:

| Setting | Behavior |
|---------|----------|
| `true` (default, recommended) | Keep the newest ~40 GiB of idle copies warm; delete older idle copies when an instance shuts down and at node startup. The in-use set is always kept and does not count against the budget. Warm and bounded. |
| `false` | Never evict automatically. Every staged copy is kept until you reclaim space manually. Warm, but unbounded: a busy node can fill its disk. |

Set `staging_keep_recent_gb` to `0` for strict evict-on-deactivate (keep only
what is in use). Raise it on nodes with large disks to keep more models warm.

The in-use set rides on top of the budget rather than inside it: a node always
keeps everything its live runners need, plus up to 40 GiB of the most recently
used idle copies.

> The store host is a special case. When a node points `node_cache_path` at the
> same directory as `store_path` (so it loads directly from the store without a
> second copy), the recency budget is skipped on that node whatever the toggle
> says. The store's canonical copies are never auto-evicted.

### Deleting a model from the store

`DELETE /store/models/{model_id}` removes the canonical copy from the store host
**and** evicts that model's staged copy from every node in the cluster at once.
This path is unconditional: it ignores both `cleanup_on_deactivate` and the
recency budget, because once the canonical copy is gone the staged copies are
orphans. Use it to remove a model everywhere and reclaim its disk fleet-wide in
one call.

### Reclaiming disk manually

`POST /store/purge-staging` clears staged copies on every node without touching
the store's canonical copies. With no body it purges the whole staging cache;
scoped to a `model_id` it purges just that model. Use it when
`cleanup_on_deactivate` is `false`, or when you want space back immediately
rather than waiting for the next deactivation.

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

Controls automatic eviction of idle staged copies. Default `true` (recommended):
when a model is shut down, or at node startup, idle staged copies beyond the
`staging_keep_recent_gb` budget are deleted, keeping the cache warm but bounded.
Set to `false` to keep every staged copy and reclaim disk only via
`POST /store/purge-staging`. See
[Staging Cache and Disk Management](#staging-cache-and-disk-management).

### `model_store.staging.staging_keep_recent_gb`

Recency budget in GiB for idle staged copies (default `40`). Eviction keeps the
newest idle copies up to this size and deletes the rest; `0` evicts everything
not in use, and larger values keep more models warm on big-disk nodes. Applies
only when `cleanup_on_deactivate` is `true`.

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

The dashboard's Store Registry view combines these registry entries with model
metadata so it can show capability-derived tags for downloaded models. Today
that includes `vision`, `thinking`, `embedding`, `tensor`, and `optiq` when the
underlying model card exposes enough metadata for Skulk to derive them.

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

### A multimodal model is in the store but the UI does not show vision support

Check:

- that the model card includes the `vision` capability
- that the dashboard is running a current Skulk build
- that `GET /v1/models` returns a `vision` tag for that model

Remember that store registration only tracks artifacts and metadata. Actual
image understanding still depends on launching the model and sending a
multimodal request through the chat APIs.

### Placements are slow even though the model is already in the store

A staged copy that falls outside the recency budget had to be re-copied from the
store host before loading. With `cleanup_on_deactivate: true` (the default), the
newest ~40 GiB of idle copies stay warm, but a model larger than the budget, or
one pushed out by other recently used models, is evicted and re-staged on its
next placement. Raise `staging_keep_recent_gb` to keep more models warm, or set
it large enough to hold the models you cycle between. Setting
`cleanup_on_deactivate: false` keeps every copy warm but lets staging grow
without bound (reclaim disk with `POST /store/purge-staging`).

### Staged files are not being cleaned up

With `cleanup_on_deactivate: false`, nothing is evicted automatically, so
staging keeps growing. Either leave it at the `true` default (which keeps the
newest ~40 GiB warm and evicts the rest on deactivation/startup) or reclaim
space on demand with `POST /store/purge-staging`. Note that the store host never
auto-evicts its own canonical copies when `node_cache_path` equals `store_path`.

## Good Defaults for Most Clusters

- use the dashboard to manage the store config
- choose one machine with the most storage as the store host
- keep `allow_hf_fallback: true` while you are getting started
- use a fast local staging path on worker nodes
- point the store host's `node_cache_path` at the store itself

## Related Docs

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- [API guide](api-guide)
- [Architecture overview](architecture)
- [skulk.yaml example](https://github.com/Foxlight-Foundation/Skulk/blob/main/skulk.yaml.example)
