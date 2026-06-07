---
id: operations
title: Operator Runbook
sidebar_position: 7
---

<!-- Copyright 2025 Foxlight Foundation -->

This is the day-two runbook for running a Skulk cluster: what to check and
what to do when storage fills up, a placement is refused, a node dies
mid-generation, the disk gets tight, or you need to trace a bad request.

It is written around the live control-plane endpoints and the behaviors that
shipped for launch. Every action here is something you can do against a
running cluster with `curl` and the dashboard — there is nothing to recompile.

Throughout, `localhost:52415` is the local node's API; the same endpoints
exist on every node, so swap the host to inspect a specific machine.

## Storage Management

### How staged copies work

With the model store enabled, each node keeps its model files in two places.
The **store** is the authoritative copy served over the LAN from the store
host. **Staging** is the node-local copy a worker writes before MLX loads it
— MLX always loads from a local filesystem path, never from the store
directly. Staged copies live under `node_cache_path` (default
`~/.exo/staging`), one `<org>--<name>` subdirectory per model.

Staged copies are cheap to recreate from the LAN store but local disk on
small-disk nodes is the scarce resource, so staging has a lifecycle.

### The eviction lifecycle and its three triggers

There is one eviction mechanism with three trigger points:

1. **Instance deactivation** — when a model instance shuts down.
2. **Node startup** — which reconciles staged copies orphaned by a crashed
   session (a node that died never got to clean up).
3. **Operator tooling** — `POST /store/purge-staging` (see below).

A staged model becomes an **eviction candidate** when no live runner uses
it. Candidates are kept newest-first by last use up to the
`staging_keep_recent_gb` grace budget, and everything beyond the budget is
deleted.

"In-use includes companions": a model is in use not only when an instance
names it directly but also when it is the **companion** of an active model —
an MTP sidecar, an assistant drafter, or split vision weights. Companions
are never eviction candidates, so eviction can never pull weights out from
under a live runner.

This lifecycle only runs when `cleanup_on_deactivate` is `true` (the
default). Set it to `false` to keep every staged copy and manage cleanup
entirely by hand.

### The grace budget and tuning it

`staging_keep_recent_gb` (default **40 GiB**) is a most-recently-used grace
budget. Eviction never reduces the staging cache below this much of
recently-used, not-in-use model data. The budget exists so that node deaths,
restarts, and repeated place/delete cycles of the same model do not re-pay
the staging copy every time.

Tune it in the `staging` section of `skulk.yaml`, or per node via
`node_overrides`:

```yaml
model_store:
  staging:
    cleanup_on_deactivate: true
    staging_keep_recent_gb: 40
```

- **Disk is tight** → lower it. Set `staging_keep_recent_gb: 0` for **strict
  evict-on-deactivate**: every not-in-use copy is removed the moment its
  instance stops.
- **Disk is plentiful and you re-launch the same few models** → raise it, so
  the next launch of a recently-used model skips the staging copy.
- **Store host that loads directly from the store** → set
  `cleanup_on_deactivate: false` in that node's override (it is loading from
  `store_path`, not making a separate staged copy).

### Seeing the per-node picture

`GET /store/storage` returns the local node's storage breakdown: every
staged model with its size, last-use time, and `in_use` flag (which already
accounts for companions), plus event-log usage and free disk on the models
volume.

```bash
curl http://localhost:52415/store/storage
```

There is no cluster-wide storage endpoint — query each node's API for the
fleet view.

### Manual cleanup

`POST /store/purge-staging` **broadcasts a purge to every node in the
cluster**, removing staged model artifacts without deleting the store copy
itself. The endpoint requires a JSON body; an empty object purges all
not-in-use staged models, and an optional `modelId` narrows the purge to one
model:

```bash
# purge all not-in-use staged copies, cluster-wide
curl -X POST http://localhost:52415/store/purge-staging \
  -H "Content-Type: application/json" -d '{}'

# purge one model's staged copies, cluster-wide
curl -X POST http://localhost:52415/store/purge-staging \
  -H "Content-Type: application/json" \
  -d '{"modelId": "mlx-community/Qwen3.5-9B-MLX-4bit"}'
```

Use this when you have set `cleanup_on_deactivate: false` and are managing
staging by hand, or to reclaim space immediately rather than waiting for the
next deactivation/startup trigger. Remember it acts on **all** nodes — for a
single node's picture before and after, use that node's `GET /store/storage`.

## Placement Failures

Impossible placements now fail loudly at the API **before** the command
reaches the master, with a specific typed reason, instead of returning
"Command received" and leaving the client with an unexplained `404`. Here is
how to read each one.

### 400 — per-node memory arithmetic

A `400` from `POST /place_instance` names the node that cannot fit and shows
the GB arithmetic. The key fact: **memory is checked per node, not summed
across the cycle.** Tensor sharding splits weights evenly and Pipeline
allocates layers proportionally to each node's free memory, and every node
must hold its share times a runtime-overhead factor (KV cache, activations,
runner) on top of the raw weight bytes. A model that exactly equals a node's
free memory is rejected — that placement would thrash rather than run.

What to do:

- **Use a smaller model or a more aggressive quant** (e.g. 4-bit instead of
  8-bit) so each shard fits with headroom.
- **Add more nodes** so the per-node share shrinks.
- **Free memory** on the named node (kill other instances) and retry.

Other `400` reasons from the same endpoint:

- no connected cycle of `min_nodes` nodes (topology gap);
- exclusions removed every candidate;
- the model does not support Tensor sharding.

### 503 — info-pending right after cluster formation

A cluster that has just formed has not finished gossiping yet: connection
edges lag node identities by a few rounds, and per-node memory info lags the
edges. Placing into that window does **not** produce a false "insufficient
memory" — it is reported as info-pending. The request internally waits up to
**15 seconds** for the info to arrive before returning `503`.

What to do: **wait a few seconds and retry.** The 15-second internal grace
covers most cases; a `503` means the info still had not arrived, so a second
attempt shortly after almost always succeeds.

### Node IDs are per-session

`excluded_nodes` (and the preview `excluded_node_ids`) take libp2p node IDs.
**Node IDs change when a cluster session restarts** — they are per-session,
not stable identifiers. Re-read current IDs from `GET /state` before
constructing an exclusion list; an old ID is simply ignored.

Preview before you place to see which combinations are valid and why the
others fail:

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit"
```

## Node and Request Failure Behavior

### A node dies mid-generation

When an instance is lost — node disconnect, crash, or deletion with a
request in flight — open requests now **error within seconds with a
retryable message** instead of hanging until the client's own timeout.

- For a lost **worker**: the master emits `TaskFailed` for in-flight API
  tasks whose instance is gone; streaming responses close with an error
  event and non-streaming requests return a `500`.
- For a lost **master**: failover starts a new cluster session that cannot
  carry the old session's tasks, so the API fails every open command stream
  at the session boundary with an error explaining the session changed and
  asking the client to retry.

End-to-end, clients receive the error within **~4–6 seconds** of a node kill
(master or worker rank). The correct client behavior in both cases is the
same: **retry the request.**

Recovery timeline:

- **Election** of a new master runs on a 3-second timeout
  (`DEFAULT_ELECTION_TIMEOUT`).
- **Orphaned runners** left by the lost session are reconciled by the runner
  supervisor's escalation path; staged copies orphaned by the crash are
  cleaned up at node startup (trigger 2 above).
- **Rejoining is automatic** when the process restarts — the node rejoins
  the cluster on startup with no manual step.

### A wedged GPU vs. a crashed process

These two failure modes look different and call for different responses.

**Wedged GPU (warmup deadline).** A faulted Metal eval can park warmup
forever at 0% CPU, uninterruptible from Python. Warmup now runs under a hard
deadline (default **300 s**, override with `SKULK_WARMUP_DEADLINE_SECONDS`).
On overrun the runner logs a **CRITICAL** diagnosis (including
reboot-if-GPU-wedged guidance) and exits, the supervisor reports
`RunnerFailed`, and **the node keeps dispatching** rather than silently
sitting in `RunnerWarmingUp` while every request queues and times out. If
you see the CRITICAL warmup line recur on the same node, the GPU is wedged —
follow the log's reboot guidance for that machine.

**Crashed process.** When a runner process exits, **GPU/Metal memory is
reclaimed on exit.** A crash is therefore self-healing from a memory
standpoint; the supervisor restarts the runner and the node keeps serving.
The dangerous case is the wedged-but-alive process above, not the clean
crash.

`POST /admin/restart` is the clean way to recycle a node: it replaces the
process image in place (releasing Metal memory) and the node rejoins
automatically.

## Logs and Disk

### Where logs go

Skulk writes human-readable logs to **stderr**. When centralized logging is
enabled (`logging.enabled: true` with `logging.ingest_url` set in
`skulk.yaml` or dashboard Settings), it additionally emits structured JSON,
one object per line, on **stdout**. A local Vector shipper reads stdout and
forwards to VictoriaLogs + Grafana. The full stack setup is in the
[External logging](external-logging.md) guide.

### Event-log retention and the free-space floor

The API-side **event log** records per-token chunk events and backs only the
`GET /events` diagnostic. It now has retention so it cannot eat the disk:

- it **ring-compacts past 256 MiB**, keeping the most recent 20k events;
- archive rotation is capped by total bytes (**1 GiB**) on top of the count
  cap;
- a proactive **free-space floor of 2 GiB** (checked every 1024 appends)
  degrades persistence *before* the disk hits zero.

**Degraded counting-only mode** is what you get when free space hits the
floor (or a write fails with ENOSPC): the node **keeps serving inference**,
but event-log *persistence* degrades — events are counted, not written.
Operationally this means `GET /events` history thins out on that node while
generation continues normally. It is a deliberate trade: before this, a
master on a full disk throttled the whole cluster to ~0.5 tok/s before
dying. Treat counting-only mode as a signal to **free disk on that node.**

Check free disk per node via `GET /store/storage` (it reports event-log
bytes and disk free alongside the staged models).

## Tracing

Runtime tracing is a **debugging** feature, not an always-on mode — leave it
off in normal operation and switch it on to investigate a specific issue.

Enable it cluster-wide for new requests:

```bash
curl -X PUT http://localhost:52415/v1/tracing \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true}'
```

It applies to **new requests only** and does not retroactively trace
in-flight work. Check the current state with `GET /v1/tracing`. Turn it back
off by sending `{"enabled": false}`.

Then send the request you want to investigate. Traces are keyed by the
**master-created task ID, which is not the chat completion's response `id`**
(that is the API command id) — so list the traces and pick yours by
`createdAt` and `modelId`, then fetch its stats by the listed `taskId`:

```bash
curl http://localhost:52415/v1/traces                       # list local traces (taskId, createdAt, modelId)
curl http://localhost:52415/v1/traces/<task_id>/stats       # timing summary for one trace
```

`GET /v1/traces*` reads artifacts stored on the current node;
`GET /v1/traces/cluster*` fans out to reachable peers and deduplicates by
`task_id` for a cluster-wide read-only view. See the
[Tracing](tracing.md) guide for the full endpoint set. Saved trace files are
pruned by an hourly janitor after `tracing.retention_days` (default 3).

## Quick Health Checklist

A fast pass to confirm a cluster is healthy and ready to serve:

1. **Cluster formed.** `GET /state` and confirm the `node_identities` count
   matches the machines you expect to be in the cluster.

   ```bash
   curl -s http://localhost:52415/state
   ```

2. **Storage headroom per node.** `GET /store/storage` on each node — check
   disk free is well above the 2 GiB event-log floor and that staging is not
   close to filling the volume.

3. **A live probe request.** A 2-token completion confirms a placement is
   actually serving:

   ```bash
   curl -X POST http://localhost:52415/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
       "messages": [{"role": "user", "content": "ping"}],
       "max_tokens": 2
     }'
   ```

   A `404 No instance found for model ...` means the placement is not ready
   or never launched; place it first via `POST /place_instance`.

4. **Speculation is engaging (carded models).** For a model that ships a
   drafter, confirm the runner log shows a healthy `MTP acceptance` line
   rather than plain decode. See
   [Speculative Decoding (MTP)](speculative-decoding.md) for what to expect
   and how to read it.
