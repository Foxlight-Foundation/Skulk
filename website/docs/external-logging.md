---
title: External logging (Vector + VictoriaLogs + Grafana)
description: Ship Skulk's structured JSON logs to a central store so you can search, alert, and graph across the whole cluster.
sidebar_label: External logging
---

# External logging

Forward Skulk's structured logs from every node to one place where you can search and graph them. This guide walks through the whole stack — what it is, how to install it, how to wire Skulk into it, and how to debug it when it doesn't work.

## What you'll have when you're done

- Every Skulk node ships its logs to one central store (VictoriaLogs).
- You can search across the whole cluster from a Grafana panel — by node, by component, by message, by time range.
- Logs survive node reboots, network blips, and the central store being down (Vector buffers up to 512 MB on disk per node).
- Skulk's inference path is **never blocked** by a slow log shipper, because the shipper runs as a separate process.

About 30 minutes for first-time setup of the central stack, then about 1 minute per node.

## Architecture in one picture

```
┌──────────────────────┐    JSON lines    ┌─────────────────────┐
│  Skulk node (laptop, │  ──────────────▶ │  ~/.skulk/logs/     │
│  Mac mini, R720…)    │   on stdout      │  skulk.stdout.log   │
└──────────────────────┘                  └─────────────────────┘
                                                    │
                                                    │ tailed by
                                                    ▼
                                          ┌─────────────────────┐
                                          │  Vector LaunchAgent │
                                          │  (skulk-vector)     │
                                          └─────────────────────┘
                                                    │
                                                    │ HTTP POST,
                                                    │ disk-buffered
                                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│  Central host (e.g. R720) running Docker Compose                   │
│                                                                    │
│   ┌──────────────────┐    queries    ┌──────────────────┐         │
│   │  VictoriaLogs    │ ◀──────────── │  Grafana         │         │
│   │  port 9428       │               │  port 3000       │         │
│   └──────────────────┘               └──────────────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

Three layers:

1. **Skulk** writes one JSON object per log line to `stdout`. The LaunchAgent / systemd unit captures stdout to a file.
2. **Vector** runs as its own process on every Skulk node. It tails that file, batches lines, and POSTs them to the central store. If the central store is down, Vector buffers to disk and ships when it comes back.
3. **VictoriaLogs + Grafana** run together (single Docker Compose stack) on whatever machine you've designated as the central host. VictoriaLogs stores the logs; Grafana queries them.

## Why a separate process for Vector?

Skulk's inference threads must never block on logging. If Vector is in the same process and the central store is slow, the kernel pipe between them fills up and every `logger.info()` call in Skulk blocks. By running Vector as a separate agent that reads from a file, slow shipping just means the file grows on disk — Skulk keeps inferring at full speed.

This is why the LaunchAgent installer (`deployment/install/install-launchd.sh`) installs **two** agents by default: the Skulk service itself, and a `skulk-vector` shipper that runs alongside it.

## Step 1 — Set up the central stack (one-time, on one machine)

Pick the machine you want logs to live on. Anything that can run Docker works — an R720, a NAS, a Mac mini, a small VPS. It needs to be reachable from every Skulk node on TCP 9428 (ingest) and 3000 (Grafana UI).

On the central host:

```bash
# Clone Skulk if you haven't already (only the deployment/ dir is needed)
git clone https://github.com/foxlight-foundation/skulk.git
cd skulk/deployment/logging

# Set the Grafana admin password (required — the compose file refuses to
# start without it)
echo "GF_SECURITY_ADMIN_PASSWORD=$(openssl rand -base64 24)" > .env
echo "Wrote a Grafana admin password to .env — keep this file safe."

# Bring the stack up
docker compose up -d
```

This launches:

- **VictoriaLogs** on port 9428 — log ingest and storage. Built-in UI at `http://<host>:9428/select/vmui/`.
- **Grafana** on port 3000 — dashboards. Username `admin`, password from your `.env` file.

Verify both are healthy:

```bash
curl -s http://localhost:9428/health
# expected: {"status":"ok"}

curl -sI http://localhost:3000/login | head -1
# expected: HTTP/1.1 200 OK
```

The Grafana stack is pre-configured to use VictoriaLogs as its default data source — no manual wiring needed.

### What gets persisted

The Compose file uses two named volumes:

- `vlogs-data` — VictoriaLogs storage, 90-day retention by default
- `grafana-data` — Grafana dashboards, users, and config

These survive container restarts and image upgrades. To wipe them, `docker compose down -v`.

## Step 2 — Point each Skulk node at the central stack

The shipping process model differs between platforms. Both ship to the same central stack:

- **macOS** runs Vector as a separate LaunchAgent (`foundation.foxlight.skulk-vector`) that tails Skulk's captured stdout file. Lifecycle is decoupled from Skulk — a slow VictoriaLogs cannot backpressure inference.
- **Linux** runs Vector as an in-process subprocess that Skulk spawns when `logging.enabled: true` is set in `skulk.yaml`. JSON is piped directly into Vector's stdin via `deployment/logging/vector.yaml` (stdin source). This release does not include a separate `skulk-vector` systemd unit.

On every node that's running Skulk:

1. **Install Vector.** Single binary; instructions at [vector.dev](https://vector.dev/docs/setup/installation/). On macOS: `brew install vectordotdev/brew/vector`. On Debian/Ubuntu: `curl -1sLf 'https://repositories.timber.io/public/vector/cfg/setup/bash.deb.sh' | sudo -E bash && sudo apt install vector`.
2. **Install the Skulk service** (if you haven't already):

   ```bash
   deployment/install/install-launchd.sh    # macOS — installs both skulk + skulk-vector agents
   deployment/install/install-systemd.sh    # Linux — installs skulk only
   ```

   On macOS, pass `--no-vector` to skip the external Vector agent and fall back to the in-process subprocess model.
3. **Tell the shipper where to ship.** Edit `~/.skulk/skulk.env` and set:

   ```bash
   EXO_LOGGING_INGEST_URL=http://<central-host>:9428/insert/jsonline?_stream_fields=node_id,component&_msg_field=msg&_time_field=ts
   ```

   On Linux, also set `logging.enabled: true` and `logging.ingest_url: <same-url>` in `skulk.yaml` so Skulk knows to spawn its in-process Vector subprocess.

   The query parameters tell VictoriaLogs which fields to use as stream identifiers (so `node_id` and `component` become indexed dimensions).
4. **Make sure Skulk is emitting JSON.** On macOS this is on by default when you install via the wrapper (`SKULK_LOGGING_EXTERNAL=1` in the env file). On Linux this is gated by `logging.enabled` in `skulk.yaml`.
5. **Restart so the new config is picked up:**

   ```bash
   # macOS — restart both agents
   launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk
   launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk-vector

   # Linux — Skulk respawns its Vector subprocess on restart
   systemctl --user restart skulk
   ```

That's it for that node. Repeat on each one.

## Step 3 — Verify logs are flowing

On any node:

```bash
# Last 5 lines of what Vector is shipping right now
tail -n 5 ~/.skulk/logs/skulk.stdout.log

# Vector's own status — should show 0 errors and recent successful POSTs
tail -f ~/.skulk/logs/vector.stderr.log
```

In Grafana (`http://<central-host>:3000`), open Explore, pick the VictoriaLogs data source, and run:

```
*
```

You should see log lines from every node that has shipped at least one event. To filter to one node:

```
node_id:"laptop-1"
```

To see only errors from a specific component:

```
level:ERROR AND component:"worker"
```

The full LogsQL syntax is in the [VictoriaLogs docs](https://docs.victoriametrics.com/victorialogs/logsql/).

## How the JSON is structured

Each line shipped by Skulk looks like this:

```json
{
  "ts": "2026-05-04T12:34:56.789Z",
  "level": "INFO",
  "node_id": "laptop-1",
  "component": "worker",
  "module": "exo.worker.runner",
  "function": "spawn",
  "line": 142,
  "msg": "spawned runner for shard 0/4 of mlx-community/Qwen3-30B"
}
```

`node_id` defaults to the machine's hostname. `component` is the second segment of the Python module path (e.g. `exo.worker.runner` → `worker`). Both are indexed by VictoriaLogs as stream fields so queries against them are fast.

## Customizing where Vector buffers and ships

All knobs live in `~/.skulk/skulk.env` and are picked up by both the Skulk and Vector agents on next restart:

| Env var | What it does | Default |
| --- | --- | --- |
| `SKULK_LOGGING_EXTERNAL` | `1` = Skulk writes JSON to stdout for the external Vector agent. `0` = Skulk spawns its own internal Vector subprocess (only useful if you ran the installer with `--no-vector`) | `1` |
| `EXO_LOGGING_INGEST_URL` | Where Vector POSTs logs | the in-house R720 endpoint |
| `EXO_VECTOR_DATA_DIR` | Where Vector keeps its disk buffer and file checkpoints | `~/.skulk/vector` |
| `SKULK_LOG_FILE` | Override the source file Vector tails | `~/.skulk/logs/skulk.stdout.log` |

After editing, restart the relevant agents (the table in the [service guide](./run-skulk-as-a-service.md#day-to-day-operations) has the commands).

## Things that go wrong

### "Logs are flowing on one node but not another"

Check that node's Vector output:

```bash
# macOS — separate LaunchAgent has its own log
tail -f ~/.skulk/logs/vector.stderr.log

# Linux — Vector runs as a Skulk subprocess; its output is folded into Skulk's
journalctl --user -u skulk -f | grep -i vector
```

Common causes:

- **`vector` isn't installed.** The agent fails fast with a clear error. Install Vector and restart.
- **The node can't reach the central host.** `curl -v http://<central-host>:9428/health` from the node tells you whether it's network or firewall.
- **The wrong ingest URL is in `~/.skulk/skulk.env`.** Vector logs the URL it's POSTing to on startup — check it matches.
- **The source file is empty.** Run `tail ~/.skulk/logs/skulk.stdout.log`. If it's empty, Skulk isn't emitting JSON — see next section.

### "The skulk.stdout.log file is empty"

The wrapper sets `SKULK_LOGGING_EXTERNAL=1` by default, which tells Skulk to emit JSON to stdout. If the file is empty:

- **Skulk isn't running through the wrapper.** Check that the LaunchAgent / systemd unit is actually live (`launchctl print …` / `systemctl --user status skulk`).
- **`SKULK_LOGGING_EXTERNAL` got set to 0** in `~/.skulk/skulk.env`. Set it back to 1.
- **Skulk crashed before logging started.** The file would have a partial early line and then stop. Check `~/.skulk/logs/skulk.stderr.log` for a traceback.

### "Vector buffered for hours, now it's catching up"

This is the design working as intended. When the central store is unreachable, Vector buffers up to 512 MB per node on disk. When connectivity returns, it drains the buffer at full speed. You'll see a temporary spike in CPU and network use until the backlog clears.

To monitor backlog: `du -sh ~/.skulk/vector/` (or wherever `EXO_VECTOR_DATA_DIR` points).

### "VictoriaLogs is full / disk pressure on the central host"

VictoriaLogs is configured for 90-day retention by default. To reduce it, edit `deployment/logging/docker-compose.yml`:

```yaml
command:
  - -retentionPeriod=30d
```

Then `docker compose up -d` to apply. VictoriaLogs reclaims space within a few minutes.

### "I changed the ingest URL but Vector still ships to the old one"

Vector reads its config at startup. Restart the Vector agent:

```bash
# macOS
launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk-vector
# Linux
systemctl --user restart skulk-vector
```

## Disabling external logging

Set `SKULK_LOGGING_EXTERNAL=0` in `~/.skulk/skulk.env` and restart the Skulk service. The Vector agent will keep tailing the (now-empty-of-new-JSON) stdout file harmlessly; you can also bootout the agent if you want it gone:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/foundation.foxlight.skulk-vector.plist
rm ~/Library/LaunchAgents/foundation.foxlight.skulk-vector.plist
```

Or re-run the installer with `--no-vector`.

## Advanced: running Vector standalone (development)

For ad-hoc runs without the LaunchAgent, you can run Vector by hand:

```bash
# Make sure ~/.skulk/skulk.env is sourced so EXO_LOGGING_INGEST_URL is set
source ~/.skulk/skulk.env

# Then run vector pointed at the same config the LaunchAgent uses
vector --config deployment/logging/vector.yaml
```

This is useful when iterating on `vector.yaml` — restart picks up changes immediately, and you see Vector's full output in the terminal.
