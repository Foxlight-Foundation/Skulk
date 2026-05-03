---
title: Use Skulk over Tailscale
description: Connect Skulk cluster nodes across the internet using Tailscale.
sidebar_label: Tailscale
---

# Use Skulk over Tailscale

By default, Skulk discovers cluster peers using mDNS, which only works on the same local network. Tailscale gives every node a stable `100.x.x.x` address that works from anywhere — at home, at a cloud provider, or behind a NAT — so you can build a cluster that spans physical locations.

**What you get:**
- Nodes on different networks join the same cluster
- Encrypted peer-to-peer traffic between nodes (Tailscale handles it)
- Works with [Headscale](https://headscale.net/) — same config, just join a different tailnet

## Before you start

On **every node** that will join the cluster:

1. Install Tailscale: [tailscale.com/download](https://tailscale.com/download)
2. Log in: `tailscale up`
3. Confirm each machine has a `100.x.x.x` address: `tailscale ip -4`

All nodes must be on the **same tailnet** (the same Tailscale account or Headscale server). Nodes on different tailnets cannot reach each other.

## Quick setup

Skulk nodes discover each other via **bootstrap peers** — a list of libp2p multiaddrs that each node dials at startup. For Tailscale, those multiaddrs use the `100.x.x.x` addresses.

### 1. Find each node's Tailscale IP

On each machine:

```bash
tailscale ip -4
```

Write down the IP for every node that should join the cluster.

### 2. Edit `skulk.yaml`

On **every node**, add a `connectivity` section listing the *other* nodes' Tailscale IPs. You only need to list peers — not yourself.

```yaml
connectivity:
  tailscale:
    enabled: true
    bootstrap_peers:
      - /ip4/100.101.102.103/tcp/52416   # Node B
      - /ip4/100.101.102.104/tcp/52416   # Node C
```

Port `52416` is Skulk's default libp2p port. If you changed it with `--libp2p-port`, use that port instead.

### 3. Restart Skulk

```bash
# If running manually:
uv run skulk

# If running as a service (macOS):
launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk

# If running as a service (Linux):
systemctl --user restart skulk
```

That's it. Skulk reads the config, logs the Tailscale status, and dials the bootstrap peers over the Tailscale overlay.

## Verify it's working

**Check the startup logs** — look for the Tailscale line:

```
INFO  Tailscale: running | IP 100.101.102.103 | my-node.tailnet-abc.ts.net
```

If you see `Tailscale connectivity configured but tailscaled is not running`, tailscaled isn't up yet — run `tailscale status` and fix that first.

**Check via the API:**

```bash
curl http://localhost:52415/v1/connectivity/tailscale | python3 -m json.tool
```

You should see something like:

```json
{
  "running": true,
  "selfIp": "100.101.102.103",
  "hostname": "my-node",
  "dnsName": "my-node.tailnet-abc.ts.net",
  "tailnet": "tailnet-abc.ts.net",
  "version": "1.66.1"
}
```

**Check the dashboard** — open the Observability panel, pick a node, and look in the Runtime section. The Tailscale row shows your Tailscale IP and DNS name, or "not running" if tailscaled is down.

**Check that peers connected** — open the dashboard cluster view. Once libp2p has dialed the bootstrap peers and gossipsub has propagated state, both nodes should appear.

## Troubleshooting

### `Tailscale: not running` in the logs

tailscaled is installed but not running. Fix:

```bash
# macOS — start the Tailscale app or:
sudo tailscaled &

# Linux:
sudo systemctl start tailscaled
tailscale up
```

### Wrong IP — Tailscale IP doesn't appear in the multiaddr

Run `tailscale ip -4` on that machine and update `skulk.yaml`. Tailscale IPs don't change unless you reinstall, but verify if anything looks off.

### Nodes can't reach each other

Check that Tailscale can ping between the machines:

```bash
ping 100.101.102.103
```

If ping fails, check your tailnet ACLs (Tailscale admin console → Access Controls). By default all nodes on the same tailnet can reach each other, but a custom ACL might block port 52416. Allow TCP on that port between Skulk nodes.

### Only some nodes are visible

Skulk peer discovery fans out from bootstrap peers via gossipsub. If Node A only lists Node B as a bootstrap peer, Node A learns about Node C indirectly once Node B connects to both. Give it 10–15 seconds after the last node restarts.

## Headscale

[Headscale](https://headscale.net/) is a self-hosted Tailscale control server. Skulk works with it identically — `tailscale status --json` reports the same structure regardless of whether the control plane is Tailscale's servers or a Headscale instance. No config changes needed.

Join each node to your Headscale server:

```bash
tailscale up --login-server https://your-headscale-server.example.com
```

Then follow the same setup steps above.
