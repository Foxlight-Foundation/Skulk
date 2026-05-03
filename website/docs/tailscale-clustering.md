---
title: Multi-network clusters via Tailscale
description: Connect Skulk cluster nodes that live on different physical networks using Tailscale.
sidebar_label: Multi-network clustering
---

# Multi-network clusters via Tailscale

By default, Skulk discovers cluster peers using mDNS, which only works on the same local network segment. If you want cluster nodes in different locations — a Mac at home, a Linux box at a colo, a cloud VM — mDNS won't reach them.

Tailscale solves this by giving every node a stable `100.x.x.x` address that works across any network. You configure Skulk to use those addresses as bootstrap peers, and the cluster forms over the Tailscale overlay.

## Prerequisites

On **every node** that will join the cluster:

1. Install Tailscale and log in: see [Remote access via Tailscale](tailscale) for install instructions
2. Confirm each machine has a `100.x.x.x` address: `tailscale ip -4`
3. All nodes must be on the **same tailnet** (same Tailscale account or Headscale server)

## Setup

### 1. Collect every node's Tailscale IP

On each machine:

```bash
tailscale ip -4
```

Write down the IP for every node in the cluster.

### 2. Edit `skulk.yaml` on every node

Add a `connectivity` section listing the **other** nodes' Tailscale IPs. You don't list yourself — only peers.

**Node A** (`100.101.102.101`) — lists B and C:

```yaml
connectivity:
  tailscale:
    enabled: true
    bootstrap_peers:
      - /ip4/100.101.102.102/tcp/52416   # Node B
      - /ip4/100.101.102.103/tcp/52416   # Node C
```

**Node B** (`100.101.102.102`) — lists A and C:

```yaml
connectivity:
  tailscale:
    enabled: true
    bootstrap_peers:
      - /ip4/100.101.102.101/tcp/52416   # Node A
      - /ip4/100.101.102.103/tcp/52416   # Node C
```

Port `52416` is Skulk's default libp2p port. If you changed it with `--libp2p-port`, use that port instead.

### 3. Restart Skulk on every node

```bash
# Running manually:
uv run skulk

# Running as a service (macOS):
launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk

# Running as a service (Linux):
systemctl --user restart skulk
```

Skulk reads the config, logs the Tailscale status, and dials the bootstrap peers over the overlay.

## Verify the cluster formed

**Check startup logs** on each node — look for the Tailscale line:

```
INFO  Tailscale: running | IP 100.101.102.101 | my-node.tailnet-abc.ts.net
```

**Check the cluster view** — open the dashboard on any node (`http://100.x.x.x:52415`). Once libp2p has dialed the bootstrap peers and gossipsub has propagated state, all nodes should appear. Allow 10–15 seconds after the last node restarts.

**Check via the API:**

```bash
curl http://localhost:52415/v1/state | python3 -m json.tool
```

Look for all expected nodes in the `nodes` map.

## How peer discovery works

Skulk's cluster uses gossipsub for state propagation. You only need to list **some** of the other nodes in `bootstrap_peers` — not all of them. Once Node A connects to Node B, and Node B already knows about Node C, Node A will learn about Node C indirectly within a few seconds. A single well-connected bootstrap node is enough to bring a new node into the cluster.

Tailscale IPs are stable — they don't change unless you reinstall Tailscale. You set `bootstrap_peers` once and leave it.

## Troubleshooting

### Nodes can't reach each other

```bash
ping 100.101.102.102
```

If ping fails between nodes, check:
- Both nodes are on the same tailnet (same Tailscale account or Headscale server)
- `tailscale status` on each node shows the other as a peer
- Your tailnet ACL policy allows TCP on port 52416 between nodes (the default "allow all" policy works; a custom ACL might block it)

### Only some nodes are visible in the dashboard

Gossipsub fans out from bootstrap peers. If Node A only lists Node B, and Node B hasn't connected to Node C yet, Node A won't see Node C immediately. Give it 10–15 seconds after all nodes have restarted. If it doesn't resolve, check that every node has at least one valid bootstrap peer in its config.

### Wrong IP in the multiaddr

Run `tailscale ip -4` on the relevant machine and update `skulk.yaml`. Tailscale IPs are stable but worth verifying if something looks off.

### `Tailscale connectivity configured but tailscaled is not running`

tailscaled is not running on that node. Fix:

```bash
# macOS:
sudo tailscaled &
tailscale up

# Linux:
sudo systemctl start tailscaled
tailscale up
```

## Using Headscale

[Headscale](https://headscale.net/) is a self-hosted Tailscale control server. Skulk works with it identically — `tailscale status --json` returns the same structure regardless of whether the control plane is Tailscale's or Headscale's. No config changes needed.

```bash
tailscale up --login-server https://your-headscale-server.example.com
```

Join every node to the same Headscale server, then follow the setup steps above.
