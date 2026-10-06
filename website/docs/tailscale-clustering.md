---
title: Multi-network clusters via Tailscale
description: Connect Skulk cluster nodes that live on different physical networks using Tailscale.
sidebar_label: Multi-network clustering
---

# Multi-network clusters via Tailscale

By default, Skulk discovers cluster peers using mDNS, which only works on the same local network segment. If you want cluster nodes in different locations (a Mac at home, a Linux box at a colo, a cloud VM), mDNS won't reach them.

Tailscale solves this by giving every node a stable `100.x.x.x` address that works across any network. Use those addresses for Skulk's control-plane bootstrap and its data-plane connections. Both paths must be reachable.

:::info Tailscale must be installed on every cluster node
Unlike the [remote access](tailscale) scenario (where only the node you want to reach needs Tailscale), multi-network clustering requires **every node** to have Tailscale installed and running. This is because Skulk dials each peer directly by its `100.x.x.x` address; there's no gateway or proxy. If a node doesn't have a Tailscale IP, the other nodes have no address to dial it on.

If all your nodes are on the same local network, you don't need Tailscale at all: mDNS handles discovery automatically. Tailscale is only needed when nodes are on physically separate networks.
:::

## Prerequisites

On **every node** that will join the cluster:

1. Install Tailscale and log in: see [Remote access via Tailscale](tailscale) for install instructions
2. Confirm each machine has a `100.x.x.x` address: `tailscale ip -4`
3. All nodes must be on the **same tailnet** (same Tailscale account or Headscale server)

## Setup

### 1. Enable Tailscale connectivity on every node

This advanced cross-network compute setup is not a control in the dashboard
Settings panel. Configure the following on **every** node; the same three lines
enable control-plane discovery, with no per-node bootstrap IP list required:

```yaml
connectivity:
  tailscale:
    enabled: true
```

With this set, each node queries its local tailnet and **auto-discovers every
other node's `100.x` address as a bootstrap peer**: there is no list to
maintain. Nodes are dialed on Skulk's libp2p port (default `52416`); peers that
aren't running Skulk simply fail Skulk's private-network handshake and are
ignored. When a node's Tailscale IP changes, discovery picks up the new one on
the next restart with no config edit.

:::tip Local Network permission not needed over Tailscale
The Tailscale overlay (a `utun` interface) is exempt from macOS Local Network
Privacy, so (unlike a plain LAN/Thunderbolt cluster) you do **not** need to
grant Local Network access for a Tailscale cluster to form. See
[Thunderbolt clustering](thunderbolt-clustering) for the local-network case.
:::

### 2. (Optional) Pin specific peers

Auto-discovery is enough for most setups. If you want to pin specific bootstrap
addresses anyway (for example to dial a node on a non-default port, or to seed
peers from outside the tailnet), list them explicitly; they are merged with the
auto-discovered set:

```yaml
connectivity:
  tailscale:
    enabled: true
    bootstrap_peers:
      - /ip4/100.101.102.103/tcp/52416   # pinned peer
```

Port `52416` is Skulk's default libp2p port. If you changed it with `--libp2p-port`, use that port instead.

### 3. Connect the data plane

mDNS and Zenoh multicast discovery do not cross routed networks. Control-plane
bootstrap alone does not establish the data path used for inference output and
media. Configure `SKULK_ZENOH_LISTEN` with a reachable listener on each node and
`SKULK_ZENOH_CONNECT` with the peers' explicit `tcp/HOST:PORT` endpoints. Use the
actual listener ports rather than assuming they are the API or libp2p port. Allow
those ports in your tailnet policy as well as the control port.

All nodes must use the same data transport and `SKULK_LIBP2P_NAMESPACE`. After
startup, check `/state` and data-plane diagnostics for transport mismatch or
isolation before starting a multi-node workload. See
[cluster communication](cluster-communication.md) for defaults and constraints.

### 4. Restart Skulk on every node

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

**Check startup logs** on each node, looking for the Tailscale line:

```
INFO  Tailscale: running | IP 100.101.102.101 | my-node.tailnet-abc.ts.net
```

**Check the cluster view**: open the dashboard on any node (`http://100.x.x.x:52415`). Once libp2p has dialed the bootstrap peers and gossipsub has propagated state, all nodes should appear. Allow 10 to 15 seconds after the last node restarts.

**Check via the API:**

```bash
curl http://localhost:52415/state | python3 -m json.tool
```

Look for all expected peers in `topology.nodes`, their identities in
`nodeIdentities`, and fresh backend/resource observations in `nodeResources`.

## How peer discovery works

Skulk's cluster uses gossipsub for state propagation. You only need to list **some** of the other nodes in `bootstrap_peers`, not all of them. Once Node A connects to Node B, and Node B already knows about Node C, Node A will learn about Node C indirectly within a few seconds. A single well-connected bootstrap node is enough to bring a new node into the cluster.

Use `tailscale status` and `tailscale ip -4` to verify current addresses when diagnosing connectivity. Explicit peer endpoints must be updated if an address or listener changes.

## Troubleshooting

### Nodes can't reach each other

```bash
ping 100.101.102.102
```

If ping fails between nodes, check:
- Both nodes are on the same tailnet (same Tailscale account or Headscale server)
- `tailscale status` on each node shows the other as a peer
- Your tailnet policy allows the configured libp2p and Zenoh TCP ports between nodes

### Only some nodes are visible in the dashboard

Gossipsub fans out from bootstrap peers. If Node A only lists Node B, and Node B hasn't connected to Node C yet, Node A won't see Node C immediately. Give it 10 to 15 seconds after all nodes have restarted. If it doesn't resolve, check that every node has at least one valid bootstrap peer in its config.

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

[Headscale](https://headscale.net/) is a self-hosted Tailscale control server. Skulk works with it identically: `tailscale status --json` returns the same structure regardless of whether the control plane is Tailscale's or Headscale's. No config changes needed.

```bash
tailscale up --login-server https://your-headscale-server.example.com
```

Join every node to the same Headscale server, then follow the setup steps above.
