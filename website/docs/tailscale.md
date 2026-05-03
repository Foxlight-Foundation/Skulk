---
title: Remote access via Tailscale
description: Access your Skulk dashboard and operator panel from anywhere using Tailscale.
sidebar_label: Remote access
---

# Remote access via Tailscale

Skulk's dashboard and API are served over plain HTTP on port 52415. On your local network that works fine — but when you want to check on your cluster from your phone, restart a node from a coffee shop, or run inference from another machine, you need a way to reach those nodes without opening ports to the internet.

Tailscale solves this cleanly. It creates a private overlay network where every device gets a stable `100.x.x.x` address. Once your cluster node and your phone (or laptop) are both on the same tailnet, you can open `http://100.x.x.x:52415` exactly like you would at home — encrypted, no port forwarding, no VPN configuration.

## What you get

- **Dashboard from anywhere** — full cluster view, observability, traces, model placement
- **Operator panel** — mobile-friendly cluster control from your phone: node health, memory, GPU, temperature, and tap-twice node restart
- **API access** — run inference or call management endpoints from any device on your tailnet
- **Works with [Headscale](https://headscale.net/)** — self-host the control plane if you prefer

## Setup

### 1. Install Tailscale on your cluster node

On the machine running Skulk:

```bash
# macOS — install from tailscale.com/download or:
brew install tailscale

# Linux:
curl -fsSL https://tailscale.com/install.sh | sh
```

Then connect it to your tailnet:

```bash
tailscale up
```

Note the IP address:

```bash
tailscale ip -4
# e.g. 100.101.102.103
```

### 2. Install Tailscale on your remote device

On your phone, tablet, or laptop — install the Tailscale app and log in to the **same Tailscale account**. That's it; both devices are now on the same tailnet.

- iOS / Android: search "Tailscale" in the App Store / Play Store
- macOS / Windows / Linux: [tailscale.com/download](https://tailscale.com/download)

### 3. Open the dashboard

In any browser on your remote device:

```
http://100.101.102.103:52415
```

Replace `100.101.102.103` with your node's Tailscale IP from step 1. You get the full Skulk dashboard — chat, cluster view, observability, everything.

:::tip Bookmark it
Save the `http://100.x.x.x:52415` URL on your phone. iOS and Android both let you add it to your home screen as a web app shortcut.
:::

## Operator panel

The dashboard includes a mobile-first operator view designed for exactly this scenario — checking on your cluster and restarting nodes from a small screen.

To open it, navigate to the dashboard and open the browser console, then run:

```js
window.__skulkNavigate?.('operator')
```

Or bookmark `http://100.x.x.x:52415` and use the direct route:

The operator panel shows:
- **Cluster summary** — total nodes, aggregate memory usage, average GPU utilization, average temperature
- **Per-node cards** — role (master/worker), memory bar, GPU usage, temperature, active placements
- **Tap-twice restart** — tap "Restart" on any node card; a "Confirm?" prompt appears; tap again within 3 seconds to send the restart command. Accidental taps do nothing.

Restarts are sent over the cluster's pub/sub channel, so you can restart any node — including remote ones — from any node's dashboard.

## API access over Tailscale

The full Skulk API is available at the same address:

```bash
# From any device on your tailnet:
curl http://100.101.102.103:52415/v1/models

# Run inference:
curl http://100.101.102.103:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [{"role": "user", "content": "Hello"}]}'

# Check Tailscale connectivity status:
curl http://100.101.102.103:52415/v1/connectivity/tailscale
```

## Verify it's working

**Check the node's Tailscale status** in the startup logs:

```
INFO  Tailscale: running | IP 100.101.102.103 | my-node.tailnet-abc.ts.net
```

**Check in the dashboard** — Observability → Node tab → Runtime section shows the Tailscale row with the node's IP and DNS name.

**Check via the API:**

```bash
curl http://100.101.102.103:52415/v1/connectivity/tailscale | python3 -m json.tool
```

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

## Troubleshooting

### Can't reach `100.x.x.x:52415`

First confirm Tailscale can reach the node at all:

```bash
ping 100.101.102.103
```

If ping fails, the devices aren't on the same tailnet — check that both are logged into the same Tailscale account (or Headscale server) and that Tailscale is running on both.

If ping succeeds but port 52415 doesn't respond, Skulk may not be running. SSH in (also works over Tailscale) and check:

```bash
# macOS:
launchctl print gui/$(id -u)/foundation.foxlight.skulk | grep "state ="

# Linux:
systemctl --user status skulk
```

### `Tailscale: not running` in the Skulk logs

tailscaled is installed but not running on the cluster node:

```bash
# macOS — open the Tailscale app, or:
sudo tailscaled &
tailscale up

# Linux:
sudo systemctl start tailscaled
tailscale up
```

### Tailscale ACLs blocking the port

By default all devices on the same tailnet can reach each other. If you've customised your ACL policy, make sure TCP 52415 (API/dashboard) is allowed between your devices.

## Using Headscale

[Headscale](https://headscale.net/) is a self-hosted Tailscale control server. Skulk treats it identically to Tailscale's own servers — no config changes needed. Join each device to your Headscale instance:

```bash
tailscale up --login-server https://your-headscale-server.example.com
```

Then follow the same setup steps above.
