# Thunderbolt clustering (local setup)

This guide walks through standing up a Skulk cluster on several Macs connected
on the same local network — typically wired together with **Thunderbolt** for
the high-bandwidth inference path. No Tailscale or cross-network configuration
is required; that is covered separately in
[Tailscale clustering](./tailscale-clustering.md).

The happy path is deliberately small:

1. Install Skulk on each node.
2. **Grant macOS Local Network access** (the one step people miss — see below).
3. Run `uv run skulk` on each node.

Skulk discovers peers automatically over mDNS on your local network, and the
inference data plane automatically prefers Thunderbolt links when present.

## 1. Prerequisites

- Two or more Apple-silicon Macs.
- The Macs on the **same local network** (any of: a shared Ethernet/Wi-Fi LAN,
  or a Thunderbolt Bridge — see [Thunderbolt wiring](#thunderbolt-wiring)).
- [`uv`](https://docs.astral.sh/uv/) and Node.js installed on each node.

## 2. Install (each node)

```bash
git clone https://github.com/Foxlight-Foundation/Skulk.git
cd Skulk
npm --prefix dashboard-react install
npm --prefix dashboard-react run build
uv sync
```

## 3. Grant macOS Local Network access (required)

> **This is the most common reason a freshly-installed cluster "won't form."**

macOS 15 (Sequoia) and macOS 26 gate access to the **local network** — your
LAN, link-local addresses, multicast (which mDNS discovery relies on), and the
Thunderbolt Bridge — behind a per-application privacy permission. Until the app
you run Skulk from is granted Local Network access, Skulk **cannot discover or
connect to peers on your local network or over Thunderbolt**. Every local
connection fails with `No route to host` (`EHOSTUNREACH`), while internet and
Tailscale traffic keep working — so the symptom is a cluster that silently
stays at one node.

When you first run Skulk, macOS shows a prompt:

> *"Skulk" / "Terminal" would like to find and connect to devices on your local
> network.*

Click **Allow**. If you missed it (or are running over SSH and never saw it):

1. Open **System Settings → Privacy & Security → Local Network**.
2. Enable the toggle for the app you launch Skulk from — usually **Terminal**
   (or iTerm, or your IDE). Tools launched from that app inherit its grant.
3. Restart Skulk.

Skulk detects this denial at startup and logs a warning telling you to grant
access, so you are never left guessing.

> **The grant follows the launching app.** macOS attributes Local Network
> access to the app a process is launched *from*. Run Skulk from the **Terminal
> you granted** (i.e. `uv run skulk` in that Terminal) and it inherits the
> grant. A process **detached** from that Terminal — `nohup … &`, or some
> background/service launchers that reparent it — is attributed separately and
> may be denied even though the foreground command works. If you run Skulk
> detached or as a background service and see the denial warning, grant Local
> Network to that launcher too (see [Run as a service](./run-skulk-as-a-service)).

> **Headless / SSH-only nodes:** macOS cannot show the Local Network prompt to
> an SSH session, and there is no command-line way to grant the permission. Use
> Screen Sharing (or a directly-attached display) once to enable Local Network
> for Terminal in System Settings; the grant then persists across reboots. Run
> Skulk in a foreground/attached Terminal session (e.g. via `tmux`/`screen` on
> the console) rather than a detached `nohup … &`, so it inherits that grant.
> Alternatively, run the cluster over [Tailscale](./tailscale-clustering.md),
> whose overlay interface is exempt from Local Network Privacy.

## 4. Run (each node)

```bash
uv run skulk
```

That's it. With Local Network access granted, each node's mDNS announcements
reach the others, the cluster forms automatically, and a master is elected. No
`--bootstrap-peers` or `--libp2p-port` flags are needed on a single local
network.

Open the dashboard on any node (`http://localhost:52415`) and confirm every
node appears in the cluster topology.

## Thunderbolt wiring

Skulk does not require any Thunderbolt-specific configuration to *cluster* — the
control plane (peer discovery, coordination) runs over whatever local network
your Macs share. Thunderbolt matters for the **data plane**: when a model is
placed across nodes, Skulk's placement automatically prefers Thunderbolt links
for the high-bandwidth tensor exchange (ranked above Ethernet, Wi-Fi, and any
overlay such as Tailscale).

To use Thunderbolt:

1. Cable the Macs together over Thunderbolt.
2. macOS automatically creates a **Thunderbolt Bridge** network service that
   carries traffic between the directly-connected Macs. The default
   self-assigned (link-local) addressing is sufficient for the inference ring;
   no manual IP assignment is required.
3. Run Skulk as above. Placement uses the Thunderbolt path for inference
   automatically — you can confirm the chosen interface in the placement
   diagnostics.

If your Macs are *only* connected by Thunderbolt (no shared Ethernet/Wi-Fi),
peer discovery happens over the Thunderbolt Bridge segment — the same Local
Network permission applies.

## With or without Tailscale

- **Without Tailscale (this guide):** nodes discover each other over mDNS on the
  local network. Requires the Local Network permission above.
- **With Tailscale:** useful when nodes are on different networks. Tailscale's
  overlay is exempt from Local Network Privacy. See
  [Tailscale clustering](./tailscale-clustering.md). You can run both — Skulk
  still prefers the direct Thunderbolt/Ethernet path for the data plane.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Cluster stays at one node; dashboard shows only the local node | Local Network access denied (most common) | Grant it (section 3), restart Skulk |
| Startup log: *"macOS Local Network access appears to be DENIED"* | Same as above | Grant it (section 3) |
| Local connections log `No route to host` / `EHOSTUNREACH` but internet works | Local Network access denied | Grant it (section 3) |
| Nodes on different subnets don't see each other | mDNS does not cross subnets | Use [Tailscale](./tailscale-clustering.md) or `--bootstrap-peers` |
