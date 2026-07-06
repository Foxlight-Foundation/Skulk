---
id: mobile-dashboard
title: Manage Your Cluster from a Phone
sidebar_label: Mobile dashboard
sidebar_position: 7
description: The Skulk dashboard works from a phone browser, so you can monitor topology, chat with models, and watch runner internals from the couch.
---

<!-- Copyright 2025 Foxlight Foundation -->

# Manage Your Cluster from a Phone

The Skulk dashboard is fully responsive. At phone widths every view adapts to
a single-column layout: the header collapses into a menu, side panels become
slide-in drawers, and observability takes over the full screen. Nothing is
cut down; topology, the model store, chat, placement, and live runner
internals all work from a phone browser.

## Opening the dashboard

Every Skulk node serves the dashboard. From a phone on the same network as
any node:

```
http://<node-address>:52415
```

Any node works, not just the master. The dashboard talks to the whole
cluster through whichever node serves it.

### From outside the local network

If your nodes are on a [Tailscale tailnet](./tailscale.md), the dashboard is
reachable from anywhere your phone has connectivity. Install the Tailscale
app on the phone, sign in to the same tailnet, and open the node's Tailscale
address (or MagicDNS name):

```
http://<node-name>.<tailnet>.ts.net:52415
```

This is the recommended remote shape: no ports exposed to the internet, and
the phone joins the same private overlay the nodes already use. The
observability Node tab shows each node's own Tailscale state, which helps
confirm the overlay is up when you are away from the cluster.

### Add it to your home screen

Both iOS Safari (Share, then "Add to Home Screen") and Android Chrome (menu,
then "Add to Home screen") will pin the dashboard as an icon that opens
full-screen, which makes it feel like an app rather than a browser tab.

## The mobile layout

<div style={{display: 'flex', gap: '12px', flexWrap: 'wrap'}}>
  <img src={require('./imgs/mobile-topology.png').default} alt="Cluster topology on a phone" width="260" />
  <img src={require('./imgs/mobile-menu.png').default} alt="The mobile navigation menu" width="260" />
</div>

The hamburger button opens a menu carrying everything the desktop header
shows inline: the Cluster, Model Store, and Chat views, plus Observability,
Settings, and the theme toggle. The cluster view renders the same live
topology as desktop, with node cards sized for the smaller canvas.

<div style={{display: 'flex', gap: '12px', flexWrap: 'wrap'}}>
  <img src={require('./imgs/mobile-chat.png').default} alt="Chat with a placed model on a phone" width="260" />
  <img src={require('./imgs/mobile-observability.png').default} alt="Observability sheet showing live runner phases" width="260" />
</div>

Chat works against any placed model, streaming tokens with the same
time-to-first-token and throughput stats as desktop. The conversation
history and active-instances panels open as drawers over the content and
close by tapping the dimmed area behind them.

Observability opens as a full-screen sheet with the same three tabs as
desktop: **Live** (runner phases and the cross-rank timeline), **Node**
(per-node hardware, memory, and connectivity details, including Tailscale
state), and **Traces** (saved generation traces). Watching a placement load
layer by layer from a phone is a good way to keep an eye on a long model
load without sitting at a desk.

## Notes

- The dashboard is served by nodes that have the built web assets. A
  headless node without them still serves the API; point the phone at a
  node that has the dashboard built.
- All dashboard traffic stays inside your network (or tailnet). Nothing
  about the mobile layout changes what is exposed.
