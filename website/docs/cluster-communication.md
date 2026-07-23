---
id: cluster-communication
title: How the cluster communicates
sidebar_position: 30
---

<!-- Copyright 2025 Foxlight Foundation -->

A Skulk cluster moves four different kinds of traffic: raw tensors between the
pieces of a model, durable decisions that keep the cluster coherent, live
observations about nodes, and request-scoped payloads on their way to or from a
model. Skulk carries each on its own **plane**, so high-volume or replaceable
traffic never clogs the ordered decisions that cluster correctness depends on.
This separation is what lets Skulk be a general fabric for multi-node compute
rather than a single-purpose server.

## The four planes

### Compute plane

The compute plane is the high-speed interconnect between the parts of a running
model. When a model is sharded across several nodes, each node holds a slice and
hands its intermediate results to the next; that exchange of activations happens
here, every step of generation. It is the most bandwidth- and latency-sensitive
traffic in the cluster, so it rides the fastest local link available
(Thunderbolt or RDMA between directly connected machines).

Speculative decoding also lives on this plane. On a multi-node pipeline, one rank
makes the accept/reject decisions and shares the draft tokens and the outcome
with the others through fixed-shape collective operations, so every rank commits
exactly the same tokens. None of that touches the other planes.

### Control plane

The control plane is how the cluster stays coherent: which node is the master,
where each model is placed, request lifecycle transitions, membership decisions,
and cluster settings. It runs over libp2p gossip. This traffic is low-volume but
order-sensitive because durable decisions have to be applied the same way
everywhere. The master indexes and persists only an explicit allowlist of these
control facts; payload and observational event types are rejected before
ordering, retention, replay, or global broadcast.

### Telemetry plane

The telemetry plane carries replaceable live observations such as heartbeat,
memory, disk, accelerator, download progress, and capability readings. Each
replica keeps only the newest value for a node and reading type in a separate
`TelemetryView`. Telemetry is gossiped last-write-wins, never indexed by the
master, and never written to the event log. Drops and duplicates therefore
affect freshness rather than cluster history.

The plane is isolated from control traffic end to end. On the wire, telemetry
rides its own gossipsub behavior with its own protocol identifier, so it has
separate protocol and handler queues from control and election messages:
control fan-out cannot starve telemetry, and telemetry pressure cannot consume
control or election capacity. On the sending side, admission is a bounded
latest-value map (256 keys, one per node and reading type) where a newer
reading replaces the stale pending one, drained through a one-packet egress
queue: at most one serialized telemetry packet ever waits on the network.

That design is **lossy by design**. When the plane is under pressure, older
pending readings are coalesced or dropped and the next reading supersedes them;
nothing is retried, and a drop costs freshness only. The one telemetry-adjacent
fact that does enter durable cluster state is the terminal outcome of a model
download (completed or failed), because placement and the operator view depend
on it being an ordered decision rather than a freshness-best-effort reading;
download *progress* and every other reading stay on this plane and in
`TelemetryView` only.

### Data plane

The data plane carries request-scoped model output, provider streams, image and
speech input, realtime audio, and completed diagnostic trace payloads. It never
passes through the master or gets written to the cluster's decision log. On
Zenoh, packets are addressed to the owning API or selected worker. On the gossip
fallback, permitted data topics are broadcast over the trusted fabric with a
target tag and receiving components discard packets not addressed to their node
before assembly or persistence; private reference audio and remote realtime
audio require Zenoh and are unavailable on that fallback. Keeping payloads off
the control plane is what stops a busy model or large upload from drowning out
cluster coordination.

## Where the planes run, and the trust model

Skulk assumes a **trusted cluster fabric**. The intended shapes are:

- **Thunderbolt or RDMA** for the compute interconnect between directly connected
  machines (a physical, point-to-point link).
- **A private LAN**, or a **Tailscale** network for nodes in different locations.
  Tailscale is the supported way to run a cluster across the internet: it gives
  every node an encrypted, authenticated link with no extra setup in Skulk.

Running a cluster across a network you do not control is not a supported
configuration. Put remote nodes on Tailscale (or another trusted overlay) rather
than exposing them directly. See [multi-network clustering](tailscale-clustering)
for the remote setup.

## The data plane in detail

The data plane can run over either of two transports:

- **libp2p gossip** (the same stack as the control plane), or
- **Eclipse Zenoh**, a transport built specifically for streaming data.

On Zenoh, each producer publishes to a key addressed to the API or worker that
owns the stream, and every node listens only for its own key, so packets are
delivered directly instead of broadcast. This includes generated `DATA`, generic
`PROVIDER_DATA`, `REALTIME_AUDIO`, `SPEECH_MEDIA`, `VISION_MEDIA`, and diagnostic
`TRACE_DATA`. Zenoh also preserves the order of a single producer's messages,
which matters for the next section. Zenoh is the shipping default, including on
a fresh install. With no transport settings, Skulk binds Zenoh to its preferred
private-LAN or CGNAT fabric IPv4 address (or loopback when offline or
public-only) and uses local multicast scouting to discover other zero-config
nodes. Supplying `SKULK_ZENOH_CONNECT` switches to explicit peer endpoints for
routed or Tailscale deployments; `SKULK_ZENOH_LISTEN` overrides the selected
local listener and is required to bind a public address. Set
`SKULK_ZENOH_DATA_PLANE=0` only to force the legacy gossip fallback.

**Every node in a cluster must use the same data-plane transport.** Skulk does not
bridge the two, so a partially configured fleet (Zenoh on some nodes, gossip on
others) cannot deliver output for a request whose serving node and requesting
node land on opposite transports. Each node advertises its resolved transport in
`nodeResources`; `/state` marks every live node with the error-level
`data_transport_mismatch` health reason when both transports are present, and the
dashboard and node diagnostics show the same condition. This detection does not
bridge the transports or make mixed operation safe. Configure any legacy
gossipsub override consistently across the whole fleet, restart it, and confirm
that `nodeResources` reports one transport.

Zenoh sessions are kept isolated per cluster: each cluster prefixes its keys with
a segment derived from its libp2p network namespace (`SKULK_LIBP2P_NAMESPACE`), so
two separate Skulk clusters on the same network do not receive each other's
output. That isolation is a partition between clusters, not a secret:
confidentiality on an untrusted network is the job of the fabric (Tailscale, or a
firewall), which is why the trusted-fabric model above matters.

## How speculative decoding rides the planes

Speculative decoding and the data plane stay out of each other's way. All of
speculation (drafting candidate tokens, verifying them in one forward pass,
deciding what to keep, and the cross-rank agreement that keeps multi-node clusters
in lockstep) happens on the **compute** plane, inside the running model. The data
plane only ever sees the **committed** tokens that come out the far end.

The one visible interaction is timing. Speculative decoding commits tokens in
bursts: a good round accepts several tokens at once, so the model emits a little
flurry of output and then pauses to verify the next round, rather than a steady
one-token drip. The data plane carries those bursts, and the client sees a clean,
correctly ordered stream regardless of how bursty the underlying generation was.
On Zenoh that ordering comes from the transport itself (a single producer's
messages arrive in order); on the gossip transport, which can reorder, each chunk
carries a sequence number and a small reorder buffer on the receiving node puts
them back in order. Either way the committed tokens reach the client in the order
they were produced. (See [speculative decoding](speculative-decoding) for how the
decode loop itself works.)
