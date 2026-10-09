---
id: cluster-communication
title: How the cluster communicates
sidebar_position: 30
---

<!-- Copyright 2025 Foxlight Foundation -->

A Skulk cluster separates raw tensors between the
pieces of a model, durable decisions that keep the cluster coherent, operator
authority, live observations about nodes, and request-scoped payloads on their way to or from a
model. Skulk carries each on its own **plane**, so high-volume or replaceable
traffic has bounded queues separate from the ordered decisions that cluster
correctness depends on. The planes still share host and network resources.
This separation is what lets Skulk be a general fabric for multi-node compute
rather than a single-purpose server.

## Compute and runtime message planes

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

### Authority plane

`AUTHORITY_MESSAGES` carries the signed operator-authority protocol separately
from ordinary cluster commands and events. It has dedicated bounded Python
egress and is not replayed as model or cluster-state history. Device membership,
credential rotation, revocation and gateway authority have their own durable
store and consensus rules. The elected inference master and the operator
authority are different roles; knowing a transport peer does not grant operator
access. See [operator authority](architecture.md#operator-identity-and-authority-foundation).

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
  Tailscale can provide authenticated private reachability across the internet;
  configure both control discovery and data-plane peer endpoints for routed links.

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
`PROVIDER_DATA`, `REALTIME_AUDIO`, `SPEECH_MEDIA`, `VISION_MEDIA`, completed
video `OUTPUT_MEDIA`, and diagnostic `TRACE_DATA`. Zenoh also preserves the order of a single producer's messages,
which matters for the next section. Zenoh is the shipping default, including on
a fresh install. With no transport settings, Skulk binds Zenoh to its preferred
private-LAN or CGNAT fabric IPv4 address (or loopback when offline or
public-only) and uses local multicast scouting to discover other zero-config
nodes. Supplying `SKULK_ZENOH_CONNECT` switches to explicit peer endpoints for
routed or Tailscale deployments; `SKULK_ZENOH_LISTEN` overrides the selected
local listener and is required to bind a public address. Both take `tls/HOST:PORT`
endpoints with an IPv4 address or a DNS name; an IPv6 address literal is refused at
startup, because the data plane's TLS links cannot dial one. Set
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

Zenoh sessions are scoped to and authenticated by the cluster, using the same
shared key that already protects the control plane (derived from the network
version and `SKULK_LIBP2P_NAMESPACE`):

- **Discovery.** Multicast scouting uses a UDP port derived from the key, so two
  clusters with different namespaces on the same network normally never see each
  other's data plane. The port is one of about 7,000, so two namespaces can
  occasionally share it; their scout packets then meet, but the TLS handshake
  refuses the other cluster, so isolation never depends on the port. The port is logged at startup; allow it in host firewalls.
- **Encryption and authentication.** Every data-plane link is TLS 1.3 with
  mutual authentication, chained to a certificate authority each node derives
  from the key. A device without the key cannot join, read, or inject, and an
  eavesdropper sees only ciphertext. Endpoints are written `tls/HOST:PORT`;
  older `tcp/` endpoints are accepted and used as `tls/`.

This is only as strong as the namespace. With the default namespace, or a blank
one, the key is public, exactly as it is for the control plane: traffic is
encrypted, but any Skulk node that can reach the network may join. Set the same private
`SKULK_LIBP2P_NAMESPACE` on every node to restrict membership, and keep the
trusted-fabric model above for anything the cluster's key does not cover.

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
