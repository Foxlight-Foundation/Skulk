---
id: cluster-communication
title: How the cluster communicates
sidebar_position: 30
---

<!-- Copyright 2025 Foxlight Foundation -->

A Skulk cluster moves three very different kinds of traffic: the raw tensors that
flow between the pieces of a model, the decisions that keep the cluster coherent,
and the generated output on its way back to whoever asked. Skulk carries each on
its own **plane**, so the high-volume traffic never clogs the low-volume traffic
that the cluster's correctness depends on. This separation is what lets Skulk be
a general fabric for multi-node compute rather than a single-purpose server.

## The three planes

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
where each model is placed, the lifecycle of each request, and the health of every
node. It runs over libp2p gossip. This traffic is low-volume but order-sensitive,
because cluster decisions have to be applied the same way everywhere, so it is
kept deliberately separate from the firehose of generated tokens.

### Data plane

The data plane carries generated output (the tokens, and other per-request
chunks) from the node running the model back to the node that received the API
request. It always goes straight to the node that needs it: it never passes
through the master or gets written to the cluster's decision log. (On Zenoh it is
addressed point-to-point to that one node; on the gossip transport it is published
on a shared topic that only the owning node consumes, see below.) Keeping output
off the control plane is what stops a busy model from drowning out the cluster's
own coordination.

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

On Zenoh, each generating node publishes a request's output to a key addressed to
the one node that asked for it, and every node listens only for its own key, so
output is delivered directly instead of broadcast. Zenoh also preserves the order
of a single producer's messages, which matters for the next section. When Zenoh
is configured (a listen endpoint is set), Skulk uses it for the data plane; a node
with no Zenoh configuration falls back to gossip, and `SKULK_ZENOH_DATA_PLANE`
forces the choice either way.

**Every node in a cluster must use the same data-plane transport.** Skulk does not
bridge the two, so a partially configured fleet (Zenoh on some nodes, gossip on
others) silently drops output for any request whose serving node and requesting
node land on opposite transports, and that stream ends only by timeout. Configure
the whole fleet the same way (the simplest rule: either set a Zenoh listen
endpoint on every node, or on none).

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
