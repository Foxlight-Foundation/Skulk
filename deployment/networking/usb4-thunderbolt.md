# USB4 / Thunderbolt networking between Linux nodes

USB4 (Thunderbolt) host-to-host networking gives a Skulk pair a direct
high-bandwidth link that placement automatically prefers over ethernet for
multi-node (RPC) tensor traffic. This guide is the bring-up and tuning recipe
for a Linux pair (validated on two AMD Strix Halo nodes, kernel 7.x), plus how
to verify Skulk is actually using the link.

## What the link buys (measured, Strix Halo pair, 2026-07)

| Metric | 2.5GbE LAN | USB4 (20G link) |
|---|---:|---:|
| TCP throughput (iperf3) | 2.36 Gbit/s | 9.4 Gbit/s |
| Ping RTT (avg) | 0.99 ms | 0.72 ms |
| Pooled 63GB model load (driver launch to ready) | 96 s | 81 s |
| Pooled decode tok/s | 44.3 | 44.0 |

Bandwidth-bound phases (model load, donor tensor push, long-prompt prefill)
benefit. Per-token decode does not: its multi-node overhead is dominated by
the llama.cpp RPC protocol, not the wire, and is the same on both transports.

## Requirements

- A USB4/Thunderbolt cable rated for the speed you want. **Cable
  certification decides the negotiated rate**: 40 Gbit/s needs a
  USB4-40Gbps-certified cable (passive, 0.8 m or shorter is the safe
  choice); a 20 Gbps-rated cable silently negotiates the link down to
  20 Gbit/s even between 40G-capable ports. Check the negotiated rate after
  connecting (below).
- `thunderbolt` and `thunderbolt_net` kernel modules (present in stock
  Ubuntu kernels). The `thunderbolt0` interface appears when the two hosts
  establish their XDomain connection; no pairing/authorization step is
  needed in the default `user` security mode for host-to-host links.

## Bring-up

### 1. Static subnet (required, not cosmetic)

Give the link a dedicated static subnet. Do NOT rely on the 169.254/16
link-local addresses the interface gets by default: Skulk's donor-endpoint
selection rejects link-local addresses by contract, because a host with two
Thunderbolt ports routes all of 169.254/16 out whichever port has the lower
metric, so TCP breaks asymmetrically while ping appears to work (observed on
the reference pair, which has a second TB port cabled to another machine).

With NetworkManager (Ubuntu desktop/server default), on the first node:

```bash
nmcli connection modify "Wired connection 1" \
  ipv4.method manual ipv4.addresses 10.99.0.1/30 \
  802-3-ethernet.mtu 65520
```

and on the second node (adjust the profile name to whichever one owns
`thunderbolt0`; `nmcli -t -f NAME,DEVICE connection show` lists them):

```bash
nmcli connection modify "Wired connection 1" \
  ipv4.method manual ipv4.addresses 10.99.0.2/30 \
  802-3-ethernet.mtu 65520
```

MTU 65520 is the thunderbolt-net maximum (it is a virtual interface, not
ethernet; jumbo-frame conventions like 9000 leave packet-count savings on
the table for bulk transfers).

`nmcli connection modify` edits the stored profile only; an already-active
`thunderbolt0` keeps its old address and MTU until the profile is
re-applied. Reactivate it on both nodes before verifying:

```bash
nmcli connection up "Wired connection 1"
```

(`nmcli device reapply thunderbolt0` also works for address changes, but an
MTU change needs the full reactivation on some NetworkManager versions.)

### 2. Queueing discipline

`fq_codel` measurably reduces TCP retransmits on thunderbolt-net links.
Persist it as the default qdisc:

```bash
echo "net.core.default_qdisc = fq_codel" | sudo tee /etc/sysctl.d/90-skulk-tb.conf
sudo sysctl --system
sudo tc qdisc replace dev thunderbolt0 root fq_codel
```

### 3. Verify the link

```bash
# Negotiated rate (per direction, per lane; 20.0 Gb/s x2 = a 20G cable):
cat /sys/bus/thunderbolt/devices/*/tx_speed /sys/bus/thunderbolt/devices/*/tx_lanes
# Reachability + latency:
ping -c 5 10.99.0.2
# Throughput (iperf3 -s on the peer):
iperf3 -c 10.99.0.2 -t 5
```

If the peer's `thunderbolt0` never appears, check `dmesg | grep -i
thunderbolt` for retimer connect/disconnect churn (reseat the cable or use
the other port) and confirm the XDomain peer shows up under
`/sys/bus/thunderbolt/devices/` (a `<domain>-<port>` entry named after the
peer host).

## How Skulk uses the link

No Skulk configuration is needed. Placement selects multi-node addresses
from the observed peer connections, ranked by the gossiped interface type
with Thunderbolt first and VPN/overlay last; RPC donor endpoints
additionally require a routable (non-link-local) address, which is what the
static subnet provides. When both the LAN and the TB path are observed, the
donor endpoint is stamped on the TB address (regression-tested in
`src/skulk/master/tests/test_multinode_gguf_placement.py`).

Verify on a live pooled instance:

```bash
curl -s http://localhost:52415/state | jq \
  '.instances[].LlamaRpcInstance.donorEndpoints'
# expect the TB subnet address, e.g. {"<donor-node-id>": "10.99.0.2:28890"}
```

One ordering caveat: candidates are the OBSERVED peer connections, and the
workers' connection probes discover a newly-up TB path within a probe cycle
(seconds). An instance placed while the link was down keeps its LAN endpoint
for its lifetime; delete and re-place it after the link is up if you want
the tensor path moved. Control-plane traffic (gossip, telemetry, Zenoh data
plane endpoints) stays on the LAN by configuration, which keeps the TB lane
clear for tensor traffic.

## Do NOT pin CPU C-states on APU nodes

Holding `/dev/cpu_dma_latency` at 0 (or capping C-states) is a standard
small-packet latency tuning trick, and it does cut this link's RTT by an
order of magnitude. But on a unified-memory APU (Strix Halo) the CPU and
GPU share one package power budget, and cores held in C0 steal enough of it
to throttle the GPU: measured ~20% decode throughput loss on the reference
pair. Inference decode is not RTT-bound, so this trade is all cost and no
benefit on APU inference nodes.
