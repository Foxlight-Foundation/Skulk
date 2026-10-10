"""Two hosts' copies of one plugin hand each other setup codes over the fabric.

A capability node may address a short public handshake to a peer host. It
rides the node's ordinary summary on the telemetry plane, so it is exactly as
authenticated as the fabric: hosts publish and deliver handshakes only on a
private namespace. An owner sees peers running its own bundles and only the
handshakes addressed to its host.
"""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from skulk.extensions.managed import (
    MAX_HANDSHAKE_PEERS,
    ManagedConnection,
    ManagedNode,
    ManagedOwner,
    summaries_for,
)
from skulk.extensions.managed_attachment import ManagedAttachment
from skulk.extensions.telemetry import CapabilityPeerView, snapshot_capability_peers
from skulk.extensions.tests.test_managed_topology import PROFILE
from skulk.extensions.tests.test_steward_tools import context
from skulk.shared.topology import Topology
from skulk.shared.types.capability_nodes import (
    CapabilityNodeHandshake,
    CapabilityNodeSummary,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import NodeIdentity
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.topology import Connection, SocketConnection

SELF = NodeId("test-node")
PEER = NodeId("peer-node")


def summary(
    bundle: str = "foxlight.runpod",
    *handshakes: CapabilityNodeHandshake,
    node: str = "node-1",
) -> CapabilityNodeSummary:
    return CapabilityNodeSummary(
        plugin_id="managed.fixture",
        node_id=node,
        bundle_id=bundle,
        version="0.4.0",
        status="disabled",
        owner_available=True,
        handshakes=handshakes,
    )


def handshake(recipient: str = SELF, content: str = '{"code":1}') -> CapabilityNodeHandshake:
    return CapabilityNodeHandshake(
        recipient=recipient, kind="runpod-cleanup-request", content=content
    )


def test_a_summary_without_handshakes_keeps_its_earlier_shape() -> None:
    plain = summary()
    assert "handshakes" not in json.loads(plain.model_dump_json(by_alias=True))
    carrying = summary("foxlight.runpod", handshake())
    wire = carrying.model_dump_json(by_alias=True)
    assert json.loads(wire)["handshakes"][0]["recipient"] == SELF
    assert CapabilityNodeSummary.model_validate_json(wire) == carrying


@pytest.mark.parametrize(
    "values",
    [
        {"recipient": "bad id", "kind": "k", "content": "x"},
        {"recipient": "peer", "kind": "Kind", "content": "x"},
        {"recipient": "peer", "kind": "k", "content": ""},
        {"recipient": "peer", "kind": "k", "content": "x" * 2049},
        {"recipient": "peer", "kind": "k", "content": "line\nbreak"},
    ],
)
def test_handshakes_are_bounded_printable_text(values: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        CapabilityNodeHandshake.model_validate(values)


def test_a_node_addresses_at_most_two_handshakes() -> None:
    with pytest.raises(ValidationError):
        summary("foxlight.runpod", handshake(), handshake(), handshake())


def _edge(source: NodeId, sink: NodeId, ip: str, *, session: bool = False) -> Connection:
    return Connection(
        source=source,
        sink=sink,
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"), session=session
        ),
    )


def test_peers_are_fresh_hosts_other_than_this_one_with_probed_addresses() -> None:
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    view = TelemetryView()
    stale = NodeId("stale-node")
    for node_id, age in ((SELF, 0), (PEER, 5), (stale, 120)):
        view.node_capability_nodes[node_id] = (summary(),)
        view.node_capability_nodes_received_at[node_id] = now - timedelta(seconds=age)
    view.node_identities[PEER] = NodeIdentity(friendly_name="kite2")
    topology = Topology()
    for node_id in (SELF, PEER):
        topology.add_node(node_id)
    topology.add_connection(_edge(SELF, PEER, "192.168.1.20"))
    topology.add_connection(_edge(SELF, PEER, "100.64.0.20"))
    # A session edge's address is the connection's observed remote end.
    topology.add_connection(_edge(SELF, PEER, "203.0.113.9", session=True))
    (peer,) = snapshot_capability_peers(view, topology, SELF, now)
    assert peer == CapabilityPeerView(
        node_id=PEER,
        friendly_name="kite2",
        addresses=("100.64.0.20", "192.168.1.20"),
        capability_nodes=(summary(),),
    )


def owner(tmp_path: Path, *, private: bool, peers: tuple[CapabilityPeerView, ...]) -> ManagedOwner:
    managed = ManagedOwner(
        ManagedConnection(plugin_id="managed.fixture", state_root=str(tmp_path)),
        attachment=ManagedAttachment(tmp_path, PROFILE),
    )
    managed.attachment = None
    managed.context = replace(
        context(),
        read_capability_peers=lambda: peers,
        private_fabric=lambda: private,
    )
    managed.nodes = (
        ManagedNode(
            node_id="node-1",
            bundle_id="foxlight.runpod",
            version="0.4.0",
            status="disabled",
            configurable=True,
            descriptors=(),
        ),
    )
    managed.available = True
    return managed


def peer(*summaries: CapabilityNodeSummary, node_id: str = PEER) -> CapabilityPeerView:
    return CapabilityPeerView(
        node_id=NodeId(node_id),
        friendly_name="kite2",
        addresses=("100.64.0.20",),
        capability_nodes=summaries,
    )


def test_the_owner_sees_its_bundles_peers_and_only_handshakes_for_this_host(
    tmp_path: Path,
) -> None:
    managed = owner(
        tmp_path,
        private=True,
        peers=(
            peer(
                summary("foxlight.runpod", handshake(), handshake("other-host")),
                summary("foxlight.video-studio", handshake(), node="node-2"),
            ),
            peer(summary("foxlight.video-studio"), node_id="unrelated-node"),
        ),
    )
    request = managed._handshake_request()  # pyright: ignore[reportPrivateUsage]
    assert request == {
        "operation": "handshakes",
        "private_fabric": True,
        "peers": [
            {
                "node_id": PEER,
                "friendly_name": "kite2",
                "addresses": ["100.64.0.20"],
                "nodes": [
                    {
                        "bundle_id": "foxlight.runpod",
                        "node_id": "node-1",
                        "status": "disabled",
                        "handshakes": [
                            {"kind": "runpod-cleanup-request", "content": '{"code":1}'}
                        ],
                    }
                ],
            }
        ],
    }


def test_peers_with_a_handshake_for_this_host_come_first(tmp_path: Path) -> None:
    quiet = tuple(peer(summary(), node_id=f"quiet-{index}") for index in range(10))
    managed = owner(tmp_path, private=True, peers=(*quiet, peer(summary("foxlight.runpod", handshake()))))
    peers = managed._handshake_request()["peers"]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(peers, list) and len(peers) == MAX_HANDSHAKE_PEERS
    first = peers[0]
    assert isinstance(first, dict) and first["node_id"] == PEER


def test_a_public_fabric_neither_delivers_nor_publishes_handshakes(
    tmp_path: Path,
) -> None:
    managed = owner(
        tmp_path, private=False, peers=(peer(summary("foxlight.runpod", handshake())),)
    )
    request = managed._handshake_request()  # pyright: ignore[reportPrivateUsage]
    assert request["private_fabric"] is False and request["peers"] == []
    published: list[CapabilityNodeSummary] = []
    assert managed.context is not None
    managed.context = replace(managed.context, publish_capability_node=published.append)
    managed.handshakes = {"node-1": (handshake(PEER),)}
    managed._publish_summaries()  # pyright: ignore[reportPrivateUsage]
    assert published[-1].handshakes == ()


async def test_the_owners_answer_rides_its_summaries(tmp_path: Path) -> None:
    managed = owner(tmp_path, private=True, peers=())
    published: list[CapabilityNodeSummary] = []
    assert managed.context is not None
    managed.context = replace(managed.context, publish_capability_node=published.append)
    answer: dict[str, object] = {
        "nodes": [
            {
                "node_id": "node-1",
                "handshakes": [
                    {"recipient": PEER, "kind": "runpod-cleanup-ready", "content": "{}"}
                ],
            },
            # A node the owner does not run is ignored, not trusted.
            {"node_id": "elsewhere", "handshakes": [{"recipient": PEER, "kind": "k", "content": "x"}]},
        ]
    }

    async def request(message: dict[str, object], *, timeout: float) -> dict[str, object]:
        assert message["operation"] == "handshakes" and timeout == 1
        return answer

    managed._request = request  # type: ignore[method-assign]
    await managed._exchange_handshakes()  # pyright: ignore[reportPrivateUsage]
    assert set(managed.handshakes) == {"node-1"}
    managed._publish_summaries()  # pyright: ignore[reportPrivateUsage]
    assert [item.kind for item in published[-1].handshakes] == ["runpod-cleanup-ready"]
    (carried,) = summaries_for(
        "managed.fixture", managed.nodes, True, managed.handshakes
    )
    assert carried.handshakes == published[-1].handshakes


@pytest.mark.parametrize("failure", ["refused", "malformed"])
async def test_an_owner_without_handshakes_is_asked_once_per_process(
    tmp_path: Path, failure: str
) -> None:
    """An older owner records each refusal, so it is not asked every refresh.

    Handshakes are a setup convenience, never a reason to drop an owner.
    """
    managed = owner(tmp_path, private=True, peers=())
    managed.handshakes = {"node-1": (handshake(PEER),)}
    process = [(1, 100)]
    asked: list[str] = []

    async def request(message: dict[str, object], *, timeout: float) -> dict[str, object]:
        asked.append(str(message["operation"]))
        if failure == "refused":
            raise ValueError("managed operation refused")
        return {"nodes": "not a list"}

    managed._request = request  # type: ignore[method-assign]
    managed._control_identity = lambda: process[0]  # type: ignore[method-assign]
    await managed._exchange_handshakes()  # pyright: ignore[reportPrivateUsage]
    assert managed.handshakes == {} and managed.available
    await managed._exchange_handshakes()  # pyright: ignore[reportPrivateUsage]
    assert asked == ["handshakes"]
    # A restarted owner, possibly upgraded, is asked again.
    process[0] = (2, 200)
    await managed._exchange_handshakes()  # pyright: ignore[reportPrivateUsage]
    assert asked == ["handshakes", "handshakes"]


async def test_a_transient_failure_is_retried(tmp_path: Path) -> None:
    managed = owner(tmp_path, private=True, peers=())
    asked: list[str] = []

    async def request(message: dict[str, object], *, timeout: float) -> dict[str, object]:
        asked.append(str(message["operation"]))
        raise TimeoutError

    managed._request = request  # type: ignore[method-assign]
    managed._control_identity = lambda: (1, 100)  # type: ignore[method-assign]
    await managed._exchange_handshakes()  # pyright: ignore[reportPrivateUsage]
    await managed._exchange_handshakes()  # pyright: ignore[reportPrivateUsage]
    assert asked == ["handshakes", "handshakes"] and managed.handshakes_refused is None
