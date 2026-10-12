"""Choosing the freshest record across nodes and fetching it."""

from datetime import datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from skulk.recovery.fetch import RECORD_PATH, candidate_addresses, fetch_peer_record
from skulk.recovery.record import (
    ClusterIntentRecord,
    PlacementIntent,
    RestartRecoveryRecordResponse,
    derive_cluster_intent,
)
from skulk.recovery.restore import awaited_peers, record_sources, unrestored_intents
from skulk.recovery.tests.fixtures import (
    INSTALL_IDS,
    KITE1,
    KITE2,
    NOW,
    SESSION,
    sample_instance,
    sample_state,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from skulk.shared.types.restart_recovery import RestartRecoveryAdvert
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding

LOCAL = NodeId("local")
PEER = NodeId("peer")
OTHER = NodeId("other")


def _record(minutes_ago: int) -> ClusterIntentRecord:
    return derive_cluster_intent(
        sample_state(),
        install_ids=INSTALL_IDS,
        session_id=SESSION,
        written_at=NOW - timedelta(minutes=minutes_ago),
    )


def _advert(minutes_ago: int | None, port: int | None = 52415) -> RestartRecoveryAdvert:
    return RestartRecoveryAdvert(
        written_at=None if minutes_ago is None else NOW - timedelta(minutes=minutes_ago),
        placements=2,
        api_port=port,
    )


def test_sources_are_freshest_first_and_skip_what_cannot_serve() -> None:
    sources = record_sources(
        _record(30),
        {
            PEER: _advert(5),
            OTHER: _advert(1, port=None),  # no API: cannot serve its record
            NodeId("gone"): _advert(0),  # not connected
            NodeId("empty"): _advert(None),  # has no record
            LOCAL: _advert(0),  # this node's own advert, looped back
        },
        local_node=LOCAL,
        connected_node_ids={LOCAL, PEER, OTHER, NodeId("empty")},
        unfetchable=set(),
    )
    assert [source.node_id for source in sources] == [PEER, None]


def test_own_record_wins_a_tie_and_unfetchable_peers_are_skipped() -> None:
    own = _record(5)
    tie = record_sources(
        own, {PEER: _advert(5)}, local_node=LOCAL, connected_node_ids={LOCAL, PEER}, unfetchable=set()
    )
    assert [source.node_id for source in tie] == [None, PEER]
    skipped = record_sources(
        own, {PEER: _advert(1)}, local_node=LOCAL, connected_node_ids={LOCAL, PEER}, unfetchable={PEER}
    )
    assert [source.node_id for source in skipped] == [None]


def _intent(model: str, width: int = 1) -> PlacementIntent:
    return PlacementIntent(
        model_id=ModelId(model),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=width,
        host_install_ids=(KITE1,),
    )


def test_placements_already_running_are_not_requested_again() -> None:
    running = sample_instance("carried", "org/a", ("node-a",))
    steward = sample_instance("steward", "org/a", ("node-a",), system_role="steward")
    remaining = unrestored_intents(
        [_intent("org/a"), _intent("org/a"), _intent("org/b"), _intent("org/a", width=2)],
        {running.instance_id: running, steward.instance_id: steward},
    )
    # One running ordinary org/a accounts for one intent; the steward counts
    # for nothing, and a different width is a different placement.
    assert [(str(intent.model_id), intent.min_nodes) for intent in remaining] == [
        ("org/a", 1),
        ("org/b", 1),
        ("org/a", 2),
    ]


def test_adverts_land_in_the_telemetry_view_and_leave_with_the_node() -> None:
    view = TelemetryView()
    view.apply(NodeTelemetry(node_id=PEER, info=_advert(3)))
    assert view.node_restart_recovery[PEER] == _advert(3)
    view.prune(PEER)
    assert PEER not in view.node_restart_recovery


def _network(*addresses: str) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[NetworkInterfaceInfo(name=f"en{i}", ip_address=a) for i, a in enumerate(addresses)]
    )


async def test_fetch_confirms_which_node_answered() -> None:
    record = _record(2)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert request.url.path == RECORD_PATH
        if request.url.host == "10.0.0.1":
            # Another node on a shared address must not supply the record.
            body = RestartRecoveryRecordResponse(node_id=OTHER, record=_record(0))
        elif request.url.host == "10.0.0.2":
            return httpx.Response(500)
        else:
            body = RestartRecoveryRecordResponse(node_id=PEER, record=record)
        return httpx.Response(200, content=body.model_dump_json(by_alias=True))

    fetched = await fetch_peer_record(
        PEER,
        _advert(2, port=52925),
        _network("10.0.0.1", "10.0.0.2", "10.0.0.3"),
        transport=httpx.MockTransport(handler),
    )

    assert fetched == record
    assert seen == [
        "http://10.0.0.1:52925" + RECORD_PATH,
        "http://10.0.0.2:52925" + RECORD_PATH,
        "http://10.0.0.3:52925" + RECORD_PATH,
    ]


async def test_a_node_without_an_interface_list_is_fetched_at_its_own_addresses() -> None:
    """A node without a worker publishes no interface list; its advert names them."""
    record = _record(2)
    advert = _advert(2, port=52925).model_copy(update={"addresses": ("10.0.0.7",)})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "10.0.0.7"
        body = RestartRecoveryRecordResponse(node_id=PEER, record=record)
        return httpx.Response(200, content=body.model_dump_json(by_alias=True))

    assert await fetch_peer_record(PEER, advert, None, transport=httpx.MockTransport(handler)) == record


def test_candidate_addresses_try_each_address_once_and_skip_link_local_ipv6() -> None:
    advert = _advert(2).model_copy(update={"addresses": ("10.0.0.1", "10.0.0.9")})
    network = _network("10.0.0.1", "fe80::1", "169.254.3.4", "fd00::5")

    assert candidate_addresses(network, advert) == ["10.0.0.1", "169.254.3.4", "fd00::5", "10.0.0.9"]


def test_advert_addresses_are_bounded_ip_literals() -> None:
    with pytest.raises(ValidationError):
        _advert(2).model_validate({**_advert(2).model_dump(), "addresses": ("not-an-ip",)})
    with pytest.raises(ValidationError):
        _advert(2).model_validate(
            {**_advert(2).model_dump(), "addresses": tuple(f"10.0.0.{i}" for i in range(17))}
        )


async def test_fetch_needs_an_api_port_and_an_address() -> None:
    assert await fetch_peer_record(PEER, _advert(2, port=None), _network("10.0.0.1")) is None
    assert await fetch_peer_record(PEER, _advert(2), None) is None



def test_a_connected_peer_is_awaited_until_its_advert_arrives() -> None:
    observed = {PEER: KITE2}
    assert awaited_peers(observed, {}, own=_record(5), local_node=LOCAL, connected_node_ids={LOCAL, PEER}, unfetchable=set(), live_install_ids={}) == {PEER}
    assert awaited_peers(
        observed, {PEER: _advert(1)}, own=_record(5), local_node=LOCAL, connected_node_ids={LOCAL, PEER}, unfetchable=set(), live_install_ids={}
    ) == set()


def test_a_departed_peer_is_awaited_only_while_it_may_hold_a_fresher_record() -> None:
    observed = {PEER: KITE2}
    connected = {LOCAL}
    # Never advertised: what it holds is unknown.
    assert awaited_peers(observed, {}, own=_record(5), local_node=LOCAL, connected_node_ids=connected, unfetchable=set(), live_install_ids={}) == {PEER}
    # Fresher than everything reachable now.
    assert awaited_peers(
        observed, {PEER: _advert(1)}, own=_record(5), local_node=LOCAL, connected_node_ids=connected, unfetchable=set(), live_install_ids={}
    ) == {PEER}
    # No fresher than this node's own record, or holding none at all.
    assert awaited_peers(
        observed, {PEER: _advert(9)}, own=_record(5), local_node=LOCAL, connected_node_ids=connected, unfetchable=set(), live_install_ids={}
    ) == set()
    assert awaited_peers(
        observed, {PEER: _advert(None)}, own=_record(5), local_node=LOCAL, connected_node_ids=connected, unfetchable=set(), live_install_ids={}
    ) == set()


def test_a_peer_back_under_a_new_node_id_is_judged_by_its_new_advert() -> None:
    returned = NodeId("peer-after-restart")
    observed = {PEER: KITE2, returned: KITE2}
    connected = {LOCAL, returned}
    assert awaited_peers(
        observed, {}, own=_record(5), local_node=LOCAL, connected_node_ids=connected, unfetchable=set(), live_install_ids={KITE2: returned}
    ) == {returned}
    assert awaited_peers(
        observed,
        {returned: _advert(1)},
        own=_record(5),
        local_node=LOCAL,
        connected_node_ids=connected,
        unfetchable=set(),
        live_install_ids={KITE2: returned},
    ) == set()


def test_adverts_must_carry_a_utc_offset() -> None:
    """A naive time cannot be ordered against record stamps."""
    with pytest.raises(ValidationError):
        RestartRecoveryAdvert(written_at=datetime(2026, 10, 11, 12, 0), placements=1, api_port=52415)
    with pytest.raises(ValidationError):
        RestartRecoveryAdvert.model_validate_json(
            '{"RestartRecoveryAdvert": {"written_at": "2026-10-11T12:00:00", "placements": 1, "api_port": 52415}}'
        )
    assert RestartRecoveryAdvert.model_validate_json(
        '{"RestartRecoveryAdvert": {"written_at": "2026-10-11T12:00:00Z", "placements": 1, "api_port": 52415}}'
    ).written_at is not None


def test_only_usable_sources_set_the_bar_a_departed_peer_must_beat() -> None:
    """A connected peer advertising a newer record it cannot serve is no bar."""
    observed = {PEER: KITE2, OTHER: None}
    adverts = {PEER: _advert(3), OTHER: _advert(1, port=None)}
    connected = {LOCAL, OTHER}

    def awaited(unfetchable: set[NodeId]) -> set[NodeId]:
        return awaited_peers(
            observed,
            adverts,
            own=_record(5),
            local_node=LOCAL,
            connected_node_ids=connected,
            unfetchable=unfetchable,
            live_install_ids={},
        )

    # OTHER runs no API, so the bar is this node's own five-minute-old record.
    assert awaited(set()) == {PEER}
    # A connected peer that failed to serve its record does not raise it either.
    adverts[OTHER] = _advert(1)
    assert awaited(set()) == set()
    assert awaited({OTHER}) == {PEER}
