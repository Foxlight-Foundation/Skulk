"""Translating a recorded placement back into this session's nodes."""

from uuid import UUID

from skulk.recovery.record import PlacementIntent
from skulk.recovery.restore import (
    intent_nodes,
    live_nodes_by_install_id,
    restore_failure_event,
)
from skulk.recovery.tests.fixtures import KITE1, KITE2, NOW
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.profiling import NodeIdentity
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding

KITE3 = UUID("33333333-3333-4333-8333-333333333333")


def _intent(hosts: tuple[UUID, ...], excluded: tuple[UUID, ...] = ()) -> PlacementIntent:
    return PlacementIntent(
        model_id=ModelId("org/model"),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=len(hosts) or 1,
        host_install_ids=hosts,
        excluded_install_ids=excluded,
    )


def test_only_connected_machines_with_an_install_id_are_mapped() -> None:
    view = TelemetryView()
    view.node_identities[NodeId("a")] = NodeIdentity(node_install_id=KITE1)
    view.node_identities[NodeId("b")] = NodeIdentity(node_install_id=KITE2)
    view.node_identities[NodeId("unnamed")] = NodeIdentity()

    live = live_nodes_by_install_id(view, {NodeId("a"), NodeId("unnamed")})

    assert live == {KITE1: NodeId("a")}


def test_returning_machines_map_back_and_absent_ones_drop_out() -> None:
    live = {KITE1: NodeId("a"), KITE3: NodeId("c")}

    preferred, excluded = intent_nodes(_intent((KITE1, KITE2), excluded=(KITE3,)), live)

    assert preferred == frozenset({NodeId("a")})
    assert excluded == frozenset({NodeId("c")})


def test_an_excluded_machine_is_never_preferred() -> None:
    live = {KITE1: NodeId("a")}
    preferred, excluded = intent_nodes(_intent((KITE1,), excluded=(KITE1,)), live)
    assert preferred == frozenset()
    assert excluded == frozenset({NodeId("a")})


def test_failure_is_recorded_where_the_operator_looks() -> None:
    event = restore_failure_event(_intent((KITE1,)), "not enough memory.", now=NOW)

    assert event.failure.error_code == "placement_failed"
    assert event.failure.model_id == ModelId("org/model")
    assert event.failure.error_message == (
        "Could not bring back org/model after the restart: not enough memory."
    )
    assert event.failure.recorded_at == NOW
