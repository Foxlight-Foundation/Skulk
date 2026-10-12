"""The restart recovery record: derivation, round trip, and restored proposals."""

import json
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from skulk.recovery.record import (
    DEPARTURE_GRACE,
    RESTARTED_BEFORE_DECISION,
    RESTARTED_BEFORE_FINISH,
    ClusterIntentRecord,
    PlacementIntent,
    RecoveryStamp,
    carry_departed_placements,
    derive_cluster_intent,
    read_record,
    restored_steward_proposals,
    write_record,
)
from skulk.recovery.tests.fixtures import (
    INSTALL_IDS,
    KITE1,
    KITE2,
    NOW,
    SESSION,
    sample_instance,
    sample_proposal,
    sample_state,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding


def test_record_keeps_ordinary_placement_intent_by_install_id() -> None:
    record = derive_cluster_intent(
        sample_state(), install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
    )

    assert [placement.model_id for placement in record.placements] == ["org/big", "org/small"]
    big, small = record.placements
    assert big.host_install_ids == (KITE1, KITE2)
    assert big.min_nodes == 2
    assert big.sharding == Sharding.Pipeline
    assert big.instance_meta == InstanceMeta.MlxRing
    assert big.requested_context_tokens == 32768
    assert small.host_install_ids == (KITE2,)
    assert small.excluded_install_ids == (KITE1,)
    assert record.stamp == RecoveryStamp(
        session_master_node_id=NodeId("node-a"),
        session_election_clock=3,
        last_event_applied_idx=42,
        written_at=NOW,
    )
    assert [proposal.status for proposal in record.steward_proposals] == ["rejected", "pending"]


def test_unknown_install_ids_are_left_out_rather_than_guessed() -> None:
    record = derive_cluster_intent(
        sample_state(),
        install_ids={NodeId("node-b"): KITE2},
        session_id=SESSION,
        written_at=NOW,
    )
    assert record.placements[0].host_install_ids == (KITE2,)
    assert record.placements[1].excluded_install_ids == ()


def test_a_placement_without_nodes_is_left_out_instead_of_failing() -> None:
    """One malformed instance must not stop the record from being kept."""
    state = sample_state()
    husk = sample_instance("husk", "org/husk", ("node-a",))
    husk = husk.model_copy(
        update={
            "shard_assignments": husk.shard_assignments.model_copy(
                update={"node_to_runner": {}, "runner_to_shard": {}}
            )
        }
    )
    state = state.model_copy(update={"instances": {**state.instances, husk.instance_id: husk}})

    record = derive_cluster_intent(
        state, install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
    )

    assert [placement.model_id for placement in record.placements] == ["org/big", "org/small"]


def test_record_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "recovery" / "cluster-intent.json"
    record = derive_cluster_intent(
        sample_state(), install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
    )

    write_record(path, record)

    assert read_record(path) == record
    assert path.stat().st_mode & 0o777 == 0o600
    assert not any(child.name.endswith(".tmp") for child in path.parent.iterdir())


def test_a_leftover_staging_file_never_exposes_the_record(tmp_path: Path) -> None:
    """A world-readable temp file from an earlier crash is narrowed before use."""
    path = tmp_path / "cluster-intent.json"
    leftover = tmp_path / ".cluster-intent.json.tmp"
    leftover.write_text("stale")
    leftover.chmod(0o644)

    write_record(
        path,
        derive_cluster_intent(
            sample_state(), install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
        ),
    )

    assert path.stat().st_mode & 0o777 == 0o600
    assert not leftover.exists()
    assert read_record(path) is not None


def test_missing_or_corrupt_record_reads_as_none(tmp_path: Path) -> None:
    path = tmp_path / "cluster-intent.json"
    assert read_record(path) is None
    path.write_text("{not json")
    assert read_record(path) is None
    path.write_text('{"formatVersion": 99}')
    assert read_record(path) is None
    # A naive time would break the recorder's arithmetic; it reads as unreadable.
    good = derive_cluster_intent(
        sample_state(), install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
    ).model_dump(mode="json", by_alias=True)
    good["stamp"]["writtenAt"] = "2026-10-11T12:00:00"
    path.write_text(json.dumps(good))
    assert read_record(path) is None
    # Damaged bytes that are not even UTF-8 must not stop the node starting.
    path.write_bytes(b"\xff\xfe\x00garbage")
    assert read_record(path) is None


def test_content_comparison_ignores_the_stamp() -> None:
    record = derive_cluster_intent(
        sample_state(), install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
    )
    later = record.model_copy(
        update={"stamp": record.stamp.model_copy(update={"written_at": NOW + timedelta(hours=1)})}
    )
    assert later.same_content(record)
    assert not record.same_content(None)
    assert not record.model_copy(update={"placements": ()}).same_content(record)


def test_restored_proposals_are_all_terminal() -> None:
    record = ClusterIntentRecord(
        stamp=RecoveryStamp(
            session_master_node_id=NodeId("node-a"),
            session_election_clock=0,
            last_event_applied_idx=0,
            written_at=NOW,
        ),
        steward_proposals=(
            sample_proposal("pending", 3),
            sample_proposal("approved", 6),
            sample_proposal("dispatched", 8),
            sample_proposal("stale", 9),
            sample_proposal("rejected", 12),
        ),
    )
    restart = NOW + timedelta(minutes=1)

    restored = restored_steward_proposals(record, now=restart)

    by_original = {
        original.proposal_id: restored[original.proposal_id]
        for original in record.steward_proposals
    }
    statuses = [by_original[p.proposal_id].status for p in record.steward_proposals]
    assert statuses == ["expired", "failed", "failed", "stale", "rejected"]
    pending, approved, dispatched, stale, rejected = (
        by_original[p.proposal_id] for p in record.steward_proposals
    )
    assert pending.outcome == RESTARTED_BEFORE_DECISION
    assert pending.decided_at == restart
    assert pending.decided_by == "fabric_restart"
    assert approved.outcome == RESTARTED_BEFORE_FINISH
    assert dispatched.outcome == RESTARTED_BEFORE_FINISH
    # A decision already made keeps its time and actor.
    assert approved.decided_by == "trusted_fabric_operator"
    assert stale == record.steward_proposals[3]
    assert rejected == record.steward_proposals[4]
    assert all(
        proposal.status not in {"pending", "approved", "dispatched"}
        for proposal in restored.values()
    )


def _intent(model: str, *hosts: UUID) -> PlacementIntent:
    return PlacementIntent(
        model_id=ModelId(model),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=max(len(hosts), 1),
        host_install_ids=hosts,
    )


def test_departed_placements_keep_their_place_and_new_ones_follow() -> None:
    first, departed, last = _intent("org/first", KITE1), _intent("org/gone", KITE2), _intent("org/last", KITE1)
    added = _intent("org/new", KITE1)
    placements, departed_since = carry_departed_placements(
        (first, departed, last),
        (first, last, added),
        connected_install_ids={KITE1},
        departed_since={},
        now=NOW,
    )
    assert placements == (first, departed, last, added)
    assert list(departed_since.items()) == [(departed, NOW)]


def test_identical_copies_on_a_departed_machine_are_both_kept() -> None:
    copy = _intent("org/model", KITE2)
    placements, _ = carry_departed_placements(
        (copy, copy), (), connected_install_ids={KITE1}, departed_since={}, now=NOW
    )
    assert placements == (copy, copy)


def test_a_placement_spanning_a_departed_machine_is_kept() -> None:
    """A two-machine placement goes away when either of its machines does."""
    wide = _intent("org/wide", KITE1, KITE2)
    placements, _ = carry_departed_placements(
        (wide,), (), connected_install_ids={KITE1}, departed_since={}, now=NOW
    )
    assert placements == (wide,)


def test_the_grace_runs_from_when_the_placement_first_went_missing() -> None:
    departed = _intent("org/gone", KITE2)
    placements, departed_since = carry_departed_placements(
        (departed,),
        (),
        connected_install_ids=set(),
        departed_since=dict([(departed, NOW - DEPARTURE_GRACE)]),
        now=NOW,
    )
    assert placements == ()
    assert departed_since == {}


def test_exclusions_of_absent_machines_stay_in_the_record() -> None:
    """A restored instance keeps an absent machine's exclusion by install ID."""
    absent = UUID("33333333-3333-4333-8333-333333333333")
    instance = sample_instance("restored", "org/model", ("node-b",), excluded=("node-a",))
    instance = instance.model_copy(update={"excluded_install_ids": [absent]})
    state = sample_state().model_copy(update={"instances": {instance.instance_id: instance}})

    record = derive_cluster_intent(state, install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW)

    assert set(record.placements[0].excluded_install_ids) == {KITE1, absent}
