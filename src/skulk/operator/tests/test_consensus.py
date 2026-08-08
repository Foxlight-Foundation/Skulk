"""Deterministic network and recovery tests for operator authority consensus."""

from __future__ import annotations

from collections.abc import Iterable
from typing import final
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.operator.consensus import (
    AuthorityAcceptMessage,
    AuthorityBallot,
    AuthorityCommitMessage,
    AuthorityCommittedEntry,
    AuthorityConsensusConflictError,
    AuthorityConsensusError,
    AuthorityConsensusParticipant,
    AuthorityConsensusRepository,
    AuthorityConsensusSnapshot,
    AuthorityConsensusState,
    AuthorityNetworkEnvelope,
    AuthorityPrepareMessage,
    AuthorityRejectedMessage,
    AuthorityVoteCollector,
)
from skulk.operator.identity import create_cluster_identity
from skulk.operator.replication import (
    AuthorityCommitDescriptor,
    AuthorityCommitPosition,
    AuthorityMember,
    AuthorityMembership,
    authority_bootstrap_position,
    authority_commit_digest,
    authority_membership_digest,
    authority_payload_digest,
    create_authority_member,
)


@final
class _MemoryConsensusRepository(AuthorityConsensusRepository):
    """CAS repository whose state survives participant object replacement."""

    def __init__(self, state: AuthorityConsensusState) -> None:
        self._snapshot = AuthorityConsensusSnapshot(revision=0, state=state)

    def load(self) -> AuthorityConsensusSnapshot:
        """Return the current immutable snapshot."""

        return self._snapshot

    def compare_and_set(
        self,
        expected_revision: int,
        state: AuthorityConsensusState,
    ) -> AuthorityConsensusSnapshot:
        """Persist exactly one revision or report a simulated local race."""

        if expected_revision != self._snapshot.revision:
            raise AuthorityConsensusConflictError("test repository CAS conflict")
        self._snapshot = AuthorityConsensusSnapshot(
            revision=expected_revision + 1,
            state=state,
        )
        return self._snapshot


def _private_key() -> bytes:
    """Return one raw Ed25519 key for an authority test member."""

    return Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


@final
class _AuthorityHarness:
    """No-socket transport delivering production messages in explicit order."""

    def __init__(self, voter_count: int, *, learner_count: int = 0) -> None:
        identity = create_cluster_identity("Consensus Test").public_identity
        self.position = authority_bootstrap_position(identity)
        self.keys: dict[UUID, bytes] = {}
        members: list[AuthorityMember] = []
        for _ in range(voter_count):
            member_id = uuid4()
            private_key = _private_key()
            self.keys[member_id] = private_key
            members.append(create_authority_member(member_id, private_key))
        for _ in range(learner_count):
            member_id = uuid4()
            private_key = _private_key()
            self.keys[member_id] = private_key
            members.append(
                create_authority_member(member_id, private_key, role="learner")
            )
        self.membership = AuthorityMembership(generation=1, members=tuple(members))
        self.repositories = {
            member_id: _MemoryConsensusRepository(
                AuthorityConsensusState.bootstrap(self.position, self.membership)
            )
            for member_id in self.keys
        }
        self.participants = {
            member_id: AuthorityConsensusParticipant(
                member_id,
                self.keys[member_id],
                self.repositories[member_id],
            )
            for member_id in self.keys
        }

    @property
    def voter_ids(self) -> tuple[UUID, ...]:
        """Return initial voter identities in membership order."""

        return tuple(
            UUID(str(member.node_install_id))
            for member in self.membership.members
            if member.role == "voter"
        )

    @property
    def learner_ids(self) -> tuple[UUID, ...]:
        """Return initial learner identities in membership order."""

        return tuple(
            UUID(str(member.node_install_id))
            for member in self.membership.members
            if member.role == "learner"
        )

    def restart(self, member_id: UUID) -> None:
        """Replace one process object while retaining its durable repository."""

        self.participants[member_id] = AuthorityConsensusParticipant(
            member_id,
            self.keys[member_id],
            self.repositories[member_id],
        )

    def descriptor(
        self,
        leader_id: UUID,
        term: int,
        record_id: str,
        *,
        memberships: tuple[AuthorityMembership, ...] | None = None,
    ) -> AuthorityCommitDescriptor:
        """Build one desired next descriptor from the leader's durable head."""

        selected = memberships if memberships is not None else (self.membership,)
        position = self.participants[leader_id].snapshot.state.position
        return AuthorityCommitDescriptor(
            cluster_id=position.cluster_id,
            authority_term=term,
            commit_index=position.commit_index + 1,
            previous_commit_digest=position.commit_digest,
            record_type="device",
            record_id=record_id,
            payload_digest=authority_payload_digest({"recordId": record_id}),
            required_membership_digests=tuple(
                sorted(authority_membership_digest(item) for item in selected)
            ),
        )

    def collector(
        self,
        leader_id: UUID,
        term: int,
        record_id: str,
        *,
        memberships: tuple[AuthorityMembership, ...] | None = None,
    ) -> AuthorityVoteCollector:
        """Create one deterministic collector at the leader's current position."""

        selected = memberships if memberships is not None else (self.membership,)
        position = self.participants[leader_id].snapshot.state.position
        return AuthorityVoteCollector(
            AuthorityBallot(
                counter=term,
                proposer_node_install_id=leader_id,
            ),
            self.descriptor(
                leader_id,
                term,
                record_id,
                memberships=selected,
            ),
            selected,
            position,
        )

    def prepare(
        self,
        leader_id: UUID,
        collector: AuthorityVoteCollector,
        reachable: Iterable[UUID],
    ) -> None:
        """Deliver prepare messages and their responses synchronously."""

        leader = self.participants[leader_id]
        payload = collector.prepare_message()
        for target_id in reachable:
            request = leader.envelope_for(target_id, payload)
            responses = self.participants[target_id].handle(request)
            assert len(responses) == 1
            collector.record_prepare_response(responses[0])

    def accept(
        self,
        leader_id: UUID,
        collector: AuthorityVoteCollector,
        reachable: Iterable[UUID],
    ) -> None:
        """Deliver accept messages and their responses synchronously."""

        leader = self.participants[leader_id]
        payload = collector.accept_message()
        for target_id in reachable:
            request = leader.envelope_for(target_id, payload)
            responses = self.participants[target_id].handle(request)
            assert len(responses) == 1
            collector.record_accept_response(responses[0])

    def commit(
        self,
        leader_id: UUID,
        collector: AuthorityVoteCollector,
        reachable: Iterable[UUID],
    ) -> AuthorityCommittedEntry:
        """Deliver one certificate to selected replicas and return its entry."""

        entry = collector.committed_entry()
        message = AuthorityCommitMessage(
            cluster_id=entry.certificate.descriptor.cluster_id,
            request_id=collector.request_id,
            entry=entry,
        )
        leader = self.participants[leader_id]
        for target_id in reachable:
            responses = self.participants[target_id].handle(
                leader.envelope_for(target_id, message)
            )
            assert responses == ()
        descriptor = entry.certificate.descriptor
        self.position = AuthorityCommitPosition(
            cluster_id=descriptor.cluster_id,
            authority_term=descriptor.authority_term,
            commit_index=descriptor.commit_index,
            commit_digest=authority_commit_digest(descriptor),
        )
        return entry

    def run_round(
        self,
        leader_id: UUID,
        term: int,
        record_id: str,
        reachable: Iterable[UUID],
        *,
        commit_targets: Iterable[UUID] | None = None,
        memberships: tuple[AuthorityMembership, ...] | None = None,
    ) -> AuthorityCommittedEntry:
        """Run prepare, accept, and commit over one explicit reachable set."""

        targets = tuple(reachable)
        collector = self.collector(
            leader_id,
            term,
            record_id,
            memberships=memberships,
        )
        self.prepare(leader_id, collector, targets)
        self.accept(leader_id, collector, targets)
        return self.commit(
            leader_id,
            collector,
            targets if commit_targets is None else tuple(commit_targets),
        )


def test_one_voter_commits_and_restores_after_restart() -> None:
    """One authority node uses the same durable protocol without failover."""

    harness = _AuthorityHarness(1)
    leader_id = harness.voter_ids[0]

    entry = harness.run_round(leader_id, 2, "device-1", (leader_id,))
    harness.restart(leader_id)

    restored = harness.participants[leader_id].snapshot.state
    assert restored.position.commit_index == 2
    assert restored.committed_entries == (entry,)
    assert restored.accepted_descriptor is None


@pytest.mark.parametrize(
    ("voter_count", "reachable_count"),
    ((2, 1), (3, 1), (4, 2)),
)
def test_quorum_loss_fails_closed(
    voter_count: int,
    reachable_count: int,
) -> None:
    """A minority or exact half cannot enter the accept phase."""

    harness = _AuthorityHarness(voter_count)
    leader_id = harness.voter_ids[0]
    reachable = harness.voter_ids[:reachable_count]
    collector = harness.collector(leader_id, 2, "blocked")

    harness.prepare(leader_id, collector, reachable)

    assert not collector.prepare_ready
    with pytest.raises(AuthorityConsensusError, match="prepare quorum"):
        collector.accept_message()


def test_three_voters_recover_accepted_value_after_leader_restart() -> None:
    """A replacement leader finishes the value accepted before a crash."""

    harness = _AuthorityHarness(3)
    first_leader, second_leader, third_voter = harness.voter_ids
    interrupted = harness.collector(first_leader, 2, "must-survive")
    harness.prepare(first_leader, interrupted, (first_leader, second_leader))
    harness.accept(first_leader, interrupted, (first_leader, second_leader))
    harness.restart(second_leader)

    replacement = harness.collector(second_leader, 3, "must-not-replace")
    harness.prepare(second_leader, replacement, (second_leader, third_voter))
    accept = replacement.accept_message()

    assert accept.descriptor.record_id == "must-survive"
    assert accept.descriptor == interrupted.accept_message().descriptor
    assert authority_commit_digest(accept.descriptor) == authority_commit_digest(
        interrupted.accept_message().descriptor
    )
    assert accept.descriptor.authority_term == 2
    assert accept.ballot.counter == 3
    harness.accept(second_leader, replacement, (second_leader, third_voter))
    entry = harness.commit(
        second_leader,
        replacement,
        (first_leader, second_leader, third_voter),
    )
    assert entry.certificate.descriptor.record_id == "must-survive"
    assert all(
        participant.snapshot.state.position.commit_index == 2
        for participant in harness.participants.values()
    )


def test_learner_cannot_promise_or_vote() -> None:
    """Enrollment as a learner conveys no authority vote before promotion."""

    harness = _AuthorityHarness(3, learner_count=1)
    leader_id = harness.voter_ids[0]
    learner_id = harness.learner_ids[0]
    collector = harness.collector(leader_id, 2, "device-1")
    prepare = collector.prepare_message()

    response = harness.participants[learner_id].handle(
        harness.participants[leader_id].envelope_for(learner_id, prepare)
    )[0]

    assert isinstance(response.payload, AuthorityRejectedMessage)
    assert response.payload.code == "not_voter"


def test_learner_cannot_create_a_vote_collector() -> None:
    """A learner cannot become proposer even though it owns a signing key."""

    harness = _AuthorityHarness(3, learner_count=1)
    learner_id = harness.learner_ids[0]

    with pytest.raises(ValueError, match="active authority voter"):
        harness.collector(learner_id, 2, "forbidden")


def test_duplicate_and_out_of_order_messages_are_idempotent_or_rejected() -> None:
    """Transport duplication cannot double-count votes or replace one accept."""

    harness = _AuthorityHarness(3)
    leader_id = harness.voter_ids[0]
    target_id = harness.voter_ids[1]
    collector = harness.collector(leader_id, 2, "device-1")
    prepare = collector.prepare_message()
    request = harness.participants[leader_id].envelope_for(target_id, prepare)

    first = harness.participants[target_id].handle(request)[0]
    second = harness.participants[target_id].handle(request)[0]
    collector.record_prepare_response(first)
    collector.record_prepare_response(second)

    assert not collector.prepare_ready


def test_envelope_tampering_fails_before_consensus_state_changes() -> None:
    """Target or payload substitution invalidates the outer member signature."""

    harness = _AuthorityHarness(3)
    leader_id, target_id, wrong_target = harness.voter_ids
    collector = harness.collector(leader_id, 2, "device-1")
    envelope = harness.participants[leader_id].envelope_for(
        target_id,
        collector.prepare_message(),
    )
    tampered = AuthorityNetworkEnvelope(
        message_id=envelope.message_id,
        source_node_install_id=envelope.source_node_install_id,
        target_node_install_id=wrong_target,
        payload=envelope.payload,
        signature=envelope.signature,
    )

    with pytest.raises(AuthorityConsensusError):
        harness.participants[wrong_target].handle(tampered)
    assert harness.participants[wrong_target].snapshot.revision == 0


def test_joint_transition_requires_both_quorums_and_fences_removed_member() -> None:
    """Four-node membership change cannot split into two valid majorities."""

    harness = _AuthorityHarness(4, learner_count=1)
    first, second, third, removed = harness.voter_ids
    promoted = harness.learner_ids[0]
    member_by_id = {
        UUID(str(member.node_install_id)): member
        for member in harness.membership.members
    }
    new_membership = AuthorityMembership(
        generation=2,
        members=tuple(
            member_by_id[member_id].model_copy(update={"role": "voter"})
            for member_id in (first, third, promoted)
        ),
    )
    memberships = (harness.membership, new_membership)
    collector = harness.collector(
        first,
        2,
        "membership-2",
        memberships=memberships,
    )
    harness.prepare(first, collector, (first, second, removed, promoted))
    assert collector.prepare_ready
    harness.accept(first, collector, (first, second, removed))
    assert not collector.accept_ready

    accept = collector.accept_message()
    promoted_accept = harness.participants[first].envelope_for(promoted, accept)
    collector.record_accept_response(
        harness.participants[promoted].handle(promoted_accept)[0]
    )
    entry = harness.commit(first, collector, tuple(harness.participants))

    assert entry.memberships == memberships
    assert all(
        participant.snapshot.state.active_membership == new_membership
        for participant in harness.participants.values()
    )
    removed_state = harness.participants[removed].snapshot.state
    assert removed not in removed_state.active_membership.voter_ids
    assert promoted in removed_state.active_membership.voter_ids


def test_lagging_replica_requests_and_applies_certified_catch_up() -> None:
    """A replica missing commits restores only a contiguous certified suffix."""

    harness = _AuthorityHarness(3)
    leader_id, follower_id, lagging_id = harness.voter_ids
    first = harness.run_round(
        leader_id,
        2,
        "device-1",
        (leader_id, follower_id),
        commit_targets=(leader_id, follower_id),
    )
    second = harness.run_round(
        leader_id,
        3,
        "device-2",
        (leader_id, follower_id),
        commit_targets=(leader_id, follower_id),
    )
    commit = AuthorityCommitMessage(
        cluster_id=second.certificate.descriptor.cluster_id,
        request_id=uuid4(),
        entry=second,
    )

    request = harness.participants[lagging_id].handle(
        harness.participants[leader_id].envelope_for(lagging_id, commit)
    )[0]
    response = harness.participants[leader_id].handle(request)[0]
    harness.participants[lagging_id].handle(response)

    recovered = harness.participants[lagging_id].snapshot.state
    assert recovered.committed_entries == (first, second)
    assert recovered.position.commit_index == 3


def test_lagging_replica_continues_across_bounded_catch_up_pages() -> None:
    """A replica more than one response behind requests every remaining page."""

    harness = _AuthorityHarness(3)
    leader_id, follower_id, lagging_id = harness.voter_ids
    entries = tuple(
        harness.run_round(
            leader_id,
            term,
            f"device-{term}",
            (leader_id, follower_id),
            commit_targets=(leader_id, follower_id),
        )
        for term in range(2, 67)
    )
    final_commit = AuthorityCommitMessage(
        cluster_id=entries[-1].certificate.descriptor.cluster_id,
        request_id=uuid4(),
        entry=entries[-1],
    )

    request = harness.participants[lagging_id].handle(
        harness.participants[leader_id].envelope_for(lagging_id, final_commit)
    )[0]
    response = harness.participants[leader_id].handle(request)[0]
    continuation = harness.participants[lagging_id].handle(response)[0]
    final_response = harness.participants[leader_id].handle(continuation)[0]
    assert harness.participants[lagging_id].handle(final_response) == ()

    recovered = harness.participants[lagging_id].snapshot.state
    assert recovered.committed_entries == entries
    assert recovered.position.commit_index == entries[-1].certificate.descriptor.commit_index


def test_stale_ballot_cannot_overwrite_newer_durable_promise() -> None:
    """Restart retains the ballot fence against an older collector."""

    harness = _AuthorityHarness(3)
    first_leader, second_leader, target_id = harness.voter_ids
    newer = harness.collector(second_leader, 3, "newer")
    harness.prepare(second_leader, newer, (target_id,))
    harness.restart(target_id)
    stale = harness.collector(first_leader, 2, "stale")
    request = harness.participants[first_leader].envelope_for(
        target_id,
        stale.prepare_message(),
    )

    response = harness.participants[target_id].handle(request)[0]

    assert isinstance(response.payload, AuthorityRejectedMessage)
    assert response.payload.code == "stale_ballot"
    assert (
        harness.participants[target_id].snapshot.state.promised_ballot == newer.ballot
    )


def test_equal_counter_ballots_use_stable_proposer_tiebreak() -> None:
    """Concurrent leaders cannot share one ambiguous integer ballot."""

    harness = _AuthorityHarness(3)
    lower, higher, target = sorted(harness.voter_ids, key=lambda item: item.int)
    higher_round = harness.collector(higher, 2, "higher")
    harness.prepare(higher, higher_round, (target,))
    lower_round = harness.collector(lower, 2, "lower")
    request = harness.participants[lower].envelope_for(
        target,
        lower_round.prepare_message(),
    )

    response = harness.participants[target].handle(request)[0]

    assert isinstance(response.payload, AuthorityRejectedMessage)
    assert response.payload.code == "stale_ballot"
    assert (
        harness.participants[target].snapshot.state.promised_ballot
        == higher_round.ballot
    )


def test_accept_before_prepare_is_valid_but_conflicting_same_ballot_is_rejected() -> (
    None
):
    """An acceptor supports phase-two replay but refuses ballot equivocation."""

    harness = _AuthorityHarness(3)
    leader_id, target_id, _ = harness.voter_ids
    first = harness.collector(leader_id, 2, "first")
    accept = AuthorityAcceptMessage(
        cluster_id=first.requested_descriptor.cluster_id,
        request_id=first.request_id,
        ballot=first.ballot,
        descriptor=first.requested_descriptor,
        memberships=first.requested_memberships,
    )
    accepted = harness.participants[target_id].handle(
        harness.participants[leader_id].envelope_for(target_id, accept)
    )[0]
    conflicting = harness.descriptor(leader_id, 2, "second")
    conflict = AuthorityAcceptMessage(
        cluster_id=conflicting.cluster_id,
        request_id=uuid4(),
        ballot=first.ballot,
        descriptor=conflicting,
        memberships=(harness.membership,),
    )
    rejected = harness.participants[target_id].handle(
        harness.participants[leader_id].envelope_for(target_id, conflict)
    )[0]

    assert not isinstance(accepted.payload, AuthorityRejectedMessage)
    assert isinstance(rejected.payload, AuthorityRejectedMessage)
    assert rejected.payload.code == "conflicting_accept"


def test_prepare_and_accept_models_round_trip_through_wire_json() -> None:
    """Production protocol messages retain strict discriminated wire types."""

    harness = _AuthorityHarness(1)
    leader_id = harness.voter_ids[0]
    collector = harness.collector(leader_id, 2, "device-1")
    prepare = harness.participants[leader_id].envelope_for(
        leader_id,
        collector.prepare_message(),
    )
    decoded_prepare = AuthorityNetworkEnvelope.model_validate_json(
        prepare.model_dump_json()
    )
    assert isinstance(decoded_prepare.payload, AuthorityPrepareMessage)

    harness.prepare(leader_id, collector, (leader_id,))
    accept = harness.participants[leader_id].envelope_for(
        leader_id,
        collector.accept_message(),
    )
    decoded_accept = AuthorityNetworkEnvelope.model_validate_json(
        accept.model_dump_json()
    )
    assert isinstance(decoded_accept.payload, AuthorityAcceptMessage)
