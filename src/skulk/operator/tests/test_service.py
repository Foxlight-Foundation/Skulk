"""Lifecycle and failure tests for the bounded authority consensus service."""

from __future__ import annotations

from typing import final
from uuid import UUID, uuid4

import anyio
import pytest
from anyio.abc import TaskGroup
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.operator.consensus import (
    AuthorityBallot,
    AuthorityCommittedEntry,
    AuthorityConsensusConflictError,
    AuthorityConsensusParticipant,
    AuthorityConsensusRepository,
    AuthorityConsensusSnapshot,
    AuthorityConsensusState,
    AuthorityNetworkEnvelope,
    AuthorityVoteCollector,
)
from skulk.operator.identity import create_cluster_identity
from skulk.operator.replication import (
    AuthorityCommitDescriptor,
    AuthorityMember,
    AuthorityMembership,
    authority_bootstrap_position,
    authority_membership_digest,
    authority_payload_digest,
    create_authority_member,
)
from skulk.operator.service import (
    AuthorityConsensusService,
    AuthorityProposalBusyError,
    AuthorityProposalUnavailableError,
    AuthorityServiceNotRunningError,
    AuthorityTransitionRequest,
)
from skulk.utils.channels import Receiver, Sender, channel


@final
class _MemoryConsensusRepository(AuthorityConsensusRepository):
    """Revisioned in-memory repository used by asynchronous lifecycle tests."""

    def __init__(self, state: AuthorityConsensusState) -> None:
        self._snapshot = AuthorityConsensusSnapshot(revision=0, state=state)

    def load(self) -> AuthorityConsensusSnapshot:
        """Return the latest immutable snapshot."""

        return self._snapshot

    def compare_and_set(
        self,
        expected_revision: int,
        state: AuthorityConsensusState,
    ) -> AuthorityConsensusSnapshot:
        """Persist one revision or report a simulated concurrent transition."""

        if self._snapshot.revision != expected_revision:
            raise AuthorityConsensusConflictError("test repository CAS conflict")
        self._snapshot = AuthorityConsensusSnapshot(
            revision=expected_revision + 1,
            state=state,
        )
        return self._snapshot


@final
class _MemoryAuthorityTransport:
    """One node's endpoint on a bounded in-memory authority network."""

    def __init__(self, node_install_id: UUID, network: "_MemoryAuthorityNetwork"):
        self._node_install_id = node_install_id
        self._network = network

    async def send(self, envelope: AuthorityNetworkEnvelope) -> None:
        """Deliver one envelope through the shared deterministic network."""

        if UUID(str(envelope.source_node_install_id)) != self._node_install_id:
            raise ValueError("test transport cannot send for another member")
        await self._network.send(envelope)

    async def receive(self) -> AuthorityNetworkEnvelope:
        """Return the next envelope addressed to this endpoint."""

        return await self._network.receive(self._node_install_id)


@final
class _MemoryAuthorityNetwork:
    """Bounded test network with explicit node and one-shot frame loss controls."""

    def __init__(self, member_ids: tuple[UUID, ...]) -> None:
        endpoints = {
            member_id: channel[AuthorityNetworkEnvelope](256)
            for member_id in member_ids
        }
        self._senders: dict[UUID, Sender[AuthorityNetworkEnvelope]] = {
            member_id: endpoint[0] for member_id, endpoint in endpoints.items()
        }
        self._receivers: dict[UUID, Receiver[AuthorityNetworkEnvelope]] = {
            member_id: endpoint[1] for member_id, endpoint in endpoints.items()
        }
        self._blocked_nodes: set[UUID] = set()
        self._drop_once: set[tuple[str, UUID]] = set()

    def transport(self, member_id: UUID) -> _MemoryAuthorityTransport:
        """Return one stable member's transport endpoint."""

        return _MemoryAuthorityTransport(member_id, self)

    def block(self, member_id: UUID) -> None:
        """Drop every frame to and from one simulated unavailable member."""

        self._blocked_nodes.add(member_id)

    def drop_once(self, kind: str, target: UUID) -> None:
        """Drop the next matching payload kind sent to one member."""

        self._drop_once.add((kind, target))

    async def send(self, envelope: AuthorityNetworkEnvelope) -> None:
        """Apply fault rules, then deliver one frame to its bounded endpoint."""

        source = UUID(str(envelope.source_node_install_id))
        target = UUID(str(envelope.target_node_install_id))
        if source in self._blocked_nodes or target in self._blocked_nodes:
            return
        drop_key = (envelope.payload.kind, target)
        if drop_key in self._drop_once:
            self._drop_once.remove(drop_key)
            return
        await self._senders[target].send(envelope)

    async def receive(self, member_id: UUID) -> AuthorityNetworkEnvelope:
        """Return one frame from the selected member's endpoint."""

        return await self._receivers[member_id].receive()


@final
class _ServiceHarness:
    """Create matching participants, services, and an in-memory network."""

    def __init__(
        self,
        voter_count: int,
        *,
        phase_timeout_seconds: float = 0.05,
        max_attempts: int = 3,
    ) -> None:
        identity = create_cluster_identity("Service Test").public_identity
        position = authority_bootstrap_position(identity)
        keys: dict[UUID, bytes] = {}
        members: list[AuthorityMember] = []
        for _ in range(voter_count):
            member_id = uuid4()
            private_key = _private_key()
            keys[member_id] = private_key
            members.append(create_authority_member(member_id, private_key))
        self.membership = AuthorityMembership(generation=1, members=tuple(members))
        self.keys = keys
        self.member_ids = tuple(keys)
        self.repositories = {
            member_id: _MemoryConsensusRepository(
                AuthorityConsensusState.bootstrap(position, self.membership)
            )
            for member_id in self.member_ids
        }
        self.participants = {
            member_id: AuthorityConsensusParticipant(
                member_id,
                keys[member_id],
                self.repositories[member_id],
            )
            for member_id in self.member_ids
        }
        self.network = _MemoryAuthorityNetwork(self.member_ids)
        self.services = {
            member_id: AuthorityConsensusService(
                self.participants[member_id],
                self.network.transport(member_id),
                phase_timeout_seconds=phase_timeout_seconds,
                retry_backoff_seconds=0,
                max_attempts=max_attempts,
            )
            for member_id in self.member_ids
        }

    def request(self, record_id: str) -> AuthorityTransitionRequest:
        """Return one valid semantic transition against current membership."""

        return AuthorityTransitionRequest(
            record_type="device",
            record_id=record_id,
            payload_digest=authority_payload_digest({"recordId": record_id}),
        )

    async def start(
        self,
        task_group: TaskGroup,
        *member_ids: UUID,
    ) -> None:
        """Start selected services and wait for their receive loops."""

        selected = member_ids if member_ids else self.member_ids
        for member_id in selected:
            task_group.start_soon(self.services[member_id].run)
        for member_id in selected:
            await self.services[member_id].wait_started()


def _private_key() -> bytes:
    """Return one raw Ed25519 key for an authority test member."""

    return Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


async def _wait_for_commit_index(
    harness: _ServiceHarness,
    member_id: UUID,
    commit_index: int,
) -> None:
    """Wait briefly for an asynchronously broadcast commit or catch-up page."""

    with anyio.fail_after(1):
        while (
            harness.repositories[member_id].load().state.position.commit_index
            < commit_index
        ):
            await anyio.sleep(0)


async def test_proposal_requires_active_service() -> None:
    """A dormant service cannot mutate consensus state through its API."""

    harness = _ServiceHarness(1)
    member_id = harness.member_ids[0]

    with pytest.raises(AuthorityServiceNotRunningError):
        await harness.services[member_id].propose(harness.request("device-1"))
    assert harness.repositories[member_id].load().state.position.commit_index == 1


async def test_one_voter_service_commits_durably() -> None:
    """The single-voter profile uses the full lifecycle without network peers."""

    harness = _ServiceHarness(1)
    member_id = harness.member_ids[0]
    entry: AuthorityCommittedEntry | None = None

    async with anyio.create_task_group() as task_group:
        await harness.start(task_group)
        entry = await harness.services[member_id].propose(
            harness.request("device-1")
        )
        task_group.cancel_scope.cancel()

    assert entry is not None
    assert entry.certificate.descriptor.commit_index == 2
    assert entry.certificate.descriptor.record_id == "device-1"
    assert harness.repositories[member_id].load().state.position.commit_index == 2
    diagnostics = harness.services[member_id].diagnostics()
    assert diagnostics.completed_proposals == 1
    assert diagnostics.failed_proposals == 0
    assert not diagnostics.running


async def test_three_voters_commit_with_one_unavailable_member() -> None:
    """A three-voter authority makes progress with one crash-faulted voter."""

    harness = _ServiceHarness(3)
    leader_id, follower_id, unavailable_id = harness.member_ids
    harness.network.block(unavailable_id)
    entry: AuthorityCommittedEntry | None = None

    async with anyio.create_task_group() as task_group:
        await harness.start(task_group, leader_id, follower_id)
        entry = await harness.services[leader_id].propose(harness.request("device-1"))
        await _wait_for_commit_index(harness, follower_id, 2)
        task_group.cancel_scope.cancel()

    assert entry is not None
    assert entry.certificate.descriptor.commit_index == 2
    assert harness.repositories[leader_id].load().state.position.commit_index == 2
    assert harness.repositories[follower_id].load().state.position.commit_index == 2
    assert harness.repositories[unavailable_id].load().state.position.commit_index == 1


async def test_two_voters_fail_closed_when_one_is_unavailable() -> None:
    """The two-voter profile cannot imply failover when either voter is lost."""

    harness = _ServiceHarness(2, phase_timeout_seconds=0.01, max_attempts=2)
    leader_id, unavailable_id = harness.member_ids
    harness.network.block(unavailable_id)

    async with anyio.create_task_group() as task_group:
        await harness.start(task_group, leader_id)
        with pytest.raises(AuthorityProposalUnavailableError):
            await harness.services[leader_id].propose(harness.request("blocked"))
        task_group.cancel_scope.cancel()

    assert harness.repositories[leader_id].load().state.position.commit_index == 1
    assert harness.services[leader_id].diagnostics().failed_proposals == 1


async def test_commit_gap_triggers_bounded_catch_up_through_service() -> None:
    """A replica missing one commit catches up when it observes the next index."""

    harness = _ServiceHarness(3)
    leader_id, _, lagging_id = harness.member_ids
    harness.network.drop_once("commit", lagging_id)

    async with anyio.create_task_group() as task_group:
        await harness.start(task_group)
        await harness.services[leader_id].propose(harness.request("device-1"))
        assert harness.repositories[lagging_id].load().state.position.commit_index == 1

        await harness.services[leader_id].propose(harness.request("device-2"))
        await _wait_for_commit_index(harness, lagging_id, 3)
        task_group.cancel_scope.cancel()

    lagging_state = harness.repositories[lagging_id].load().state
    assert lagging_state.position.commit_index == 3
    assert tuple(
        entry.certificate.descriptor.record_id
        for entry in lagging_state.committed_entries
    ) == ("device-1", "device-2")


async def test_new_proposer_recovers_accepted_value_before_its_request() -> None:
    """Leader replacement commits accepted history before advancing its intent."""

    harness = _ServiceHarness(3)
    first_leader, replacement_leader, third_voter = harness.member_ids
    position = harness.repositories[first_leader].load().state.position
    descriptor = AuthorityCommitDescriptor(
        cluster_id=position.cluster_id,
        authority_term=2,
        commit_index=2,
        previous_commit_digest=position.commit_digest,
        record_type="device",
        record_id="must-survive",
        payload_digest=authority_payload_digest({"recordId": "must-survive"}),
        required_membership_digests=(
            authority_membership_digest(harness.membership),
        ),
    )
    interrupted = AuthorityVoteCollector(
        AuthorityBallot(
            counter=2,
            proposer_node_install_id=first_leader,
        ),
        descriptor,
        (harness.membership,),
        position,
    )
    prepare = interrupted.prepare_message()
    for target in (first_leader, replacement_leader):
        response = harness.participants[target].handle(
            harness.participants[first_leader].envelope_for(target, prepare)
        )
        interrupted.record_prepare_response(response[0])
    accept = interrupted.accept_message()
    for target in (first_leader, replacement_leader):
        response = harness.participants[target].handle(
            harness.participants[first_leader].envelope_for(target, accept)
        )
        interrupted.record_accept_response(response[0])
    assert interrupted.accept_ready
    requested: AuthorityCommittedEntry | None = None

    async with anyio.create_task_group() as task_group:
        await harness.start(task_group)
        requested = await harness.services[replacement_leader].propose(
            harness.request("replacement-intent")
        )
        await _wait_for_commit_index(harness, third_voter, 3)
        task_group.cancel_scope.cancel()

    assert requested is not None
    assert requested.certificate.descriptor.record_id == "replacement-intent"
    replacement_state = harness.repositories[replacement_leader].load().state
    assert tuple(
        entry.certificate.descriptor.record_id
        for entry in replacement_state.committed_entries
    ) == ("must-survive", "replacement-intent")


async def test_concurrent_local_proposal_is_rejected_without_queueing() -> None:
    """Only one local proposal is admitted while a quorum attempt is active."""

    harness = _ServiceHarness(2, phase_timeout_seconds=0.1, max_attempts=1)
    leader_id, unavailable_id = harness.member_ids
    harness.network.block(unavailable_id)
    first_finished = anyio.Event()

    async def first_proposal() -> None:
        """Hold the proposal slot until the intentionally missing quorum times out."""

        with pytest.raises(AuthorityProposalUnavailableError):
            await harness.services[leader_id].propose(harness.request("first"))
        first_finished.set()

    async with anyio.create_task_group() as task_group:
        await harness.start(task_group, leader_id)
        task_group.start_soon(first_proposal)
        await anyio.sleep(0)
        with pytest.raises(AuthorityProposalBusyError):
            await harness.services[leader_id].propose(harness.request("second"))
        await first_finished.wait()
        task_group.cancel_scope.cancel()
