# pyright: reportUnusedFunction=false
"""Crash-fault consensus protocol for replicated operator authority.

The protocol carries only signed public consensus metadata: ballots, membership
configurations, payload digests, votes, and quorum certificates. Secret-bearing
authority payloads use a separate encrypted replication path and never enter
these messages, Skulk's event-sourced State, or the ordinary event log.

The state machine is transport- and storage-injected. Production networking may
deliver signed envelopes over a dedicated Skulk topic, while deterministic tests
can deliver the same envelopes without sockets, sleeps, or wall-clock time.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from typing import Annotated, Literal, Protocol, final
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import UUID4, ConfigDict, Field, field_validator, model_validator

from skulk.operator.replication import (
    AuthorityCertificateError,
    AuthorityCommitDescriptor,
    AuthorityCommitPosition,
    AuthorityMembership,
    AuthorityQuorumCertificate,
    AuthorityVote,
    authority_commit_digest,
    authority_membership_digest,
    create_authority_vote,
    validate_authority_descriptor,
    verify_quorum_certificate,
)
from skulk.utils.pydantic_ext import CamelCaseModel, FrozenModel

_AUTHORITY_ENVELOPE_CONTEXT = b"skulk-operator-authority-envelope-v1\x00"
_MAX_CATCH_UP_ENTRIES = 64


class AuthorityConsensusError(RuntimeError):
    """Raised when a consensus round cannot safely make progress."""


class AuthorityEnvelopeError(AuthorityConsensusError):
    """Raised when a network envelope is malformed or unauthenticated."""


class AuthorityConsensusConflictError(AuthorityConsensusError):
    """Raised when durable consensus state changes during a local transition."""


@final
class AuthorityBallot(FrozenModel):
    """Unique, totally ordered leadership attempt for one authority index."""

    counter: int = Field(
        ge=1,
        description="Monotonic counter greater than the last committed term.",
    )
    proposer_node_install_id: UUID4 = Field(
        description="Stable proposer identity used to break concurrent-term ties.",
    )


@final
class AuthorityCommittedEntry(FrozenModel):
    """One quorum-certified public authority-log entry used for catch-up."""

    certificate: AuthorityQuorumCertificate = Field(
        description="Descriptor and voter signatures proving the committed value.",
    )
    memberships: tuple[AuthorityMembership, ...] = Field(
        min_length=1,
        max_length=2,
        description="Active or joint configurations that certified the entry.",
    )

    @model_validator(mode="after")
    def _memberships_match_descriptor(self) -> "AuthorityCommittedEntry":
        """Reject entries whose supplied membership material is ambiguous."""

        ordered = tuple(sorted(self.memberships, key=lambda item: item.generation))
        if ordered != self.memberships:
            raise ValueError("authority entry memberships must be generation ordered")
        digests = tuple(
            sorted(authority_membership_digest(item) for item in self.memberships)
        )
        if self.certificate.descriptor.required_membership_digests != digests:
            raise ValueError("authority entry memberships do not match its descriptor")
        return self


@final
class AuthorityConsensusState(FrozenModel):
    """Durable promise, accepted value, membership, and certified log state."""

    bootstrap_position: AuthorityCommitPosition = Field(
        description="Immutable cluster bootstrap position anchoring log verification.",
    )
    bootstrap_membership: AuthorityMembership = Field(
        description="Immutable initial membership anchoring configuration recovery.",
    )
    position: AuthorityCommitPosition = Field(
        description="Last locally verified committed authority position.",
    )
    active_membership: AuthorityMembership = Field(
        description="Single committed membership governing the next proposal.",
    )
    promised_ballot: AuthorityBallot | None = Field(
        default=None,
        description="Highest ballot this node has promised not to undercut.",
    )
    accepted_ballot: AuthorityBallot | None = Field(
        default=None,
        description="Ballot of the uncommitted descriptor accepted at the next index.",
    )
    accepted_descriptor: AuthorityCommitDescriptor | None = Field(
        default=None,
        description="Uncommitted descriptor retained for leader-failure recovery.",
    )
    accepted_memberships: tuple[AuthorityMembership, ...] | None = Field(
        default=None,
        description="Configurations that must certify the accepted descriptor.",
    )
    committed_entries: tuple[AuthorityCommittedEntry, ...] = Field(
        default=(),
        description="Contiguous certified entries retained for replica catch-up.",
    )

    @model_validator(mode="after")
    def _accepted_state_is_complete(self) -> "AuthorityConsensusState":
        """Require immutable bootstrap anchors and complete accepted state."""

        if self.bootstrap_position.commit_index != 1:
            raise ValueError("authority bootstrap position must be commit index one")
        if self.bootstrap_position.cluster_id != self.position.cluster_id:
            raise ValueError("authority bootstrap position belongs to another cluster")
        if self.position.commit_index < self.bootstrap_position.commit_index:
            raise ValueError("authority position precedes its bootstrap anchor")
        if self.position.commit_index == self.bootstrap_position.commit_index:
            if self.position != self.bootstrap_position:
                raise ValueError("authority bootstrap position cannot be replaced")
            if self.active_membership != self.bootstrap_membership:
                raise ValueError("authority bootstrap membership cannot be replaced")
        if self.active_membership.generation < self.bootstrap_membership.generation:
            raise ValueError("authority membership moved behind its bootstrap anchor")

        accepted_parts = (
            self.accepted_ballot,
            self.accepted_descriptor,
            self.accepted_memberships,
        )
        if any(part is None for part in accepted_parts) != all(
            part is None for part in accepted_parts
        ):
            raise ValueError(
                "accepted authority state must be wholly present or absent"
            )
        if self.accepted_descriptor is not None:
            if self.accepted_descriptor.commit_index != self.position.commit_index + 1:
                raise ValueError(
                    "accepted descriptor must target the next commit index"
                )
            assert self.accepted_ballot is not None
            assert self.accepted_memberships is not None
            if self.accepted_descriptor.authority_term > self.accepted_ballot.counter:
                raise ValueError("accepted descriptor term exceeds its ballot")
            if self.accepted_ballot.counter <= self.position.authority_term:
                raise ValueError("accepted ballot must exceed the committed term")
            try:
                _validate_transition_memberships(
                    self.active_membership,
                    self.accepted_memberships,
                )
                validate_authority_descriptor(
                    self.accepted_descriptor,
                    self.accepted_memberships,
                    self.position,
                )
            except AuthorityCertificateError as exc:
                raise ValueError("accepted authority value is invalid") from exc
        if self.promised_ballot is not None:
            if self.promised_ballot.counter < self.position.authority_term:
                raise ValueError("promised ballot precedes the committed term")
            if self.accepted_ballot is not None and _ballot_key(
                self.accepted_ballot
            ) > _ballot_key(self.promised_ballot):
                raise ValueError("accepted ballot exceeds the durable promise")
        if self.committed_entries:
            tail = self.committed_entries[-1].certificate.descriptor
            if (
                tail.commit_index != self.position.commit_index
                or authority_commit_digest(tail) != self.position.commit_digest
            ):
                raise ValueError("committed entry tail must match the current position")
        return self

    @classmethod
    def bootstrap(
        cls,
        position: AuthorityCommitPosition,
        membership: AuthorityMembership,
    ) -> "AuthorityConsensusState":
        """Create the initial durable consensus state for an enrolled member.

        Args:
            position: Deterministic cluster bootstrap position.
            membership: Initial committed voter and learner configuration.

        Returns:
            Consensus state ready to promise the next authority index.
        """

        return cls(
            bootstrap_position=position,
            bootstrap_membership=membership,
            position=position,
            active_membership=membership,
        )


@final
class AuthorityConsensusSnapshot(FrozenModel):
    """Versioned repository snapshot used for local compare-and-set writes."""

    revision: int = Field(
        ge=0,
        description="Local repository revision changed by every durable transition.",
    )
    state: AuthorityConsensusState = Field(
        description="Durable authority consensus state at this revision.",
    )


class AuthorityConsensusRepository(Protocol):
    """Injectable durable storage boundary for one authority participant."""

    def load(self) -> AuthorityConsensusSnapshot:
        """Return the latest durable consensus snapshot."""

        ...

    def compare_and_set(
        self,
        expected_revision: int,
        state: AuthorityConsensusState,
    ) -> AuthorityConsensusSnapshot:
        """Persist state only if the current revision equals the expectation.

        Args:
            expected_revision: Revision read before computing the transition.
            state: Complete replacement state to persist atomically.

        Returns:
            Newly persisted snapshot with its incremented revision.

        Raises:
            AuthorityConsensusConflictError: Another local transition won the race.
        """

        ...


class AuthorityTransport(Protocol):
    """Asynchronous delivery boundary for signed authority envelopes."""

    async def send(self, envelope: AuthorityNetworkEnvelope) -> None:
        """Deliver one signed envelope to its stable target."""

        ...

    async def receive(self) -> AuthorityNetworkEnvelope:
        """Return the next envelope addressed to the local participant."""

        ...


@final
class AuthorityPrepareMessage(FrozenModel):
    """Phase-one request asking voters to promise one ordered ballot."""

    kind: Literal["prepare"] = Field(
        default="prepare",
        description="Wire discriminator for a phase-one request.",
    )
    cluster_id: UUID4 = Field(description="Cluster owning this consensus round.")
    request_id: UUID4 = Field(description="Unique identifier shared by both phases.")
    ballot: AuthorityBallot = Field(
        description="Ordered leadership attempt to promise."
    )
    requested_descriptor: AuthorityCommitDescriptor = Field(
        description="Caller's desired next value, subject to accepted-value recovery."
    )
    memberships: tuple[AuthorityMembership, ...] = Field(
        min_length=1,
        max_length=2,
        description="Active or joint configurations required by the desired value.",
    )

    @model_validator(mode="after")
    def _request_is_internally_bound(self) -> "AuthorityPrepareMessage":
        """Bind cluster, ballot, descriptor, and memberships before delivery."""

        _validate_message_descriptor(
            self.cluster_id,
            self.ballot,
            self.requested_descriptor,
            self.memberships,
        )
        return self


@final
class AuthorityPromiseMessage(FrozenModel):
    """Durable phase-one promise plus any value accepted at this index."""

    kind: Literal["promise"] = Field(
        default="promise",
        description="Wire discriminator for a phase-one response.",
    )
    cluster_id: UUID4 = Field(description="Cluster owning this consensus round.")
    request_id: UUID4 = Field(description="Prepare request being answered.")
    ballot: AuthorityBallot = Field(description="Ballot durably promised by the voter.")
    commit_index: int = Field(
        ge=2,
        description="Next authority index governed by the promise.",
    )
    accepted_ballot: AuthorityBallot | None = Field(
        default=None,
        description="Prior accepted ballot returned for leader-failure recovery.",
    )
    accepted_descriptor: AuthorityCommitDescriptor | None = Field(
        default=None,
        description="Value accepted under the returned prior ballot.",
    )
    accepted_memberships: tuple[AuthorityMembership, ...] | None = Field(
        default=None,
        description="Configurations required to certify the accepted value.",
    )

    @model_validator(mode="after")
    def _accepted_value_is_complete(self) -> "AuthorityPromiseMessage":
        """Keep accepted recovery evidence structurally complete."""

        parts = (
            self.accepted_ballot,
            self.accepted_descriptor,
            self.accepted_memberships,
        )
        if any(part is None for part in parts) != all(part is None for part in parts):
            raise ValueError("promise accepted value must be wholly present or absent")
        if self.accepted_descriptor is not None:
            assert self.accepted_ballot is not None
            assert self.accepted_memberships is not None
            _validate_message_descriptor(
                self.cluster_id,
                self.accepted_ballot,
                self.accepted_descriptor,
                self.accepted_memberships,
            )
            if self.accepted_descriptor.commit_index != self.commit_index:
                raise ValueError("promise accepted value names another commit index")
        return self


@final
class AuthorityAcceptMessage(FrozenModel):
    """Phase-two request asking voters to durably accept and sign one value."""

    kind: Literal["accept"] = Field(
        default="accept",
        description="Wire discriminator for a phase-two request.",
    )
    cluster_id: UUID4 = Field(description="Cluster owning this consensus round.")
    request_id: UUID4 = Field(description="Prepare round that selected this value.")
    ballot: AuthorityBallot = Field(description="Ballot under which to accept.")
    descriptor: AuthorityCommitDescriptor = Field(
        description="Recovered or requested descriptor selected after prepare quorum."
    )
    memberships: tuple[AuthorityMembership, ...] = Field(
        min_length=1,
        max_length=2,
        description="Configurations whose voters must sign the descriptor.",
    )

    @model_validator(mode="after")
    def _request_is_internally_bound(self) -> "AuthorityAcceptMessage":
        """Bind cluster, ballot, descriptor, and memberships before delivery."""

        _validate_message_descriptor(
            self.cluster_id,
            self.ballot,
            self.descriptor,
            self.memberships,
        )
        return self


@final
class AuthorityAcceptedMessage(FrozenModel):
    """Phase-two response containing one descriptor-bound voter signature."""

    kind: Literal["accepted"] = Field(
        default="accepted",
        description="Wire discriminator for a phase-two response.",
    )
    cluster_id: UUID4 = Field(description="Cluster owning this consensus round.")
    request_id: UUID4 = Field(description="Accept request being answered.")
    ballot: AuthorityBallot = Field(
        description="Ballot under which the vote persisted."
    )
    vote: AuthorityVote = Field(description="Descriptor-bound member signature.")


@final
class AuthorityCommitMessage(FrozenModel):
    """Certified commit announcement for replicas and learners."""

    kind: Literal["commit"] = Field(
        default="commit",
        description="Wire discriminator for a certified commit announcement.",
    )
    cluster_id: UUID4 = Field(description="Cluster owning the committed entry.")
    request_id: UUID4 = Field(description="Round that produced the certificate.")
    entry: AuthorityCommittedEntry = Field(
        description="Public certificate and memberships replicas must verify."
    )

    @model_validator(mode="after")
    def _entry_belongs_to_cluster(self) -> "AuthorityCommitMessage":
        """Prevent an envelope from naming a different entry cluster."""

        if self.entry.certificate.descriptor.cluster_id != self.cluster_id:
            raise ValueError("authority commit entry belongs to another cluster")
        return self


@final
class AuthorityCatchUpRequestMessage(FrozenModel):
    """Request a bounded suffix after the requester's current commit index."""

    kind: Literal["catch_up_request"] = Field(
        default="catch_up_request",
        description="Wire discriminator for a certified-suffix request.",
    )
    cluster_id: UUID4 = Field(description="Cluster whose log suffix is requested.")
    request_id: UUID4 = Field(description="Identifier correlating the suffix response.")
    after_commit_index: int = Field(
        ge=1,
        description="Last certified index already held by the requester.",
    )


@final
class AuthorityCatchUpResponseMessage(FrozenModel):
    """Bounded, contiguous certified log suffix returned to one replica."""

    kind: Literal["catch_up_response"] = Field(
        default="catch_up_response",
        description="Wire discriminator for a certified-suffix response.",
    )
    cluster_id: UUID4 = Field(description="Cluster owning every returned entry.")
    request_id: UUID4 = Field(description="Catch-up request being answered.")
    entries: tuple[AuthorityCommittedEntry, ...] = Field(
        max_length=_MAX_CATCH_UP_ENTRIES,
        description="Bounded ascending certified suffix with no secret payloads.",
    )
    has_more: bool = Field(
        default=False,
        description="Whether another bounded suffix remains after these entries.",
    )

    @model_validator(mode="after")
    def _entries_are_cluster_bound_and_ordered(
        self,
    ) -> "AuthorityCatchUpResponseMessage":
        """Reject mixed-cluster or non-contiguous catch-up responses."""

        indexes = tuple(
            entry.certificate.descriptor.commit_index for entry in self.entries
        )
        if indexes and indexes != tuple(range(indexes[0], indexes[0] + len(indexes))):
            raise ValueError("authority catch-up entries must be contiguous")
        if any(
            entry.certificate.descriptor.cluster_id != self.cluster_id
            for entry in self.entries
        ):
            raise ValueError("authority catch-up entries belong to another cluster")
        if self.has_more and not self.entries:
            raise ValueError("authority catch-up continuation requires progress")
        return self


type AuthorityRejectionCode = Literal[
    "conflicting_accept",
    "membership_mismatch",
    "not_voter",
    "stale_ballot",
    "stale_position",
]


@final
class AuthorityRejectedMessage(FrozenModel):
    """Authenticated fail-closed response to a valid but inadmissible request."""

    kind: Literal["rejected"] = Field(
        default="rejected",
        description="Wire discriminator for a fail-closed response.",
    )
    cluster_id: UUID4 = Field(description="Cluster whose request was rejected.")
    request_id: UUID4 = Field(description="Request being rejected.")
    code: AuthorityRejectionCode = Field(description="Bounded non-secret reason code.")
    promised_ballot: AuthorityBallot | None = Field(
        default=None,
        description="Higher durable promise that fenced a stale request, when present.",
    )
    current_position: AuthorityCommitPosition = Field(
        description="Replica's current certified position for recovery decisions."
    )


type AuthorityProtocolPayload = Annotated[
    AuthorityPrepareMessage
    | AuthorityPromiseMessage
    | AuthorityAcceptMessage
    | AuthorityAcceptedMessage
    | AuthorityCommitMessage
    | AuthorityCatchUpRequestMessage
    | AuthorityCatchUpResponseMessage
    | AuthorityRejectedMessage,
    Field(discriminator="kind"),
]


@final
class AuthorityNetworkEnvelope(CamelCaseModel):
    """Signed node-addressed authority protocol message safe for an opaque bus."""

    model_config = ConfigDict(frozen=True)

    message_id: UUID4 = Field(description="Unique transport replay identifier.")
    source_node_install_id: UUID4 = Field(description="Stable signing member identity.")
    target_node_install_id: UUID4 = Field(
        description="Stable intended recipient identity."
    )
    payload: AuthorityProtocolPayload = Field(description="Typed consensus message.")
    signature: str = Field(
        description="Unpadded URL-safe base64 Ed25519 envelope signature."
    )

    @field_validator("signature")
    @classmethod
    def _signature_is_canonical_ed25519(cls, value: str) -> str:
        """Reject malformed signatures and normalize equivalent encodings."""

        try:
            decoded = _base64url_decode(value)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("signature is not valid URL-safe base64") from exc
        if len(decoded) != 64:
            raise ValueError("signature must contain one Ed25519 signature")
        return _base64url_encode(decoded)


def create_authority_envelope(
    payload: AuthorityProtocolPayload,
    source_node_install_id: UUID,
    target_node_install_id: UUID,
    private_key: bytes,
    *,
    message_id: UUID | None = None,
) -> AuthorityNetworkEnvelope:
    """Sign one node-addressed authority protocol message.

    Args:
        payload: Typed public consensus message.
        source_node_install_id: Stable identity owning the signing key.
        target_node_install_id: Stable recipient identity.
        private_key: Raw Ed25519 authority signing key retained by the source.
        message_id: Optional injected identifier for deterministic tests.

    Returns:
        Signed envelope whose source, target, identifier, and payload are bound.
    """

    if len(private_key) != 32:
        raise ValueError("authority private key must contain 32 bytes")
    selected_message_id = message_id if message_id is not None else uuid4()
    signature = Ed25519PrivateKey.from_private_bytes(private_key).sign(
        _envelope_message(
            selected_message_id,
            source_node_install_id,
            target_node_install_id,
            payload,
        )
    )
    return AuthorityNetworkEnvelope(
        message_id=selected_message_id,
        source_node_install_id=source_node_install_id,
        target_node_install_id=target_node_install_id,
        payload=payload,
        signature=_base64url_encode(signature),
    )


def verify_authority_envelope(
    envelope: AuthorityNetworkEnvelope,
    memberships: tuple[AuthorityMembership, ...],
) -> None:
    """Verify the stable source identity and signature of one envelope.

    Args:
        envelope: Untrusted decoded network message.
        memberships: Locally trusted membership material allowed to authenticate
            the source.

    Raises:
        AuthorityEnvelopeError: The source is unknown, ambiguously keyed, or the
            signature is invalid.
    """

    member_keys: dict[UUID, str] = {}
    key_owners: dict[str, UUID] = {}
    for membership in memberships:
        for member in membership.members:
            member_id = UUID(str(member.node_install_id))
            existing_key = member_keys.get(member_id)
            if existing_key is not None and existing_key != member.public_key:
                raise AuthorityEnvelopeError(
                    "authority source has conflicting membership keys"
                )
            existing_owner = key_owners.get(member.public_key)
            if existing_owner is not None and existing_owner != member_id:
                raise AuthorityEnvelopeError(
                    "authority membership reuses one signing key"
                )
            member_keys[member_id] = member.public_key
            key_owners[member.public_key] = member_id

    source_id = UUID(str(envelope.source_node_install_id))
    public_key_text = member_keys.get(source_id)
    if public_key_text is None:
        raise AuthorityEnvelopeError("authority envelope source is not a member")
    try:
        Ed25519PublicKey.from_public_bytes(_base64url_decode(public_key_text)).verify(
            _base64url_decode(envelope.signature),
            _envelope_message(
                UUID(str(envelope.message_id)),
                source_id,
                UUID(str(envelope.target_node_install_id)),
                envelope.payload,
            ),
        )
    except (InvalidSignature, ValueError) as exc:
        raise AuthorityEnvelopeError("authority envelope signature is invalid") from exc


@final
class AuthorityVoteCollector:
    """Deterministic leader-side collector for one prepare/accept round."""

    def __init__(
        self,
        ballot: AuthorityBallot,
        descriptor: AuthorityCommitDescriptor,
        memberships: tuple[AuthorityMembership, ...],
        previous_position: AuthorityCommitPosition,
        *,
        request_id: UUID | None = None,
    ) -> None:
        """Create a collector without sending or persisting any message.

        Args:
            ballot: Unique proposer ballot for this attempt.
            descriptor: Caller's desired next authority value.
            memberships: Active or joint configuration for that desired value.
            previous_position: Last certified position known by the proposer.
            request_id: Optional deterministic round identifier.
        """

        _validate_membership_order(memberships)
        validate_authority_descriptor(descriptor, memberships, previous_position)
        if descriptor.authority_term != ballot.counter:
            raise ValueError("descriptor term must match the collector ballot")
        if ballot.counter <= previous_position.authority_term:
            raise ValueError("collector ballot must exceed the committed term")
        if UUID(str(ballot.proposer_node_install_id)) not in memberships[0].voter_ids:
            raise ValueError("collector proposer must be an active authority voter")
        self.ballot = ballot
        self.request_id = request_id if request_id is not None else uuid4()
        self.previous_position = previous_position
        self.requested_descriptor = descriptor
        self.requested_memberships = memberships
        self._promises: dict[UUID, AuthorityPromiseMessage] = {}
        self._votes: dict[UUID, AuthorityVote] = {}
        self._selected_descriptor: AuthorityCommitDescriptor | None = None
        self._selected_memberships: tuple[AuthorityMembership, ...] | None = None
        self._rejections: dict[UUID, AuthorityRejectedMessage] = {}

    @property
    def prepare_targets(self) -> frozenset[UUID]:
        """Return all voter identities needed by the requested prepare phase."""

        return _voter_union(self.requested_memberships)

    @property
    def accept_targets(self) -> frozenset[UUID]:
        """Return voter identities for the value selected after prepare."""

        memberships = self._require_selected_memberships()
        return _voter_union(memberships)

    @property
    def prepare_ready(self) -> bool:
        """Return whether promises satisfy every requested membership quorum."""

        return _ids_satisfy_quorums(
            frozenset(self._promises), self.requested_memberships
        )

    @property
    def accept_ready(self) -> bool:
        """Return whether collected votes satisfy every selected quorum."""

        if self._selected_memberships is None:
            return False
        return _ids_satisfy_quorums(frozenset(self._votes), self._selected_memberships)

    @property
    def rejections(self) -> tuple[tuple[UUID, AuthorityRejectedMessage], ...]:
        """Return authenticated rejections sorted by stable member identity."""

        return tuple(sorted(self._rejections.items(), key=lambda item: item[0].int))

    def prepare_message(self) -> AuthorityPrepareMessage:
        """Return the phase-one message to address to every prepare target."""

        return AuthorityPrepareMessage(
            cluster_id=self.requested_descriptor.cluster_id,
            request_id=self.request_id,
            ballot=self.ballot,
            requested_descriptor=self.requested_descriptor,
            memberships=self.requested_memberships,
        )

    def record_prepare_response(self, envelope: AuthorityNetworkEnvelope) -> None:
        """Authenticate and record one promise or rejection.

        Args:
            envelope: Response addressed to the ballot proposer.

        Raises:
            AuthorityEnvelopeError: Envelope authentication or routing fails.
            AuthorityConsensusError: Response identifies another round.
        """

        self._verify_response_envelope(envelope, self.requested_memberships)
        payload = envelope.payload
        source_id = UUID(str(envelope.source_node_install_id))
        if isinstance(payload, AuthorityRejectedMessage):
            self._rejections[source_id] = payload
            return
        if not isinstance(payload, AuthorityPromiseMessage):
            raise AuthorityConsensusError("prepare round received a non-promise")
        if payload.ballot != self.ballot:
            raise AuthorityConsensusError("promise belongs to another ballot")
        if payload.commit_index != self.previous_position.commit_index + 1:
            raise AuthorityConsensusError("promise names another commit index")
        if source_id not in self.prepare_targets:
            raise AuthorityConsensusError("promise came from a non-voter")
        self._promises[source_id] = payload

    def accept_message(self) -> AuthorityAcceptMessage:
        """Select the Paxos-safe value and return its phase-two request.

        Returns:
            Accept request preserving the highest previously accepted value, or
            the caller's requested value when no accepted evidence exists.

        Raises:
            AuthorityConsensusError: Prepare quorum is absent or contradictory.
        """

        if not self.prepare_ready:
            raise AuthorityConsensusError("prepare quorum is not available")
        accepted = [
            promise
            for promise in self._promises.values()
            if promise.accepted_ballot is not None
        ]
        if not accepted:
            selected_descriptor = self.requested_descriptor
            selected_memberships = self.requested_memberships
        else:
            highest_ballot = max(
                (
                    promise.accepted_ballot
                    for promise in accepted
                    if promise.accepted_ballot is not None
                ),
                key=_ballot_key,
            )
            highest = [
                promise
                for promise in accepted
                if promise.accepted_ballot == highest_ballot
            ]
            descriptor_digests = {
                authority_commit_digest(promise.accepted_descriptor)
                for promise in highest
                if promise.accepted_descriptor is not None
            }
            if len(descriptor_digests) != 1:
                raise AuthorityConsensusError(
                    "same-ballot acceptors reported conflicting authority values"
                )
            recovered = highest[0]
            assert recovered.accepted_descriptor is not None
            assert recovered.accepted_memberships is not None
            selected_memberships = recovered.accepted_memberships
            _validate_membership_order(selected_memberships)
            selected_descriptor = recovered.accepted_descriptor
            validate_authority_descriptor(
                selected_descriptor,
                selected_memberships,
                self.previous_position,
            )
        self._selected_descriptor = selected_descriptor
        self._selected_memberships = selected_memberships
        return AuthorityAcceptMessage(
            cluster_id=selected_descriptor.cluster_id,
            request_id=self.request_id,
            ballot=self.ballot,
            descriptor=selected_descriptor,
            memberships=selected_memberships,
        )

    def record_accept_response(self, envelope: AuthorityNetworkEnvelope) -> None:
        """Authenticate and record one accepted vote or rejection.

        Args:
            envelope: Response addressed to the ballot proposer.

        Raises:
            AuthorityEnvelopeError: Envelope authentication or routing fails.
            AuthorityConsensusError: Response does not match the selected value.
        """

        memberships = self._require_selected_memberships()
        self._require_selected_descriptor()
        self._verify_response_envelope(envelope, memberships)
        payload = envelope.payload
        source_id = UUID(str(envelope.source_node_install_id))
        if isinstance(payload, AuthorityRejectedMessage):
            self._rejections[source_id] = payload
            return
        if not isinstance(payload, AuthorityAcceptedMessage):
            raise AuthorityConsensusError("accept round received a non-vote")
        if payload.ballot != self.ballot:
            raise AuthorityConsensusError("accepted vote belongs to another ballot")
        if UUID(str(payload.vote.node_install_id)) != source_id:
            raise AuthorityConsensusError("accepted vote claims another member")
        if source_id not in self.accept_targets:
            raise AuthorityConsensusError("accepted vote came from a non-voter")
        existing_vote = self._votes.get(source_id)
        if existing_vote is not None and existing_vote != payload.vote:
            raise AuthorityConsensusError("member returned conflicting accepted votes")
        self._votes[source_id] = payload.vote

    def certificate(self) -> AuthorityQuorumCertificate:
        """Return and verify the completed quorum certificate.

        Raises:
            AuthorityConsensusError: Phase two has not reached every quorum.
            AuthorityCertificateError: A collected signature is invalid.
        """

        if not self.accept_ready:
            raise AuthorityConsensusError("accept quorum is not available")
        descriptor = self._require_selected_descriptor()
        memberships = self._require_selected_memberships()
        certificate = AuthorityQuorumCertificate(
            descriptor=descriptor,
            votes=tuple(
                self._votes[member_id]
                for member_id in sorted(self._votes, key=lambda item: item.int)
            ),
        )
        verify_quorum_certificate(certificate, memberships, self.previous_position)
        return certificate

    def committed_entry(self) -> AuthorityCommittedEntry:
        """Return the certified entry selected by this completed round."""

        return AuthorityCommittedEntry(
            certificate=self.certificate(),
            memberships=self._require_selected_memberships(),
        )

    def _verify_response_envelope(
        self,
        envelope: AuthorityNetworkEnvelope,
        memberships: tuple[AuthorityMembership, ...],
    ) -> None:
        """Verify common routing, round, cluster, and signature bindings."""

        if UUID(str(envelope.target_node_install_id)) != UUID(
            str(self.ballot.proposer_node_install_id)
        ):
            raise AuthorityEnvelopeError("authority response targets another proposer")
        verify_authority_envelope(envelope, memberships)
        payload = envelope.payload
        if payload.cluster_id != self.requested_descriptor.cluster_id:
            raise AuthorityConsensusError(
                "authority response belongs to another cluster"
            )
        if payload.request_id != self.request_id:
            raise AuthorityConsensusError("authority response belongs to another round")

    def _require_selected_descriptor(self) -> AuthorityCommitDescriptor:
        """Return the selected descriptor or fail before phase two."""

        if self._selected_descriptor is None:
            raise AuthorityConsensusError("prepare phase has not selected a value")
        return self._selected_descriptor

    def _require_selected_memberships(self) -> tuple[AuthorityMembership, ...]:
        """Return selected memberships or fail before phase two."""

        if self._selected_memberships is None:
            raise AuthorityConsensusError("prepare phase has not selected a value")
        return self._selected_memberships


@final
class AuthorityConsensusParticipant:
    """Storage-backed acceptor and replica for one stable authority member."""

    def __init__(
        self,
        node_install_id: UUID,
        private_key: bytes,
        repository: AuthorityConsensusRepository,
    ) -> None:
        """Create a participant around injected signing and durable storage.

        Args:
            node_install_id: Stable member identity owning this process.
            private_key: Raw Ed25519 signing key retained by this host.
            repository: Durable compare-and-set consensus state boundary.
        """

        if len(private_key) != 32:
            raise ValueError("authority private key must contain 32 bytes")
        self.node_install_id = node_install_id
        self._private_key = private_key
        self._repository = repository

    @property
    def snapshot(self) -> AuthorityConsensusSnapshot:
        """Return this participant's latest durable state."""

        return self._repository.load()

    def envelope_for(
        self,
        target_node_install_id: UUID,
        payload: AuthorityProtocolPayload,
        *,
        message_id: UUID | None = None,
    ) -> AuthorityNetworkEnvelope:
        """Sign one outbound protocol payload for a stable target."""

        return create_authority_envelope(
            payload,
            self.node_install_id,
            target_node_install_id,
            self._private_key,
            message_id=message_id,
        )

    def handle(
        self,
        envelope: AuthorityNetworkEnvelope,
    ) -> tuple[AuthorityNetworkEnvelope, ...]:
        """Process one authenticated message and return deterministic responses.

        Args:
            envelope: Untrusted message delivered by the injected transport.

        Returns:
            Zero or one signed response envelope. A commit gap returns a catch-up
            request rather than applying an out-of-order certificate.

        Raises:
            AuthorityEnvelopeError: Routing or source authentication fails.
            AuthorityConsensusConflictError: Local state changed concurrently.
        """

        if UUID(str(envelope.target_node_install_id)) != self.node_install_id:
            raise AuthorityEnvelopeError("authority envelope targets another member")
        snapshot = self._repository.load()
        state = snapshot.state
        if envelope.payload.cluster_id != state.position.cluster_id:
            raise AuthorityEnvelopeError(
                "authority envelope belongs to another cluster"
            )
        verify_authority_envelope(envelope, (state.active_membership,))
        payload = envelope.payload
        if isinstance(payload, AuthorityPrepareMessage):
            response = self._handle_prepare(snapshot, envelope, payload)
            return (response,)
        if isinstance(payload, AuthorityAcceptMessage):
            response = self._handle_accept(snapshot, envelope, payload)
            return (response,)
        if isinstance(payload, AuthorityCommitMessage):
            return self._handle_commit(snapshot, envelope, payload)
        if isinstance(payload, AuthorityCatchUpRequestMessage):
            response = self._handle_catch_up_request(snapshot, envelope, payload)
            return (response,)
        if isinstance(payload, AuthorityCatchUpResponseMessage):
            return self._handle_catch_up_response(snapshot, envelope, payload)
        raise AuthorityEnvelopeError(
            "authority participant received a leader-side response message"
        )

    def _handle_prepare(
        self,
        snapshot: AuthorityConsensusSnapshot,
        envelope: AuthorityNetworkEnvelope,
        payload: AuthorityPrepareMessage,
    ) -> AuthorityNetworkEnvelope:
        """Persist one ballot promise before returning accepted evidence."""

        state = snapshot.state
        rejection = self._validate_phase_request(
            state,
            envelope,
            payload.request_id,
            payload.ballot,
            payload.requested_descriptor,
            payload.memberships,
        )
        if rejection is not None:
            return rejection
        if state.promised_ballot is not None and _ballot_key(
            payload.ballot
        ) < _ballot_key(state.promised_ballot):
            return self._rejection(
                envelope,
                payload.request_id,
                "stale_ballot",
                state,
            )
        next_state = _state_with_promise(state, payload.ballot)
        self._persist(snapshot.revision, next_state)
        promise = AuthorityPromiseMessage(
            cluster_id=state.position.cluster_id,
            request_id=payload.request_id,
            ballot=payload.ballot,
            commit_index=state.position.commit_index + 1,
            accepted_ballot=state.accepted_ballot,
            accepted_descriptor=state.accepted_descriptor,
            accepted_memberships=state.accepted_memberships,
        )
        return self.envelope_for(envelope.source_node_install_id, promise)

    def _handle_accept(
        self,
        snapshot: AuthorityConsensusSnapshot,
        envelope: AuthorityNetworkEnvelope,
        payload: AuthorityAcceptMessage,
    ) -> AuthorityNetworkEnvelope:
        """Durably accept one value before signing its certificate vote."""

        state = snapshot.state
        rejection = self._validate_phase_request(
            state,
            envelope,
            payload.request_id,
            payload.ballot,
            payload.descriptor,
            payload.memberships,
        )
        if rejection is not None:
            return rejection
        if state.promised_ballot is not None and _ballot_key(
            payload.ballot
        ) < _ballot_key(state.promised_ballot):
            return self._rejection(
                envelope,
                payload.request_id,
                "stale_ballot",
                state,
            )
        if (
            state.accepted_ballot == payload.ballot
            and state.accepted_descriptor is not None
            and authority_commit_digest(state.accepted_descriptor)
            != authority_commit_digest(payload.descriptor)
        ):
            return self._rejection(
                envelope,
                payload.request_id,
                "conflicting_accept",
                state,
            )
        next_state = _state_with_accept(
            state,
            payload.ballot,
            payload.descriptor,
            payload.memberships,
        )
        self._persist(snapshot.revision, next_state)
        vote = create_authority_vote(
            payload.descriptor,
            self.node_install_id,
            self._private_key,
        )
        accepted = AuthorityAcceptedMessage(
            cluster_id=state.position.cluster_id,
            request_id=payload.request_id,
            ballot=payload.ballot,
            vote=vote,
        )
        return self.envelope_for(envelope.source_node_install_id, accepted)

    def _handle_commit(
        self,
        snapshot: AuthorityConsensusSnapshot,
        envelope: AuthorityNetworkEnvelope,
        payload: AuthorityCommitMessage,
    ) -> tuple[AuthorityNetworkEnvelope, ...]:
        """Apply one contiguous certificate or request the missing suffix."""

        descriptor = payload.entry.certificate.descriptor
        if descriptor.commit_index > snapshot.state.position.commit_index + 1:
            request = AuthorityCatchUpRequestMessage(
                cluster_id=snapshot.state.position.cluster_id,
                request_id=payload.request_id,
                after_commit_index=snapshot.state.position.commit_index,
            )
            return (self.envelope_for(envelope.source_node_install_id, request),)
        self._apply_entry(snapshot, payload.entry)
        return ()

    def _handle_catch_up_request(
        self,
        snapshot: AuthorityConsensusSnapshot,
        envelope: AuthorityNetworkEnvelope,
        payload: AuthorityCatchUpRequestMessage,
    ) -> AuthorityNetworkEnvelope:
        """Return a bounded certified suffix without any secret payload."""

        available = tuple(
            entry
            for entry in snapshot.state.committed_entries
            if entry.certificate.descriptor.commit_index > payload.after_commit_index
        )
        entries = available[:_MAX_CATCH_UP_ENTRIES]
        response = AuthorityCatchUpResponseMessage(
            cluster_id=snapshot.state.position.cluster_id,
            request_id=payload.request_id,
            entries=entries,
            has_more=len(available) > len(entries),
        )
        return self.envelope_for(envelope.source_node_install_id, response)

    def _handle_catch_up_response(
        self,
        snapshot: AuthorityConsensusSnapshot,
        envelope: AuthorityNetworkEnvelope,
        payload: AuthorityCatchUpResponseMessage,
    ) -> tuple[AuthorityNetworkEnvelope, ...]:
        """Apply one certified page and request the next page when advertised."""

        current = snapshot
        for entry in payload.entries:
            before = current.state.position.commit_index
            self._apply_entry(current, entry)
            current = self._repository.load()
            if current.state.position.commit_index == before:
                continue
        if not payload.has_more:
            return ()
        request = AuthorityCatchUpRequestMessage(
            cluster_id=current.state.position.cluster_id,
            request_id=uuid4(),
            after_commit_index=current.state.position.commit_index,
        )
        return (self.envelope_for(envelope.source_node_install_id, request),)

    def _apply_entry(
        self,
        snapshot: AuthorityConsensusSnapshot,
        entry: AuthorityCommittedEntry,
    ) -> None:
        """Verify continuity and persist one idempotent certified transition."""

        state = snapshot.state
        descriptor = entry.certificate.descriptor
        if descriptor.commit_index <= state.position.commit_index:
            if (
                descriptor.commit_index == state.position.commit_index
                and authority_commit_digest(descriptor) == state.position.commit_digest
            ):
                return
            raise AuthorityConsensusError(
                "authority catch-up attempted to replace committed history"
            )
        _validate_transition_memberships(state.active_membership, entry.memberships)
        position = verify_quorum_certificate(
            entry.certificate,
            entry.memberships,
            state.position,
        )
        next_membership = (
            max(entry.memberships, key=lambda membership: membership.generation)
            if len(entry.memberships) == 2
            else state.active_membership
        )
        next_state = AuthorityConsensusState(
            bootstrap_position=state.bootstrap_position,
            bootstrap_membership=state.bootstrap_membership,
            position=position,
            active_membership=next_membership,
            committed_entries=tuple([*state.committed_entries, entry]),
        )
        self._persist(snapshot.revision, next_state)

    def _validate_phase_request(
        self,
        state: AuthorityConsensusState,
        envelope: AuthorityNetworkEnvelope,
        request_id: UUID,
        ballot: AuthorityBallot,
        descriptor: AuthorityCommitDescriptor,
        memberships: tuple[AuthorityMembership, ...],
    ) -> AuthorityNetworkEnvelope | None:
        """Return an authenticated rejection when a phase request cannot proceed."""

        if UUID(str(ballot.proposer_node_install_id)) != UUID(
            str(envelope.source_node_install_id)
        ):
            raise AuthorityEnvelopeError(
                "authority ballot proposer does not sign request"
            )
        try:
            _validate_transition_memberships(state.active_membership, memberships)
            validate_authority_descriptor(descriptor, memberships, state.position)
        except AuthorityCertificateError:
            return self._rejection(
                envelope,
                request_id,
                "membership_mismatch",
                state,
            )
        if descriptor.authority_term > ballot.counter:
            return self._rejection(
                envelope,
                request_id,
                "stale_ballot",
                state,
            )
        if ballot.counter <= state.position.authority_term:
            return self._rejection(
                envelope,
                request_id,
                "stale_ballot",
                state,
            )
        if UUID(str(envelope.source_node_install_id)) not in (
            state.active_membership.voter_ids
        ):
            return self._rejection(
                envelope,
                request_id,
                "not_voter",
                state,
            )
        if self.node_install_id not in _voter_union(memberships):
            return self._rejection(
                envelope,
                request_id,
                "not_voter",
                state,
            )
        return None

    def _rejection(
        self,
        envelope: AuthorityNetworkEnvelope,
        request_id: UUID,
        code: AuthorityRejectionCode,
        state: AuthorityConsensusState,
    ) -> AuthorityNetworkEnvelope:
        """Sign one bounded rejection without echoing untrusted payload data."""

        rejection = AuthorityRejectedMessage(
            cluster_id=state.position.cluster_id,
            request_id=request_id,
            code=code,
            promised_ballot=state.promised_ballot,
            current_position=state.position,
        )
        return self.envelope_for(envelope.source_node_install_id, rejection)

    def _persist(
        self,
        expected_revision: int,
        state: AuthorityConsensusState,
    ) -> None:
        """Persist a computed transition through the repository CAS boundary."""

        try:
            self._repository.compare_and_set(expected_revision, state)
        except AuthorityConsensusConflictError:
            raise
        except Exception as exc:
            raise AuthorityConsensusConflictError(
                "authority consensus repository rejected local CAS"
            ) from exc


def _state_with_promise(
    state: AuthorityConsensusState,
    ballot: AuthorityBallot,
) -> AuthorityConsensusState:
    """Return state with a durable promise and unchanged accepted evidence."""

    return AuthorityConsensusState(
        bootstrap_position=state.bootstrap_position,
        bootstrap_membership=state.bootstrap_membership,
        position=state.position,
        active_membership=state.active_membership,
        promised_ballot=ballot,
        accepted_ballot=state.accepted_ballot,
        accepted_descriptor=state.accepted_descriptor,
        accepted_memberships=state.accepted_memberships,
        committed_entries=state.committed_entries,
    )


def _state_with_accept(
    state: AuthorityConsensusState,
    ballot: AuthorityBallot,
    descriptor: AuthorityCommitDescriptor,
    memberships: tuple[AuthorityMembership, ...],
) -> AuthorityConsensusState:
    """Return state with one ballot and descriptor durably accepted."""

    return AuthorityConsensusState(
        bootstrap_position=state.bootstrap_position,
        bootstrap_membership=state.bootstrap_membership,
        position=state.position,
        active_membership=state.active_membership,
        promised_ballot=ballot,
        accepted_ballot=ballot,
        accepted_descriptor=descriptor,
        accepted_memberships=memberships,
        committed_entries=state.committed_entries,
    )


def _validate_transition_memberships(
    active_membership: AuthorityMembership,
    memberships: tuple[AuthorityMembership, ...],
) -> None:
    """Require a single active config or a consecutive active-to-new transition."""

    _validate_membership_order(memberships)
    if authority_membership_digest(memberships[0]) != authority_membership_digest(
        active_membership
    ):
        raise AuthorityCertificateError(
            "authority proposal does not start from active membership"
        )


def _validate_message_descriptor(
    cluster_id: UUID,
    ballot: AuthorityBallot,
    descriptor: AuthorityCommitDescriptor,
    memberships: tuple[AuthorityMembership, ...],
) -> None:
    """Validate internal wire bindings that do not require local log state."""

    _validate_membership_order(memberships)
    if UUID(str(descriptor.cluster_id)) != UUID(str(cluster_id)):
        raise ValueError("authority descriptor belongs to another cluster")
    if descriptor.authority_term > ballot.counter:
        raise ValueError("authority descriptor term exceeds its ballot")
    expected_digests = tuple(
        sorted(authority_membership_digest(membership) for membership in memberships)
    )
    if descriptor.required_membership_digests != expected_digests:
        raise ValueError("authority descriptor names different memberships")


def _validate_membership_order(
    memberships: tuple[AuthorityMembership, ...],
) -> None:
    """Require one config or two strictly generation-ordered configs."""

    if not 1 <= len(memberships) <= 2:
        raise AuthorityCertificateError(
            "authority proposal requires one or two memberships"
        )
    ordered = tuple(sorted(memberships, key=lambda item: item.generation))
    if ordered != memberships:
        raise AuthorityCertificateError(
            "authority proposal memberships must be generation ordered"
        )
    if len(memberships) == 2 and (
        memberships[1].generation != memberships[0].generation + 1
    ):
        raise AuthorityCertificateError(
            "joint authority memberships must be consecutive generations"
        )


def _voter_union(memberships: Iterable[AuthorityMembership]) -> frozenset[UUID]:
    """Return the union of stable voter identities across configurations."""

    voters: set[UUID] = set()
    for membership in memberships:
        voters.update(membership.voter_ids)
    return frozenset(voters)


def _ids_satisfy_quorums(
    member_ids: frozenset[UUID],
    memberships: tuple[AuthorityMembership, ...],
) -> bool:
    """Return whether identities satisfy every strict-majority requirement."""

    return all(
        len(member_ids.intersection(membership.voter_ids)) >= membership.quorum_size
        for membership in memberships
    )


def _ballot_key(ballot: AuthorityBallot) -> tuple[int, int]:
    """Return the deterministic total ordering for concurrent ballots."""

    return ballot.counter, UUID(str(ballot.proposer_node_install_id)).int


def _base64url_encode(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    """Decode a padded or unpadded URL-safe base64 value."""

    padding = "=" * (-len(value) % 4)
    return base64.b64decode(
        f"{value}{padding}",
        altchars=b"-_",
        validate=True,
    )


def _canonical_json(payload: object) -> bytes:
    """Serialize finite signed protocol metadata deterministically."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _envelope_message(
    message_id: UUID,
    source_node_install_id: UUID,
    target_node_install_id: UUID,
    payload: AuthorityProtocolPayload,
) -> bytes:
    """Build the domain-separated bytes signed by an authority envelope."""

    return _AUTHORITY_ENVELOPE_CONTEXT + _canonical_json(
        {
            "messageId": str(message_id),
            "payload": payload.model_dump(mode="json", by_alias=True),
            "sourceNodeInstallId": str(source_node_install_id),
            "targetNodeInstallId": str(target_node_install_id),
        }
    )
