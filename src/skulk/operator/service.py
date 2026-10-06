# pyright: reportUnusedFunction=false
"""Bounded runtime lifecycle for the dormant operator-authority protocol.

The consensus types remain deterministic and storage-backed. This module adds
the asynchronous machinery that receives participant messages and drives one
serialized proposal at a time without exposing an API route or starting the
service from :class:`skulk.main.Node`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, final
from uuid import UUID

from anyio import Event, Lock, WouldBlock, create_task_group, move_on_after, sleep
from loguru import logger
from pydantic import Field, model_validator

from skulk.operator.consensus import (
    AuthorityAcceptedMessage,
    AuthorityAcceptMessage,
    AuthorityBallot,
    AuthorityCommitMessage,
    AuthorityCommittedEntry,
    AuthorityConsensusError,
    AuthorityConsensusParticipant,
    AuthorityNetworkEnvelope,
    AuthorityPrepareMessage,
    AuthorityPromiseMessage,
    AuthorityRejectedMessage,
    AuthorityTransport,
    AuthorityVoteCollector,
)
from skulk.operator.replication import (
    AuthorityCertificateError,
    AuthorityCommitDescriptor,
    AuthorityMembership,
    authority_membership_digest,
)
from skulk.utils.channels import channel
from skulk.utils.pydantic_ext import FrozenModel

_DEFAULT_RUNTIME_QUEUE_CAPACITY = 128
_DEFAULT_PHASE_TIMEOUT_SECONDS = 2.0
_DEFAULT_RETRY_BACKOFF_SECONDS = 0.05
_DEFAULT_MAX_ATTEMPTS = 3


class AuthorityServiceError(RuntimeError):
    """Base class for fail-closed authority runtime failures."""


class AuthorityServiceNotRunningError(AuthorityServiceError):
    """Raised when a proposal is attempted outside the service lifecycle."""


class AuthorityProposalBusyError(AuthorityServiceError):
    """Raised when another local proposal already owns the single admission slot."""


class AuthorityProposalUnavailableError(AuthorityServiceError):
    """Raised when a proposal cannot reach a safe quorum within bounded retries."""


class AuthorityProposalConfigurationError(AuthorityServiceError):
    """Raised when authenticated voters reject the requested configuration."""


@final
class AuthorityTransitionRequest(FrozenModel):
    """Non-secret semantic intent for one replicated authority transition."""

    record_type: str = Field(
        min_length=1,
        max_length=80,
        description="Bounded non-secret authority record type.",
    )
    record_id: str = Field(
        min_length=1,
        max_length=160,
        description="Stable opaque record identity; never secret-bearing data.",
    )
    payload_digest: str = Field(
        min_length=1,
        description="Digest of the separately encrypted authority payload.",
    )
    memberships: tuple[AuthorityMembership, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=2,
        description=(
            "Explicit active or joint configurations, or null to use the "
            "participant's current committed membership on each attempt."
        ),
    )

    @model_validator(mode="after")
    def _request_is_canonical(self) -> "AuthorityTransitionRequest":
        """Reject ambiguous record types and unordered explicit memberships."""

        if any(character.isspace() for character in self.record_type):
            raise ValueError("record_type must not contain whitespace")
        if self.memberships is not None:
            ordered = tuple(
                sorted(self.memberships, key=lambda membership: membership.generation)
            )
            if self.memberships != ordered:
                raise ValueError("authority memberships must be generation ordered")
        return self


@final
class AuthorityServiceDiagnostics(FrozenModel):
    """Payload-free counters and queue depths for one authority runtime."""

    running: bool = Field(description="Whether the single-use runtime is active.")
    outbound_queue_depth: int = Field(
        ge=0,
        description="Signed envelopes awaiting transport delivery.",
    )
    response_queue_depth: int = Field(
        ge=0,
        description="Leader-side responses awaiting the active proposal.",
    )
    rejected_envelopes: int = Field(
        ge=0,
        description="Authenticated or routing-invalid envelopes rejected locally.",
    )
    dropped_outbound_envelopes: int = Field(
        ge=0,
        description="Responses dropped because bounded outbound admission was full.",
    )
    dropped_response_envelopes: int = Field(
        ge=0,
        description="Leader responses dropped because bounded admission was full.",
    )
    stale_response_envelopes: int = Field(
        ge=0,
        description="Late responses ignored after their proposal attempt ended.",
    )
    completed_proposals: int = Field(
        ge=0,
        description="Requested transitions committed by this runtime.",
    )
    failed_proposals: int = Field(
        ge=0,
        description="Locally admitted proposals that failed closed.",
    )


type _ProposalPhase = Literal["prepare", "accept"]


class _RetryAuthorityRoundError(RuntimeError):
    """Internal signal that a higher ballot or timeout requires another attempt."""


@final
class AuthorityConsensusService:
    """Run one bounded authority participant and serialize local proposals.

    The service is deliberately not wired into node startup yet. Production
    activation depends on the later OS-protected key, encrypted payload, and
    gateway-fencing slices. A caller must run :meth:`run` in a task group before
    invoking :meth:`propose`.
    """

    def __init__(
        self,
        participant: AuthorityConsensusParticipant,
        transport: AuthorityTransport,
        *,
        queue_capacity: int = _DEFAULT_RUNTIME_QUEUE_CAPACITY,
        phase_timeout_seconds: float = _DEFAULT_PHASE_TIMEOUT_SECONDS,
        retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        """Create a dormant authority runtime around injected effects.

        Args:
            participant: Storage-backed deterministic consensus participant.
            transport: Signed-envelope delivery boundary.
            queue_capacity: Maximum pending outbound and leader-response frames.
            phase_timeout_seconds: Deadline for each prepare and accept phase.
            retry_backoff_seconds: Base delay between failed ballot attempts.
            max_attempts: Maximum ballot attempts before failing closed.

        Raises:
            ValueError: A bound or timeout is not positive.
        """

        if queue_capacity < 1:
            raise ValueError("authority queue_capacity must be positive")
        if phase_timeout_seconds <= 0:
            raise ValueError("authority phase_timeout_seconds must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("authority retry_backoff_seconds cannot be negative")
        if max_attempts < 1:
            raise ValueError("authority max_attempts must be positive")
        self._participant = participant
        self._transport = transport
        self._phase_timeout_seconds = phase_timeout_seconds
        self._retry_backoff_seconds = retry_backoff_seconds
        self._max_attempts = max_attempts
        self._outbound_send, self._outbound_receive = channel[
            AuthorityNetworkEnvelope
        ](queue_capacity)
        self._response_send, self._response_receive = channel[
            AuthorityNetworkEnvelope
        ](queue_capacity)
        self._participant_lock = Lock()
        self._proposal_lock = Lock()
        self._started = Event()
        self._running = False
        self._has_run = False
        self._rejected_envelopes = 0
        self._dropped_outbound_envelopes = 0
        self._dropped_response_envelopes = 0
        self._stale_response_envelopes = 0
        self._completed_proposals = 0
        self._failed_proposals = 0

    async def run(self) -> None:
        """Receive, validate, and deliver authority messages until cancelled.

        Raises:
            RuntimeError: This single-use service is started more than once.
        """

        if self._has_run:
            raise RuntimeError("authority consensus service is single-use")
        self._has_run = True
        self._running = True
        self._started.set()
        try:
            async with create_task_group() as task_group:
                task_group.start_soon(self._receive_loop)
                task_group.start_soon(self._outbound_loop)
        finally:
            self._running = False

    async def wait_started(self) -> None:
        """Wait until :meth:`run` has entered its active lifecycle."""

        await self._started.wait()

    def diagnostics(self) -> AuthorityServiceDiagnostics:
        """Return bounded operational counters without authority payload data."""

        return AuthorityServiceDiagnostics(
            running=self._running,
            outbound_queue_depth=self._outbound_send.statistics().current_buffer_used,
            response_queue_depth=self._response_send.statistics().current_buffer_used,
            rejected_envelopes=self._rejected_envelopes,
            dropped_outbound_envelopes=self._dropped_outbound_envelopes,
            dropped_response_envelopes=self._dropped_response_envelopes,
            stale_response_envelopes=self._stale_response_envelopes,
            completed_proposals=self._completed_proposals,
            failed_proposals=self._failed_proposals,
        )

    async def propose(
        self,
        request: AuthorityTransitionRequest,
    ) -> AuthorityCommittedEntry:
        """Commit one transition or fail closed after bounded retries.

        A prior leader's accepted value may be recovered and committed first.
        The service then advances to the next index and retries the caller's
        semantic intent. Concurrent local proposals are rejected rather than
        accumulated in an unbounded waiter queue.

        Args:
            request: Non-secret transition identity and payload digest.

        Returns:
            Certified entry that committed the requested semantic transition.

        Raises:
            AuthorityServiceNotRunningError: The runtime is not active.
            AuthorityProposalBusyError: Another local proposal is active.
            AuthorityProposalConfigurationError: Voters reject the membership.
            AuthorityProposalUnavailableError: A safe quorum is unavailable.
        """

        if not self._running:
            raise AuthorityServiceNotRunningError(
                "authority consensus service is not running"
            )
        try:
            self._proposal_lock.acquire_nowait()
        except WouldBlock as exc:
            raise AuthorityProposalBusyError(
                "another authority proposal is already active"
            ) from exc
        try:
            for attempt in range(self._max_attempts):
                try:
                    entry = await self._attempt(request)
                except _RetryAuthorityRoundError:
                    if attempt + 1 == self._max_attempts:
                        raise AuthorityProposalUnavailableError(
                            "authority proposal exhausted bounded retries"
                        ) from None
                    if self._retry_backoff_seconds > 0:
                        await sleep(self._retry_backoff_seconds * (attempt + 1))
                    continue
                if self._entry_matches_request(entry, request):
                    self._completed_proposals += 1
                    return entry
                if attempt + 1 == self._max_attempts:
                    raise AuthorityProposalUnavailableError(
                        "authority recovery consumed bounded proposal attempts"
                    )
            raise AssertionError("authority proposal attempts were not exhausted")
        except AuthorityServiceError:
            self._failed_proposals += 1
            raise
        except AuthorityConsensusError as exc:
            self._failed_proposals += 1
            raise AuthorityProposalUnavailableError(
                "authority consensus rejected the local proposal"
            ) from exc
        finally:
            self._proposal_lock.release()

    async def _attempt(
        self,
        request: AuthorityTransitionRequest,
    ) -> AuthorityCommittedEntry:
        """Drive one prepare, accept, and local durable commit attempt."""

        snapshot = self._participant.snapshot
        state = snapshot.state
        memberships = (
            request.memberships
            if request.memberships is not None
            else (state.active_membership,)
        )
        promised_counter = (
            state.promised_ballot.counter if state.promised_ballot is not None else 0
        )
        accepted_counter = (
            state.accepted_ballot.counter if state.accepted_ballot is not None else 0
        )
        ballot = AuthorityBallot(
            counter=max(
                state.position.authority_term,
                promised_counter,
                accepted_counter,
            )
            + 1,
            proposer_node_install_id=self._participant.node_install_id,
        )
        try:
            descriptor = AuthorityCommitDescriptor(
                cluster_id=state.position.cluster_id,
                authority_term=ballot.counter,
                commit_index=state.position.commit_index + 1,
                previous_commit_digest=state.position.commit_digest,
                record_type=request.record_type,
                record_id=request.record_id,
                payload_digest=request.payload_digest,
                required_membership_digests=tuple(
                    sorted(authority_membership_digest(item) for item in memberships)
                ),
            )
            collector = AuthorityVoteCollector(
                ballot,
                descriptor,
                memberships,
                state.position,
            )
        except (AuthorityCertificateError, ValueError) as exc:
            raise AuthorityProposalConfigurationError(
                "authority transition does not match committed membership"
            ) from exc
        await self._send_phase(
            collector.prepare_message(),
            collector.prepare_targets,
            collector,
            "prepare",
        )
        accept_message = collector.accept_message()
        await self._send_phase(
            accept_message,
            collector.accept_targets,
            collector,
            "accept",
        )
        entry = collector.committed_entry()
        commit = AuthorityCommitMessage(
            cluster_id=entry.certificate.descriptor.cluster_id,
            request_id=collector.request_id,
            entry=entry,
        )
        await self._commit_locally(commit)
        self._broadcast_commit(commit, entry.memberships)
        return entry

    async def _send_phase(
        self,
        payload: AuthorityPrepareMessage | AuthorityAcceptMessage,
        targets: frozenset[UUID],
        collector: AuthorityVoteCollector,
        phase: _ProposalPhase,
    ) -> None:
        """Address one phase to every voter and await its required quorums."""

        for target in sorted(targets, key=lambda item: item.int):
            envelope = self._participant.envelope_for(target, payload)
            if target == self._participant.node_install_id:
                responses = await self._handle_locally(envelope)
                if len(responses) != 1:
                    raise AuthorityProposalUnavailableError(
                        "local authority voter returned an invalid response count"
                    )
                self._record_phase_response(collector, phase, responses[0])
            else:
                self._offer_outbound(envelope, required=True)
        await self._await_phase_quorum(collector, phase)

    async def _await_phase_quorum(
        self,
        collector: AuthorityVoteCollector,
        phase: _ProposalPhase,
    ) -> None:
        """Collect only the active round until quorum or its deadline."""

        ready = (
            (lambda: collector.prepare_ready)
            if phase == "prepare"
            else (lambda: collector.accept_ready)
        )
        if ready():
            return
        with move_on_after(self._phase_timeout_seconds) as timeout_scope:
            while not ready():
                envelope = await self._response_receive.receive()
                if envelope.payload.request_id != collector.request_id:
                    self._stale_response_envelopes += 1
                    continue
                try:
                    self._record_phase_response(collector, phase, envelope)
                except AuthorityConsensusError:
                    self._rejected_envelopes += 1
                    continue
                self._raise_for_rejection(collector)
        if timeout_scope.cancel_called:
            raise _RetryAuthorityRoundError

    @staticmethod
    def _record_phase_response(
        collector: AuthorityVoteCollector,
        phase: _ProposalPhase,
        envelope: AuthorityNetworkEnvelope,
    ) -> None:
        """Record one phase-specific authenticated response."""

        if phase == "prepare":
            collector.record_prepare_response(envelope)
        else:
            collector.record_accept_response(envelope)

    @staticmethod
    def _raise_for_rejection(collector: AuthorityVoteCollector) -> None:
        """Convert authenticated fail-closed responses into lifecycle decisions."""

        for _, rejection in collector.rejections:
            if rejection.code == "conflicting_accept":
                raise AuthorityProposalConfigurationError(
                    "authority voters rejected the requested configuration"
                )
            if rejection.code in {"stale_ballot", "stale_position"}:
                raise _RetryAuthorityRoundError

    async def _commit_locally(self, commit: AuthorityCommitMessage) -> None:
        """Persist the certified entry on the proposer before reporting success."""

        envelope = self._participant.envelope_for(
            self._participant.node_install_id,
            commit,
        )
        responses = await self._handle_locally(envelope)
        if responses:
            raise AuthorityProposalUnavailableError(
                "local authority commit required unexpected catch-up"
            )

    def _broadcast_commit(
        self,
        commit: AuthorityCommitMessage,
        memberships: tuple[AuthorityMembership, ...],
    ) -> None:
        """Offer a certified commit to every remote voter and learner."""

        for target in self._member_union(memberships):
            if target == self._participant.node_install_id:
                continue
            self._offer_outbound(
                self._participant.envelope_for(target, commit),
                required=False,
            )

    async def _handle_locally(
        self,
        envelope: AuthorityNetworkEnvelope,
    ) -> tuple[AuthorityNetworkEnvelope, ...]:
        """Serialize access to participant state shared with the receive loop."""

        async with self._participant_lock:
            return self._participant.handle(envelope)

    async def _receive_loop(self) -> None:
        """Route leader responses or handle participant messages indefinitely."""

        while True:
            envelope = await self._transport.receive()
            if isinstance(
                envelope.payload,
                (
                    AuthorityPromiseMessage,
                    AuthorityAcceptedMessage,
                    AuthorityRejectedMessage,
                ),
            ):
                try:
                    self._response_send.send_nowait(envelope)
                except WouldBlock:
                    self._dropped_response_envelopes += 1
                continue
            try:
                responses = await self._handle_locally(envelope)
            except AuthorityConsensusError as exc:
                self._rejected_envelopes += 1
                logger.warning(
                    "Rejected operator authority envelope ({})",
                    type(exc).__name__,
                )
                continue
            for response in responses:
                self._offer_outbound(response, required=False)

    async def _outbound_loop(self) -> None:
        """Drain bounded signed-envelope admission into the transport."""

        with self._outbound_receive as outbound_envelopes:
            async for envelope in outbound_envelopes:
                await self._transport.send(envelope)

    def _offer_outbound(
        self,
        envelope: AuthorityNetworkEnvelope,
        *,
        required: bool,
    ) -> None:
        """Admit one envelope without allowing producer-side growth."""

        try:
            self._outbound_send.send_nowait(envelope)
        except WouldBlock as exc:
            self._dropped_outbound_envelopes += 1
            if required:
                raise _RetryAuthorityRoundError from exc

    @staticmethod
    def _member_union(
        memberships: Iterable[AuthorityMembership],
    ) -> tuple[UUID, ...]:
        """Return all configured voter and learner IDs in stable order."""

        member_ids = {
            UUID(str(member.node_install_id))
            for membership in memberships
            for member in membership.members
        }
        return tuple(sorted(member_ids, key=lambda member_id: member_id.int))

    @staticmethod
    def _entry_matches_request(
        entry: AuthorityCommittedEntry,
        request: AuthorityTransitionRequest,
    ) -> bool:
        """Return whether a recovered or new entry satisfies caller intent."""

        descriptor = entry.certificate.descriptor
        return (
            descriptor.record_type == request.record_type
            and descriptor.record_id == request.record_id
            and descriptor.payload_digest == request.payload_digest
        )
