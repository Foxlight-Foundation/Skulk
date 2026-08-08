# pyright: reportUnusedFunction=false
"""Deterministic quorum certification for replicated operator authority.

This module owns the pure cryptographic boundary between a proposed authority
record and the encrypted local projection. Networking, leader election, log
replication, and retry policy remain separate concerns: a caller may append an
authority record only after this layer verifies a majority certificate for the
active membership, or for both memberships during a joint transition.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from typing import Literal, final
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import UUID4, Field, field_validator, model_validator

from skulk.operator.authority import AuthorityRecord, EncryptedAuthorityStore
from skulk.operator.identity import ClusterPublicIdentity
from skulk.utils.pydantic_ext import FrozenModel

_AUTHORITY_SIGNATURE_CONTEXT = b"skulk-operator-authority-commit-v1\x00"
_AUTHORITY_MEMBERSHIP_CONTEXT = b"skulk-operator-authority-membership-v1\x00"
_AUTHORITY_PAYLOAD_CONTEXT = b"skulk-operator-authority-payload-v1\x00"
_AUTHORITY_BOOTSTRAP_CONTEXT = b"skulk-operator-authority-bootstrap-v1\x00"
_SHA256_PREFIX = "sha256:"
_SHA256_BASE64URL_LENGTH = 43


class AuthorityCertificateError(RuntimeError):
    """Raised when an authority commit lacks a valid quorum certificate."""


def _base64url_encode(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    """Decode an unpadded URL-safe base64 value."""

    padding = "=" * (-len(value) % 4)
    return base64.b64decode(
        f"{value}{padding}",
        altchars=b"-_",
        validate=True,
    )


def _canonical_json(payload: object) -> bytes:
    """Serialize a signed object with deterministic, finite JSON semantics."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_digest(context: bytes, payload: bytes) -> str:
    """Hash one domain-separated authority value."""

    return f"{_SHA256_PREFIX}{_base64url_encode(hashlib.sha256(context + payload).digest())}"


def _validate_sha256_digest(value: str) -> str:
    """Validate one unpadded URL-safe SHA-256 digest."""

    if not value.startswith(_SHA256_PREFIX):
        raise ValueError("digest must use the sha256 prefix")
    encoded = value.removeprefix(_SHA256_PREFIX)
    if len(encoded) != _SHA256_BASE64URL_LENGTH:
        raise ValueError("digest must contain one SHA-256 value")
    try:
        decoded = _base64url_decode(encoded)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("digest must contain URL-safe base64") from exc
    if len(decoded) != hashlib.sha256().digest_size:
        raise ValueError("digest must contain one SHA-256 value")
    return value


@final
class AuthorityMember(FrozenModel):
    """One stable node identity participating in operator authority."""

    node_install_id: UUID4 = Field(
        description="Stable installation identity; never a runtime libp2p peer ID.",
    )
    public_key: str = Field(
        description="Unpadded URL-safe base64 Ed25519 authority signing key.",
    )
    role: Literal["voter", "learner"] = Field(
        description="Whether the member counts toward quorum or only catches up.",
    )

    @field_validator("public_key")
    @classmethod
    def _public_key_is_ed25519(cls, value: str) -> str:
        """Reject malformed authority signing keys at the membership boundary."""

        try:
            decoded = _base64url_decode(value)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("public_key is not valid URL-safe base64") from exc
        if len(decoded) != 32:
            raise ValueError("public_key must contain one raw Ed25519 public key")
        return _base64url_encode(decoded)


@final
class AuthorityMembership(FrozenModel):
    """One versioned voter and learner configuration."""

    generation: int = Field(
        ge=1,
        description="Monotonic membership generation selected by authority state.",
    )
    members: tuple[AuthorityMember, ...] = Field(
        min_length=1,
        max_length=128,
        description="Unique stable members in this configuration.",
    )

    @model_validator(mode="after")
    def _membership_is_unambiguous(self) -> "AuthorityMembership":
        """Require unique members and at least one voting authority node."""

        member_ids = tuple(member.node_install_id for member in self.members)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("authority membership contains a duplicate node")
        public_keys = tuple(member.public_key for member in self.members)
        if len(set(public_keys)) != len(public_keys):
            raise ValueError("authority membership contains a duplicate signing key")
        if not any(member.role == "voter" for member in self.members):
            raise ValueError("authority membership requires at least one voter")
        return self

    @property
    def voter_ids(self) -> frozenset[UUID]:
        """Return stable identities whose signatures count toward quorum."""

        return frozenset(
            UUID(str(member.node_install_id))
            for member in self.members
            if member.role == "voter"
        )

    @property
    def quorum_size(self) -> int:
        """Return the strict majority required by this membership."""

        return (len(self.voter_ids) // 2) + 1


@final
class AuthorityCommitDescriptor(FrozenModel):
    """Canonical authority transition signed by stable cluster voters."""

    cluster_id: UUID4 = Field(
        description="Cluster identity that owns this authorization transition.",
    )
    authority_term: int = Field(
        ge=1,
        description="Monotonic authority leadership term proposing the transition.",
    )
    commit_index: int = Field(
        ge=1,
        description="Next contiguous authority journal index.",
    )
    previous_commit_digest: str = Field(
        description="Digest of the immediately preceding certified commit.",
    )
    record_type: str = Field(
        min_length=1,
        max_length=80,
        description="Bounded non-secret semantic record type.",
    )
    record_id: str = Field(
        min_length=1,
        max_length=160,
        description="Stable opaque record identity; never secret-bearing data.",
    )
    payload_digest: str = Field(
        description="Domain-separated digest of the encrypted record payload.",
    )
    required_membership_digests: tuple[str, ...] = Field(
        min_length=1,
        max_length=2,
        description=(
            "One active membership digest, or old and new digests during a joint "
            "membership transition."
        ),
    )

    @field_validator("previous_commit_digest", "payload_digest")
    @classmethod
    def _digest_is_sha256(cls, value: str) -> str:
        """Reject ambiguous or malformed commit-chain digests."""

        return _validate_sha256_digest(value)

    @field_validator("record_type")
    @classmethod
    def _record_type_has_no_whitespace(cls, value: str) -> str:
        """Keep record-type metadata bounded and unambiguous."""

        if any(character.isspace() for character in value):
            raise ValueError("record_type must not contain whitespace")
        return value

    @field_validator("required_membership_digests")
    @classmethod
    def _membership_digests_are_canonical(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require unique sorted membership digests before signatures exist."""

        for digest in value:
            _validate_sha256_digest(digest)
        if len(set(value)) != len(value):
            raise ValueError("required membership digests must be unique")
        if tuple(sorted(value)) != value:
            raise ValueError("required membership digests must be sorted")
        return value


@final
class AuthorityVote(FrozenModel):
    """One node's signature over an authority commit descriptor."""

    node_install_id: UUID4 = Field(
        description="Stable identity of the signing authority member.",
    )
    signature: str = Field(
        description="Unpadded URL-safe base64 Ed25519 descriptor signature.",
    )

    @field_validator("signature")
    @classmethod
    def _signature_is_ed25519(cls, value: str) -> str:
        """Reject malformed signature encodings before quorum evaluation."""

        try:
            decoded = _base64url_decode(value)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("signature is not valid URL-safe base64") from exc
        if len(decoded) != 64:
            raise ValueError("signature must contain one Ed25519 signature")
        return _base64url_encode(decoded)


@final
class AuthorityQuorumCertificate(FrozenModel):
    """A commit descriptor plus unique voter signatures authorizing it."""

    descriptor: AuthorityCommitDescriptor = Field(
        description="Exact immutable authority transition signed by every vote.",
    )
    votes: tuple[AuthorityVote, ...] = Field(
        min_length=1,
        max_length=128,
        description="Unique signatures offered as quorum evidence.",
    )

    @model_validator(mode="after")
    def _votes_are_unique(self) -> "AuthorityQuorumCertificate":
        """Prevent one member signature from being counted more than once."""

        voter_ids = tuple(vote.node_install_id for vote in self.votes)
        if len(set(voter_ids)) != len(voter_ids):
            raise ValueError("quorum certificate contains a duplicate vote")
        return self


@final
class AuthorityCommitPosition(FrozenModel):
    """Verified position and digest at the head of an authority log."""

    cluster_id: UUID4 = Field(
        description="Cluster identity owning the verified log position.",
    )
    authority_term: int = Field(
        ge=1,
        description="Authority term of the verified commit.",
    )
    commit_index: int = Field(
        ge=1,
        description="Contiguous verified authority journal index.",
    )
    commit_digest: str = Field(
        description="Domain-separated digest of the verified descriptor.",
    )

    @field_validator("commit_digest")
    @classmethod
    def _commit_digest_is_sha256(cls, value: str) -> str:
        """Reject malformed log-head digests."""

        return _validate_sha256_digest(value)


@final
class AuthorityAppliedCommit(FrozenModel):
    """Local encrypted projection result for one certified authority commit."""

    position: AuthorityCommitPosition = Field(
        description="New verified head of the authority log.",
    )
    record: AuthorityRecord = Field(
        description="Public metadata for the encrypted local journal append.",
    )


def create_authority_member(
    node_install_id: UUID,
    private_key: bytes,
    *,
    role: Literal["voter", "learner"] = "voter",
) -> AuthorityMember:
    """Derive one authority membership record from private signing material.

    Args:
        node_install_id: Stable installation identity assigned to the host.
        private_key: Raw Ed25519 private key retained by that host.
        role: Whether this member votes or catches up as a learner.

    Returns:
        Public authority member suitable for a membership configuration.
    """

    if len(private_key) != 32:
        raise ValueError("authority private key must contain 32 bytes")
    public_key = (
        Ed25519PrivateKey.from_private_bytes(private_key)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    return AuthorityMember(
        node_install_id=node_install_id,
        public_key=_base64url_encode(public_key),
        role=role,
    )


def authority_payload_digest(payload: Mapping[str, object]) -> str:
    """Return the signed digest for one secret-bearing authority payload.

    Args:
        payload: JSON object that will be encrypted by the local projection.

    Returns:
        Domain-separated SHA-256 digest.

    Raises:
        ValueError: The payload contains non-finite or non-JSON data.
    """

    try:
        encoded = _canonical_json(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("authority payload must be a finite JSON object") from exc
    return _sha256_digest(_AUTHORITY_PAYLOAD_CONTEXT, encoded)


def authority_bootstrap_position(
    identity: ClusterPublicIdentity,
    *,
    authority_term: int = 1,
) -> AuthorityCommitPosition:
    """Derive the common verified log head created by cluster bootstrap.

    The operator-visible name is deliberately excluded because it is editable
    metadata. Stable key-bound identity fields establish the first commit that
    every replica uses as the previous position for membership enrollment.

    Args:
        identity: Validated public identity committed by authority bootstrap.
        authority_term: Initial authority term persisted with commit index one.

    Returns:
        Deterministic bootstrap position at commit index one.

    Raises:
        ValueError: The initial authority term is not positive.
    """

    if authority_term < 1:
        raise ValueError("authority_term must be positive")
    try:
        public_key_bytes = _base64url_decode(identity.public_key)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("cluster public key is not valid URL-safe base64") from exc
    if len(public_key_bytes) != 32:
        raise ValueError("cluster public key must contain one raw Ed25519 public key")
    canonical_public_key = _base64url_encode(public_key_bytes)
    digest = _sha256_digest(
        _AUTHORITY_BOOTSTRAP_CONTEXT,
        _canonical_json(
            {
                "clusterId": str(identity.cluster_id),
                "fingerprint": identity.fingerprint,
                "formatVersion": identity.format_version,
                "publicKey": canonical_public_key,
            }
        ),
    )
    return AuthorityCommitPosition(
        cluster_id=identity.cluster_id,
        authority_term=authority_term,
        commit_index=1,
        commit_digest=digest,
    )


def authority_membership_digest(membership: AuthorityMembership) -> str:
    """Return the stable digest for one membership configuration.

    Args:
        membership: Versioned authority membership.

    Returns:
        Domain-separated SHA-256 digest independent of tuple order.
    """

    members = sorted(
        (
            {
                "nodeInstallId": str(member.node_install_id),
                "publicKey": member.public_key,
                "role": member.role,
            }
            for member in membership.members
        ),
        key=lambda member: str(member["nodeInstallId"]),
    )
    return _sha256_digest(
        _AUTHORITY_MEMBERSHIP_CONTEXT,
        _canonical_json(
            {
                "generation": membership.generation,
                "members": members,
            }
        ),
    )


def authority_commit_digest(descriptor: AuthorityCommitDescriptor) -> str:
    """Return the stable digest for one signed authority descriptor.

    Args:
        descriptor: Immutable authority transition.

    Returns:
        Domain-separated SHA-256 commit digest.
    """

    return _sha256_digest(
        _AUTHORITY_SIGNATURE_CONTEXT,
        _descriptor_bytes(descriptor),
    )


def create_authority_vote(
    descriptor: AuthorityCommitDescriptor,
    node_install_id: UUID,
    private_key: bytes,
) -> AuthorityVote:
    """Sign one exact authority descriptor with a member's Ed25519 key.

    Args:
        descriptor: Proposed transition to authorize.
        node_install_id: Stable identity of the signing member.
        private_key: Raw Ed25519 signing key retained by that member.

    Returns:
        Signature bound to the descriptor and stable node identity.
    """

    if len(private_key) != 32:
        raise ValueError("authority private key must contain 32 bytes")
    signature = Ed25519PrivateKey.from_private_bytes(private_key).sign(
        _vote_message(descriptor, node_install_id)
    )
    return AuthorityVote(
        node_install_id=node_install_id,
        signature=_base64url_encode(signature),
    )


def verify_quorum_certificate(
    certificate: AuthorityQuorumCertificate,
    memberships: tuple[AuthorityMembership, ...],
    previous_position: AuthorityCommitPosition,
) -> AuthorityCommitPosition:
    """Verify commit continuity, membership, signatures, and every majority.

    Args:
        certificate: Proposed commit and its collected signatures.
        memberships: Active membership, or old and new memberships during a
            joint transition.
        previous_position: Last locally verified authority commit.

    Returns:
        New verified authority-log position.

    Raises:
        AuthorityCertificateError: Any continuity, membership, signature, or
            quorum check fails.
    """

    descriptor = certificate.descriptor
    validate_authority_descriptor(descriptor, memberships, previous_position)

    member_by_id: dict[UUID, AuthorityMember] = {}
    member_id_by_public_key: dict[str, UUID] = {}
    voting_members: set[UUID] = set()
    for membership in memberships:
        voting_members.update(membership.voter_ids)
        for member in membership.members:
            member_id = UUID(str(member.node_install_id))
            existing = member_by_id.get(member_id)
            if existing is not None and existing.public_key != member.public_key:
                raise AuthorityCertificateError(
                    "one authority member has conflicting signing keys"
                )
            existing_member_id = member_id_by_public_key.get(member.public_key)
            if existing_member_id is not None and existing_member_id != member_id:
                raise AuthorityCertificateError(
                    "one authority signing key belongs to conflicting members"
                )
            member_by_id[member_id] = member
            member_id_by_public_key[member.public_key] = member_id

    valid_vote_ids: set[UUID] = set()
    for vote in certificate.votes:
        voter_id = UUID(str(vote.node_install_id))
        member = member_by_id.get(voter_id)
        if member is None:
            raise AuthorityCertificateError(
                "authority vote came from an unknown member"
            )
        if voter_id not in voting_members:
            raise AuthorityCertificateError("authority learner vote cannot count")
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                _base64url_decode(member.public_key)
            )
            public_key.verify(
                _base64url_decode(vote.signature),
                _vote_message(descriptor, voter_id),
            )
        except (InvalidSignature, ValueError) as exc:
            raise AuthorityCertificateError(
                "authority vote signature is invalid"
            ) from exc
        valid_vote_ids.add(voter_id)

    for membership in memberships:
        votes_in_membership = len(valid_vote_ids.intersection(membership.voter_ids))
        if votes_in_membership < membership.quorum_size:
            raise AuthorityCertificateError(
                "authority certificate does not satisfy every required quorum"
            )

    return AuthorityCommitPosition(
        cluster_id=descriptor.cluster_id,
        authority_term=descriptor.authority_term,
        commit_index=descriptor.commit_index,
        commit_digest=authority_commit_digest(descriptor),
    )


def validate_authority_descriptor(
    descriptor: AuthorityCommitDescriptor,
    memberships: tuple[AuthorityMembership, ...],
    previous_position: AuthorityCommitPosition,
) -> AuthorityCommitPosition:
    """Validate one unsigned proposal before an authority voter signs it.

    Args:
        descriptor: Exact proposed authority transition.
        memberships: Active membership, or old and new memberships for a
            joint transition.
        previous_position: Last locally verified authority commit.

    Returns:
        The position the descriptor would occupy after quorum certification.

    Raises:
        AuthorityCertificateError: Continuity, term, or membership binding is
            invalid.
    """

    if not 1 <= len(memberships) <= 2:
        raise AuthorityCertificateError(
            "authority verification requires one or two memberships"
        )
    if UUID(str(descriptor.cluster_id)) != UUID(str(previous_position.cluster_id)):
        raise AuthorityCertificateError("authority commit belongs to another cluster")
    if descriptor.commit_index != previous_position.commit_index + 1:
        raise AuthorityCertificateError("authority commit index is not contiguous")
    if descriptor.authority_term < previous_position.authority_term:
        raise AuthorityCertificateError("authority term moved backwards")
    if descriptor.previous_commit_digest != previous_position.commit_digest:
        raise AuthorityCertificateError("authority previous-commit digest is stale")

    membership_digests = tuple(
        sorted(authority_membership_digest(membership) for membership in memberships)
    )
    if len(set(membership_digests)) != len(membership_digests):
        raise AuthorityCertificateError("joint memberships must be distinct")
    if descriptor.required_membership_digests != membership_digests:
        raise AuthorityCertificateError("authority commit names a different membership")
    if len(memberships) == 2:
        generations = sorted(membership.generation for membership in memberships)
        if generations[1] != generations[0] + 1:
            raise AuthorityCertificateError(
                "joint authority memberships must be consecutive generations"
            )

    return AuthorityCommitPosition(
        cluster_id=descriptor.cluster_id,
        authority_term=descriptor.authority_term,
        commit_index=descriptor.commit_index,
        commit_digest=authority_commit_digest(descriptor),
    )


def apply_quorum_certified_payload(
    store: EncryptedAuthorityStore,
    certificate: AuthorityQuorumCertificate,
    memberships: tuple[AuthorityMembership, ...],
    previous_position: AuthorityCommitPosition,
    payload: Mapping[str, object],
) -> AuthorityAppliedCommit:
    """Verify and append one certified payload to the encrypted projection.

    Args:
        store: Initialized encrypted authority projection.
        certificate: Proposed commit plus quorum signatures.
        memberships: Active or joint authority memberships.
        previous_position: Last locally verified authority-log position.
        payload: Exact secret-bearing payload named by the descriptor digest.

    Returns:
        Verified log head and public metadata for the encrypted append.

    Raises:
        AuthorityCertificateError: Certificate, continuity, cluster binding, or
            payload integrity is invalid.
        AuthorityCommitConflictError: The local projection has advanced.
    """

    descriptor = certificate.descriptor
    cluster_identity = store.cluster_identity()
    if UUID(str(cluster_identity.cluster_id)) != UUID(str(descriptor.cluster_id)):
        raise AuthorityCertificateError("authority store belongs to another cluster")
    if authority_payload_digest(payload) != descriptor.payload_digest:
        raise AuthorityCertificateError("authority payload digest does not match")
    position = verify_quorum_certificate(
        certificate,
        memberships,
        previous_position,
    )
    record = store.append(
        expected_commit_index=previous_position.commit_index,
        authority_term=descriptor.authority_term,
        record_type=descriptor.record_type,
        record_id=descriptor.record_id,
        payload=payload,
    )
    if (
        record.commit_index != position.commit_index
        or record.authority_term != position.authority_term
    ):
        raise AuthorityCertificateError(
            "authority projection returned an unexpected commit position"
        )
    return AuthorityAppliedCommit(position=position, record=record)


def _descriptor_bytes(descriptor: AuthorityCommitDescriptor) -> bytes:
    """Serialize the exact signed descriptor independently of field aliases."""

    return _canonical_json(
        {
            "authorityTerm": descriptor.authority_term,
            "clusterId": str(descriptor.cluster_id),
            "commitIndex": descriptor.commit_index,
            "payloadDigest": descriptor.payload_digest,
            "previousCommitDigest": descriptor.previous_commit_digest,
            "recordId": descriptor.record_id,
            "recordType": descriptor.record_type,
            "requiredMembershipDigests": list(descriptor.required_membership_digests),
        }
    )


def _vote_message(
    descriptor: AuthorityCommitDescriptor,
    node_install_id: UUID,
) -> bytes:
    """Bind a descriptor signature to one stable member identity."""

    return (
        _AUTHORITY_SIGNATURE_CONTEXT
        + str(node_install_id).encode("ascii")
        + b"\x00"
        + _descriptor_bytes(descriptor)
    )
