"""Quorum, continuity, and encrypted-apply tests for operator authority."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import final
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from skulk.operator.authority import (
    AuthorityCommitConflictError,
    EncryptedAuthorityStore,
)
from skulk.operator.identity import create_cluster_identity
from skulk.operator.replication import (
    AuthorityCertificateError,
    AuthorityCommitDescriptor,
    AuthorityCommitPosition,
    AuthorityMember,
    AuthorityMembership,
    AuthorityQuorumCertificate,
    AuthorityVote,
    apply_quorum_certified_payload,
    authority_bootstrap_position,
    authority_commit_digest,
    authority_membership_digest,
    authority_payload_digest,
    create_authority_member,
    create_authority_vote,
    verify_quorum_certificate,
)


@final
class _StaticKeyProvider:
    """Test-only provider for the encrypted projection data key."""

    active_key_id = "test-key-v1"

    def load_data_key(self, key_id: str) -> bytes:
        """Return the deterministic test key version."""

        if key_id != self.active_key_id:
            raise KeyError(key_id)
        return b"a" * 32


def _private_key() -> bytes:
    """Create one raw Ed25519 signing key for a test member."""

    return Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _member() -> tuple[UUID, bytes]:
    """Create a stable member identity and its private signing key."""

    return uuid4(), _private_key()


def _membership(
    voters: int,
    *,
    learners: int = 0,
    generation: int = 1,
) -> tuple[AuthorityMembership, dict[UUID, bytes]]:
    """Create a deterministic-shape membership and retained test keys."""

    keys: dict[UUID, bytes] = {}
    members: list[AuthorityMember] = []
    for _ in range(voters):
        member_id, private_key = _member()
        keys[member_id] = private_key
        members.append(create_authority_member(member_id, private_key))
    for _ in range(learners):
        member_id, private_key = _member()
        keys[member_id] = private_key
        members.append(
            create_authority_member(member_id, private_key, role="learner")
        )
    return AuthorityMembership(generation=generation, members=tuple(members)), keys


def _previous_position(cluster_id: UUID) -> AuthorityCommitPosition:
    """Return the verified bootstrap position preceding test commits."""

    return AuthorityCommitPosition(
        cluster_id=cluster_id,
        authority_term=1,
        commit_index=1,
        commit_digest=authority_payload_digest({"bootstrap": True}),
    )


def test_bootstrap_position_is_stable_across_display_name_changes() -> None:
    """Editable cluster naming cannot fork the authority trust anchor."""

    identity = create_cluster_identity("Fox Den").public_identity
    renamed = identity.model_copy(update={"name": "Home Fabric"})

    assert authority_bootstrap_position(identity) == authority_bootstrap_position(
        renamed
    )


def _descriptor(
    cluster_id: UUID,
    previous_position: AuthorityCommitPosition,
    memberships: tuple[AuthorityMembership, ...],
    payload: Mapping[str, object] | None = None,
    *,
    authority_term: int = 1,
) -> AuthorityCommitDescriptor:
    """Build the next descriptor for active or joint test membership."""

    selected_payload = payload if payload is not None else {"enabled": True}
    return AuthorityCommitDescriptor(
        cluster_id=cluster_id,
        authority_term=authority_term,
        commit_index=previous_position.commit_index + 1,
        previous_commit_digest=previous_position.commit_digest,
        record_type="device",
        record_id="device-1",
        payload_digest=authority_payload_digest(selected_payload),
        required_membership_digests=tuple(
            sorted(
                authority_membership_digest(membership)
                for membership in memberships
            )
        ),
    )


def _certificate(
    descriptor: AuthorityCommitDescriptor,
    keys: dict[UUID, bytes],
    voter_ids: tuple[UUID, ...],
) -> AuthorityQuorumCertificate:
    """Sign a descriptor with the selected test members."""

    return AuthorityQuorumCertificate(
        descriptor=descriptor,
        votes=tuple(
            create_authority_vote(descriptor, voter_id, keys[voter_id])
            for voter_id in voter_ids
        ),
    )


@pytest.mark.parametrize(
    ("voter_count", "vote_count"),
    ((1, 1), (2, 2), (3, 2), (4, 3)),
)
def test_certificate_accepts_strict_majority_for_one_to_four_voters(
    voter_count: int,
    vote_count: int,
) -> None:
    """Common deployment sizes use strict-majority quorum mathematics."""

    cluster_id = uuid4()
    membership, keys = _membership(voter_count)
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, (membership,))
    selected_voters = tuple(keys)[:vote_count]

    position = verify_quorum_certificate(
        _certificate(descriptor, keys, selected_voters),
        (membership,),
        previous,
    )

    assert position.commit_index == 2
    assert position.commit_digest == authority_commit_digest(descriptor)


@pytest.mark.parametrize(
    ("voter_count", "insufficient_vote_count"),
    ((2, 1), (3, 1), (4, 2)),
)
def test_certificate_rejects_less_than_a_strict_majority(
    voter_count: int,
    insufficient_vote_count: int,
) -> None:
    """One local writer or an exact half cannot authorize cluster mutation."""

    cluster_id = uuid4()
    membership, keys = _membership(voter_count)
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, (membership,))

    with pytest.raises(AuthorityCertificateError, match="every required quorum"):
        verify_quorum_certificate(
            _certificate(
                descriptor,
                keys,
                tuple(keys)[:insufficient_vote_count],
            ),
            (membership,),
            previous,
        )


def test_joint_membership_requires_majority_of_old_and_new_voters() -> None:
    """A configuration change cannot be captured by only one voter set."""

    cluster_id = uuid4()
    first_id, first_key = _member()
    shared_id, shared_key = _member()
    second_shared_id, second_shared_key = _member()
    new_id, new_key = _member()
    keys = {
        first_id: first_key,
        shared_id: shared_key,
        second_shared_id: second_shared_key,
        new_id: new_key,
    }
    old_membership = AuthorityMembership(
        generation=1,
        members=tuple(
            create_authority_member(member_id, keys[member_id])
            for member_id in (first_id, shared_id, second_shared_id)
        ),
    )
    new_membership = AuthorityMembership(
        generation=2,
        members=tuple(
            create_authority_member(member_id, keys[member_id])
            for member_id in (shared_id, second_shared_id, new_id)
        ),
    )
    memberships = (old_membership, new_membership)
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, memberships)

    with pytest.raises(AuthorityCertificateError, match="every required quorum"):
        verify_quorum_certificate(
            _certificate(descriptor, keys, (first_id, shared_id)),
            memberships,
            previous,
        )

    verified = verify_quorum_certificate(
        _certificate(descriptor, keys, (shared_id, second_shared_id)),
        memberships,
        previous,
    )
    assert verified.commit_index == 2


def test_learner_signature_never_counts_as_a_vote() -> None:
    """A catch-up replica cannot authorize mutations before voter promotion."""

    cluster_id = uuid4()
    membership, keys = _membership(1, learners=1)
    learner_id = next(
        UUID(str(member.node_install_id))
        for member in membership.members
        if member.role == "learner"
    )
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, (membership,))

    with pytest.raises(AuthorityCertificateError, match="learner vote"):
        verify_quorum_certificate(
            _certificate(descriptor, keys, (learner_id,)),
            (membership,),
            previous,
        )


def test_duplicate_vote_is_rejected_before_quorum_counting() -> None:
    """Repeating one valid signature cannot manufacture a majority."""

    cluster_id = uuid4()
    membership, keys = _membership(1)
    voter_id = next(iter(keys))
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, (membership,))
    vote = create_authority_vote(descriptor, voter_id, keys[voter_id])

    with pytest.raises(ValidationError, match="duplicate vote"):
        AuthorityQuorumCertificate(descriptor=descriptor, votes=(vote, vote))


def test_membership_rejects_one_signing_key_for_two_nodes() -> None:
    """One private key cannot be registered as multiple apparent voters."""

    private_key = _private_key()

    with pytest.raises(ValidationError, match="duplicate signing key"):
        AuthorityMembership(
            generation=1,
            members=(
                create_authority_member(uuid4(), private_key),
                create_authority_member(uuid4(), private_key),
            ),
        )


def test_signature_cannot_be_relabelled_as_another_member() -> None:
    """The signed message includes the stable identity claiming the vote."""

    cluster_id = uuid4()
    membership, keys = _membership(2)
    first_id, second_id = tuple(keys)
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, (membership,))
    first_vote = create_authority_vote(descriptor, first_id, keys[first_id])
    relabelled_vote = first_vote.model_copy(update={"node_install_id": second_id})

    with pytest.raises(AuthorityCertificateError, match="signature is invalid"):
        verify_quorum_certificate(
            AuthorityQuorumCertificate(
                descriptor=descriptor,
                votes=(relabelled_vote,),
            ),
            (membership,),
            previous,
        )


def test_signature_cannot_be_replayed_for_another_descriptor() -> None:
    """A valid vote binds every payload, position, membership, and target field."""

    cluster_id = uuid4()
    membership, keys = _membership(1)
    voter_id = next(iter(keys))
    previous = _previous_position(cluster_id)
    original = _descriptor(cluster_id, previous, (membership,))
    changed = original.model_copy(update={"record_id": "device-2"})
    certificate = AuthorityQuorumCertificate(
        descriptor=changed,
        votes=(create_authority_vote(original, voter_id, keys[voter_id]),),
    )

    with pytest.raises(AuthorityCertificateError, match="signature is invalid"):
        verify_quorum_certificate(certificate, (membership,), previous)


def test_certificate_rejects_stale_chain_position() -> None:
    """A valid majority cannot append against a different verified log head."""

    cluster_id = uuid4()
    membership, keys = _membership(1)
    voter_id = next(iter(keys))
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, (membership,))
    certificate = _certificate(descriptor, keys, (voter_id,))
    stale_position = previous.model_copy(
        update={"commit_digest": authority_payload_digest({"stale": True})}
    )

    with pytest.raises(AuthorityCertificateError, match="digest is stale"):
        verify_quorum_certificate(certificate, (membership,), stale_position)


def test_joint_membership_requires_consecutive_generations() -> None:
    """Joint consensus cannot skip an unverified membership generation."""

    cluster_id = uuid4()
    old_membership, old_keys = _membership(1, generation=1)
    new_membership, new_keys = _membership(1, generation=3)
    memberships = (old_membership, new_membership)
    previous = _previous_position(cluster_id)
    descriptor = _descriptor(cluster_id, previous, memberships)
    keys = old_keys | new_keys

    with pytest.raises(AuthorityCertificateError, match="consecutive generations"):
        verify_quorum_certificate(
            _certificate(descriptor, keys, tuple(keys)),
            memberships,
            previous,
        )


def test_certified_payload_is_applied_only_after_digest_and_quorum_match(
    tmp_path: Path,
) -> None:
    """The encrypted store accepts exactly the payload authorized by voters."""

    store = EncryptedAuthorityStore(
        _StaticKeyProvider(),
        tmp_path / "authority" / "authority.sqlite3",
    )
    cluster = create_cluster_identity("Fox Den")
    store.initialize_cluster(cluster.public_identity, cluster.private_key)
    cluster_id = UUID(str(cluster.public_identity.cluster_id))
    membership, keys = _membership(1)
    voter_id = next(iter(keys))
    previous = authority_bootstrap_position(cluster.public_identity)
    payload = {"refreshVerifier": "secret"}
    descriptor = _descriptor(cluster_id, previous, (membership,), payload)
    certificate = _certificate(descriptor, keys, (voter_id,))

    with pytest.raises(AuthorityCertificateError, match="payload digest"):
        apply_quorum_certified_payload(
            store,
            certificate,
            (membership,),
            previous,
            {"refreshVerifier": "substituted"},
        )
    assert len(store.records()) == 1

    applied = apply_quorum_certified_payload(
        store,
        certificate,
        (membership,),
        previous,
        payload,
    )
    assert applied.record.commit_index == 2
    assert applied.position.commit_digest == authority_commit_digest(descriptor)
    assert store.read_latest_payload("device", "device-1") == payload


def test_local_compare_and_set_still_fences_a_stale_certified_apply(
    tmp_path: Path,
) -> None:
    """Certification never bypasses the local projection's final CAS fence."""

    store = EncryptedAuthorityStore(
        _StaticKeyProvider(),
        tmp_path / "authority" / "authority.sqlite3",
    )
    cluster = create_cluster_identity()
    store.initialize_cluster(cluster.public_identity, cluster.private_key)
    cluster_id = UUID(str(cluster.public_identity.cluster_id))
    membership, keys = _membership(1)
    voter_id = next(iter(keys))
    previous = authority_bootstrap_position(cluster.public_identity)
    payload = {"enabled": True}
    descriptor = _descriptor(cluster_id, previous, (membership,), payload)
    certificate = _certificate(descriptor, keys, (voter_id,))
    store.append(
        expected_commit_index=1,
        authority_term=1,
        record_type="device",
        record_id="other-device",
        payload={"enabled": False},
    )

    with pytest.raises(AuthorityCommitConflictError):
        apply_quorum_certified_payload(
            store,
            certificate,
            (membership,),
            previous,
            payload,
        )


def test_payload_digest_rejects_non_finite_values() -> None:
    """Signed payload truth is portable across standards-compliant JSON peers."""

    with pytest.raises(ValueError, match="finite JSON"):
        authority_payload_digest({"invalid": float("inf")})


def test_vote_encoding_is_validated() -> None:
    """Malformed wire signatures fail at strict model validation."""

    with pytest.raises(ValidationError, match="Ed25519 signature"):
        AuthorityVote(node_install_id=uuid4(), signature="AAAA")
