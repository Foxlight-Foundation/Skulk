"""Durability and integrity tests for authority consensus persistence."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.operator.consensus import (
    AuthorityBallot,
    AuthorityCommittedEntry,
    AuthorityConsensusConflictError,
    AuthorityConsensusState,
)
from skulk.operator.consensus_store import (
    AuthorityConsensusAlreadyInitializedError,
    AuthorityConsensusIntegrityError,
    AuthorityConsensusNotInitializedError,
    SqliteAuthorityConsensusRepository,
)
from skulk.operator.identity import create_cluster_identity
from skulk.operator.replication import (
    AuthorityCommitDescriptor,
    AuthorityMembership,
    AuthorityQuorumCertificate,
    authority_bootstrap_position,
    authority_membership_digest,
    authority_payload_digest,
    create_authority_member,
    create_authority_vote,
    verify_quorum_certificate,
)


def _private_key() -> bytes:
    """Return one raw Ed25519 authority key."""

    return Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _bootstrap_state() -> tuple[AuthorityConsensusState, UUID, bytes]:
    """Return one-voter bootstrap state and its signing material."""

    identity = create_cluster_identity("Persistence Test").public_identity
    member_id = uuid4()
    private_key = _private_key()
    membership = AuthorityMembership(
        generation=1,
        members=(create_authority_member(member_id, private_key),),
    )
    return (
        AuthorityConsensusState.bootstrap(
            authority_bootstrap_position(identity),
            membership,
        ),
        member_id,
        private_key,
    )


def _committed_state(
    bootstrap: AuthorityConsensusState,
    member_id: UUID,
    private_key: bytes,
) -> AuthorityConsensusState:
    """Return bootstrap state extended by one valid certified entry."""

    descriptor = AuthorityCommitDescriptor(
        cluster_id=bootstrap.position.cluster_id,
        authority_term=2,
        commit_index=2,
        previous_commit_digest=bootstrap.position.commit_digest,
        record_type="device",
        record_id="device-1",
        payload_digest=authority_payload_digest({"enabled": True}),
        required_membership_digests=(
            authority_membership_digest(bootstrap.active_membership),
        ),
    )
    certificate = AuthorityQuorumCertificate(
        descriptor=descriptor,
        votes=(create_authority_vote(descriptor, member_id, private_key),),
    )
    position = verify_quorum_certificate(
        certificate,
        (bootstrap.active_membership,),
        bootstrap.position,
    )
    return AuthorityConsensusState(
        bootstrap_position=bootstrap.bootstrap_position,
        bootstrap_membership=bootstrap.bootstrap_membership,
        position=position,
        active_membership=bootstrap.active_membership,
        promised_ballot=AuthorityBallot(
            counter=2,
            proposer_node_install_id=member_id,
        ),
        committed_entries=(
            AuthorityCommittedEntry(
                certificate=certificate,
                memberships=(bootstrap.active_membership,),
            ),
        ),
    )


def test_repository_requires_explicit_single_initialization(tmp_path: Path) -> None:
    """Missing state fails closed and bootstrap cannot overwrite it."""

    repository = SqliteAuthorityConsensusRepository(tmp_path / "authority.sqlite3")
    state, _, _ = _bootstrap_state()

    with pytest.raises(AuthorityConsensusNotInitializedError):
        repository.load()
    assert repository.initialize(state).revision == 0
    with pytest.raises(AuthorityConsensusAlreadyInitializedError):
        repository.initialize(state)


def test_restart_restores_promise_acceptance_and_certified_log(tmp_path: Path) -> None:
    """All safety state survives closing and recreating the repository handle."""

    path = tmp_path / "authority.sqlite3"
    repository = SqliteAuthorityConsensusRepository(path)
    bootstrap, member_id, private_key = _bootstrap_state()
    initial = repository.initialize(bootstrap)
    committed = _committed_state(bootstrap, member_id, private_key)
    ballot = AuthorityBallot(counter=2, proposer_node_install_id=member_id)
    accepted = AuthorityConsensusState(
        bootstrap_position=bootstrap.bootstrap_position,
        bootstrap_membership=bootstrap.bootstrap_membership,
        position=bootstrap.position,
        active_membership=bootstrap.active_membership,
        promised_ballot=ballot,
        accepted_ballot=ballot,
        accepted_descriptor=committed.committed_entries[0].certificate.descriptor,
        accepted_memberships=(bootstrap.active_membership,),
    )

    accepted_snapshot = repository.compare_and_set(initial.revision, accepted)
    accepted_after_restart = SqliteAuthorityConsensusRepository(path).load()
    persisted = repository.compare_and_set(accepted_snapshot.revision, committed)
    restored = SqliteAuthorityConsensusRepository(path).load()

    assert accepted_after_restart == accepted_snapshot
    assert persisted.revision == 2
    assert restored == persisted
    assert restored.state.committed_entries == committed.committed_entries


def test_compare_and_set_rejects_stale_revision(tmp_path: Path) -> None:
    """Two local writers cannot both commit from one snapshot."""

    repository = SqliteAuthorityConsensusRepository(tmp_path / "authority.sqlite3")
    bootstrap, member_id, _ = _bootstrap_state()
    snapshot = repository.initialize(bootstrap)
    promised = AuthorityConsensusState(
        bootstrap_position=bootstrap.bootstrap_position,
        bootstrap_membership=bootstrap.bootstrap_membership,
        position=bootstrap.position,
        active_membership=bootstrap.active_membership,
        promised_ballot=AuthorityBallot(
            counter=2,
            proposer_node_install_id=member_id,
        ),
    )
    repository.compare_and_set(snapshot.revision, promised)

    with pytest.raises(AuthorityConsensusConflictError, match="revision changed"):
        repository.compare_and_set(snapshot.revision, promised)


def test_compare_and_set_cannot_replace_bootstrap_trust_anchor(tmp_path: Path) -> None:
    """Revision-zero writes cannot substitute another cluster or voter root."""

    repository = SqliteAuthorityConsensusRepository(tmp_path / "authority.sqlite3")
    bootstrap, _, _ = _bootstrap_state()
    snapshot = repository.initialize(bootstrap)
    replacement, _, _ = _bootstrap_state()

    with pytest.raises(
        AuthorityConsensusConflictError,
        match="bootstrap anchor changed",
    ):
        repository.compare_and_set(snapshot.revision, replacement)


def test_compare_and_set_cannot_remove_committed_history(tmp_path: Path) -> None:
    """A local replacement cannot roll the certified log back to bootstrap."""

    repository = SqliteAuthorityConsensusRepository(tmp_path / "authority.sqlite3")
    bootstrap, member_id, private_key = _bootstrap_state()
    initial = repository.initialize(bootstrap)
    committed = _committed_state(bootstrap, member_id, private_key)
    current = repository.compare_and_set(initial.revision, committed)

    with pytest.raises(
        AuthorityConsensusIntegrityError,
        match="missing committed entries",
    ):
        repository.compare_and_set(
            current.revision,
            AuthorityConsensusState(
                bootstrap_position=committed.bootstrap_position,
                bootstrap_membership=committed.bootstrap_membership,
                position=committed.position,
                active_membership=committed.active_membership,
            ),
        )


def test_corrupt_state_json_fails_closed(tmp_path: Path) -> None:
    """Malformed durable metadata never becomes an empty or permissive state."""

    path = tmp_path / "authority.sqlite3"
    repository = SqliteAuthorityConsensusRepository(path)
    state, _, _ = _bootstrap_state()
    repository.initialize(state)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE consensus_state SET state_json = ? WHERE singleton = 1",
            ("{not-json",),
        )

    with pytest.raises(AuthorityConsensusIntegrityError):
        repository.load()


def test_restart_reverifies_persisted_certificate_signatures(tmp_path: Path) -> None:
    """Well-formed but forged durable votes fail closed during recovery."""

    path = tmp_path / "authority.sqlite3"
    repository = SqliteAuthorityConsensusRepository(path)
    bootstrap, member_id, private_key = _bootstrap_state()
    initial = repository.initialize(bootstrap)
    repository.compare_and_set(
        initial.revision,
        _committed_state(bootstrap, member_id, private_key),
    )
    with sqlite3.connect(path) as connection:
        row = cast(
            tuple[object] | None,
            connection.execute(
                "SELECT entry_json FROM consensus_commits WHERE commit_index = 2"
            ).fetchone(),
        )
        assert row is not None
        assert isinstance(row[0], str)
        entry = AuthorityCommittedEntry.model_validate_json(row[0])
        signature = entry.certificate.votes[0].signature
        forged_entry_json = row[0].replace(signature, "A" * 86)
        assert forged_entry_json != row[0]
        connection.execute(
            "UPDATE consensus_commits SET entry_json = ? WHERE commit_index = 2",
            (forged_entry_json,),
        )

    with pytest.raises(
        AuthorityConsensusIntegrityError,
        match="certificate failed verification",
    ):
        SqliteAuthorityConsensusRepository(path).load()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics only")
def test_repository_repairs_directory_and_database_permissions(tmp_path: Path) -> None:
    """Every open repairs permissive authority metadata paths."""

    directory = tmp_path / "operator"
    path = directory / "authority.sqlite3"
    repository = SqliteAuthorityConsensusRepository(path)
    state, _, _ = _bootstrap_state()
    repository.initialize(state)
    os.chmod(directory, 0o777)
    os.chmod(path, 0o666)

    repository.load()

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
