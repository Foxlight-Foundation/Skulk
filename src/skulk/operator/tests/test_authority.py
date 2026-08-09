"""Authenticated-encryption and CAS tests for operator authority persistence."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path
from typing import cast, final

import pytest

from skulk.operator.authority import (
    AuthorityAlreadyInitializedError,
    AuthorityCommitConflictError,
    AuthorityIntegrityError,
    EncryptedAuthorityStore,
)
from skulk.operator.identity import create_cluster_identity


@final
class _StaticKeyProvider:
    """Test-only provider that keeps key material out of the database."""

    def __init__(self, key: bytes, key_id: str = "test-key-v1") -> None:
        self._key = key
        self._key_id = key_id

    @property
    def active_key_id(self) -> str:
        """Return the test key version."""

        return self._key_id

    def load_data_key(self, key_id: str) -> bytes:
        """Return test-only in-memory key material."""

        if key_id != self._key_id:
            raise KeyError(key_id)
        return self._key


@final
class _RotatingKeyProvider:
    """Test provider retaining old versions during data-key rotation."""

    def __init__(self) -> None:
        self.active_key_id = "key-v1"
        self._keys = {"key-v1": b"a" * 32, "key-v2": b"b" * 32}

    def load_data_key(self, key_id: str) -> bytes:
        """Return one retained test key version."""

        return self._keys[key_id]


def _store(tmp_path: Path, key: bytes | None = None) -> EncryptedAuthorityStore:
    """Create a store with deterministic test key ownership."""

    return EncryptedAuthorityStore(
        _StaticKeyProvider(key if key is not None else b"a" * 32),
        tmp_path / "authority" / "authority.sqlite3",
    )


def test_cluster_private_key_is_encrypted_at_rest(tmp_path: Path) -> None:
    """Cluster private bytes never appear in SQLite or its journal sidecars."""

    store = _store(tmp_path)
    material = create_cluster_identity("Fox Den")
    record = store.initialize_cluster(
        material.public_identity,
        material.private_key,
    )

    assert record.commit_index == 1
    assert store.cluster_identity() == material.public_identity
    assert store.load_cluster_private_key() == material.private_key
    persisted = b"".join(
        candidate.read_bytes()
        for candidate in store.path.parent.iterdir()
        if candidate.is_file()
    )
    assert material.private_key not in persisted
    encoded_private_key = base64.urlsafe_b64encode(material.private_key).rstrip(b"=")
    assert encoded_private_key not in persisted


def test_initialize_cluster_is_single_use(tmp_path: Path) -> None:
    """A second bootstrap cannot silently replace cluster identity or keys."""

    store = _store(tmp_path)
    first = create_cluster_identity()
    store.initialize_cluster(first.public_identity, first.private_key)
    second = create_cluster_identity()

    with pytest.raises(AuthorityAlreadyInitializedError):
        store.initialize_cluster(second.public_identity, second.private_key)

    assert store.cluster_identity() == first.public_identity


def test_initialize_cluster_rejects_mismatched_key_pair(tmp_path: Path) -> None:
    """Bootstrap cannot permanently bind public identity to another private key."""

    store = _store(tmp_path)
    identity = create_cluster_identity()
    unrelated = create_cluster_identity()

    with pytest.raises(
        ValueError,
        match="private key does not match the public identity",
    ):
        store.initialize_cluster(identity.public_identity, unrelated.private_key)

    record = store.initialize_cluster(identity.public_identity, identity.private_key)
    assert record.commit_index == 1
    assert store.load_cluster_private_key() == identity.private_key


def test_append_requires_exact_commit_index(tmp_path: Path) -> None:
    """Concurrent authority transitions cannot both commit from stale state."""

    store = _store(tmp_path)
    material = create_cluster_identity()
    store.initialize_cluster(material.public_identity, material.private_key)
    appended = store.append(
        expected_commit_index=1,
        authority_term=1,
        record_type="device",
        record_id="device-1",
        payload={"refreshVerifier": "secret-verifier"},
    )

    with pytest.raises(AuthorityCommitConflictError):
        store.append(
            expected_commit_index=1,
            authority_term=1,
            record_type="device",
            record_id="device-2",
            payload={"refreshVerifier": "another-secret"},
        )

    assert appended.commit_index == 2
    assert [record.commit_index for record in store.records()] == [1, 2]
    assert store.read_latest_payload("device", "device-1") == {
        "refreshVerifier": "secret-verifier"
    }
    persisted = store.path.read_bytes()
    assert b"secret-verifier" not in persisted


def test_old_records_remain_readable_during_data_key_rotation(tmp_path: Path) -> None:
    """The provider selects key versions per record during staged rotation."""

    provider = _RotatingKeyProvider()
    store = EncryptedAuthorityStore(
        provider,
        tmp_path / "authority" / "authority.sqlite3",
    )
    material = create_cluster_identity()
    store.initialize_cluster(material.public_identity, material.private_key)
    provider.active_key_id = "key-v2"
    record = store.append(
        expected_commit_index=1,
        authority_term=1,
        record_type="device",
        record_id="device-1",
        payload={"refreshVerifier": "v2-secret"},
    )

    assert record.key_id == "key-v2"
    assert store.load_cluster_private_key() == material.private_key
    assert store.read_latest_payload("device", "device-1") == {
        "refreshVerifier": "v2-secret"
    }


def test_wrong_external_key_fails_authenticated_decryption(tmp_path: Path) -> None:
    """Copied ciphertext is unusable without the enrolled wrapping boundary."""

    original = _store(tmp_path, b"a" * 32)
    material = create_cluster_identity()
    original.initialize_cluster(material.public_identity, material.private_key)
    wrong_key = EncryptedAuthorityStore(
        _StaticKeyProvider(b"b" * 32),
        original.path,
    )

    with pytest.raises(AuthorityIntegrityError):
        wrong_key.load_cluster_private_key()


def test_ciphertext_tamper_fails_closed(tmp_path: Path) -> None:
    """Authenticated encryption detects database-level secret modification."""

    store = _store(tmp_path)
    material = create_cluster_identity()
    store.initialize_cluster(material.public_identity, material.private_key)
    with sqlite3.connect(store.path) as connection:
        row = cast(
            tuple[bytes] | None,
            connection.execute(
                "SELECT ciphertext FROM authority_journal WHERE commit_index = 1"
            ).fetchone(),
        )
        assert row is not None
        ciphertext = bytearray(row[0])
        ciphertext[0] ^= 0x01
        connection.execute(
            "UPDATE authority_journal SET ciphertext = ? WHERE commit_index = 1",
            (bytes(ciphertext),),
        )

    with pytest.raises(AuthorityIntegrityError):
        store.load_cluster_private_key()


def test_public_identity_is_rebound_to_encrypted_private_key(tmp_path: Path) -> None:
    """Valid but substituted public metadata cannot redefine the cluster."""

    store = _store(tmp_path)
    original = create_cluster_identity("Fox Den")
    store.initialize_cluster(original.public_identity, original.private_key)
    substituted = create_cluster_identity("Impostor")
    substituted_payload = substituted.public_identity.model_dump(
        mode="json",
        by_alias=True,
    )
    substituted_payload["clusterId"] = str(original.public_identity.cluster_id)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE authority_metadata SET value = ? WHERE name = ?",
            (
                json.dumps(substituted_payload),
                "cluster_public_identity",
            ),
        )

    with pytest.raises(
        AuthorityIntegrityError,
        match="does not match encrypted private key",
    ):
        store.cluster_identity()


def test_append_rejects_non_finite_json(tmp_path: Path) -> None:
    """NaN and infinity cannot create non-portable authority digests."""

    store = _store(tmp_path)
    material = create_cluster_identity()
    store.initialize_cluster(material.public_identity, material.private_key)

    with pytest.raises(ValueError, match="JSON object"):
        store.append(
            expected_commit_index=1,
            authority_term=1,
            record_type="device",
            record_id="device-1",
            payload={"invalid": float("nan")},
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_authority_files_are_service_account_only(tmp_path: Path) -> None:
    """The database and parent directory never inherit permissive umasks."""

    store = _store(tmp_path)
    material = create_cluster_identity()
    store.initialize_cluster(material.public_identity, material.private_key)

    assert store.path.parent.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_authority_open_repairs_overly_broad_directory_mode(tmp_path: Path) -> None:
    """Every database open repairs a directory broadened after bootstrap."""

    store = _store(tmp_path)
    material = create_cluster_identity()
    store.initialize_cluster(material.public_identity, material.private_key)
    store.path.parent.chmod(0o755)

    assert store.records()
    assert store.path.parent.stat().st_mode & 0o777 == 0o700
