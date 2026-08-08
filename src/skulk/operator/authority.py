# pyright: reportUnusedFunction=false
"""Encrypted local journal for operator authorization state.

This module establishes the storage and compare-and-set boundary used by the
future replicated authority service. It is not itself a quorum implementation:
the caller supplies committed term/index decisions and an external key
provider. That separation prevents a local SQLite transaction from being
mistaken for distributed authorization consensus.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast, final
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import Field, field_validator

from skulk.operator.identity import ClusterPublicIdentity
from skulk.shared.constants import SKULK_DATA_HOME
from skulk.utils.pydantic_ext import FrozenModel

_AUTHORITY_SCHEMA_VERSION = 1
_AUTHORITY_DATA_KEY_BYTES = 32
_AUTHORITY_NONCE_BYTES = 12
_CLUSTER_PRIVATE_KEY_RECORD_TYPE = "cluster_identity_private_key"
_CLUSTER_PRIVATE_KEY_RECORD_ID = "primary"

type _SqliteRow = tuple[object, ...]


class AuthorityCommitConflictError(RuntimeError):
    """Raised when a compare-and-set append observes a different commit index."""


class AuthorityIntegrityError(RuntimeError):
    """Raised when encrypted authority state fails authenticated decryption."""


class AuthorityAlreadyInitializedError(RuntimeError):
    """Raised when cluster bootstrap is attempted on an initialized store."""


class AuthorityNotInitializedError(RuntimeError):
    """Raised when authority state is read before cluster bootstrap."""


class AuthorityKeyProvider(Protocol):
    """External provider of the active cluster authority data key.

    Production implementations must unwrap this key through an OS-protected
    host key. The authority database stores only the provider key ID and
    encrypted records; it never stores the returned plaintext key.
    """

    @property
    def active_key_id(self) -> str:
        """Return the identifier used to encrypt newly committed records."""

        ...

    def load_data_key(self, key_id: str) -> bytes:
        """Return the requested 32-byte key version in process memory.

        Args:
            key_id: Persisted key version required to decrypt or encrypt a
                record. Providers retain old versions during staged rotation.
        """

        ...


@final
class AuthorityRecord(FrozenModel):
    """Public metadata for one committed encrypted authority journal record."""

    commit_index: int = Field(
        ge=1,
        description="Monotonic local projection of the replicated commit index.",
    )
    authority_term: int = Field(
        ge=1,
        description="Authority consensus term that committed this record.",
    )
    record_type: str = Field(
        min_length=1,
        max_length=80,
        description="Bounded semantic record type; never contains secret data.",
    )
    record_id: str = Field(
        min_length=1,
        max_length=160,
        description="Stable opaque record identity; never contains secret data.",
    )
    key_id: str = Field(
        min_length=1,
        max_length=160,
        description="External authority data-key version used for encryption.",
    )
    created_at: datetime = Field(
        description="Timezone-aware UTC commit timestamp.",
    )

    @field_validator("created_at")
    @classmethod
    def _created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous journal timestamps."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value


def _base64url_encode(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _cluster_public_key_bytes(identity: ClusterPublicIdentity) -> bytes:
    """Decode the validated raw Ed25519 public key from a cluster identity."""

    padding = "=" * (-len(identity.public_key) % 4)
    return base64.b64decode(
        f"{identity.public_key}{padding}",
        altchars=b"-_",
        validate=True,
    )


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(tz=timezone.utc)


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    """Serialize one secret payload deterministically before encryption."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _record_aad(
    *,
    cluster_id: UUID,
    authority_term: int,
    commit_index: int,
    record_type: str,
    record_id: str,
    key_id: str,
) -> bytes:
    """Build authenticated metadata binding ciphertext to its journal position."""

    return _canonical_json(
        {
            "authorityTerm": authority_term,
            "clusterId": str(cluster_id),
            "commitIndex": commit_index,
            "keyId": key_id,
            "recordId": record_id,
            "recordType": record_type,
            "schemaVersion": _AUTHORITY_SCHEMA_VERSION,
        }
    )


@final
class EncryptedAuthorityStore:
    """Encrypted SQLite projection of committed operator authority records.

    The store provides durable local compare-and-set and authenticated
    encryption. A later replication layer remains responsible for assigning a
    term/index after quorum commit and for fencing stale writers.
    """

    def __init__(
        self,
        key_provider: AuthorityKeyProvider,
        path: Path | None = None,
    ) -> None:
        """Create a local authority-store handle.

        Args:
            key_provider: External data-key provider. The key is never persisted
                by this class.
            path: Explicit SQLite path. Defaults to the Skulk data directory.
        """

        self._key_provider = key_provider
        self._path = (
            path
            if path is not None
            else SKULK_DATA_HOME / "operator" / "authority.sqlite3"
        )

    @property
    def path(self) -> Path:
        """Return the local encrypted authority database path."""

        return self._path

    def initialize_cluster(
        self,
        identity: ClusterPublicIdentity,
        private_key: bytes,
        *,
        authority_term: int = 1,
    ) -> AuthorityRecord:
        """Atomically initialize cluster metadata and encrypted private key.

        Args:
            identity: New public cluster identity.
            private_key: Raw Ed25519 private key bytes. They are encrypted before
                entering SQLite and never returned by this operation.
            authority_term: Initial authority term assigned by bootstrap.

        Returns:
            Metadata for the first committed journal record.

        Raises:
            AuthorityAlreadyInitializedError: The store already owns a cluster.
            ValueError: Key material or term is invalid.
        """

        if len(private_key) != 32:
            raise ValueError("cluster identity private key must contain 32 bytes")
        derived_public_key = (
            Ed25519PrivateKey.from_private_bytes(private_key)
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
        if derived_public_key != _cluster_public_key_bytes(identity):
            raise ValueError(
                "cluster identity private key does not match the public identity"
            )
        if authority_term < 1:
            raise ValueError("authority_term must be positive")
        self._prepare_path()
        with closing(self._connect()) as connection, connection:
            self._create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            if self._metadata_value(connection, "cluster_public_identity") is not None:
                raise AuthorityAlreadyInitializedError(
                    "operator authority store already has a cluster identity"
                )
            cluster_id = UUID(str(identity.cluster_id))
            record = self._insert_encrypted_record(
                connection,
                cluster_id=cluster_id,
                expected_commit_index=0,
                authority_term=authority_term,
                record_type=_CLUSTER_PRIVATE_KEY_RECORD_TYPE,
                record_id=_CLUSTER_PRIVATE_KEY_RECORD_ID,
                payload={"privateKey": _base64url_encode(private_key)},
            )
            connection.execute(
                "INSERT INTO authority_metadata(name, value) VALUES (?, ?)",
                (
                    "cluster_public_identity",
                    identity.model_dump_json(by_alias=True),
                ),
            )
            connection.execute(
                "INSERT INTO authority_metadata(name, value) VALUES (?, ?)",
                ("cluster_id", str(identity.cluster_id)),
            )
            connection.commit()
        self._harden_database_files()
        return record

    def cluster_identity(self) -> ClusterPublicIdentity:
        """Return the initialized cluster's public identity."""

        with closing(self._connect_existing()) as connection:
            raw = self._metadata_value(connection, "cluster_public_identity")
        if raw is None:
            raise AuthorityNotInitializedError(
                "operator authority store is not initialized"
            )
        return ClusterPublicIdentity.model_validate_json(raw)

    def load_cluster_private_key(self) -> bytes:
        """Decrypt the cluster private key into short-lived process memory.

        Returns:
            Raw Ed25519 private-key bytes.

        Raises:
            AuthorityIntegrityError: The record, metadata, or key is invalid.
            AuthorityNotInitializedError: Cluster bootstrap has not completed.
        """

        payload = self.read_latest_payload(
            _CLUSTER_PRIVATE_KEY_RECORD_TYPE,
            _CLUSTER_PRIVATE_KEY_RECORD_ID,
        )
        encoded = payload.get("privateKey")
        if not isinstance(encoded, str):
            raise AuthorityIntegrityError("cluster private-key record is malformed")
        padding = "=" * (-len(encoded) % 4)
        try:
            decoded = base64.b64decode(
                f"{encoded}{padding}",
                altchars=b"-_",
                validate=True,
            )
        except (ValueError, UnicodeError) as exc:
            raise AuthorityIntegrityError(
                "cluster private-key record is malformed"
            ) from exc
        if len(decoded) != 32:
            raise AuthorityIntegrityError("cluster private-key record is malformed")
        return decoded

    def append(
        self,
        *,
        expected_commit_index: int,
        authority_term: int,
        record_type: str,
        record_id: str,
        payload: Mapping[str, object],
    ) -> AuthorityRecord:
        """Append one encrypted record when the expected index still matches.

        Args:
            expected_commit_index: Last commit index observed by the caller.
            authority_term: Consensus term that authorized this append.
            record_type: Bounded non-secret semantic record type.
            record_id: Bounded non-secret opaque record identity.
            payload: Secret-bearing JSON object encrypted before persistence.

        Returns:
            Public metadata for the committed record.

        Raises:
            AuthorityCommitConflictError: Another commit changed the index.
            AuthorityNotInitializedError: Cluster bootstrap has not completed.
            ValueError: Inputs are invalid or the payload is not JSON-safe.
        """

        self._validate_record_identity(record_type, record_id)
        if expected_commit_index < 1:
            raise ValueError("expected_commit_index must be positive")
        if authority_term < 1:
            raise ValueError("authority_term must be positive")
        cluster_identity = self.cluster_identity()
        with closing(self._connect_existing()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            record = self._insert_encrypted_record(
                connection,
                cluster_id=UUID(str(cluster_identity.cluster_id)),
                expected_commit_index=expected_commit_index,
                authority_term=authority_term,
                record_type=record_type,
                record_id=record_id,
                payload=payload,
            )
            connection.commit()
        self._harden_database_files()
        return record

    def records(self, *, after_commit_index: int = 0) -> tuple[AuthorityRecord, ...]:
        """List public journal metadata after a commit index.

        Args:
            after_commit_index: Exclusive lower bound, or zero for all records.

        Returns:
            Records in ascending commit order, without ciphertext or plaintext.
        """

        if after_commit_index < 0:
            raise ValueError("after_commit_index must not be negative")
        with closing(self._connect_existing()) as connection:
            rows = self._fetch_all(
                connection.execute(
                    """
                    SELECT commit_index, authority_term, record_type, record_id,
                           key_id, created_at
                    FROM authority_journal
                    WHERE commit_index > ?
                    ORDER BY commit_index ASC
                    """,
                    (after_commit_index,),
                )
            )
        return tuple(self._record_from_row(row) for row in rows)

    def read_latest_payload(
        self,
        record_type: str,
        record_id: str,
    ) -> dict[str, object]:
        """Decrypt the newest payload for one logical record.

        Args:
            record_type: Exact record type.
            record_id: Exact record identity.

        Returns:
            Decrypted JSON object.

        Raises:
            AuthorityNotInitializedError: No matching record exists.
            AuthorityIntegrityError: Authenticated decryption or validation fails.
        """

        self._validate_record_identity(record_type, record_id)
        cluster_identity = self.cluster_identity()
        with closing(self._connect_existing()) as connection:
            row = self._fetch_one(
                connection.execute(
                    """
                    SELECT commit_index, authority_term, record_type, record_id,
                           key_id, nonce, ciphertext, created_at
                    FROM authority_journal
                    WHERE record_type = ? AND record_id = ?
                    ORDER BY commit_index DESC
                    LIMIT 1
                    """,
                    (record_type, record_id),
                )
            )
        if row is None:
            raise AuthorityNotInitializedError("authority record does not exist")
        commit_index = self._required_int(row[0], "commit_index")
        authority_term = self._required_int(row[1], "authority_term")
        persisted_record_type = self._required_str(row[2], "record_type")
        persisted_record_id = self._required_str(row[3], "record_id")
        persisted_key_id = self._required_str(row[4], "key_id")
        nonce = self._required_bytes(row[5], "nonce")
        ciphertext = self._required_bytes(row[6], "ciphertext")
        key = self._load_data_key(key_id=persisted_key_id)
        aad = _record_aad(
            cluster_id=UUID(str(cluster_identity.cluster_id)),
            authority_term=authority_term,
            commit_index=commit_index,
            record_type=persisted_record_type,
            record_id=persisted_record_id,
            key_id=persisted_key_id,
        )
        try:
            plaintext = AESGCM(key).decrypt(
                nonce,
                ciphertext,
                aad,
            )
        except (InvalidTag, ValueError) as exc:
            raise AuthorityIntegrityError(
                "authority record failed authenticated decryption"
            ) from exc
        try:
            decoded = cast(object, json.loads(plaintext))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuthorityIntegrityError("authority record is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise AuthorityIntegrityError("authority record must contain a JSON object")
        decoded_mapping = cast(dict[object, object], decoded)
        result: dict[str, object] = {}
        for key_name, value in decoded_mapping.items():
            if not isinstance(key_name, str):
                raise AuthorityIntegrityError(
                    "authority record must contain string object keys"
                )
            result[key_name] = value
        return result

    def _insert_encrypted_record(
        self,
        connection: sqlite3.Connection,
        *,
        cluster_id: UUID,
        expected_commit_index: int,
        authority_term: int,
        record_type: str,
        record_id: str,
        payload: Mapping[str, object],
    ) -> AuthorityRecord:
        """Encrypt and insert one record inside the caller's transaction."""

        self._validate_record_identity(record_type, record_id)
        current_index = self._current_commit_index(connection)
        if current_index != expected_commit_index:
            raise AuthorityCommitConflictError(
                "authority commit index changed before append"
            )
        commit_index = current_index + 1
        key_id = self._key_provider.active_key_id
        key = self._load_data_key(key_id=key_id)
        nonce = os.urandom(_AUTHORITY_NONCE_BYTES)
        aad = _record_aad(
            cluster_id=cluster_id,
            authority_term=authority_term,
            commit_index=commit_index,
            record_type=record_type,
            record_id=record_id,
            key_id=key_id,
        )
        try:
            plaintext = _canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("authority payload must be a JSON object") from exc
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
        created_at = _utc_now()
        connection.execute(
            """
            INSERT INTO authority_journal(
                commit_index, authority_term, record_type, record_id,
                key_id, nonce, ciphertext, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                commit_index,
                authority_term,
                record_type,
                record_id,
                key_id,
                nonce,
                ciphertext,
                created_at.isoformat(),
            ),
        )
        return AuthorityRecord(
            commit_index=commit_index,
            authority_term=authority_term,
            record_type=record_type,
            record_id=record_id,
            key_id=key_id,
            created_at=created_at,
        )

    def _load_data_key(self, *, key_id: str) -> bytes:
        """Load and validate the externally protected active data key."""

        try:
            key = self._key_provider.load_data_key(key_id)
        except KeyError as exc:
            raise AuthorityIntegrityError(
                "authority record key version is not available"
            ) from exc
        if len(key) != _AUTHORITY_DATA_KEY_BYTES:
            raise AuthorityIntegrityError(
                "authority key provider returned invalid key material"
            )
        return key

    def _prepare_path(self) -> None:
        """Create and harden the authority database directory."""

        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            self._path.parent.chmod(0o700)

    def _connect(self) -> sqlite3.Connection:
        """Open the authority database with durable local settings."""

        connection = sqlite3.connect(self._path, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._harden_database_files()
        return connection

    def _connect_existing(self) -> sqlite3.Connection:
        """Open an initialized authority database without creating a new one."""

        if not self._path.exists():
            raise AuthorityNotInitializedError(
                "operator authority store is not initialized"
            )
        uri = f"{self._path.resolve().as_uri()}?mode=rw"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._harden_database_files()
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create the version-one local authority schema."""

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS authority_metadata(
                name TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS authority_journal(
                commit_index INTEGER PRIMARY KEY NOT NULL CHECK(commit_index > 0),
                authority_term INTEGER NOT NULL CHECK(authority_term > 0),
                record_type TEXT NOT NULL,
                record_id TEXT NOT NULL,
                key_id TEXT NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL,
                created_at TEXT NOT NULL
            ) STRICT;
            CREATE INDEX IF NOT EXISTS authority_record_latest
            ON authority_journal(record_type, record_id, commit_index DESC);
            """
        )

    @staticmethod
    def _metadata_value(connection: sqlite3.Connection, name: str) -> str | None:
        """Read one public metadata value inside the current transaction."""

        row = EncryptedAuthorityStore._fetch_one(
            connection.execute(
                "SELECT value FROM authority_metadata WHERE name = ?",
                (name,),
            )
        )
        if row is None:
            return None
        return EncryptedAuthorityStore._required_str(row[0], "metadata value")

    @staticmethod
    def _current_commit_index(connection: sqlite3.Connection) -> int:
        """Return the greatest locally applied commit index."""

        row = EncryptedAuthorityStore._fetch_one(
            connection.execute(
                "SELECT COALESCE(MAX(commit_index), 0) FROM authority_journal"
            )
        )
        if row is None:
            return 0
        return EncryptedAuthorityStore._required_int(row[0], "commit_index")

    @staticmethod
    def _validate_record_identity(record_type: str, record_id: str) -> None:
        """Reject unbounded or ambiguous public journal metadata."""

        if not record_type or len(record_type) > 80:
            raise ValueError("record_type must contain 1 to 80 characters")
        if not record_id or len(record_id) > 160:
            raise ValueError("record_id must contain 1 to 160 characters")
        if any(character.isspace() for character in record_type):
            raise ValueError("record_type must not contain whitespace")

    @staticmethod
    def _record_from_row(row: _SqliteRow) -> AuthorityRecord:
        """Validate one public journal row as an API-neutral record."""

        return AuthorityRecord(
            commit_index=EncryptedAuthorityStore._required_int(row[0], "commit_index"),
            authority_term=EncryptedAuthorityStore._required_int(
                row[1], "authority_term"
            ),
            record_type=EncryptedAuthorityStore._required_str(row[2], "record_type"),
            record_id=EncryptedAuthorityStore._required_str(row[3], "record_id"),
            key_id=EncryptedAuthorityStore._required_str(row[4], "key_id"),
            created_at=datetime.fromisoformat(
                EncryptedAuthorityStore._required_str(row[5], "created_at")
            ),
        )

    @staticmethod
    def _fetch_one(cursor: sqlite3.Cursor) -> _SqliteRow | None:
        """Narrow sqlite's untyped row result at one audited boundary."""

        return cast(_SqliteRow | None, cursor.fetchone())

    @staticmethod
    def _fetch_all(cursor: sqlite3.Cursor) -> list[_SqliteRow]:
        """Narrow sqlite's untyped row sequence at one audited boundary."""

        return cast(list[_SqliteRow], cursor.fetchall())

    @staticmethod
    def _required_int(value: object, field_name: str) -> int:
        """Validate an integer loaded from the strict SQLite schema."""

        if not isinstance(value, int):
            raise AuthorityIntegrityError(f"authority {field_name} is not an integer")
        return value

    @staticmethod
    def _required_str(value: object, field_name: str) -> str:
        """Validate text loaded from the strict SQLite schema."""

        if not isinstance(value, str):
            raise AuthorityIntegrityError(f"authority {field_name} is not text")
        return value

    @staticmethod
    def _required_bytes(value: object, field_name: str) -> bytes:
        """Validate bytes loaded from the strict SQLite schema."""

        if not isinstance(value, bytes):
            raise AuthorityIntegrityError(f"authority {field_name} is not binary")
        return value

    def _harden_database_files(self) -> None:
        """Restrict SQLite database and sidecar files to the service account."""

        if os.name != "posix":
            return
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self._path}{suffix}")
            if candidate.exists():
                candidate.chmod(0o600)


def iter_authority_record_types(records: tuple[AuthorityRecord, ...]) -> Iterator[str]:
    """Yield record types without revealing record payloads.

    Args:
        records: Public journal metadata.

    Yields:
        Record type for each committed item.
    """

    for record in records:
        yield record.record_type
