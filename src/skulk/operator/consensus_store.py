# pyright: reportUnusedFunction=false
"""Crash-safe SQLite repository for public operator consensus metadata.

The repository stores ballots, membership, payload digests, signatures, and
certificates. It never accepts secret-bearing authority payloads or plaintext
credentials. Current state and new committed entries share one SQLite
transaction so a restart observes either the complete transition or its prior
revision.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast, final

from pydantic import ValidationError

from skulk.operator.consensus import (
    AuthorityCommittedEntry,
    AuthorityConsensusConflictError,
    AuthorityConsensusSnapshot,
    AuthorityConsensusState,
)
from skulk.operator.replication import (
    AuthorityCertificateError,
    authority_membership_digest,
    verify_quorum_certificate,
)
from skulk.shared.constants import SKULK_DATA_HOME

_CONSENSUS_SCHEMA_VERSION = 1
type _SqliteRow = tuple[object, ...]


class AuthorityConsensusAlreadyInitializedError(RuntimeError):
    """Raised when an initialized consensus repository is initialized again."""


class AuthorityConsensusNotInitializedError(RuntimeError):
    """Raised when consensus state is read before explicit initialization."""


class AuthorityConsensusIntegrityError(RuntimeError):
    """Raised when persisted consensus state or its certified log is corrupt."""


@final
class SqliteAuthorityConsensusRepository:
    """SQLite-backed compare-and-set repository for one authority participant."""

    def __init__(self, path: Path | None = None) -> None:
        """Create a repository handle without implicitly initializing state.

        Args:
            path: Explicit database path. Defaults under Skulk's data directory.
        """

        self._path = (
            path
            if path is not None
            else SKULK_DATA_HOME / "operator" / "authority-consensus.sqlite3"
        )

    @property
    def path(self) -> Path:
        """Return the consensus database path."""

        return self._path

    def initialize(self, state: AuthorityConsensusState) -> AuthorityConsensusSnapshot:
        """Atomically create revision zero and its initial certified log.

        Args:
            state: Bootstrap position, initial membership, and optional imported
                certified suffix.

        Returns:
            Newly persisted revision-zero snapshot.

        Raises:
            AuthorityConsensusAlreadyInitializedError: State already exists.
            AuthorityConsensusIntegrityError: The initial log is not contiguous.
        """

        _validate_log_continuity(state)
        self._prepare_path()
        with closing(self._connect()) as connection:
            self._create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = cast(
                    _SqliteRow | None,
                    connection.execute(
                        "SELECT 1 FROM consensus_state WHERE singleton = 1"
                    ).fetchone(),
                )
                if existing is not None:
                    raise AuthorityConsensusAlreadyInitializedError(
                        "authority consensus repository is already initialized"
                    )
                for entry in state.committed_entries:
                    self._insert_entry(connection, entry)
                connection.execute(
                    """
                    INSERT INTO consensus_state (
                        singleton, schema_version, revision, state_json
                    ) VALUES (1, ?, 0, ?)
                    """,
                    (
                        _CONSENSUS_SCHEMA_VERSION,
                        _state_json_without_entries(state),
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return AuthorityConsensusSnapshot(revision=0, state=state)

    def load(self) -> AuthorityConsensusSnapshot:
        """Return and fully validate the latest durable snapshot.

        Raises:
            AuthorityConsensusNotInitializedError: No bootstrap state exists.
            AuthorityConsensusIntegrityError: Metadata or log validation fails.
        """

        self._prepare_path()
        with closing(self._connect()) as connection:
            self._create_schema(connection)
            return self._load_from_connection(connection)

    def compare_and_set(
        self,
        expected_revision: int,
        state: AuthorityConsensusState,
    ) -> AuthorityConsensusSnapshot:
        """Persist one complete state transition at the expected revision.

        Existing committed entries are immutable and the replacement may only
        append a contiguous certified suffix. The state row and new entries are
        committed in one SQLite transaction.

        Args:
            expected_revision: Revision previously returned by :meth:`load`.
            state: Complete replacement state.

        Returns:
            Snapshot at ``expected_revision + 1``.

        Raises:
            AuthorityConsensusConflictError: The revision changed or committed
                history was removed or replaced.
            AuthorityConsensusIntegrityError: Replacement history is not valid.
        """

        _validate_log_continuity(state)
        self._prepare_path()
        with closing(self._connect()) as connection:
            self._create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._load_from_connection(connection)
                if current.revision != expected_revision:
                    raise AuthorityConsensusConflictError(
                        "authority consensus revision changed"
                    )
                if (
                    state.bootstrap_position != current.state.bootstrap_position
                    or state.bootstrap_membership != current.state.bootstrap_membership
                ):
                    raise AuthorityConsensusConflictError(
                        "authority consensus bootstrap anchor changed"
                    )
                retained_count = len(current.state.committed_entries)
                if (
                    len(state.committed_entries) < retained_count
                    or state.committed_entries[:retained_count]
                    != current.state.committed_entries
                ):
                    raise AuthorityConsensusConflictError(
                        "authority consensus transition replaced committed history"
                    )
                for entry in state.committed_entries[retained_count:]:
                    self._insert_entry(connection, entry)
                next_revision = expected_revision + 1
                updated = connection.execute(
                    """
                    UPDATE consensus_state
                    SET revision = ?, state_json = ?
                    WHERE singleton = 1 AND revision = ?
                    """,
                    (
                        next_revision,
                        _state_json_without_entries(state),
                        expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise AuthorityConsensusConflictError(
                        "authority consensus revision changed"
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return AuthorityConsensusSnapshot(revision=next_revision, state=state)

    def _prepare_path(self) -> None:
        """Create and re-harden the private repository directory."""

        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(self._path.parent, 0o700)

    def _connect(self) -> sqlite3.Connection:
        """Open one fully synchronous WAL connection and harden its file mode."""

        connection = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        self._harden_database_files()
        return connection

    def _harden_database_files(self) -> None:
        """Repair database, WAL, and shared-memory sidecar permissions."""

        if os.name != "posix":
            return
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self._path}{suffix}")
            try:
                os.chmod(candidate, 0o600)
            except FileNotFoundError:
                # SQLite creates and removes sidecars with connection lifetime.
                # Missing files are normal; every later open re-runs this repair.
                continue

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create the versioned state and append-only certificate tables."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consensus_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                revision INTEGER NOT NULL CHECK (revision >= 0),
                state_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consensus_commits (
                commit_index INTEGER PRIMARY KEY CHECK (commit_index >= 2),
                entry_json TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _insert_entry(
        connection: sqlite3.Connection,
        entry: AuthorityCommittedEntry,
    ) -> None:
        """Append one immutable certified entry inside the caller transaction."""

        connection.execute(
            """
            INSERT INTO consensus_commits (commit_index, entry_json)
            VALUES (?, ?)
            """,
            (
                entry.certificate.descriptor.commit_index,
                _model_json(entry),
            ),
        )

    @staticmethod
    def _load_from_connection(
        connection: sqlite3.Connection,
    ) -> AuthorityConsensusSnapshot:
        """Load one snapshot using the caller's connection or transaction."""

        row = cast(
            _SqliteRow | None,
            connection.execute(
                """
                SELECT schema_version, revision, state_json
                FROM consensus_state
                WHERE singleton = 1
                """
            ).fetchone(),
        )
        if row is None:
            raise AuthorityConsensusNotInitializedError(
                "authority consensus repository is not initialized"
            )
        schema_version, revision, state_json = row
        if schema_version != _CONSENSUS_SCHEMA_VERSION:
            raise AuthorityConsensusIntegrityError(
                "authority consensus schema version is unsupported"
            )
        if not isinstance(revision, int) or revision < 0:
            raise AuthorityConsensusIntegrityError(
                "authority consensus revision is invalid"
            )
        if not isinstance(state_json, str):
            raise AuthorityConsensusIntegrityError(
                "authority consensus state encoding is invalid"
            )
        try:
            base_state = AuthorityConsensusState.model_validate_json(state_json)
            entry_rows = cast(
                list[_SqliteRow],
                connection.execute(
                    """
                    SELECT entry_json
                    FROM consensus_commits
                    ORDER BY commit_index ASC
                    """
                ).fetchall(),
            )
            entry_json_values: list[str] = []
            for entry_row in entry_rows:
                if len(entry_row) != 1 or not isinstance(entry_row[0], str):
                    raise AuthorityConsensusIntegrityError(
                        "authority consensus entry encoding is invalid"
                    )
                entry_json_values.append(entry_row[0])
            entries = tuple(
                AuthorityCommittedEntry.model_validate_json(entry_json)
                for entry_json in entry_json_values
            )
            state = AuthorityConsensusState(
                bootstrap_position=base_state.bootstrap_position,
                bootstrap_membership=base_state.bootstrap_membership,
                position=base_state.position,
                active_membership=base_state.active_membership,
                promised_ballot=base_state.promised_ballot,
                accepted_ballot=base_state.accepted_ballot,
                accepted_descriptor=base_state.accepted_descriptor,
                accepted_memberships=base_state.accepted_memberships,
                committed_entries=entries,
            )
            _validate_log_continuity(state)
        except (TypeError, ValueError, ValidationError) as exc:
            raise AuthorityConsensusIntegrityError(
                "authority consensus state failed validation"
            ) from exc
        return AuthorityConsensusSnapshot(revision=revision, state=state)


def _state_json_without_entries(state: AuthorityConsensusState) -> str:
    """Serialize mutable state independently from the append-only commit log."""

    without_entries = AuthorityConsensusState(
        bootstrap_position=state.bootstrap_position,
        bootstrap_membership=state.bootstrap_membership,
        position=state.position,
        active_membership=state.active_membership,
        promised_ballot=state.promised_ballot,
        accepted_ballot=state.accepted_ballot,
        accepted_descriptor=state.accepted_descriptor,
        accepted_memberships=state.accepted_memberships,
    )
    return _model_json(without_entries)


def _model_json(model: AuthorityConsensusState | AuthorityCommittedEntry) -> str:
    """Return deterministic finite JSON for one consensus model."""

    return json.dumps(
        model.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_log_continuity(state: AuthorityConsensusState) -> None:
    """Reverify the complete certificate and membership chain from bootstrap."""

    entries = state.committed_entries
    if not entries:
        if (
            state.position != state.bootstrap_position
            or state.active_membership != state.bootstrap_membership
        ):
            raise AuthorityConsensusIntegrityError(
                "authority consensus log is missing committed entries"
            )
        return
    expected_indexes = tuple(
        range(
            state.bootstrap_position.commit_index + 1,
            state.position.commit_index + 1,
        )
    )
    actual_indexes = tuple(
        entry.certificate.descriptor.commit_index for entry in entries
    )
    if actual_indexes != expected_indexes:
        raise AuthorityConsensusIntegrityError(
            "authority consensus log is not contiguous from bootstrap"
        )

    position = state.bootstrap_position
    active_membership = state.bootstrap_membership
    try:
        for entry in entries:
            if authority_membership_digest(
                entry.memberships[0]
            ) != authority_membership_digest(active_membership):
                raise AuthorityConsensusIntegrityError(
                    "authority consensus log does not continue active membership"
                )
            position = verify_quorum_certificate(
                entry.certificate,
                entry.memberships,
                position,
            )
            if len(entry.memberships) == 2:
                active_membership = entry.memberships[1]
    except AuthorityCertificateError as exc:
        raise AuthorityConsensusIntegrityError(
            "authority consensus log certificate failed verification"
        ) from exc

    if position != state.position:
        raise AuthorityConsensusIntegrityError(
            "authority consensus log head does not match durable state"
        )
    if active_membership != state.active_membership:
        raise AuthorityConsensusIntegrityError(
            "authority consensus membership does not match certified history"
        )
