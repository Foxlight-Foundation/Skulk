"""Private durable plans/jobs with retained idempotency and exclusive host ownership."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast, final

from pydantic import JsonValue

from skulk.store.preservation_contracts import (
    PreservationCopy,
    PreservationJob,
    PreservationPlan,
    PreservationStart,
)


class PreservationHostError(ValueError):
    """Stable safe host refusal; raw paths, keys and exception text stay private."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def secure_directory(path: Path) -> None:
    """Create or validate an owner-only local journal/custody directory."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise PreservationHostError("unsafe_private_directory")


def safe_json(value: object) -> str:
    """Encode trusted private state deterministically, without protocol type coercion."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@final
class PreservationJournal:
    """One private owner, bounded job display and permanent replay reservations."""

    def __init__(self, root: Path) -> None:
        secure_directory(root)
        self.root = root
        self._mutex = threading.RLock()
        self._lock = os.open(
            root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(self._lock)
            self._lock = -1
            raise PreservationHostError("preservation_host_already_running") from error
        try:
            path = root / "journal.sqlite3"
            descriptor = os.open(
                path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
            )
            info = os.fstat(descriptor)
            os.close(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                self.close()
                raise PreservationHostError("unsafe_private_journal")
            self._path = path
            with self._connection() as db:
                version = cast(
                    tuple[int], db.execute("PRAGMA user_version").fetchone()
                )[0]
                if version not in {0, 1}:
                    raise PreservationHostError("unsupported_journal_version")
                db.executescript("""
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=FULL;
                    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, expires INTEGER NOT NULL, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, updated INTEGER NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS requests (owner TEXT NOT NULL, operation TEXT NOT NULL, digest TEXT NOT NULL, job TEXT NOT NULL, PRIMARY KEY(owner,operation));
                    CREATE TABLE IF NOT EXISTS copies (id TEXT PRIMARY KEY, created INTEGER NOT NULL, value TEXT NOT NULL);
                    PRAGMA user_version=1;
                """)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._mutex:
            if self._lock < 0:
                raise PreservationHostError("preservation_host_closed")
            connection = sqlite3.connect(self._path, timeout=5, isolation_level=None)
            try:
                connection.execute("PRAGMA synchronous=FULL")
                yield connection
            finally:
                connection.close()

    def close(self) -> None:
        """Release ownership only after the caller has stopped every owned worker."""
        if self._lock >= 0:
            os.close(self._lock)
            self._lock = -1

    def setting(self, key: str) -> JsonValue:
        """Read trusted private configuration; this is not a browser response."""
        with self._connection() as db:
            row = cast(
                tuple[str] | None,
                db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone(),
            )
        return cast(JsonValue, json.loads(row[0])) if row is not None else None

    def set_setting(self, key: str, value: JsonValue) -> None:
        """Persist private enrollment/custody metadata without returning secrets."""
        encoded = safe_json(value)
        if len(encoded.encode()) > 8 * 1024 * 1024:
            raise PreservationHostError("private_metadata_too_large")
        with self._connection() as db:
            db.execute(
                "INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def add_plan(self, plan: PreservationPlan, payload: dict[str, JsonValue]) -> None:
        """Retain the exact reviewed summary and host-only immutable inputs."""
        value = safe_json({"view": plan.model_dump(mode="json"), "payload": payload})
        if len(value.encode()) > 8 * 1024 * 1024:
            raise PreservationHostError("plan_too_large")
        with self._connection() as db:
            db.execute(
                "DELETE FROM plans WHERE expires<? AND id NOT IN (SELECT json_extract(value, '$.plan_id') FROM jobs)",
                (int(time.time()),),
            )
            count = cast(
                tuple[int], db.execute("SELECT count(*) FROM plans").fetchone()
            )[0]
            if count >= 1024:
                raise PreservationHostError("plan_capacity_reached")
            existing = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM plans WHERE id=?", (plan.plan_id,)
                ).fetchone(),
            )
            if existing is not None and existing[0] != value:
                raise PreservationHostError("plan_identity_changed")
            db.execute(
                "INSERT OR IGNORE INTO plans VALUES (?,?,?)",
                (plan.plan_id, plan.expires_at, value),
            )

    def plan(self, plan_id: str) -> tuple[PreservationPlan, dict[str, JsonValue]]:
        """Read an existing plan without causing any archive/store effect."""
        with self._connection() as db:
            row = cast(
                tuple[str] | None,
                db.execute("SELECT value FROM plans WHERE id=?", (plan_id,)).fetchone(),
            )
        if row is None:
            raise PreservationHostError("plan_missing_or_expired")
        record = cast(dict[str, JsonValue], json.loads(row[0]))
        return (
            PreservationPlan.model_validate_json(safe_json(record["view"])),
            cast(dict[str, JsonValue], record["payload"]),
        )

    def reserve(self, request: PreservationStart) -> tuple[PreservationJob, bool]:
        """Atomically reserve one exact request; retries never create another job."""
        request_digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        job_id = hashlib.sha256(
            (request.owner_id + "\x00" + request.operation_id).encode()
        ).hexdigest()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                reservation = cast(
                    tuple[str, str] | None,
                    db.execute(
                        "SELECT digest,job FROM requests WHERE owner=? AND operation=?",
                        (request.owner_id, request.operation_id),
                    ).fetchone(),
                )
                if reservation is not None:
                    if reservation[0] != request_digest:
                        raise PreservationHostError("request_changed")
                    row = cast(
                        tuple[str] | None,
                        db.execute(
                            "SELECT value FROM jobs WHERE id=?", (reservation[1],)
                        ).fetchone(),
                    )
                    if row is None:
                        raise PreservationHostError("operation_retired")
                    result = PreservationJob.model_validate_json(row[0])
                    db.execute("COMMIT")
                    return result, False
                plan_row = cast(
                    tuple[str] | None,
                    db.execute(
                        "SELECT value FROM plans WHERE id=?", (request.plan_id,)
                    ).fetchone(),
                )
                if plan_row is None:
                    raise PreservationHostError("plan_missing_or_expired")
                record = cast(dict[str, JsonValue], json.loads(plan_row[0]))
                plan = PreservationPlan.model_validate_json(safe_json(record["view"]))
                now = int(time.time())
                if plan.expires_at < now:
                    raise PreservationHostError("plan_missing_or_expired")
                if plan.requires_owner_acceptance and not request.accept_owner_capture:
                    raise PreservationHostError("owner_capture_acceptance_required")
                if plan.requires_code_permission and not request.allow_repository_code:
                    raise PreservationHostError("repository_code_permission_required")
                if plan.deleted_aliases and not request.restore_deleted:
                    raise PreservationHostError("removed_models_need_confirmation")
                if (
                    plan.kind in {"http_share", "torrent_share"}
                    and not request.public_participation
                ):
                    raise PreservationHostError("public_participation_required")
                # Retire display history, never its permanent request reservation.
                # Uncertain outcomes without a confirmed copy stay inspectable.
                retire = "SELECT id FROM jobs WHERE state IN ('succeeded','failed','cancelled') ORDER BY updated DESC,id LIMIT -1 OFFSET 256"
                db.execute(
                    "DELETE FROM settings WHERE key IN (SELECT 'request:' || id FROM ("
                    + retire
                    + "))"
                )
                db.execute("DELETE FROM jobs WHERE id IN (" + retire + ")")
                uncertain = cast(
                    tuple[int],
                    db.execute(
                        "SELECT count(*) FROM jobs WHERE state='interrupted'"
                    ).fetchone(),
                )[0]
                if uncertain >= 256:
                    raise PreservationHostError(
                        "uncertain_history_needs_reconciliation"
                    )
                count = cast(
                    tuple[int], db.execute("SELECT count(*) FROM requests").fetchone()
                )[0]
                if count >= 65536:
                    raise PreservationHostError("reservation_capacity_reached")
                pending = cast(
                    tuple[int],
                    db.execute(
                        "SELECT count(*) FROM jobs WHERE state IN ('queued','running','cancelling')"
                    ).fetchone(),
                )[0]
                if pending >= 32:
                    raise PreservationHostError("job_queue_full")
                job = PreservationJob(
                    job_id=job_id,
                    operation_id=request.operation_id,
                    owner_id=request.owner_id,
                    plan_id=plan.plan_id,
                    kind=plan.kind,
                    model_id=plan.model_id,
                    state="queued",
                    created_at=now,
                    updated_at=now,
                )
                db.execute(
                    "INSERT INTO requests VALUES (?,?,?,?)",
                    (request.owner_id, request.operation_id, request_digest, job_id),
                )
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?)",
                    (job_id, now, job.state, job.model_dump_json()),
                )
                db.execute(
                    "INSERT INTO settings VALUES (?,?)",
                    ("request:" + job_id, request.model_dump_json()),
                )
                db.execute("COMMIT")
                return job, True
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def job(self, job_id: str) -> PreservationJob:
        """Read actual durable status; a read never queues or replays work."""
        with self._connection() as db:
            row = cast(
                tuple[str] | None,
                db.execute("SELECT value FROM jobs WHERE id=?", (job_id,)).fetchone(),
            )
        if row is None:
            raise PreservationHostError("job_not_found")
        return PreservationJob.model_validate_json(row[0])

    def update_job(self, job: PreservationJob) -> None:
        """Persist an owned worker observation, preserving immutable request binding."""
        encoded = job.model_dump_json()
        if len(encoded.encode()) > 16384 or (
            job.result is not None and len(safe_json(job.result).encode()) > 8192
        ):
            raise PreservationHostError("job_receipt_too_large")
        with self._connection() as db:
            row = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM jobs WHERE id=?", (job.job_id,)
                ).fetchone(),
            )
            if row is None:
                raise PreservationHostError("job_not_found")
            previous = PreservationJob.model_validate_json(row[0])
            if (
                previous.plan_id,
                previous.operation_id,
                previous.owner_id,
                previous.kind,
                previous.model_id,
                previous.created_at,
            ) != (
                job.plan_id,
                job.operation_id,
                job.owner_id,
                job.kind,
                job.model_id,
                job.created_at,
            ):
                raise PreservationHostError("job_identity_changed")
            if (
                previous.state in {"succeeded", "failed", "cancelled", "interrupted"}
                and previous != job
            ):
                raise PreservationHostError("job_already_terminal")
            if previous.state == "cancelling" and job.state in {"queued", "running"}:
                raise PreservationHostError("cancellation_already_requested")
            if (
                job.attempts < previous.attempts
                or job.bytes_copied < previous.bytes_copied
                or job.bytes_hashed < previous.bytes_hashed
            ):
                raise PreservationHostError("job_observation_regressed")
            if (
                previous.expected_manifest_id is not None
                and previous.expected_manifest_id != job.expected_manifest_id
            ):
                raise PreservationHostError("job_outcome_identity_changed")
            db.execute(
                "UPDATE jobs SET updated=?,state=?,value=? WHERE id=?",
                (job.updated_at, job.state, encoded, job.job_id),
            )

    def jobs(
        self, offset: int = 0, *, include_results: bool = False
    ) -> tuple[PreservationJob, ...]:
        """Read one bounded display page; large results require an exact job read."""
        if not 0 <= offset <= 2**31:
            raise PreservationHostError("invalid_offset")
        with self._connection() as db:
            rows = cast(
                list[tuple[str]],
                db.execute(
                    "SELECT value FROM jobs ORDER BY updated DESC,id LIMIT 16 OFFSET ?",
                    (offset,),
                ).fetchall(),
            )
        jobs = tuple(PreservationJob.model_validate_json(row[0]) for row in rows)
        return (
            jobs
            if include_results
            else tuple(job.model_copy(update={"result": None}) for job in jobs)
        )

    def interrupt_active(self) -> None:
        """On exclusive host restart, retain intent and require explicit reconciliation."""
        with self._connection() as db:
            rows = cast(
                list[tuple[str]],
                db.execute(
                    "SELECT value FROM jobs WHERE state IN ('queued','running','cancelling')"
                ).fetchall(),
            )
            for row in rows:
                job = PreservationJob.model_validate_json(row[0]).model_copy(
                    update={
                        "state": "interrupted",
                        "stage": "check_outcome",
                        "updated_at": int(time.time()),
                    }
                )
                db.execute(
                    "UPDATE jobs SET state=?,updated=?,value=? WHERE id=?",
                    (job.state, job.updated_at, job.model_dump_json(), job.job_id),
                )

    def add_copy(self, copy: PreservationCopy) -> None:
        """Retain a confirmed copy after actual verification; never infer from a process start."""
        with self._connection() as db:
            existing = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM copies WHERE id=?", (copy.copy_id,)
                ).fetchone(),
            )
            if existing is not None:
                prior = PreservationCopy.model_validate_json(existing[0])
                if (prior.manifest_id, prior.location_id, prior.model_id) != (
                    copy.manifest_id,
                    copy.location_id,
                    copy.model_id,
                ):
                    raise PreservationHostError("copy_identity_changed")
            db.execute(
                "INSERT INTO copies VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                (copy.copy_id, copy.created_at, copy.model_dump_json()),
            )

    def copy(self, copy_id: str) -> PreservationCopy:
        """Read one exact retained copy without checking bytes or granting sharing."""
        with self._connection() as db:
            row = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM copies WHERE id=?", (copy_id,)
                ).fetchone(),
            )
        if row is None:
            raise PreservationHostError("copy_not_found")
        return PreservationCopy.model_validate_json(row[0])

    def copies(self, offset: int = 0) -> tuple[PreservationCopy, ...]:
        """Read one bounded copy-library page."""
        if not 0 <= offset <= 2**31:
            raise PreservationHostError("invalid_offset")
        with self._connection() as db:
            rows = cast(
                list[tuple[str]],
                db.execute(
                    "SELECT value FROM copies ORDER BY created DESC,id LIMIT 16 OFFSET ?",
                    (offset,),
                ).fetchall(),
            )
        return tuple(PreservationCopy.model_validate_json(row[0]) for row in rows)
