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

from pydantic import JsonValue, TypeAdapter

from skulk.store.discovery_contracts import DiscoveryJob, DiscoveryRequest
from skulk.store.mirror_contracts import MirrorJob, MirrorRequest
from skulk.store.preservation_contracts import (
    Digest,
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


def _receive_resume_digest(job_id: str, request: PreservationStart) -> str:
    return hashlib.sha256(
        ("receive-resume\x00" + job_id + "\x00" + request.model_dump_json()).encode()
    ).hexdigest()


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
                    CREATE TABLE IF NOT EXISTS discovery_jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, operation TEXT NOT NULL, digest TEXT NOT NULL, updated INTEGER NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, request TEXT NOT NULL, UNIQUE(owner,operation));
                    CREATE TABLE IF NOT EXISTS mirror_jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, operation TEXT NOT NULL, digest TEXT NOT NULL, updated INTEGER NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, request TEXT NOT NULL, UNIQUE(owner,operation));
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

    def private_location_ids(self) -> tuple[str, ...]:
        """List bounded private enrollments directly, without a separate index."""
        with self._connection() as db:
            rows = cast(
                list[tuple[str]],
                db.execute(
                    "SELECT key FROM settings WHERE key LIKE 'location:%' AND COALESCE(json_extract(value, '$.purpose'), 'private')='private' ORDER BY key LIMIT 17"
                ).fetchall(),
            )
        if len(rows) > 16:
            raise PreservationHostError("storage_location_limit")
        return tuple(row[0][9:] for row in rows)

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
                if plan.resume_job_id is not None:
                    raise PreservationHostError("receive_resume_action_required")
                if plan.requires_owner_acceptance and not request.accept_owner_capture:
                    raise PreservationHostError("owner_capture_acceptance_required")
                if plan.requires_code_permission and not request.allow_repository_code:
                    raise PreservationHostError("repository_code_permission_required")
                if plan.deleted_aliases and not request.restore_deleted:
                    raise PreservationHostError("removed_models_need_confirmation")
                if (
                    plan.kind
                    in {"publish", "http_share", "torrent_share", "peer_share"}
                    and not request.public_participation
                ):
                    raise PreservationHostError("public_participation_required")
                # Retire display history, never its permanent request reservation.
                # Uncertain outcomes and stopped work stay inspectable. Presence
                # pins even legacy or unsafe work: cleanup authenticates custody
                # separately, and history retirement must never strand its handle.
                retire = "SELECT id FROM jobs WHERE state IN ('succeeded','failed','cancelled') ORDER BY updated DESC,id LIMIT -1 OFFSET 256"
                candidates = cast(list[tuple[str]], db.execute(retire).fetchall())
                for (retired_id,) in candidates:
                    if any(
                        path.exists() or path.is_symlink()
                        for path in (
                            self.root / "work" / retired_id,
                            self.root / "work-trash" / retired_id,
                        )
                    ):
                        continue
                    db.execute(
                        "DELETE FROM settings WHERE key IN (?,?)",
                        ("request:" + retired_id, "temporary-work:" + retired_id),
                    )
                    db.execute("DELETE FROM jobs WHERE id=?", (retired_id,))
                uncertain = cast(
                    tuple[int],
                    db.execute(
                        "SELECT count(*) FROM jobs WHERE state='interrupted'"
                    ).fetchone(),
                )[0]
                if uncertain >= 256 and plan.kind != "check":
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
                    location_id=plan.location_id,
                    sharing_policy=plan.sharing,
                    restore_deleted=request.restore_deleted,
                    accept_owner_capture=request.accept_owner_capture,
                    allow_repository_code=request.allow_repository_code,
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

    def receive_resume_reserved(
        self, job_id: str, request: PreservationStart
    ) -> PreservationJob | None:
        """Observe a previously reserved exact resume action, without queueing."""
        with self._connection() as db:
            reservation = cast(
                tuple[str, str] | None,
                db.execute(
                    "SELECT digest,job FROM requests WHERE owner=? AND operation=?",
                    (request.owner_id, request.operation_id),
                ).fetchone(),
            )
            if reservation is None:
                return None
            if reservation != (_receive_resume_digest(job_id, request), job_id):
                raise PreservationHostError("request_changed")
            row = cast(
                tuple[str] | None,
                db.execute("SELECT value FROM jobs WHERE id=?", (job_id,)).fetchone(),
            )
            if row is None:
                raise PreservationHostError("operation_retired")
            return PreservationJob.model_validate_json(row[0])

    def resume_receive(
        self, job_id: str, request: PreservationStart
    ) -> tuple[PreservationJob, bool]:
        """Reserve a fresh explicit receive attempt without refilling its allowance.

        The review must bind unchanged original inputs and this same stopped job.
        The original execution plan, operation identity and custody remain intact;
        the new action gets its own permanent replay reservation. Repeating it
        observes the job and never queues another attempt. This cannot restart
        sharing, publication, recovery or a successful receive.
        """
        request_digest = _receive_resume_digest(job_id, request)
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
                row = cast(
                    tuple[str] | None,
                    db.execute(
                        "SELECT value FROM jobs WHERE id=?", (job_id,)
                    ).fetchone(),
                )
                if row is None:
                    raise PreservationHostError("job_not_found")
                previous = PreservationJob.model_validate_json(row[0])
                if reservation is not None:
                    if reservation != (request_digest, job_id):
                        raise PreservationHostError("request_changed")
                    db.execute("COMMIT")
                    return previous, False
                if (
                    previous.kind != "receive"
                    or previous.state not in {"failed", "cancelled", "interrupted"}
                    or previous.owner_id != request.owner_id
                ):
                    raise PreservationHostError("stopped_owned_receive_required")
                records: list[dict[str, JsonValue]] = []
                for plan_id in (previous.plan_id, request.plan_id):
                    plan_row = cast(
                        tuple[str] | None,
                        db.execute(
                            "SELECT value FROM plans WHERE id=?", (plan_id,)
                        ).fetchone(),
                    )
                    if plan_row is None:
                        raise PreservationHostError("plan_missing_or_expired")
                    records.append(cast(dict[str, JsonValue], json.loads(plan_row[0])))
                original, reviewed = records
                plan = PreservationPlan.model_validate_json(safe_json(reviewed["view"]))
                original_plan = PreservationPlan.model_validate_json(
                    safe_json(original["view"])
                )
                payload = cast(dict[str, JsonValue], reviewed["payload"])
                peer_resume = (
                    original_plan.receiving is not None
                    and original_plan.receiving.transport == "peer"
                )
                ignored = (
                    {"resume_job_id", "peer_setup_id"}
                    if peer_resume
                    else {"resume_job_id"}
                )
                original_payload = cast(dict[str, JsonValue], original["payload"])
                now = int(time.time())
                if plan.expires_at < now:
                    raise PreservationHostError("plan_missing_or_expired")
                if (
                    plan.kind != "receive"
                    or plan.resume_job_id != job_id
                    or original_plan.kind != "receive"
                    or plan.manifest_id != original_plan.manifest_id
                    or payload.get("resume_job_id") != job_id
                    or safe_json(
                        {
                            key: value
                            for key, value in payload.items()
                            if key not in ignored
                        }
                    )
                    != safe_json(
                        {
                            key: value
                            for key, value in original_payload.items()
                            if key not in ignored
                        }
                    )
                    or plan.receiving != original_plan.receiving
                ):
                    raise PreservationHostError("receive_resume_inputs_changed")
                if not request.accept_owner_capture:
                    raise PreservationHostError("owner_capture_acceptance_required")
                count = cast(
                    tuple[int], db.execute("SELECT count(*) FROM requests").fetchone()
                )[0]
                pending = cast(
                    tuple[int],
                    db.execute(
                        "SELECT count(*) FROM jobs WHERE state IN ('queued','running','cancelling')"
                    ).fetchone(),
                )[0]
                if count >= 65536:
                    raise PreservationHostError("reservation_capacity_reached")
                if pending >= 32:
                    raise PreservationHostError("job_queue_full")
                job = previous.model_copy(
                    update={
                        "state": "queued",
                        "stage": "queued",
                        "updated_at": now,
                        "finished_at": None,
                        "error_code": None,
                        "result": None,
                    }
                )
                if peer_resume:
                    setup_id = TypeAdapter[str](Digest).validate_python(
                        payload.get("peer_setup_id")
                    )
                    db.execute(
                        "INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        ("peer-attempt:" + job_id, safe_json(setup_id)),
                    )
                db.execute(
                    "INSERT INTO requests VALUES (?,?,?,?)",
                    (request.owner_id, request.operation_id, request_digest, job_id),
                )
                db.execute(
                    "UPDATE jobs SET updated=?,state=?,value=? WHERE id=?",
                    (now, job.state, job.model_dump_json(), job_id),
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
                previous.location_id,
                previous.restore_deleted,
                previous.accept_owner_capture,
                previous.allow_repository_code,
                previous.sharing_policy,
            ) != (
                job.plan_id,
                job.operation_id,
                job.owner_id,
                job.kind,
                job.model_id,
                job.created_at,
                job.location_id,
                job.restore_deleted,
                job.accept_owner_capture,
                job.allow_repository_code,
                job.sharing_policy,
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
                or job.bytes_received < previous.bytes_received
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
                previous = PreservationJob.model_validate_json(row[0])
                sharing = previous.sharing
                serving = previous.kind in {"http_share", "torrent_share", "peer_share"}
                if sharing is not None:
                    sharing = sharing.model_copy(
                        update={"ready": False, "listen_port": 0, "connected_peers": 0}
                    )
                uncertain = previous.expected_manifest_id is not None and not serving
                state = (
                    "interrupted"
                    if uncertain
                    else "cancelled"
                    if previous.state == "cancelling"
                    else "failed"
                )
                now = int(time.time())
                job = previous.model_copy(
                    update={
                        "state": state,
                        "sharing": sharing,
                        "stage": "stopped"
                        if serving
                        else "check_outcome"
                        if uncertain
                        else state,
                        "error_code": "host_restart_stopped_sharing"
                        if serving
                        else None
                        if uncertain
                        else "host_restart_before_publication",
                        "updated_at": now,
                        "finished_at": None if uncertain else now,
                    }
                )
                db.execute(
                    "UPDATE jobs SET state=?,updated=?,value=? WHERE id=?",
                    (job.state, job.updated_at, job.model_dump_json(), job.job_id),
                )

    def confirm_outcome(self, original_id: str, copy_id: str) -> None:
        """Close a recorded uncertain outcome only after matching copy verification.

        The host calls this after its actual Check worker succeeded. Ordinary
        copy checks have no original job and do nothing here. Immutable model,
        location and expected manifest bindings are rechecked under the journal
        lock; reads never perform this transition or replay the original effect.
        """
        with self._connection() as db:
            row = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM jobs WHERE id=?", (original_id,)
                ).fetchone(),
            )
            if row is None:
                return
            original = PreservationJob.model_validate_json(row[0])
            if original.state != "interrupted":
                return
            retained = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM copies WHERE id=?", (copy_id,)
                ).fetchone(),
            )
            plan_row = cast(
                tuple[str] | None,
                db.execute(
                    "SELECT value FROM plans WHERE id=?", (original.plan_id,)
                ).fetchone(),
            )
            if retained is None or plan_row is None:
                raise PreservationHostError("verified_outcome_missing")
            copy = PreservationCopy.model_validate_json(retained[0])
            record = cast(dict[str, JsonValue], json.loads(plan_row[0]))
            plan = PreservationPlan.model_validate_json(safe_json(record["view"]))
            if (copy.manifest_id, copy.location_id, copy.model_id) != (
                original.expected_manifest_id,
                plan.location_id,
                original.model_id,
            ):
                raise PreservationHostError("verified_outcome_mismatch")
            now = int(time.time())
            confirmed = original.model_copy(
                update={
                    "state": "succeeded",
                    "stage": "reconciled",
                    "copy_id": copy_id,
                    "updated_at": now,
                    "finished_at": now,
                    "error_code": None,
                    "result": {
                        "copy_id": copy_id,
                        "manifest_id": copy.manifest_id,
                    },
                }
            )
            db.execute(
                "UPDATE jobs SET state=?,updated=?,value=? WHERE id=?",
                (
                    confirmed.state,
                    confirmed.updated_at,
                    confirmed.model_dump_json(),
                    original_id,
                ),
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

    def reserve_discovery(self, request: DiscoveryRequest) -> tuple[DiscoveryJob, bool]:
        """Atomically retain a bounded metadata request; replay never queues work."""
        encoded = request.model_dump_json()
        request_digest = hashlib.sha256(encoded.encode()).hexdigest()
        identity = hashlib.sha256(
            (
                "discovery\x00" + request.owner_id + "\x00" + request.operation_id
            ).encode()
        ).hexdigest()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                previous = cast(
                    tuple[str, str] | None,
                    db.execute(
                        "SELECT digest,value FROM discovery_jobs WHERE owner=? AND operation=?",
                        (request.owner_id, request.operation_id),
                    ).fetchone(),
                )
                if previous is not None:
                    if previous[0] != request_digest:
                        raise PreservationHostError("request_changed")
                    job = DiscoveryJob.model_validate_json(previous[1])
                    db.execute("COMMIT")
                    return job, False
                count = cast(
                    tuple[int],
                    db.execute("SELECT count(*) FROM discovery_jobs").fetchone(),
                )[0]
                pending = cast(
                    tuple[int],
                    db.execute(
                        "SELECT count(*) FROM discovery_jobs WHERE state IN ('queued','running','cancelling')"
                    ).fetchone(),
                )[0]
                if count >= 1024 or pending >= 4:
                    raise PreservationHostError("discovery_job_capacity_reached")
                now = int(time.time())
                job = DiscoveryJob(
                    job_id=identity,
                    owner_id=request.owner_id,
                    operation_id=request.operation_id,
                    kind=request.kind,
                    source_id=request.source_id,
                    manifest_id=request.manifest_id,
                    state="queued",
                    created_at=now,
                    updated_at=now,
                )
                db.execute(
                    "INSERT INTO discovery_jobs VALUES (?,?,?,?,?,?,?,?)",
                    (
                        identity,
                        request.owner_id,
                        request.operation_id,
                        request_digest,
                        now,
                        job.state,
                        job.model_dump_json(),
                        encoded,
                    ),
                )
                db.execute("COMMIT")
                return job, True
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def discovery_job(self, job_id: str) -> tuple[DiscoveryJob, DiscoveryRequest]:
        """Read one retained typed metadata record without replay or network IO."""
        with self._connection() as db:
            row = cast(
                tuple[str, str] | None,
                db.execute(
                    "SELECT value,request FROM discovery_jobs WHERE id=?", (job_id,)
                ).fetchone(),
            )
        if row is None:
            raise PreservationHostError("discovery_job_not_found")
        return DiscoveryJob.model_validate_json(
            row[0]
        ), DiscoveryRequest.model_validate_json(row[1])

    def update_discovery_job(self, job: DiscoveryJob) -> None:
        """Persist a host-observed metadata state without modifying its request."""
        with self._connection() as db:
            db.execute(
                "UPDATE discovery_jobs SET updated=?,state=?,value=? WHERE id=?",
                (job.updated_at, job.state, job.model_dump_json(), job.job_id),
            )

    def discovery_jobs(
        self, owner_id: str | None = None, offset: int = 0
    ) -> tuple[DiscoveryJob, ...]:
        """Read a bounded job page, or all at most four active metadata jobs."""
        if not 0 <= offset <= 1024:
            raise PreservationHostError("invalid_offset")
        with self._connection() as db:
            rows = cast(
                list[tuple[str]],
                db.execute(
                    "SELECT value FROM discovery_jobs WHERE (owner=? OR ? IS NULL AND state IN ('queued','running','cancelling')) ORDER BY updated DESC,id LIMIT 5 OFFSET ?",
                    (owner_id, owner_id, offset),
                ).fetchall(),
            )
        return tuple(DiscoveryJob.model_validate_json(row[0]) for row in rows)

    def reserve_mirror(self, request: MirrorRequest) -> tuple[MirrorJob, bool]:
        """Retain one permanent mirror decision before any filesystem/network work.

        Repeated semantic requests observe the original receipt, even when its
        policy later changes. Two active slots allow a controller plus one live
        model control. Exhaustion never prevents observing or stopping work.
        """
        request = MirrorRequest.model_validate_json(request.model_dump_json())
        encoded = request.model_dump_json(exclude_defaults=True)
        request_digest = hashlib.sha256(encoded.encode()).hexdigest()
        identity = hashlib.sha256(
            ("mirror\x00" + request.owner_id + "\x00" + request.operation_id).encode()
        ).hexdigest()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                previous = cast(
                    tuple[str, str] | None,
                    db.execute(
                        "SELECT digest,value FROM mirror_jobs WHERE owner=? AND operation=?",
                        (request.owner_id, request.operation_id),
                    ).fetchone(),
                )
                if previous is not None:
                    if previous[0] != request_digest:
                        raise PreservationHostError("request_changed")
                    job = MirrorJob.model_validate_json(previous[1])
                    db.execute("COMMIT")
                    return job, False
                count = cast(
                    tuple[int],
                    db.execute("SELECT count(*) FROM mirror_jobs").fetchone(),
                )[0]
                pending = cast(
                    tuple[int],
                    db.execute(
                        "SELECT count(*) FROM mirror_jobs WHERE state IN ('queued','running','cancelling')"
                    ).fetchone(),
                )[0]
                if count >= 4096 or pending >= 2:
                    raise PreservationHostError("mirror_job_capacity_reached")
                now = int(time.time())
                job = MirrorJob(
                    job_id=identity,
                    owner_id=request.owner_id,
                    operation_id=request.operation_id,
                    kind=request.kind,
                    state="queued",
                    created_at=now,
                    updated_at=now,
                    manifest_id=request.manifest_id,
                )
                db.execute(
                    "INSERT INTO mirror_jobs VALUES (?,?,?,?,?,?,?,?)",
                    (
                        identity,
                        request.owner_id,
                        request.operation_id,
                        request_digest,
                        now,
                        job.state,
                        job.model_dump_json(),
                        encoded,
                    ),
                )
                db.execute("COMMIT")
                return job, True
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def mirror_job(self, job_id: str) -> tuple[MirrorJob, MirrorRequest]:
        """Read the original typed mirror decision and its last durable receipt."""
        with self._connection() as db:
            row = cast(
                tuple[str, str] | None,
                db.execute(
                    "SELECT value,request FROM mirror_jobs WHERE id=?", (job_id,)
                ).fetchone(),
            )
        if row is None:
            raise PreservationHostError("mirror_job_not_found")
        return MirrorJob.model_validate_json(row[0]), MirrorRequest.model_validate_json(
            row[1]
        )

    def update_mirror_job(self, job: MirrorJob) -> None:
        """Persist a bounded mirror observation without rewriting its authority."""
        job = MirrorJob.model_validate_json(job.model_dump_json())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = cast(
                    tuple[str] | None,
                    db.execute(
                        "SELECT value FROM mirror_jobs WHERE id=?", (job.job_id,)
                    ).fetchone(),
                )
                if row is None:
                    raise PreservationHostError("mirror_job_not_found")
                old = MirrorJob.model_validate_json(row[0])
                if (old.owner_id, old.operation_id, old.kind, old.manifest_id) != (
                    job.owner_id,
                    job.operation_id,
                    job.kind,
                    job.manifest_id,
                ):
                    raise PreservationHostError("request_changed")
                if old.state == "cancelling" and job.state in {"queued", "running"}:
                    raise PreservationHostError("cancellation_already_requested")
                if (
                    old.state not in {"queued", "running", "cancelling"}
                    and job.state != old.state
                ):
                    raise PreservationHostError("mirror_job_terminal")
                if (
                    job.state not in {"queued", "running", "cancelling"}
                    and job.activity is not None
                    and job.activity.running
                ):
                    raise PreservationHostError("terminal_mirror_cannot_be_running")
                db.execute(
                    "UPDATE mirror_jobs SET updated=?,state=?,value=? WHERE id=?",
                    (job.updated_at, job.state, job.model_dump_json(), job.job_id),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def mirror_jobs(
        self, owner_id: str | None = None, offset: int = 0
    ) -> tuple[MirrorJob, ...]:
        """Read at most 21 mirror receipts; a caller pages after the first 20."""
        if not 0 <= offset <= 4096:
            raise PreservationHostError("invalid_offset")
        with self._connection() as db:
            rows = cast(
                list[tuple[str]],
                db.execute(
                    "SELECT value FROM mirror_jobs WHERE owner=? OR ? IS NULL ORDER BY updated DESC,id LIMIT 21 OFFSET ?",
                    (owner_id, owner_id, offset),
                ).fetchall(),
            )
        return tuple(MirrorJob.model_validate_json(row[0]) for row in rows)

    def interrupt_mirror_jobs(self) -> None:
        """Mark prior processes interrupted; never replay their inputs or ports."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = cast(
                    list[tuple[str]],
                    db.execute(
                        "SELECT value FROM mirror_jobs WHERE state IN ('queued','running','cancelling')"
                    ).fetchall(),
                )
                for row in rows:
                    old = MirrorJob.model_validate_json(row[0])
                    job = old.model_copy(
                        update={
                            "state": "interrupted",
                            "activity": None,
                            "updated_at": int(time.time()),
                            "error_code": "core_restarted",
                        }
                    )
                    db.execute(
                        "UPDATE mirror_jobs SET updated=?,state=?,value=? WHERE id=?",
                        (job.updated_at, job.state, job.model_dump_json(), job.job_id),
                    )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def add_discovery_review(self, identity: str, value: dict[str, JsonValue]) -> None:
        """Retain at most 64 live source-file reviews, outside model-copy plans."""
        now = int(time.time())
        with self._connection() as db:
            db.execute(
                "DELETE FROM settings WHERE key LIKE 'discovery-review:%' AND json_extract(value,'$.expires_at')<?",
                (now,),
            )
            count = cast(
                tuple[int],
                db.execute(
                    "SELECT count(*) FROM settings WHERE key LIKE 'discovery-review:%'"
                ).fetchone(),
            )[0]
            if count >= 64:
                raise PreservationHostError("discovery_review_capacity_reached")
            db.execute(
                "INSERT INTO settings VALUES (?,?)",
                ("discovery-review:" + identity, safe_json(value)),
            )
