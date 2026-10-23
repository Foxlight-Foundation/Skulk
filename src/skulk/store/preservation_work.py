"""Explicit, inode-bound maintenance of stopped public-sharing job work."""

from __future__ import annotations

import os
import shutil
import stat
import time
from contextlib import ExitStack
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from skulk.store.preservation_contracts import PreservationJob, TemporaryWork
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
)
from skulk.store.preservation_worker import WorkerRequest

WORK_KINDS = frozenset(
    {"mirror_import", "http_share", "torrent_share", "peer_share", "publish", "receive"}
)
MAX_MEASURE_ENTRIES = 100_000
MEASURE_SECONDS = 2.0


def _directory(stack: ExitStack, path: str | Path, parent: int | None = None) -> int:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
    )
    stack.callback(os.close, descriptor)
    info = os.fstat(descriptor)
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise PreservationHostError("unsafe_temporary_directory")
    return descriptor


def _optional(stack: ExitStack, name: str, parent: int) -> int | None:
    try:
        return _directory(stack, name, parent)
    except FileNotFoundError:
        return None


def _exists(parent: int | None, name: str) -> bool:
    if parent is None:
        return False
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _identity(root: int, selected: int, job: PreservationJob) -> dict[str, JsonValue]:
    parent, child = os.fstat(root), os.fstat(selected)
    return {
        "root_device": parent.st_dev,
        "root_inode": parent.st_ino,
        "device": child.st_dev,
        "inode": child.st_ino,
        "kind": job.kind,
        "plan_id": job.plan_id,
    }


def register_temporary_work(
    root: Path, journal: PreservationJournal, job: PreservationJob
) -> None:
    """Record owned work identity before a sharing worker can start.

    The host has created the private request directory. The durable receipt
    permits explicit cleanup even after a crash removed part of that directory.
    This records custody only; it starts no work and grants no runtime authority.
    """
    if job.kind not in WORK_KINDS:
        return
    with ExitStack() as stack:
        parent = _directory(stack, root)
        work = _directory(stack, "work", parent)
        selected = _directory(stack, job.job_id, work)
        identity = _identity(parent, selected, job)
        key = "temporary-work:" + job.job_id
        previous = journal.setting(key)
        if previous is not None and safe_json(previous) != safe_json(identity):
            raise PreservationHostError("temporary_work_identity_changed")
        journal.set_setting(key, identity)


def temporary_work_pending(root: Path, job: PreservationJob) -> bool:
    """Read a cheap presence hint for an exact stopped job; never scan contents."""
    if job.kind not in WORK_KINDS or job.state in {"queued", "running", "cancelling"}:
        return False
    return any(
        path.exists() or path.is_symlink()
        for path in (root / "work" / job.job_id, root / "work-trash" / job.job_id)
    )


def _legacy_request(
    selected: int, root: Path, job: PreservationJob, manifest: str
) -> None:
    descriptor = os.open(
        "request.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=selected
    )
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise PreservationHostError("unknown_temporary_work")
        encoded = stream.read(1024 * 1024 + 1)
    if len(encoded) > 1024 * 1024:
        raise PreservationHostError("unknown_temporary_work")
    request = WorkerRequest.model_validate_json(encoded)
    if (
        request.kind != job.kind
        or request.manifest_id != manifest
        or request.snapshot_parent.resolve() != (root / "work" / job.job_id).resolve()
    ):
        raise PreservationHostError("temporary_work_identity_changed")


def _measure(descriptor: int) -> tuple[int, int, bool]:
    total = files = entries = 0
    deadline = time.monotonic() + MEASURE_SECONDS

    def visit(parent: int, depth: int) -> bool:
        nonlocal total, files, entries
        if depth > 64:
            return False
        with os.scandir(parent) as children:
            for child in children:
                entries += 1
                if entries > MAX_MEASURE_ENTRIES or time.monotonic() > deadline:
                    return False
                info = child.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                    files += 1
                elif stat.S_ISDIR(info.st_mode):
                    nested = os.open(
                        child.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent,
                    )
                    try:
                        if not visit(nested, depth + 1):
                            return False
                    finally:
                        os.close(nested)
        return True

    complete = visit(descriptor, 0)
    return total, files, complete


def manage_temporary_work(
    root: Path, journal: PreservationJournal, job: PreservationJob, *, remove: bool
) -> TemporaryWork:
    """Inspect or reclaim one host-authorized, drained sharing job directory.

    Args:
        root: Private custody root belonging to the exclusive journal owner.
        journal: The live owner's durable journal, never an imported grant.
        job: Exact terminal job; the caller excludes owned tasks/processes.
        remove: Explicit operator cleanup, false for a bounded read-only report.

    Returns:
        Logical regular-file counts observed before optional removal. A limited
        measurement is a lower bound, not an allocation or freed-space claim.

    Keeps archives, public copies, keys, kits and torrent artifacts intact. A
    durable inode receipt precedes quarantine so interrupted deletion is retryable.
    Legacy work needs its matching host-authored request before initial cleanup.
    """
    if (
        job.kind not in WORK_KINDS
        or job.state in {"queued", "running", "cancelling"}
        or len(job.job_id) != 64
        or any(c not in "0123456789abcdef" for c in job.job_id)
    ):
        raise PreservationHostError("temporary_job_not_stopped")
    with ExitStack() as stack:
        parent = _directory(stack, root)
        work = _optional(stack, "work", parent)
        trash = _optional(stack, "work-trash", parent)
        original, quarantined = _exists(work, job.job_id), _exists(trash, job.job_id)
        if not original and not quarantined:
            return TemporaryWork(job_id=job.job_id)
        if original and quarantined:
            raise PreservationHostError("temporary_work_collision")
        selected = _directory(stack, job.job_id, work if original else trash)
        identity = _identity(parent, selected, job)
        key = "temporary-work:" + job.job_id
        previous = journal.setting(key)
        if previous is None:
            if quarantined:
                raise PreservationHostError("unknown_temporary_work")
            plan = journal.plan(job.plan_id)[0]
            if plan.manifest_id is None:
                raise PreservationHostError("unknown_temporary_work")
            _legacy_request(selected, root, job, plan.manifest_id)
        elif safe_json(previous) != safe_json(identity):
            raise PreservationHostError("temporary_work_identity_changed")
        size, files, complete = _measure(selected)
        if remove:
            if not shutil.rmtree.avoids_symlink_attacks:
                raise PreservationHostError("temporary_cleanup_unsupported")
            if previous is None:
                journal.set_setting(key, identity)
            if original:
                if trash is None:
                    os.mkdir("work-trash", mode=0o700, dir_fd=parent)
                    trash = _directory(stack, "work-trash", parent)
                # The exclusive host queue owns this private namespace. Avoid a
                # separate reservation directory that could strand a collision
                # if Core died between reservation and the atomic rename.
                os.rename(job.job_id, job.job_id, src_dir_fd=work, dst_dir_fd=trash)
                os.fsync(cast(int, work))
                os.fsync(trash)
            info = os.stat(job.job_id, dir_fd=trash, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != (identity["device"], identity["inode"]):
                raise PreservationHostError("temporary_work_identity_changed")
            shutil.rmtree(job.job_id, dir_fd=trash)
            os.fsync(cast(int, trash))
        return TemporaryWork(
            job_id=job.job_id,
            found=True,
            logical_bytes=size,
            files=files,
            complete=complete,
            removed=remove,
        )
