"""Explicit reclamation of known, unindexed recovery work under the live lease."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import cast

from foxlight_model_preservation.models import MAX_METADATA_BYTES, TrustProfile
from foxlight_model_preservation.signing import verify_manifest
from pydantic import JsonValue, TypeAdapter

from skulk.store.model_store import ModelStore, StoreModelEntry, StoreRegistryIndex
from skulk.store.preservation_contracts import RecoveryCleanup
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.store.preservation_worker import WorkerRequest


def _metadata(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise PreservationHostError("invalid_recovery_metadata")
        data = stream.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES:
        raise PreservationHostError("recovery_metadata_too_large")
    return data


def _entries(store: Path) -> dict[str, StoreModelEntry]:
    # Destructive maintenance must not use the ordinary rebuildable index
    # reader's empty-on-error fallback: corruption is not proof of no references.
    path = store / "registry.json"
    if not path.exists() and not path.is_symlink():
        raise PreservationHostError("recovery_index_unreadable")
    data = _metadata(path)

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise PreservationHostError("recovery_index_unreadable")
            result[key] = item
        return result

    value = cast(object, json.loads(data, object_pairs_hook=unique))
    if not isinstance(value, dict):
        raise PreservationHostError("recovery_index_unreadable")
    if "schema_version" in value or "entries" in value:
        return StoreRegistryIndex.model_validate_json(data).entries
    return TypeAdapter(dict[str, StoreModelEntry]).validate_json(data)


def _referenced(
    store: Path, entries: dict[str, StoreModelEntry], targets: tuple[Path, ...]
) -> bool:
    for entry in entries.values():
        selected = (store / entry.store_path).resolve()
        if any(
            selected == path
            or selected.is_relative_to(path)
            or path.is_relative_to(selected)
            for path in targets
        ):
            return True
    return False


def _flush(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_directory(path: Path) -> None:
    # Earlier workers created the intermediate work parent using the process
    # umask. Read/traverse permission is not write authority over its children.
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o022
    ):
        raise PreservationHostError("unsafe_recovery_directory")


def reclaim_recovery_files(
    store: ModelStore,
    journal: PreservationJournal,
    job_id: str,
    manifest_id: str,
    trust: TrustProfile,
) -> RecoveryCleanup:
    """Remove only stopped job work proven unreferenced by a readable index.

    The caller holds the actual store owner's mutation lease and has drained
    this job. A durable local inode receipt precedes quarantine, so another
    crash during deletion can be retried without depending on deleted metadata.
    No hidden directory scan, model admission or permission replay occurs.
    """
    if len(job_id) != 64 or any(c not in "0123456789abcdef" for c in job_id):
        raise PreservationHostError("invalid_recovery_identity")
    root = store.store_path.resolve()
    entries = _entries(root)
    removed: dict[str, bool] = {}
    retained = False
    for kind, parent in (("generation", ".recoveries"), ("work", ".recovery-work")):
        source = root / parent / job_id
        trash = root / ".recovery-trash" / (job_id + "-" + kind)
        removed[kind] = False
        # Check the parent before resolving references or inspecting any child.
        for directory in (source.parent, trash.parent):
            if directory.exists() or directory.is_symlink():
                _owned_directory(directory)
        if _referenced(root, entries, (source, trash)):
            retained = True
            continue
        key = "reclaim:" + job_id + ":" + kind
        receipt = journal.setting(key)
        source_exists = source.exists() or source.is_symlink()
        trash_exists = trash.exists() or trash.is_symlink()
        if not source_exists and not trash_exists:
            continue
        if source_exists and trash_exists:
            raise PreservationHostError("recovery_cleanup_collision")
        selected = source if source_exists else trash
        _owned_directory(selected)
        info = selected.stat(follow_symlinks=False)
        identity: dict[str, JsonValue] = {
            "manifest_id": manifest_id,
            "device": info.st_dev,
            "inode": info.st_ino,
        }
        if receipt is None:
            if trash_exists:
                raise PreservationHostError("unknown_recovery_temporary_files")
            if kind == "generation":
                verify_manifest(
                    _metadata(source / "preservation" / "envelope.json"),
                    trust,
                    manifest_id,
                )
            else:
                request = WorkerRequest.model_validate_json(
                    _metadata(source / "request.json")
                )
                if (
                    request.kind != "recover"
                    or request.manifest_id != manifest_id
                    or request.trust != trust
                    or request.source_root.resolve() != root
                    or request.snapshot_parent != source
                    or request.output_path != source / "recovered"
                ):
                    raise PreservationHostError("recovery_work_identity_changed")
            journal.set_setting(key, identity)
        elif safe_json(receipt) != safe_json(identity):
            raise PreservationHostError("recovery_cleanup_identity_changed")
        if source_exists:
            secure_directory(trash.parent)
            source.rename(trash)
            _flush(source.parent)
            _flush(trash.parent)
        # shutil uses descriptor-relative symlink-safe deletion on supported
        # POSIX hosts. Refuse a platform where that guarantee is unavailable.
        if not shutil.rmtree.avoids_symlink_attacks:
            raise PreservationHostError("recovery_cleanup_unsupported")
        shutil.rmtree(trash)
        _flush(trash.parent)
        removed[kind] = True
    return RecoveryCleanup(
        job_id=job_id,
        generation_removed=removed["generation"],
        temporary_removed=removed["work"],
        indexed_generation_retained=retained,
    )
