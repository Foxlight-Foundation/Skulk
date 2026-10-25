"""Enrolled private kit generations, immutable content seals and explicit removal.

The fixed byte worker hashes prepared content. Metadata/download admission uses
the saved identity and file fingerprint, never a new full-model or kit scan.
Only an operator-confirmed job generation can be removed; no retention timer or
restart performs cleanup. The signed kit remains independently portable.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal, final

from foxlight_model_preservation.operations import OperationControl
from foxlight_model_preservation.storage import CHUNK_SIZE, directory, open_file
from pydantic import Field

from skulk.store.mirror_contracts import MirrorKit
from skulk.store.preservation_contracts import Digest, PreservationContract
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)

KIT_FILENAME = "bundle.foxkit"


@final
class KitFingerprint(PreservationContract):
    """Local descriptor identity and modification fence, never portable authority."""

    device: int = Field(description="Prepared file filesystem device.")
    inode: int = Field(description="Prepared file inode.")
    size_bytes: int = Field(
        ge=1, le=2**53 - 1, description="Exact prepared byte length."
    )
    modified_ns: int = Field(
        description="Prepared file modification timestamp in nanoseconds."
    )
    changed_ns: int = Field(
        description="Prepared inode change timestamp in nanoseconds."
    )
    mode: int = Field(description="Exact owner-only prepared file mode.")


@final
class KitSeal(PreservationContract):
    """Known-worker content hash plus local descriptor fence for that exact output."""

    manifest_id: Digest = Field(description="Expected original signed model identity.")
    sha256: Digest = Field(description="Hash of every actual prepared kit byte.")
    fingerprint: KitFingerprint = Field(
        description="Fingerprint captured after content and permissions are durable."
    )

    def public(self, *, removed: bool = False) -> MirrorKit:
        """Project content identity without leaking inode, paths or local permissions."""
        return MirrorKit(
            state="removed" if removed else "prepared",
            manifest_id=self.manifest_id,
            size_bytes=self.fingerprint.size_bytes,
            sha256=self.sha256,
        )


class _Root(PreservationContract):
    """Enrolled kit collection identity; missing storage is never recreated."""

    identity: tuple[int, int] = Field(
        description="Original collection device and inode."
    )


class _Intent(PreservationContract):
    """Durable private generation and removal intent, retained across crashes."""

    manifest_id: Digest = Field(description="Exact selected signed model identity.")
    identity: tuple[int, int] | None = Field(
        default=None,
        description="Enrolled generation device/inode, absent only before creation.",
    )
    phase: Literal["active", "removing", "removed"] = Field(
        default="active",
        description="Explicit generation/removal phase; restart never performs deletion.",
    )
    seal: KitSeal | None = Field(
        default=None, description="Known-worker prepared content receipt."
    )


def kit_fingerprint(descriptor: int) -> KitFingerprint:
    """Observe a private regular file descriptor without reading its content."""
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise PreservationHostError("mirror_kit_file_changed")
    return KitFingerprint(
        device=info.st_dev,
        inode=info.st_ino,
        size_bytes=info.st_size,
        modified_ns=info.st_mtime_ns,
        changed_ns=info.st_ctime_ns,
        mode=stat.S_IMODE(info.st_mode),
    )


def seal_kit(target: int, manifest_id: str, control: OperationControl) -> KitSeal:
    """Hash one completed kit in the owned byte worker, preserving cancellation.

    The caller has verified the original model closure. This extra content digest
    binds resumable downloads to the prepared artifact, without doing byte work
    in GET handlers. No data outside the fixed output filename is selected.
    """
    with open_file(target, KIT_FILENAME) as source:
        before = kit_fingerprint(source.fileno())
        checksum = hashlib.sha256()
        while True:
            control.checkpoint()
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            checksum.update(chunk)
            control.advance(hashed=len(chunk))
        if kit_fingerprint(source.fileno()) != before:
            raise PreservationHostError("mirror_kit_file_changed")
        os.fchmod(source.fileno(), 0o400)
        os.fsync(source.fileno())
        return KitSeal(
            manifest_id=manifest_id,
            sha256=checksum.hexdigest(),
            fingerprint=kit_fingerprint(source.fileno()),
        )


@final
class MirrorKitStore:
    """Fixed owner-local generations bound to permanent preparation jobs."""

    def __init__(self, root: Path, journal: PreservationJournal) -> None:
        """Select the fixed private collection without creating or scanning storage."""
        self.path, self.journal = root / "mirror-kits", journal
        self._mutex = threading.RLock()

    @staticmethod
    def _key(job_id: str) -> str:
        if re.fullmatch(r"[a-f0-9]{64}", job_id) is None:
            raise PreservationHostError("invalid_job_id")
        return "mirror-kit:" + job_id

    def _intent(self, job_id: str) -> _Intent:
        raw = self.journal.setting(self._key(job_id))
        if raw is None:
            raise PreservationHostError("mirror_kit_not_prepared")
        return _Intent.model_validate_json(safe_json(raw))

    def _save(self, job_id: str, value: _Intent) -> None:
        self.journal.set_setting(self._key(job_id), value.model_dump(mode="json"))

    @contextmanager
    def _root(self, *, create: bool = False) -> Generator[int]:
        with self._mutex, self._enrolled_root(create=create) as root:
            yield root

    @contextmanager
    def _enrolled_root(self, *, create: bool = False) -> Generator[int]:
        raw = self.journal.setting("mirror-kits-root")
        if raw is None:
            if not create:
                raise PreservationHostError("mirror_kit_not_prepared")
            secure_directory(self.path)
            with directory(self.path) as root:
                if os.listdir(root):
                    raise PreservationHostError("mirror_kit_directory_changed")
                info = os.fstat(root)
                selected = _Root(identity=(info.st_dev, info.st_ino))
                with directory(self.path.parent) as parent:
                    os.fsync(parent)
                self.journal.set_setting(
                    "mirror-kits-root", selected.model_dump(mode="json")
                )
        else:
            selected = _Root.model_validate_json(safe_json(raw))
        with directory(self.path) as root:
            info = os.fstat(root)
            if (
                (info.st_dev, info.st_ino) != selected.identity
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise PreservationHostError("mirror_kit_directory_changed")
            yield root

    @contextmanager
    def _generation(self, root: int, job_id: str, intent: _Intent) -> Generator[int]:
        name = ".removed-" + job_id if intent.phase == "removing" else job_id
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
        try:
            info = os.fstat(child)
            if (
                (info.st_dev, info.st_ino) != intent.identity
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise PreservationHostError("mirror_kit_directory_changed")
            yield child
        finally:
            os.close(child)

    def prepare(self, job_id: str, manifest_id: str) -> None:
        """Enroll an empty private job directory before the fixed worker can write."""
        key = self._key(job_id)
        with self._root(create=True) as root:
            raw = self.journal.setting(key)
            intent = (
                _Intent.model_validate_json(safe_json(raw))
                if raw is not None
                else _Intent(manifest_id=manifest_id)
            )
            if (
                intent.manifest_id != manifest_id
                or intent.phase != "active"
                or intent.seal is not None
            ):
                raise PreservationHostError("mirror_kit_generation_changed")
            if raw is None:
                self._save(job_id, intent)
            if intent.identity is None:
                with suppress(FileExistsError):
                    os.mkdir(job_id, 0o700, dir_fd=root)
                child = os.open(
                    job_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
                )
                try:
                    info = os.fstat(child)
                    if (
                        os.listdir(child)
                        or info.st_uid != os.getuid()
                        or info.st_mode & 0o077
                    ):
                        raise PreservationHostError("mirror_kit_directory_changed")
                    os.fsync(root)
                    intent = intent.model_copy(
                        update={"identity": (info.st_dev, info.st_ino)}
                    )
                    self._save(job_id, intent)
                finally:
                    os.close(child)

    def target(self, job_id: str) -> int:
        """Open the enrolled private output directory for one inherited worker lease."""
        intent = self._intent(job_id)
        if intent.phase != "active" or intent.identity is None:
            raise PreservationHostError("mirror_kit_not_prepared")
        with self._root() as root, self._generation(root, job_id, intent) as target:
            return os.dup(target)

    def record(self, job_id: str, seal: KitSeal) -> MirrorKit:
        """Commit the known worker's seal only after matching the actual output."""
        intent = self._intent(job_id)
        if intent.phase != "active" or intent.manifest_id != seal.manifest_id:
            raise PreservationHostError("mirror_kit_generation_changed")
        if intent.seal is not None and intent.seal != seal:
            raise PreservationHostError("mirror_kit_generation_changed")
        with (
            self._root() as root,
            self._generation(root, job_id, intent) as target,
            open_file(target, KIT_FILENAME) as source,
        ):
            if kit_fingerprint(source.fileno()) != seal.fingerprint:
                raise PreservationHostError("mirror_kit_file_changed")
            self._save(job_id, intent.model_copy(update={"seal": seal}))
        return seal.public()

    def open(self, job_id: str) -> tuple[int, KitSeal]:
        """Pin an unchanged prepared file; reads never repeat full-content hashing."""
        intent = self._intent(job_id)
        if intent.phase != "active" or intent.seal is None:
            raise PreservationHostError("mirror_kit_not_ready")
        with self._root() as root, self._generation(root, job_id, intent) as target:
            descriptor = os.open(
                KIT_FILENAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=target
            )
            try:
                if kit_fingerprint(descriptor) != intent.seal.fingerprint:
                    raise PreservationHostError("mirror_kit_file_changed")
                return descriptor, intent.seal
            except BaseException:
                os.close(descriptor)
                raise

    def remove(self, job_id: str, confirmation: str) -> MirrorKit | None:
        """Remove only the explicitly confirmed stopped job's output generation.

        The host must first establish that no task/process can still write this
        generation. Removal intent survives crashes; only explicit retry finishes
        it. Unexpected entries, symlinks or replaced identities refuse deletion.
        """
        if confirmation != job_id:
            raise PreservationHostError("mirror_kit_confirmation_required")
        if self.journal.setting(self._key(job_id)) is None:
            return None
        intent = self._intent(job_id)
        if intent.phase == "removed":
            return intent.seal.public(removed=True) if intent.seal else None
        with self._root() as root:
            if intent.identity is None:
                # A crash before directory creation can leave only the intent.
                # Enroll only an empty directory; unknown bytes are never adopted.
                try:
                    child = os.open(
                        job_id,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=root,
                    )
                except FileNotFoundError:
                    self._save(job_id, intent.model_copy(update={"phase": "removed"}))
                    return None
                try:
                    info = os.fstat(child)
                    if (
                        os.listdir(child)
                        or info.st_uid != os.getuid()
                        or info.st_mode & 0o077
                    ):
                        raise PreservationHostError("mirror_kit_directory_changed")
                    intent = intent.model_copy(
                        update={"identity": (info.st_dev, info.st_ino)}
                    )
                    self._save(job_id, intent)
                finally:
                    os.close(child)
            if intent.phase == "active":
                # Preserve the exact quarantined name before physical changes.
                intent = intent.model_copy(update={"phase": "removing"})
                self._save(job_id, intent)
            quarantine = ".removed-" + job_id
            try:
                with self._generation(root, job_id, intent):
                    pass
            except FileNotFoundError:
                try:
                    original = intent.model_copy(update={"phase": "active"})
                    with self._generation(root, job_id, original):
                        os.rename(job_id, quarantine, src_dir_fd=root, dst_dir_fd=root)
                    os.fsync(root)
                except FileNotFoundError:
                    self._save(job_id, intent.model_copy(update={"phase": "removed"}))
                    return intent.seal.public(removed=True) if intent.seal else None
            with self._generation(root, job_id, intent) as target:
                names = os.listdir(target)
                for name in names:
                    info = os.stat(name, dir_fd=target, follow_symlinks=False)
                    if (
                        (
                            name != KIT_FILENAME
                            and re.fullmatch(r"\.kit-[a-f0-9]{32}", name) is None
                        )
                        or not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                    ):
                        raise PreservationHostError("mirror_kit_directory_changed")
                for name in names:
                    os.unlink(name, dir_fd=target)
                os.fsync(target)
            os.rmdir(quarantine, dir_fd=root)
            os.fsync(root)
            self._save(job_id, intent.model_copy(update={"phase": "removed"}))
        return intent.seal.public(removed=True) if intent.seal else None
