"""Bounded private source copies for admitted native separation tasks.

This file owner reuses the generated-media assembler for raw bytes, then checks
the original WAV contract independently. It registers no transport receiver.
The receiver must select a nonterminal authoritative local task before admission;
neither a packet nor an API-supplied path can manufacture such a task.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import final

from skulk.api.video_store import VideoStore
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    source_from_wav,
)
from skulk.shared.types.common import CommandId
from skulk.shared.types.tasks import AudioSeparation, TaskStatus
from skulk.utils.runtime_files import RuntimeLock, read_private
from skulk.worker.audio_source_types import AudioSourceTask
from skulk.worker.separation_custody import (
    SOURCE_OWNERSHIP_FILE,
    SeparationSourceCustody,
)


@final
class SeparationSourceStore[SourceTask: AudioSourceTask = AudioSeparation]:
    """Own up to four encoded originals, including verified waiting inputs.

    File I/O belongs off the worker loop. Copies are ephemeral: worker/API restart
    fails interrupted jobs, and only the original owner can clean these copies.
    This store never receives a Studio asset path or deletes a user's original.
    Acknowledgement follows full encoded identity and decoded WAV-fact checks.
    """

    def __init__(
        self, root: Path, *, custody: SeparationSourceCustody[SourceTask] | None = None
    ) -> None:
        """Take an exclusive private input root without starting native execution."""
        self.root = root
        self._guard = threading.RLock()
        if custody is not None and custody.root != root:
            raise ValueError("source custody journal belongs to a different root")
        self._custody = custody
        self._ownership: RuntimeLock | None = RuntimeLock(root, name="service.lock")
        self._tasks: dict[CommandId, SourceTask] = {}
        self._ready: dict[CommandId, Path] = {}
        try:
            if custody is not None:
                custody.load()
                custody.recover()
            # A new process has no ownership proof for directories left by a
            # prior worker or placed here externally. Preserve them, but charge
            # every unresolved entry against admission across restarts. Stop the
            # bounded scan at overflow rather than retaining an arbitrary list.
            self._unresolved_count = self._count_unresolved()
            self._store = VideoStore(
                root, max_total_bytes=4 * MAX_SEPARATION_SOURCE_BYTES
            )
        except BaseException:
            self._ownership.close()
            self._ownership = None
            raise

    def _count_unresolved(self) -> int:
        count = 0
        with os.scandir(self.root) as children:
            for child in children:
                if (
                    child.name == "service.lock"
                    or self._custody is not None
                    and child.name == SOURCE_OWNERSHIP_FILE
                    or CommandId(child.name) in self._tasks
                ):
                    continue
                count += 1
                if count > 4:
                    raise ValueError(
                        "separation source root needs ownership-checked stale-copy recovery"
                    )
        return count

    def _check(self, command_id: CommandId) -> None:
        if self._ownership is None:
            raise RuntimeError("separation source owner is closed")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(command_id)) is None:
            raise ValueError("invalid separation source identity")

    def unresolved_source_count(self) -> int:
        """Count preserved prior/unknown entries that still consume input capacity.

        The startup owner must prove prior reader drain and file ownership before
        externally reconciling these entries and reopening the store. Merely
        observing terminal State never authorizes deleting an unknown file.
        """
        with self._guard:
            return self._unresolved_count

    def recover_stale(self) -> int:
        """Retry ownership-checked stale cleanup after prior native readers exit."""
        with self._guard:
            if self._ownership is None:
                raise RuntimeError("separation source owner is closed")
            if self._custody is None:
                return 0
            recovered = self._custody.recover(tuple(self._tasks))
            self._unresolved_count = self._count_unresolved()
            return recovered

    def admit(self, task: SourceTask) -> None:
        """Reserve bytes for an authoritative local task, preserving identical opens.

        The receiver must correlate current State, target instance and source
        API node before calling. Completed, failed and cancelled tasks are never
        admitted. All four slots include already verified waiting/running input.
        """
        with self._guard:
            self._check(task.command_id)
            if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
                raise ValueError("terminal separation task cannot receive source media")
            current = self._tasks.get(task.command_id)
            if current is not None:
                if (
                    current.task_id != task.task_id
                    or current.instance_id != task.instance_id
                    or current.owner_node != task.owner_node
                    or current.task_params != task.task_params
                ):
                    raise ValueError(
                        "separation source admission changed its owner or identity"
                    )
                return
            if len(self._tasks) + self._unresolved_count >= 4:
                raise ValueError(
                    "separation source capacity is full; unresolved copies require ownership-checked recovery"
                )
            directory = self.root / str(task.command_id)
            if directory.exists() or directory.is_symlink():
                raise ValueError("separation source directory needs stale-copy cleanup")
            if self._custody is not None:
                self._custody.claim(task)
            try:
                directory.mkdir(mode=0o700)
            except BaseException:
                if self._custody is not None:
                    self._custody.forget(task.command_id)
                self._unresolved_count = self._count_unresolved()
                raise
            self._tasks[task.command_id] = task
            try:
                if self._custody is not None:
                    self._custody.remember_directory(task.command_id)
                self._store.open_assembly(
                    task.command_id,
                    "music",
                    content_type="audio/wav",
                    total_bytes=task.task_params.source.size_bytes,
                    total_chunks=task.task_params.total_input_chunks,
                    keep=tuple(self._tasks),
                )
                # The directory is private throughout, including the short interval
                # in which the shared assembler creates its default-mode staging file.
                (directory / "output.wav.part").chmod(0o600)
            except BaseException:
                self.delete(task.command_id)
                raise

    def append(self, command_id: CommandId, sequence: int, data: bytes) -> None:
        """Receive a bounded raw frame only for an admitted incomplete source."""
        with self._guard:
            self._check(command_id)
            task = self._tasks.get(command_id)
            if task is None or command_id in self._ready:
                raise ValueError("no receiving separation source for this command")
            if (
                not 1 <= sequence <= task.task_params.total_input_chunks
                or not 1 <= len(data) <= SKULK_MAX_CHUNK_SIZE
            ):
                raise ValueError("separation source frame exceeds its admitted bounds")
            self._store.append(command_id, "music", sequence, data)

    def complete(self, command_id: CommandId) -> Path | None:
        """Verify a complete WAV or wait for reordered frames; return a private path.

        Receivers separately check the transport completion declaration against
        the task. This method hashes against the authoritative source, then
        validates every measured WAV fact. Any failure drops only this copy.
        """
        with self._guard:
            self._check(command_id)
            task = self._tasks.get(command_id)
            if task is None:
                raise ValueError("no admitted separation source for this command")
            ready = self._ready.get(command_id)
            if ready is not None:
                return ready
            if not self._store.assembly_complete(command_id, "music"):
                return None
            try:
                stored = self._store.commit(
                    command_id, "music", sha256=task.task_params.source.sha256,
                    total_chunks=task.task_params.total_input_chunks,
                )
                measured = source_from_wav(read_private(stored.file_path, MAX_SEPARATION_SOURCE_BYTES))
                if measured != task.task_params.source:
                    raise ValueError("separation source differs from admitted WAV facts")
                target = stored.file_path.with_name("original.wav")
                stored.file_path.replace(target)
                self._ready[command_id] = target
                return target
            except BaseException:
                self.delete(command_id)
                raise

    def ready_path(self, command_id: CommandId) -> Path | None:
        """Observe verified input only; the resident rechecks it before inference."""
        with self._guard:
            self._check(command_id)
            return self._ready.get(command_id)

    def delete(self, command_id: CommandId) -> None:
        """Release this store's source copy after its runner has stopped using it."""
        with self._guard:
            self._check(command_id)
            if command_id not in self._tasks:
                return
            if (
                self._custody is not None
                and not self._custody.reclaim_native_results(command_id)
            ):
                raise OSError("native separation result still needs owned cleanup")
            self._store.delete(command_id)
            # The shared assembler intentionally tolerates deletion failures.
            # This source owner must retain its slot/fence until bytes are gone;
            # otherwise repeated cleanup failures could bypass the input bound.
            if (self.root / str(command_id)).exists():
                raise OSError("separation source cleanup is incomplete; retry delete")
            if self._custody is not None:
                self._custody.forget(command_id)
            self._tasks.pop(command_id)
            self._ready.pop(command_id, None)

    def close(self) -> None:
        """Drop owned copies and release the fence after all resident readers drain."""
        with self._guard:
            if self._ownership is None:
                return
            for command_id in tuple(self._tasks):
                self.delete(command_id)
            self._ownership.close()
            self._ownership = None

    def detach(self) -> None:
        """Release this service fence without deleting potentially live originals.

        The worker calls this only after its receiver/file effects have joined.
        A replacement owner uses durable custody and the separate inherited
        native-reader fence to recover these files; parent shutdown is no drain
        proof. This differs from close(), which requires all readers drained.
        """
        with self._guard:
            if self._ownership is not None:
                self._ownership.close()
                self._ownership = None
