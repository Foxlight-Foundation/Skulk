"""Bounded ownership intents for crash cleanup after native readers have exited.

The source store holds its exclusive root fence before loading or mutating this
journal. A separate per-model reader fence is inherited by the runner and native
process, so a replacement worker cannot mistake parent death for reader drain.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Literal, final

from pydantic import Field, model_validator

from skulk.shared.types.audio_separation import MAX_SEPARATION_SOURCE_BYTES
from skulk.shared.types.common import CommandId
from skulk.shared.types.tasks import AudioSeparation
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId
from skulk.utils.pydantic_ext import FrozenModel
from skulk.utils.runtime_files import RuntimeLock, read_private, write_private
from skulk.worker.runner.demucs.model import separation_result_directory
from skulk.worker.runner.demucs.result_custody import reclaim_result_directory

SOURCE_OWNERSHIP_FILE = "source-ownership.json"


def separation_reader_key(instance_id: InstanceId, runner_id: RunnerId) -> str:
    """Derive a fixed local reader directory from the authoritative placement."""
    return hashlib.sha256(
        (str(instance_id) + "\0" + str(runner_id)).encode()
    ).hexdigest()


@final
class _Ownership(FrozenModel):
    """One durable private source intent and its observed directory identity."""

    command_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    reader_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    filesystem_device: int | None = Field(
        default=None,
        ge=0,
        description="Observed filesystem device before source writes.",
    )
    inode: int | None = Field(default=None, ge=0)


@final
class _Journal(FrozenModel):
    """Bounded local ownership records, never a replicated state contract."""

    version: Literal[1] = 1
    records: tuple[_Ownership, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def validate_unique_owners(self) -> "_Journal":
        """Reject duplicate command intents rather than understating occupied slots."""
        if len({record.command_id for record in self.records}) != len(self.records):
            raise ValueError("source ownership journal repeats a command")
        return self


@final
class SeparationSourceCustody:
    """Record at most four source owners and reclaim only drained, owned copies."""

    def __init__(
        self,
        root: Path,
        readers_root: Path,
        reader_key: Callable[[AudioSeparation], str],
    ) -> None:
        """Bind private paths and authoritative placement resolution without effects."""
        self.root, self.readers_root, self.reader_key = root, readers_root, reader_key
        self._records: dict[CommandId, _Ownership] = {}

    def load(self) -> None:
        """Load bounded private ownership metadata under the source owner's fence."""
        try:
            data = read_private(self.root / SOURCE_OWNERSHIP_FILE, 16 * 1024)
        except FileNotFoundError:
            return
        journal = _Journal.model_validate_json(data)
        self._records = {
            CommandId(record.command_id): record for record in journal.records
        }

    def _save(self) -> None:
        journal = _Journal(records=tuple(self._records.values()))
        write_private(
            self.root / SOURCE_OWNERSHIP_FILE, journal.model_dump_json().encode()
        )

    def claim(self, task: AudioSeparation) -> None:
        """Persist ownership intent before creating any source directory or writer."""
        key = self.reader_key(task)
        if re.fullmatch(r"[a-f0-9]{64}", key) is None:
            raise ValueError("source owner has no authoritative native reader identity")
        if len(self._records) >= 4 or task.command_id in self._records:
            raise ValueError("source ownership journal is full or already claimed")
        record = _Ownership(command_id=str(task.command_id), reader_key=key)
        self._records[task.command_id] = record
        try:
            self._save()
        except BaseException:
            self._records.pop(task.command_id)
            raise

    def remember_directory(self, command: CommandId) -> None:
        """Persist the created directory's identity before its first input bytes."""
        info = (self.root / str(command)).lstat()
        self._records[command] = self._records[command].model_copy(
            update={"filesystem_device": info.st_dev, "inode": info.st_ino}
        )
        self._save()

    def forget(self, command: CommandId) -> None:
        """Remove the durable intent only after its source directory is absent."""
        record = self._records.pop(command, None)
        if record is None:
            return
        try:
            self._save()
        except BaseException:
            self._records[command] = record
            raise

    def reclaim_native_results(self, command: CommandId) -> bool:
        """Reclaim this command's result before forgetting its durable source intent.

        The caller must prove native read/output drain. A result can exist before
        the supervisor observes its completion, so producer registration is not
        sufficient evidence that no native files remain. False retains custody.
        """
        record = self._records.get(command)
        if record is None:
            return True
        directory = separation_result_directory(
            self.readers_root / record.reader_key, command
        )
        return reclaim_result_directory(directory)

    def recover(self, active: Collection[CommandId] = ()) -> int:
        """Reclaim stale owned copies only after an actual exclusive reader fence.

        Busy native readers, changed directory identities, foreign files and
        symbolic links stay untouched and continue to consume source capacity.
        This method never reads a PID as proof of exit or deletes recursively.
        """
        recovered = 0
        for command, record in tuple(self._records.items()):
            if command in active:
                continue
            try:
                reader = RuntimeLock(
                    self.readers_root / record.reader_key, name="attachment.lock"
                )
            except BlockingIOError:
                continue
            try:
                if not self.reclaim_native_results(command):
                    # Keep the only durable source/job intent until the exact
                    # result inode and every remaining file can be reclaimed.
                    continue
                directory = self.root / str(command)
                try:
                    info = directory.lstat()
                except FileNotFoundError:
                    self.forget(command)
                    recovered += 1
                    continue
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_mode & 0o077
                    or record.inode is not None
                    and (info.st_dev, info.st_ino)
                    != (record.filesystem_device, record.inode)
                ):
                    continue
                owned: list[Path] = []
                with os.scandir(directory) as children:
                    for entry in children:
                        details = entry.stat(follow_symlinks=False)
                        if (
                            entry.name
                            not in {"original.wav", "output.wav", "output.wav.part"}
                            or not stat.S_ISREG(details.st_mode)
                            or details.st_uid != os.getuid()
                            or details.st_nlink != 1
                            or details.st_size > MAX_SEPARATION_SOURCE_BYTES
                        ):
                            break
                        if record.inode is None:
                            # An intent without a directory receipt can precede
                            # mkdir, but never precedes legitimate input bytes.
                            # Preserve any content in this ambiguous crash gap.
                            break
                        owned.append(directory / entry.name)
                    else:
                        for path in owned:
                            path.unlink()
                        directory.rmdir()
                        descriptor = os.open(
                            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                        )
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                        self.forget(command)
                        recovered += 1
            finally:
                reader.close()
        return recovered
