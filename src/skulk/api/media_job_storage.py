"""Shared bounded file ownership and uncertain-write fencing for media jobs.

Domain registries still validate their own strict indexes and transitions. This
boundary handles the common private-file lifetime and durable replacement; it
never stores media, interprets model requests or launches/replays work.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, final

from skulk.utils.runtime_files import RuntimeLock, read_private, write_private


@final
class PrivateMediaJobFile:
    """Fence one typed registry's bounded bytes until its API handlers have stopped.

    The registry serializes callers under its own mutex. Any write/fsync error
    makes subsequent access fail closed: replacement may already have happened,
    so only closing and reopening can recover authoritative durable state.
    """

    def __init__(
        self, path: Path, *, limit: int,
        kind: Literal["separation", "music edit"],
        capacity_error: type[ValueError], storage_error: type[RuntimeError],
        write: Callable[[Path, bytes], None] = write_private,
    ) -> None:
        """Take an OS ownership fence without parsing or changing retained records."""
        if not 1 <= limit <= 4 * 1024**2:
            raise ValueError("Media job index limit must fit within four MiB")
        self.path, self.limit, self.kind = path, limit, kind
        self.capacity_error, self.storage_error = capacity_error, storage_error
        self.write = write
        self._fenced, self._closed = False, False
        self._ownership = RuntimeLock(path.parent, name="service.lock")

    def check(self) -> None:
        """Refuse access after close or an uncertain durable replacement."""
        if self._closed or self._fenced:
            raise self.storage_error(f"{self.kind} index must be reopened before use")

    def read(self) -> bytes | None:
        """Read bounded private bytes for domain validation; absence is a fresh index."""
        self.check()
        try:
            return read_private(self.path, self.limit)
        except FileNotFoundError:
            return None

    def replace(self, record: bytes) -> None:
        """Durably replace validated records before the caller publishes new memory."""
        self.check()
        if len(record) > self.limit:
            raise self.capacity_error(f"{self.kind} metadata exceeds its bounded index")
        try:
            self.write(self.path, record)
        except OSError as error:
            self._fenced = True
            raise self.storage_error(
                f"could not durably retain {self.kind} state; reopen its index"
            ) from error

    def close(self) -> None:
        """Release file ownership after the API's active effects have joined."""
        self._closed = True
        self._ownership.close()
