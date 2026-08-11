"""Short-lived capabilities for exporting node-local staged artifacts."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import final

from skulk.store.installed_cards import (
    InstalledCardRecord,
    read_installed_card_with_fallback,
    verify_installed_card,
)

type FileSnapshot = tuple[int, int, int, int]


@dataclass(frozen=True)
class ArtifactExportGrant:
    """One random, bounded and target-bound artifact export capability."""

    token: str
    model_directory: Path
    record: InstalledCardRecord
    target_node_id: str
    byte_ceiling: int
    expires_at: float


@final
class ArtifactExportManager:
    """Issue and validate process-local artifact export capabilities."""

    def __init__(self, lifetime_seconds: float = 900.0) -> None:
        self._lifetime_seconds = lifetime_seconds
        self._grants: dict[str, ArtifactExportGrant] = {}
        self._remaining_bytes: dict[str, int] = {}
        self._file_snapshots: dict[str, dict[str, FileSnapshot]] = {}

    def issue(
        self,
        model_directory: Path,
        *,
        manifest_sha256: str,
        target_node_id: str,
    ) -> ArtifactExportGrant:
        """Issue a random capability for one exact complete manifest."""

        record = read_installed_card_with_fallback(model_directory)
        if (
            record is None
            or record.manifest_sha256 != manifest_sha256
            or not verify_installed_card(model_directory, record)
        ):
            raise ValueError("artifact manifest is unavailable or changed")
        now = time.time()
        self._discard_expired(now)
        token = secrets.token_urlsafe(32)
        grant = ArtifactExportGrant(
            token=token,
            model_directory=model_directory.resolve(),
            record=record,
            target_node_id=target_node_id,
            byte_ceiling=sum(entry.size_bytes for entry in record.files),
            expires_at=now + self._lifetime_seconds,
        )
        self._grants[token] = grant
        self._remaining_bytes[token] = grant.byte_ceiling
        self._file_snapshots[token] = {
            entry.path: _file_snapshot(model_directory / entry.path)
            for entry in record.files
        }
        return grant

    def resolve(
        self,
        token: str,
        *,
        target_node_id: str,
        relative_path: str,
        range_header: str | None = None,
    ) -> tuple[Path, ArtifactExportGrant]:
        """Authorize one bounded read without crossing the artifact boundary.

        Args:
            token: Random capability token returned by :meth:`issue`.
            target_node_id: Store node identity bound into the grant.
            relative_path: Manifest-relative file requested by the store.
            range_header: Optional single HTTP ``bytes`` range for resumption.

        Returns:
            The authorized local path and immutable grant metadata.

        Side effects:
            Reserves the response bytes against the capability's cumulative
            transfer ceiling before the file response begins.
        """

        now = time.time()
        self._discard_expired(now)
        grant = self._grants.get(token)
        if grant is None or grant.target_node_id != target_node_id:
            raise PermissionError("invalid artifact export capability")
        manifest_entry = next(
            (entry for entry in grant.record.files if entry.path == relative_path),
            None,
        )
        if manifest_entry is None:
            raise PermissionError("file is outside the granted manifest")
        candidate = (grant.model_directory / relative_path).resolve()
        if (
            not candidate.is_relative_to(grant.model_directory)
            or not candidate.is_file()
        ):
            raise FileNotFoundError(relative_path)
        issued_snapshot = self._file_snapshots.get(token, {}).get(relative_path)
        if issued_snapshot is None or _file_snapshot(candidate) != issued_snapshot:
            raise PermissionError("granted artifact file changed")
        response_bytes = _range_response_bytes(
            manifest_entry.size_bytes,
            range_header,
        )
        remaining = self._remaining_bytes.get(token, 0)
        if response_bytes > remaining:
            raise PermissionError("artifact export byte ceiling exhausted")
        self._remaining_bytes[token] = remaining - response_bytes
        return candidate, grant

    def _discard_expired(self, now: float) -> None:
        for token, grant in tuple(self._grants.items()):
            if grant.expires_at <= now:
                del self._grants[token]
                self._remaining_bytes.pop(token, None)
                self._file_snapshots.pop(token, None)


def _file_snapshot(path: Path) -> FileSnapshot:
    """Capture stable local-file identity for one short-lived capability."""

    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _range_response_bytes(file_size: int, range_header: str | None) -> int:
    """Return bytes selected by one HTTP range, rejecting ambiguous requests."""

    if range_header is None:
        return file_size
    if not range_header.startswith("bytes=") or "," in range_header:
        raise PermissionError("unsupported artifact export range")
    bounds = range_header.removeprefix("bytes=").split("-", 1)
    if len(bounds) != 2 or not bounds[0].isdigit():
        raise PermissionError("unsupported artifact export range")
    start = int(bounds[0])
    end_text = bounds[1]
    end = int(end_text) if end_text.isdigit() else file_size - 1
    if start >= file_size or end < start or end >= file_size:
        raise PermissionError("artifact export range is outside the granted file")
    return end - start + 1
