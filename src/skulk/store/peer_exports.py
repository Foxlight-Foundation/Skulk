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
)


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

    def issue(
        self,
        model_directory: Path,
        *,
        manifest_sha256: str,
        target_node_id: str,
    ) -> ArtifactExportGrant:
        """Issue a random capability for one exact complete manifest."""

        record = read_installed_card_with_fallback(model_directory)
        if record is None or record.manifest_sha256 != manifest_sha256:
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
        return grant

    def resolve(
        self,
        token: str,
        *,
        target_node_id: str,
        relative_path: str,
    ) -> tuple[Path, ArtifactExportGrant]:
        """Resolve one permitted file without crossing the artifact boundary."""

        now = time.time()
        self._discard_expired(now)
        grant = self._grants.get(token)
        if grant is None or grant.target_node_id != target_node_id:
            raise PermissionError("invalid artifact export capability")
        if relative_path not in {entry.path for entry in grant.record.files}:
            raise PermissionError("file is outside the granted manifest")
        candidate = (grant.model_directory / relative_path).resolve()
        if (
            not candidate.is_relative_to(grant.model_directory)
            or not candidate.is_file()
        ):
            raise FileNotFoundError(relative_path)
        return candidate, grant

    def _discard_expired(self, now: float) -> None:
        for token, grant in tuple(self._grants.items()):
            if grant.expires_at <= now:
                del self._grants[token]
