import contextlib
from pathlib import Path

import zstandard
from loguru import logger

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state_sync import StateSnapshot

_MAX_SNAPSHOTS = 3


class StateSnapshotStore:
    """Persist and prune compressed state snapshots for master bootstrap."""

    def __init__(self, directory: Path, max_snapshots: int = _MAX_SNAPSHOTS) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._max_snapshots = max_snapshots

    def write(self, snapshot: StateSnapshot) -> Path:
        safe_master_id = str(snapshot.session_id.master_node_id).replace("/", "_")
        path = (
            self._directory
            / (
                f"snapshot.{safe_master_id}.{snapshot.session_id.election_clock}."
                f"{snapshot.last_event_applied_idx}.json.zst"
            )
        )
        compressor = zstandard.ZstdCompressor()
        with open(path, "wb") as f_out, compressor.stream_writer(f_out) as writer:
            writer.write(snapshot.model_dump_json().encode("utf-8"))
        self._prune()
        return path

    def latest_for_session(self, session_id: SessionId) -> StateSnapshot | None:
        matches = [
            path
            for path in self._directory.glob("snapshot.*.json.zst")
            if self._parse_snapshot_path(path) == session_id
        ]
        if not matches:
            return None

        latest = max(matches, key=self._snapshot_index)
        decompressor = zstandard.ZstdDecompressor()
        with open(latest, "rb") as f_in, decompressor.stream_reader(f_in) as reader:
            raw = reader.read()
        return StateSnapshot.model_validate_json(raw.decode("utf-8"))

    def _prune(self) -> None:
        snapshots = sorted(
            self._directory.glob("snapshot.*.json.zst"),
            key=self._snapshot_index,
        )
        for old_snapshot in snapshots[:-self._max_snapshots]:
            with contextlib.suppress(OSError):
                old_snapshot.unlink()

    def _snapshot_index(self, path: Path) -> int:
        try:
            return int(path.name.split(".")[-3])
        except (IndexError, ValueError):
            logger.warning(f"Unexpected snapshot filename {path.name}")
            return -1

    def _parse_snapshot_path(self, path: Path) -> SessionId | None:
        parts = path.name.split(".")
        if len(parts) < 6:
            return None
        try:
            return SessionId(
                master_node_id=NodeId(parts[1]),
                election_clock=int(parts[2]),
            )
        except ValueError:
            logger.warning(f"Unexpected snapshot filename {path.name}")
            return None
