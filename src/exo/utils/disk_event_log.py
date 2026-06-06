import contextlib
import json
import shutil
from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BufferedRandom, BufferedReader
from pathlib import Path

import msgspec
import zstandard
from loguru import logger
from pydantic import Field, TypeAdapter

from exo.shared.types.events import Event
from exo.utils.pydantic_ext import CamelCaseModel

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

_HEADER_SIZE = 4  # uint32 big-endian
_OFFSET_CACHE_SIZE = 128
_MAX_ARCHIVES = 5

_MAX_ARCHIVE_TOTAL_BYTES = 1024 * 1024 * 1024  # 1 GiB
"""Total size budget for compressed archives, enforced alongside the count
cap. Five archives of unbounded size defeated the count cap in practice
(3.5 GB of event logs observed on a 16 GB node during the launch smoke)."""

_FREE_SPACE_FLOOR_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
"""Free-disk floor below which the log proactively stops persisting.

Running the disk to zero kills nodes in ways the append-time ENOSPC guard
cannot fully contain (metadata writes at init/compact, and everything else
on the machine starts failing too — the 2026-06-06 launch smoke lost a
node this way and throttled the whole cluster first). Dropping into the
degraded counting-only mode while 2 GiB remain keeps the node alive and
loudly diagnosable instead."""

_DISK_CHECK_INTERVAL_APPENDS = 1024
"""How many appends between free-space checks (a statvfs call is cheap but
not free; per-token chunk events make append a hot path)."""


class EventLogMetadata(CamelCaseModel):
    """Metadata describing the absolute index range of the active replay tail."""

    start_idx: int = Field(ge=0)
    count: int = Field(ge=0)


def _serialize_event(event: Event) -> bytes:
    return msgspec.msgpack.encode(event.model_dump(mode="json"))


def _deserialize_event(raw: bytes) -> Event:
    # Decode msgpack into a Python dict, then re-encode as JSON for Pydantic.
    # Pydantic's validate_json() uses JSON-mode coercion (e.g. string -> enum)
    # even under strict=True, whereas validate_python() does not. Going through
    # JSON is the only way to get correct round-trip deserialization without
    # disabling strict mode or adding casts everywhere.
    as_json = json.dumps(msgspec.msgpack.decode(raw, type=dict))
    return _EVENT_ADAPTER.validate_json(as_json)


def _unpack_header(header: bytes) -> int:
    return int.from_bytes(header, byteorder="big")


def _skip_record(f: BufferedReader) -> bool:
    """Skip one length-prefixed record. Returns False on EOF."""
    header = f.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        return False
    f.seek(_unpack_header(header), 1)
    return True


def _read_record(f: BufferedReader) -> Event | None:
    """Read one length-prefixed record. Returns None on EOF."""
    header = f.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        return None
    length = _unpack_header(header)
    payload = f.read(length)
    if len(payload) < length:
        return None
    return _deserialize_event(payload)


class DiskEventLog:
    """Append-only event log backed by a file on disk.

    On-disk format: sequence of length-prefixed msgpack records.
    Each record is [4-byte big-endian uint32 length][msgpack payload].

    Uses a bounded LRU cache of event index → byte offset for efficient
    random access without storing an offset per event.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._active_path = directory / "events.bin"
        self._metadata_path = directory / "events.meta.json"
        self._offset_cache: OrderedDict[int, int] = OrderedDict()
        self._base_idx: int = 0
        self._count: int = 0
        # Set on the first failed disk write (e.g. ENOSPC); the log then
        # counts events without persisting so indices stay coherent.
        self._persistence_failed: bool = False
        self._appends_since_disk_check: int = 0
        self._file: BufferedRandom | None = None

        # A full disk at construction time must not take the component down
        # (the API event log's init-time metadata write crashed a node with
        # ENOSPC during the launch smoke): degrade to counting-only from
        # birth instead.
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            # Rotate stale active file from a previous session/crash
            if self._active_path.exists():
                self._rotate(self._active_path, self._directory)
            with contextlib.suppress(FileNotFoundError):
                self._metadata_path.unlink()

            self._file = open(self._active_path, "w+b")  # noqa: SIM115
            self._write_metadata()
        except OSError as error:
            self._enter_degraded_mode(f"initialization failed: {error}")

    @property
    def start_idx(self) -> int:
        return self._base_idx

    @property
    def active_size_bytes(self) -> int:
        """Size of the active (uncompressed) log file in bytes.

        Returns 0 in degraded mode — there is no coherent active file to
        measure, and callers use this to drive retention decisions that no
        longer apply.
        """
        if self._persistence_failed or self._file is None:
            return 0
        try:
            return self._file.tell()
        except (OSError, ValueError):
            return 0

    def _write_metadata(self) -> None:
        self._metadata_path.write_text(
            EventLogMetadata(
                start_idx=self._base_idx,
                count=self._count,
            ).model_dump_json()
        )

    def _cache_offset(self, idx: int, offset: int) -> None:
        self._offset_cache[idx] = offset
        self._offset_cache.move_to_end(idx)
        if len(self._offset_cache) > _OFFSET_CACHE_SIZE:
            self._offset_cache.popitem(last=False)

    def _seek_to(self, f: BufferedReader, target_idx: int) -> None:
        """Seek f to the byte offset of event target_idx, using cache or scanning forward."""
        if target_idx in self._offset_cache:
            self._offset_cache.move_to_end(target_idx)
            f.seek(self._offset_cache[target_idx])
            return

        # Find the highest cached index before target_idx
        scan_from_idx = self._base_idx
        scan_from_offset = 0
        for cached_idx in self._offset_cache:
            if cached_idx < target_idx:
                scan_from_idx = cached_idx
                scan_from_offset = self._offset_cache[cached_idx]

        # Scan forward, skipping records
        f.seek(scan_from_offset)
        for _ in range(scan_from_idx, target_idx):
            _skip_record(f)

        self._cache_offset(target_idx, f.tell())

    def _enter_degraded_mode(self, reason: str) -> None:
        """Switch to counting-only persistence with one CRITICAL line.

        Disposes the (possibly dirty) file handle: buffered record bytes may
        be stranded and any later flush would raise again. All write and
        read paths are short-circuited by the flag from here.
        """
        if self._persistence_failed:
            return
        self._persistence_failed = True
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
        logger.critical(
            "Event-log persistence is now DISABLED for this session "
            f"({reason}); the node continues with in-memory state only — "
            "follower replay from this node's disk log will be unavailable. "
            "Free disk space and restart to restore persistence."
        )

    def append(self, event: Event) -> None:
        # Persistence failures (disk full being the canonical case) must
        # NEVER take down the event processor: the in-memory state apply
        # and the broadcast are the critical path, and event indices derive
        # from len(self) — so the count keeps advancing even when the disk
        # write fails, keeping follower replay indices coherent. The log
        # degrades to in-memory-only with one CRITICAL line (launch E2E
        # smoke 2026-06-05: ENOSPC in _write_metadata killed the node).
        if self._persistence_failed:
            self._count += 1
            return
        assert self._file is not None
        counted = False
        try:
            packed = _serialize_event(event)
            self._file.write(len(packed).to_bytes(_HEADER_SIZE, byteorder="big"))
            self._file.write(packed)
            self._count += 1
            counted = True
            self._write_metadata()
        except OSError as error:
            # The count must advance EXACTLY once per append: a failure in
            # _write_metadata (the observed ENOSPC site) arrives here with
            # the increment already done.
            if not counted:
                self._count += 1
            self._enter_degraded_mode(f"append failed: {error}")
            return

        # Proactive free-space floor: stop persisting BEFORE the disk hits
        # zero. Running a node's disk to 0 bytes fails everything on the
        # machine (and a master in that state throttled the whole cluster
        # before dying in the 2026-06-06 smoke); degrading at the floor
        # keeps the node alive and loudly diagnosable.
        self._appends_since_disk_check += 1
        if self._appends_since_disk_check >= _DISK_CHECK_INTERVAL_APPENDS:
            self._appends_since_disk_check = 0
            try:
                free_bytes = shutil.disk_usage(self._directory).free
            except OSError:
                return
            if free_bytes < _FREE_SPACE_FLOOR_BYTES:
                self._enter_degraded_mode(
                    f"free disk space below the {_FREE_SPACE_FLOOR_BYTES / 2**30:.0f} GiB "
                    f"floor ({free_bytes / 2**30:.2f} GiB left on the event-log volume)"
                )

    def compact(self, keep_from_idx: int) -> None:
        """Discard events before ``keep_from_idx`` while preserving absolute indices."""

        if self._persistence_failed:
            # Counting-only mode: there is no coherent disk tail to rebuild.
            return

        absolute_end = len(self)
        keep_from_idx = max(keep_from_idx, self._base_idx)
        keep_from_idx = min(keep_from_idx, absolute_end)
        if keep_from_idx == self._base_idx:
            return

        expected_retained_count = absolute_end - keep_from_idx

        # Compaction must rebuild the retained tail from authoritative disk
        # positions. If a cached absolute index points at the wrong byte
        # offset, rewriting a partial tail would make the active log's absolute
        # length fall behind the in-memory state index and break future event
        # application on the master.
        self._offset_cache.clear()
        try:
            retained_events = list(self.read_range(keep_from_idx, absolute_end))
        except Exception as exc:
            self._offset_cache.clear()
            logger.opt(exception=exc).error(
                "Refusing to compact event log because the retained tail could "
                f"not be read (keep_from_idx={keep_from_idx}, "
                f"absolute_end={absolute_end}); active log left intact"
            )
            return
        if len(retained_events) != expected_retained_count:
            logger.error(
                "Refusing to compact event log because the retained tail was read "
                f"incompletely (expected={expected_retained_count}, "
                f"actual={len(retained_events)}, keep_from_idx={keep_from_idx}, "
                f"absolute_end={absolute_end})"
            )
            return

        assert self._file is not None
        # Compaction writes a full replacement file plus fresh metadata —
        # both can hit ENOSPC. Degrade to counting-only instead of letting
        # the exception escape into the master's snapshot path (the
        # init/compact metadata writes were the unguarded ENOSPC sites that
        # killed a node in the 2026-06-06 smoke).
        try:
            replacement_path = self._directory / "events.bin.tmp"
            with open(replacement_path, "w+b") as replacement_file:
                for event in retained_events:
                    packed = _serialize_event(event)
                    replacement_file.write(
                        len(packed).to_bytes(_HEADER_SIZE, byteorder="big")
                    )
                    replacement_file.write(packed)

            self._file.close()
            replacement_path.replace(self._active_path)
            self._file = open(self._active_path, "r+b")  # noqa: SIM115
            self._base_idx = keep_from_idx
            self._count = len(retained_events)
            self._offset_cache.clear()
            self._write_metadata()
        except OSError as error:
            # Index bookkeeping must stay coherent even when the rewrite
            # failed partway: keep the pre-compaction logical range (the
            # in-memory count is authoritative for future indices).
            self._enter_degraded_mode(f"compaction failed: {error}")

    def read_range(self, start: int, end: int) -> Iterator[Event]:
        """Yield events from index start (inclusive) to end (exclusive)."""
        if self._persistence_failed:
            # Counting-only mode: the disk tail is incomplete and the file
            # handle is closed — replay from this log is unavailable (the
            # append-time CRITICAL log already told the operator).
            return
        if start < 0 or end < 0:
            return
        start = max(start, self._base_idx)
        end = min(end, len(self))
        if start >= end:
            return

        base_idx_at_open = self._base_idx
        assert self._file is not None
        self._file.flush()
        with open(self._active_path, "rb") as f:
            self._seek_to(f, start)
            for idx in range(start, end):
                try:
                    event = _read_record(f)
                except Exception as exc:
                    self._offset_cache.clear()
                    logger.opt(exception=exc).error(
                        "Stopping event log read because a record could not be "
                        f"decoded (idx={idx}, start={start}, end={end})"
                    )
                    break
                if event is None:
                    break
                yield event

            # Cache where we ended up so the next sequential read is a hit
            if base_idx_at_open == self._base_idx and end < len(self):
                self._cache_offset(end, f.tell())

    def read_all(self) -> Iterator[Event]:
        """Yield all events from the log one at a time."""
        if self._persistence_failed or self._count == 0:
            return
        assert self._file is not None
        self._file.flush()
        with open(self._active_path, "rb") as f:
            for _ in range(self._count):
                event = _read_record(f)
                if event is None:
                    break
                yield event

    def __len__(self) -> int:
        return self._base_idx + self._count

    def close(self) -> None:
        """Close the file and rotate active file to compressed archive."""
        if self._persistence_failed:
            # The active file holds an incomplete tail; archiving it would
            # preserve a corrupt log. Leave it for post-mortem inspection.
            if self._file is not None:
                with contextlib.suppress(Exception):
                    self._file.close()
            return
        assert self._file is not None
        if self._file.closed:
            return
        self._file.close()
        if self._active_path.exists() and self._count > 0:
            self._rotate(self._active_path, self._directory)
        elif self._active_path.exists():
            self._active_path.unlink()
        with contextlib.suppress(FileNotFoundError):
            self._metadata_path.unlink()

    @staticmethod
    def _rotate(source: Path, directory: Path) -> None:
        """Compress source into a timestamped archive.

        Keeps at most ``_MAX_ARCHIVES`` compressed copies.  Oldest beyond
        the limit are deleted.
        """
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
            dest = directory / f"events.{stamp}.bin.zst"
            compressor = zstandard.ZstdCompressor()
            with open(source, "rb") as f_in, open(dest, "wb") as f_out:
                compressor.copy_stream(f_in, f_out)
            source.unlink()
            logger.info(f"Rotated event log: {source} -> {dest}")

            # Prune oldest archives beyond the count limit, then enforce the
            # total-bytes budget — five archives of unbounded size defeated
            # the count cap in practice.
            archives = sorted(directory.glob("events.*.bin.zst"))
            for old in archives[:-_MAX_ARCHIVES]:
                old.unlink()
            archives = sorted(directory.glob("events.*.bin.zst"))
            total_bytes = sum(archive.stat().st_size for archive in archives)
            for old in archives:
                if total_bytes <= _MAX_ARCHIVE_TOTAL_BYTES:
                    break
                total_bytes -= old.stat().st_size
                old.unlink()
                logger.info(
                    f"Pruned event-log archive {old.name} to honor the "
                    f"{_MAX_ARCHIVE_TOTAL_BYTES / 2**30:.0f} GiB archive budget"
                )
        except Exception as e:
            logger.opt(exception=e).warning(f"Failed to rotate event log {source}")
            # Clean up the source even if compression fails
            with contextlib.suppress(OSError):
                source.unlink()
