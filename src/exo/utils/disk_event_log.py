import contextlib
import json
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
        self._directory.mkdir(parents=True, exist_ok=True)
        self._active_path = directory / "events.bin"
        self._metadata_path = directory / "events.meta.json"
        self._offset_cache: OrderedDict[int, int] = OrderedDict()
        self._base_idx: int = 0
        self._count: int = 0

        # Rotate stale active file from a previous session/crash
        if self._active_path.exists():
            self._rotate(self._active_path, self._directory)
        with contextlib.suppress(FileNotFoundError):
            self._metadata_path.unlink()

        self._file: BufferedRandom = open(self._active_path, "w+b")  # noqa: SIM115
        self._write_metadata()

    @property
    def start_idx(self) -> int:
        return self._base_idx

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

    def append(self, event: Event) -> None:
        packed = _serialize_event(event)
        self._file.write(len(packed).to_bytes(_HEADER_SIZE, byteorder="big"))
        self._file.write(packed)
        self._count += 1
        self._write_metadata()

    def compact(self, keep_from_idx: int) -> None:
        """Discard events before ``keep_from_idx`` while preserving absolute indices."""

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

    def read_range(self, start: int, end: int) -> Iterator[Event]:
        """Yield events from index start (inclusive) to end (exclusive)."""
        if start < 0 or end < 0:
            return
        start = max(start, self._base_idx)
        end = min(end, len(self))
        if start >= end:
            return

        base_idx_at_open = self._base_idx
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
        if self._count == 0:
            return
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

            # Prune oldest archives beyond the limit
            archives = sorted(directory.glob("events.*.bin.zst"))
            for old in archives[:-_MAX_ARCHIVES]:
                old.unlink()
        except Exception as e:
            logger.opt(exception=e).warning(f"Failed to rotate event log {source}")
            # Clean up the source even if compression fails
            with contextlib.suppress(OSError):
                source.unlink()
