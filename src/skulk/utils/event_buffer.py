from loguru import logger
from pydantic import BaseModel


class ConflictingDuplicateIndexError(ValueError):
    """Raised when two different payloads claim the same sequence index."""

    def __init__(self, idx: int, existing: object, incoming: object) -> None:
        super().__init__(
            "Received different messages with identical indices, probable race condition"
        )
        self.idx = idx
        self.existing = existing
        self.incoming = incoming


class OrderedBuffer[T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Currently this buffer doesn't raise any errors if an event is lost
    This buffer is NOT thread safe, and is designed to only be polled from one
    source at a time.
    """

    def __init__(self):
        self.store: dict[int, T] = {}
        self.skipped: set[int] = set()
        self.next_idx_to_release: int = 0

    def ingest(self, idx: int, t: T):
        """Ingest a sequence into the buffer"""
        logger.trace(f"Ingested event {t}")
        if idx < self.next_idx_to_release:
            return
        if idx in self.skipped:
            return
        if idx in self.store:
            if self._messages_match(self.store[idx], t):
                return
            raise ConflictingDuplicateIndexError(idx, self.store[idx], t)
        self.store[idx] = t

    def truncate_from(self, idx: int) -> None:
        """Drop the buffered tail from ``idx`` onward so it can be replayed."""
        self.store = {
            stored_idx: event
            for stored_idx, event in self.store.items()
            if stored_idx < idx
        }
        self.skipped = {skipped_idx for skipped_idx in self.skipped if skipped_idx < idx}
        self.next_idx_to_release = min(self.next_idx_to_release, idx)

    def skip(self, idx: int) -> None:
        """Advance past one rejected sequence without retaining its payload."""

        if idx < self.next_idx_to_release or idx in self.store:
            return
        self.skipped.add(idx)

    def drain(self) -> list[T]:
        """Drain all available events from the buffer"""
        ret: list[T] = []
        while (
            self.next_idx_to_release in self.store
            or self.next_idx_to_release in self.skipped
        ):
            idx = self.next_idx_to_release
            if idx in self.skipped:
                self.skipped.remove(idx)
            else:
                event = self.store.pop(idx)
                ret.append(event)
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret

    def drain_indexed(self) -> list[tuple[int, T]]:
        """Drain all available events from the buffer"""
        ret: list[tuple[int, T]] = []
        while (
            self.next_idx_to_release in self.store
            or self.next_idx_to_release in self.skipped
        ):
            idx = self.next_idx_to_release
            if idx in self.skipped:
                self.skipped.remove(idx)
            else:
                event = self.store.pop(idx)
                ret.append((idx, event))
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret

    @staticmethod
    def _messages_match(existing: T, incoming: T) -> bool:
        """Compare payload identity while ignoring private debug-only fields."""
        if existing == incoming:
            return True
        if isinstance(existing, BaseModel) and isinstance(incoming, BaseModel):
            if type(existing) is not type(incoming):
                return False
            return existing.model_dump(mode="json") == incoming.model_dump(
                mode="json"
            )
        return False


class MultiSourceBuffer[SourceId, T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Tracks events with multiple sources
    """

    def __init__(self):
        self.stores: dict[SourceId, OrderedBuffer[T]] = {}

    def ingest(self, idx: int, t: T, source: SourceId):
        if source not in self.stores:
            self.stores[source] = OrderedBuffer()
        buffer = self.stores[source]
        buffer.ingest(idx, t)

    def skip(self, idx: int, source: SourceId) -> None:
        """Reject one source sequence while preserving later ordering progress."""

        if source not in self.stores:
            self.stores[source] = OrderedBuffer()
        self.stores[source].skip(idx)

    def drain(self) -> list[T]:
        ret: list[T] = []
        for store in self.stores.values():
            ret.extend(store.drain())
        return ret
