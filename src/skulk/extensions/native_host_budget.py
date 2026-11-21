"""Event-loop-owned byte/frame admission for native HTTP and host ingress."""

import asyncio
from typing import final


@final
class PayloadReservation:
    """One host-owned resource charge; it is neither a grant nor a usage cap."""

    def __init__(self, budget: "NativePayloadBudget") -> None:
        """Receive custody only after the budget registers this exact handle."""
        self._budget = budget
        self._released = False

    def shrink(self, size: int) -> None:
        """Reduce a bounded HTTP reservation to its completely received body."""
        self._budget.shrink(self, size)

    def release(self) -> None:
        """Release only this charge once, including repeated cancellation cleanup."""
        self._budget.check_loop()
        if not self._released:
            self._budget.release(self)
            self._released = True


@final
class NativePayloadBudget:
    """Bound retained raw bytes and frames across actual host connections."""

    def __init__(self, capacity: int = 64 * 1024 * 1024, frames: int = 256) -> None:
        """Bind finite concurrent resource capacity to the owning event loop."""
        if type(capacity) is not int or not 1 <= capacity <= 1024**3:
            raise ValueError("native byte capacity invalid")
        if type(frames) is not int or not 1 <= frames <= 4096:
            raise ValueError("native frame capacity invalid")
        self._capacity, self._frames = capacity, frames
        self._charges: dict[PayloadReservation, int] = {}
        self._used = 0
        self._loop = asyncio.get_running_loop()

    def check_loop(self) -> None:
        """Keep synchronous accounting on one actual owner loop."""
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("native budget loop differs")

    @property
    def used(self) -> int:
        """Return bounded resource diagnostics without payloads or call IDs."""
        return self._used

    def reserve(self, size: int) -> PayloadReservation:
        """Charge before reading a declared body; zero bodies still consume a slot."""
        self.check_loop()
        if type(size) is not int or size < 0:
            raise ValueError("native payload size invalid")
        if size > self._capacity - self._used or len(self._charges) >= self._frames:
            raise ValueError("native payload capacity unavailable")
        reservation = PayloadReservation(self)
        self._charges[reservation] = size
        self._used += size
        return reservation

    def shrink(self, reservation: PayloadReservation, size: int) -> None:
        """Return unused capacity without transferring ownership of retained bytes."""
        self.check_loop()
        old = self._charges.get(reservation)
        if old is None or type(size) is not int or not 0 <= size <= old:
            raise ValueError("native reservation shrink invalid")
        self._charges[reservation] = size
        self._used -= old - size

    def release(self, reservation: PayloadReservation) -> None:
        """Retire the exact registered handle; a forged handle cannot free capacity."""
        self.check_loop()
        if reservation not in self._charges:
            raise ValueError("native reservation differs")
        self._used -= self._charges.pop(reservation)
