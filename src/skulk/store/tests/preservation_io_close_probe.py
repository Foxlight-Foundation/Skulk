"""Exercise completion-callback retirement in an independently bounded process."""

import asyncio

import pytest

from skulk.store.preservation_io import PreservationIO


async def main() -> None:
    """Close a finished effect before its queued retirement callback runs."""
    storage = PreservationIO()
    future: asyncio.Future[object] = asyncio.Future()
    pending = {future}
    # Reproduce the scheduler window after an effect completes but before the
    # owner's queued callback removes it. Use a real Future and event loop.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(storage, "_pending", pending)
        future.add_done_callback(pending.discard)
        future.set_result(None)
        await storage.close()
        assert not pending
    print("retirement callback drained", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
