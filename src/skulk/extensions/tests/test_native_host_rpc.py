"""Real protected RPC correlation, cancellation and raw lease ownership."""

import asyncio

import pytest

from skulk.extensions.native_host_protocol import Operation, RequestFrame, ResponseFrame
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.tests.test_native_host_protocol import (
    Channels,
)


@pytest.mark.parametrize("roots", [16, 32])
async def test_stream_control_roots_reserve_media_progress(
    channels: Channels, roots: int
) -> None:
    """A full ordinary lane cannot starve admitted streams' readiness/input/output."""
    entered: asyncio.Queue[None] = asyncio.Queue()
    release = asyncio.Event()

    async def hold(_peer: Peer, _frame: RequestFrame, _body: bytes) -> Reply:
        entered.put_nowait(None)
        await release.wait()
        return Reply("complete", {}, b"ok")

    core = Peer(channels.core, hold, frozenset(), lambda: None)
    owner = Peer(
        channels.owner,
        hold,
        frozenset(
            {"infer_stream", "describe", "stream_ready", "stream_input", "stream_next"}
        ),
        lambda: None,
    )
    operations: list[Operation] = []
    operations.extend("infer_stream" for _ in range(roots))
    operations.extend("describe" for _ in range(32 - roots))
    calls = [
        asyncio.create_task(core.request(operation, {})) for operation in operations
    ]
    try:
        async with asyncio.timeout(5):
            for _ in calls:
                await entered.get()
            for operation in ("stream_ready", "stream_input", "stream_next"):
                for _ in range(roots):
                    calls.append(asyncio.create_task(core.request(operation, {})))
            for _ in range(roots * 3):
                await entered.get()
            with pytest.raises(ValueError, match="capacity"):
                await core.request("describe", {})
            if roots == 32:
                with pytest.raises(ValueError, match="capacity"):
                    await core.request("stream_next", {})
            core.check()
            owner.check()
            release.set()
            for value in await asyncio.gather(*calls):
                assert value.payload == b"ok"
                value.release()
            # All slots and byte leases are reusable after actual delivery.
            result = await core.request("describe", {})
            result.release()
            assert channels.core_budget.used == channels.owner_budget.used == 0
    finally:
        release.set()
        for call in calls:
            if not call.done():
                call.cancel()
        for value in await asyncio.gather(*calls, return_exceptions=True):
            if not isinstance(value, BaseException):
                value.release()
        await asyncio.gather(core.close(), owner.close())


async def test_bidirectional_delegate_does_not_compete_for_the_reader(
    channels: Channels,
) -> None:
    """An owner handling a root can await a Core delegation on the same socket."""

    async def core(_peer: Peer, _request: RequestFrame, body: bytes) -> Reply:
        return Reply("complete", {}, body)

    async def owner(peer: Peer, _request: RequestFrame, body: bytes) -> Reply:
        response = await peer.request("delegate", {}, body)
        assert isinstance(response.frame, ResponseFrame)
        return Reply(response.frame.outcome, {}, response.payload, response.release)

    core_peer = Peer(channels.core, core, frozenset({"delegate"}), lambda: None)
    owner_peer = Peer(channels.owner, owner, frozenset({"invoke"}), lambda: None)
    try:
        raw = b'{ "exact": "raw bytes" }'
        result = await core_peer.request("invoke", {}, raw)
        assert result.payload == raw
        assert channels.core_budget.used == len(raw)
        result.release()
    finally:
        await asyncio.gather(core_peer.close(), owner_peer.close())


async def test_cancel_waits_for_remote_cleanup_without_replaying(
    channels: Channels,
) -> None:
    """A cancelled root owns its cancellation acknowledgment and remote cleanup."""
    entered, cleaned = asyncio.Event(), asyncio.Event()
    executions = 0

    async def owner(_peer: Peer, _request: RequestFrame, _body: bytes) -> Reply:
        nonlocal executions
        executions += 1
        entered.set()
        try:
            await asyncio.Future[None]()
        finally:
            cleaned.set()
        return Reply("complete", {})

    core_peer = Peer(channels.core, owner, frozenset(), lambda: None)
    owner_peer = Peer(channels.owner, owner, frozenset({"invoke"}), lambda: None)
    try:
        call = asyncio.create_task(core_peer.request("invoke", {}, b"{}"))
        await entered.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert cleaned.is_set() and executions == 1
        assert not channels.core.closed
    finally:
        await asyncio.gather(core_peer.close(), owner_peer.close())


async def test_queued_refusal_preserves_correlation_and_the_live_channel(
    channels: Channels,
) -> None:
    """A cancelled preparation emits no request and does not reuse its sequence."""
    seen: list[int] = []

    async def echo(_peer: Peer, request: RequestFrame, body: bytes) -> Reply:
        seen.append(request.request_id)
        return Reply("complete", {}, body)

    core = Peer(channels.core, echo, frozenset(), lambda: None)
    owner = Peer(channels.owner, echo, frozenset({"invoke"}), lambda: None)
    try:

        def refused() -> None:
            raise ValueError("synthetic revoked policy")

        with pytest.raises(ValueError, match="revoked"):
            await core.request("invoke", {}, b"{}", check=refused)
        result = await core.request("invoke", {}, b"{}")
        result.release()
        assert seen == [2]
    finally:
        await asyncio.gather(core.close(), owner.close())


@pytest.mark.parametrize("partial", [False, True])
async def test_response_cancellation_distinguishes_queued_and_partial_writes(
    channels: Channels, monkeypatch: pytest.MonkeyPatch, partial: bool
) -> None:
    """Queued cancellation withdraws one reply; an uncertain write fences the peer."""
    entered = {label: asyncio.Event() for label in ("cancel", "keep")}
    withdrawn = asyncio.Event()
    released: list[str] = []

    async def reply(_peer: Peer, frame: RequestFrame, _body: bytes) -> Reply:
        label = frame.parameters["label"]
        assert isinstance(label, str)
        entered[label].set()
        return Reply(
            "complete",
            {},
            label.encode(),
            lambda: released.append(label),
            cancelled=withdrawn.set if label == "cancel" else lambda: None,
        )

    core = Peer(channels.core, reply, frozenset(), lambda: None)
    owner = Peer(
        channels.owner, reply, frozenset({"stream_next", "describe"}), lambda: None
    )
    if partial:
        write = channels.owner.writer.write
        started = False

        def prefix_only(data: bytes) -> None:
            nonlocal started
            if not started:
                started = True
                write(data)

        async def stalled_drain() -> None:
            await asyncio.Future[None]()

        # Deliver a real authenticated frame prefix, then stall the remaining
        # socket write. Cancellation can no longer prove that nothing was sent.
        monkeypatch.setattr(channels.owner.writer, "write", prefix_only)
        monkeypatch.setattr(channels.owner.writer, "drain", stalled_drain)
    else:
        await channels.owner.write_lock.acquire()
    first = asyncio.create_task(core.request("stream_next", {"label": "cancel"}))
    second = None
    try:
        async with asyncio.timeout(5):
            await entered["cancel"].wait()
            second = asyncio.create_task(core.request("describe", {"label": "keep"}))
            await entered["keep"].wait()
            first.cancel()
            if not partial:
                await withdrawn.wait()
                core.check()
                owner.check()
                channels.owner.write_lock.release()
            with pytest.raises(asyncio.CancelledError):
                await first
            if partial:
                with pytest.raises(ValueError):
                    await second
                assert channels.owner.closed and channels.core.closed
                assert not withdrawn.is_set()
            else:
                result = await second
                assert result.payload == b"keep"
                result.release()
                again = await core.request("describe", {"label": "keep"})
                again.release()
                core.check()
                owner.check()
                assert sorted(released) == ["cancel", "keep", "keep"]
    finally:
        if channels.owner.write_lock.locked() and not partial:
            channels.owner.write_lock.release()
        calls = [first] if second is None else [first, second]
        for call in calls:
            if not call.done():
                call.cancel()
        for value in await asyncio.gather(*calls, return_exceptions=True):
            if not isinstance(value, BaseException):
                value.release()
        await asyncio.gather(core.close(), owner.close())
        assert channels.owner_budget.used == channels.core_budget.used == 0
