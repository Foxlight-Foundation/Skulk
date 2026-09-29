import multiprocessing as mp
import time

import pytest
from anyio import fail_after
from loguru import logger

from skulk.utils.channels import MpReceiver, MpSender, mp_channel


def foo(recv: MpReceiver[str]):
    expected = ["hi", "hi 2", "bye"]
    with recv as r:
        for item in r:
            assert item == expected.pop(0)


def bar(send: MpSender[str]):
    logger.warning("hi")
    send.send("hi")
    time.sleep(0.1)
    logger.warning("hi 2")
    send.send("hi 2")
    time.sleep(0.1)
    logger.warning("bye")
    send.send("bye")
    time.sleep(0.1)
    send.close()


@pytest.mark.anyio
async def test_channel_ipc():
    with fail_after(0.5):
        s, r = mp_channel[str]()
        p1 = mp.Process(target=foo, args=(r,))
        p2 = mp.Process(target=bar, args=(s,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()


def test_receive_timeout_returns_queued_item():
    s, r = mp_channel[str]()
    s.send("hello")
    # multiprocessing queues flush through a feeder thread; a short blocking
    # timeout absorbs that latency without a sleep.
    assert r.receive_timeout(1.0) == "hello"


def test_receive_timeout_raises_wouldblock_when_empty():
    from anyio import WouldBlock

    _, r = mp_channel[str]()
    start = time.monotonic()
    with pytest.raises(WouldBlock):
        r.receive_timeout(0.1)
    # It actually waited (blocking receive with deadline, not an instant fail).
    assert time.monotonic() - start >= 0.05


def test_receive_timeout_closed_on_sender_close_in_one_process():
    # In one process both ends share the queue object and the sender's close
    # closes it, so a later receive raises ClosedResourceError. Across
    # processes, delivered items drain first (the tests below).
    from anyio import ClosedResourceError

    s, r = mp_channel[str]()
    s.close()
    with pytest.raises(ClosedResourceError):
        r.receive_timeout(1.0)


def _send_then_close(send: MpSender[str], items: list[str]) -> None:
    for item in items:
        send.send(item)
    send.close()
    send.join()


def _closed_after_sends(*items: str) -> MpReceiver[str]:
    """A receiver whose sender process sent ``items``, closed, and exited.

    The sender is gone before anything is read, so every receive starts with
    the shared closed flag set: the order a runner's final events arrive in
    when it exits right after sending them.
    """
    s, r = mp_channel[str]()
    sender = mp.Process(target=_send_then_close, args=(s, list(items)))
    sender.start()
    sender.join(timeout=30)
    assert sender.exitcode == 0
    return r


def test_items_sent_before_close_reach_a_blocking_receive():
    """A runner's last events before it exits are delivered, not dropped.

    The receiver used to check the shared closed flag before reading, so a
    Shutdown task's Complete status sent just before the runner closed its
    channel was lost, and the supervisor waited for it forever.
    """
    from anyio import EndOfStream

    r = _closed_after_sends("complete", "shutdown")
    assert r.receive() == "complete"
    assert r.receive() == "shutdown"
    with pytest.raises(EndOfStream):
        r.receive()


def test_items_sent_before_close_reach_receive_nowait():
    from anyio import EndOfStream

    r = _closed_after_sends("complete")
    assert r.receive_nowait() == "complete"
    with pytest.raises(EndOfStream):
        r.receive_nowait()


def test_items_sent_before_close_reach_receive_timeout():
    from anyio import EndOfStream

    r = _closed_after_sends("complete")
    assert r.receive_timeout(1.0) == "complete"
    with pytest.raises(EndOfStream):
        r.receive_timeout(1.0)


def test_iteration_drains_a_closed_channel():
    r = _closed_after_sends("a", "b", "c")
    with r as items:
        assert list(items) == ["a", "b", "c"]


def test_receive_after_own_close_raises_closed():
    from anyio import ClosedResourceError

    _, r = mp_channel[str]()
    r.close()
    with pytest.raises(ClosedResourceError):
        r.receive_nowait()
    with pytest.raises(ClosedResourceError):
        r.receive()


def test_closed_stream_without_its_marker_raises_closed():
    """A sender that vanished before its marker arrived ends as closed."""
    from anyio import ClosedResourceError

    _, r = mp_channel[str]()
    r._state.closed.set()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ClosedResourceError):
        r.receive_nowait()
    started = time.monotonic()
    with pytest.raises(ClosedResourceError):
        r.receive_timeout(0.2)
    # A closed channel waits no longer than the caller allowed.
    assert time.monotonic() - started < 1.0
