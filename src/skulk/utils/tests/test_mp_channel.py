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


def test_receive_timeout_closed_on_sender_close():
    # Sender close sets the shared closed flag, so a receive that STARTS after
    # the close raises ClosedResourceError (same semantics as receive_nowait);
    # EndOfStream is only surfaced to a receive already blocked in get().
    from anyio import ClosedResourceError

    s, r = mp_channel[str]()
    s.close()
    with pytest.raises(ClosedResourceError):
        r.receive_timeout(1.0)
