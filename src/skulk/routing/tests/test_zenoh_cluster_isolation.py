"""Two-process check that the Zenoh data plane is scoped to its cluster key.

Each process derives its cluster key from ``SKULK_LIBP2P_NAMESPACE`` exactly as
a node does, so this exercises the shipped path end to end: the key-derived
multicast scouting port, mutual TLS rooted in the key-derived authority, and
the membership policy. Processes listen on loopback only and scout on ports
derived from unique test namespaces, so they cannot meet any real cluster.

Marked slow because it depends on the host delivering link-local multicast
between local processes. Run it explicitly with
``uv run pytest -m slow src/skulk/routing/tests/test_zenoh_cluster_isolation.py``.
The handshake, plaintext-client, and foreign-certificate cases run in CI as
Rust tests in ``rust/networking/src/zenoh_session.rs``.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

import pytest

# The child script prints one JSON object per line: its scouting address first,
# then its outcome. Both roles stay up for the whole window so each side's peer
# count is sampled while the other is alive; the publisher keeps publishing so
# subscriber declarations have time to propagate.
_CHILD = r"""
import asyncio
import json
import os
import sys

from skulk_pyo3_bindings import ZenohHandle


async def main(role: str, seconds: float) -> None:
    handle = ZenohHandle(["tls/127.0.0.1:0"], [], "nsisolationtest", True)
    print(json.dumps({"scouting": handle.scouting_address()}), flush=True)
    key = "data/subscriber"
    received = None
    most_peers = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    if role == "subscriber":
        await handle.zenoh_subscribe(key)
    while loop.time() < deadline:
        if role == "subscriber":
            try:
                message = await asyncio.wait_for(handle.recv(), timeout=0.2)
                received = bytes(message.data).decode()
            except TimeoutError:
                pass
        else:
            await handle.zenoh_publish(key, b"hello")
            await asyncio.sleep(0.2)
        most_peers = max(most_peers, await handle.zenoh_connected_peer_count())
    print(json.dumps({"received": received, "peers": most_peers}), flush=True)
    # Leave without interpreter teardown. Dropping a session while the GIL is
    # held can deadlock when both peers close within the same few
    # milliseconds (Zenoh logs the peer's close through pyo3-log, which then
    # waits for the GIL). That shutdown path is outside what this test checks.
    os._exit(0)


asyncio.run(main(sys.argv[1], float(sys.argv[2])))
"""


@dataclass(frozen=True)
class _Outcome:
    scouting: str | None
    received: str | None
    peers: int


def _spawn(namespace: str, role: str, seconds: float) -> subprocess.Popen[str]:
    environment = {**os.environ, "SKULK_LIBP2P_NAMESPACE": namespace}
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD, role, str(seconds)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _line(raw: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(raw))


def _optional_text(value: object) -> str | None:
    assert value is None or isinstance(value, str)
    return value


def _outcome(process: subprocess.Popen[str], seconds: float) -> _Outcome:
    try:
        stdout, stderr = process.communicate(timeout=seconds + 30)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"child timed out\nstdout={stdout}\nstderr={stderr}")
    assert process.returncode == 0, stderr
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    assert len(lines) == 2, stdout
    first, second = _line(lines[0]), _line(lines[1])
    peers = second["peers"]
    assert isinstance(peers, int)
    return _Outcome(
        scouting=_optional_text(first["scouting"]),
        received=_optional_text(second["received"]),
        peers=peers,
    )


def _run_pair(
    subscriber_namespace: str, publisher_namespace: str, seconds: float
) -> tuple[_Outcome, _Outcome]:
    subscriber = _spawn(subscriber_namespace, "subscriber", seconds)
    publisher = _spawn(publisher_namespace, "publisher", seconds)
    try:
        return _outcome(subscriber, seconds), _outcome(publisher, seconds)
    finally:
        for process in (subscriber, publisher):
            if process.poll() is None:
                process.kill()


@pytest.mark.slow
def test_same_namespace_processes_discover_and_exchange_samples() -> None:
    namespace = "isolation-same-" + uuid4().hex
    subscriber, publisher = _run_pair(namespace, namespace, seconds=20)

    assert subscriber.scouting is not None
    assert subscriber.scouting == publisher.scouting
    assert subscriber.scouting.startswith("224.0.0.224:")
    assert subscriber.received == "hello"
    assert subscriber.peers == 1
    assert publisher.peers == 1


@pytest.mark.slow
def test_different_namespaces_never_discover_each_other() -> None:
    # Different keys scout on different ports. Two random namespaces collide
    # about once in 7000 draws; redraw rather than flake, since a collision
    # only means the TLS handshake (covered by the Rust tests) does the work.
    subscriber, publisher = _run_pair(
        "isolation-a-" + uuid4().hex, "isolation-b-" + uuid4().hex, seconds=8
    )
    for _ in range(2):
        if subscriber.scouting != publisher.scouting:
            break
        subscriber, publisher = _run_pair(
            "isolation-a-" + uuid4().hex, "isolation-b-" + uuid4().hex, seconds=8
        )
    assert None not in (subscriber.scouting, publisher.scouting)
    assert subscriber.scouting != publisher.scouting
    assert subscriber.received is None
    assert subscriber.peers == 0
    assert publisher.peers == 0
