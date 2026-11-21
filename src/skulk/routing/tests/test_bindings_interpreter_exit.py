"""The networking bindings stay quiet when the interpreter exits.

The bindings' runtime threads outlive the interpreter. They used to enter
Python after it had finalized, to log or to deliver a result, and panic. The
panic needs live traffic to show, so each exit here happens with a peer
connected and messages flowing.
"""

import os
import re
import subprocess
import sys
import uuid

_PEER = r"""
import asyncio
import logging
import os
import sys

import skulk_pyo3_bindings as bindings

# Rust log records reach Python only at an enabled level; DEBUG is what makes
# the runtime threads log continuously.
logging.basicConfig(level=logging.DEBUG, stream=open(os.devnull, "w"))
serve = sys.argv[1] == "serve"
bootstrap = sys.argv[2:]


async def main() -> None:
    handle = bindings.NetworkingHandle(bindings.Keypair.generate(), bootstrap, 0)

    async def read() -> None:
        while True:
            await handle.recv()

    async def chatter() -> None:
        while True:
            try:
                await handle.gossipsub_publish("exit-test", b"x" * 64)
            except Exception:
                pass
            await asyncio.sleep(0.01)

    reader = asyncio.create_task(read())
    await handle.gossipsub_subscribe("exit-test")
    talker = asyncio.create_task(chatter())
    if serve:
        # The listener is bound shortly after the swarm starts, not before.
        addresses: list[str] = []
        while not any("/tcp/" in address for address in addresses):
            await asyncio.sleep(0.05)
            addresses = await handle.listen_addresses()
        print(" ".join(addresses), flush=True)
        await asyncio.sleep(120)
    else:
        await asyncio.sleep(1.5)
    for task in (talker, reader):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


asyncio.run(main())
"""

_EXITS = 4


def test_interpreter_exit_with_live_traffic_does_not_panic() -> None:
    # An isolated private-network key: these swarms must not be able to join
    # any other Skulk network reachable from this machine.
    env = dict(os.environ, SKULK_LIBP2P_NAMESPACE=f"exit-test-{uuid.uuid4().hex}")
    server = subprocess.Popen(
        [sys.executable, "-c", _PEER, "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert server.stdout is not None
        port = re.search(r"/tcp/(\d+)", server.stdout.readline())
        assert port is not None, "the serving peer reported no TCP listener"
        bootstrap = f"/ip4/127.0.0.1/tcp/{port.group(1)}"

        for _ in range(_EXITS):
            peer = subprocess.run(
                [sys.executable, "-c", _PEER, "join", bootstrap],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert "panicked" not in peer.stderr, peer.stderr
            assert peer.returncode == 0, peer.stderr
    finally:
        server.kill()
        server.wait()
