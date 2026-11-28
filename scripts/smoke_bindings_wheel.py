"""Import the installed bindings wheel and start the swarm.

Run by ``.github/workflows/bindings-wheel.yml`` with the interpreter of an
environment that holds nothing but the wheel under test. The caller sets
``SKULK_LIBP2P_NAMESPACE`` to an isolated private-network key, so this swarm
can never join another Skulk network reachable from the runner.
"""

import asyncio

import skulk_pyo3_bindings as bindings


async def main() -> None:
    """Start a swarm on a free port, subscribe to one topic, and print its addresses.

    Raises when the bindings cannot be used or do not answer within 30 seconds.
    """
    handle = bindings.NetworkingHandle(bindings.Keypair.generate(), listen_port=0)

    async def read_events() -> None:
        # The swarm only processes commands while its event stream
        # is being read.
        while True:
            await handle.recv()

    reader = asyncio.create_task(read_events())
    addresses = await asyncio.wait_for(handle.listen_addresses(), 30)
    assert await asyncio.wait_for(handle.gossipsub_subscribe("smoke"), 30)
    reader.cancel()
    print("listening on", addresses)


if __name__ == "__main__":
    asyncio.run(main())
