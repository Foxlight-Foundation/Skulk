"""Bounded loopback TCP observation bridge; forwards opaque inner TLS bytes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from bench.operator_fixture_observer import FixtureObservationError, FixtureObserver


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while chunk := await reader.read(65536):
        writer.write(chunk)
        await writer.drain()


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        async with asyncio.timeout(2):
            await writer.wait_closed()
    except (ConnectionError, TimeoutError):
        # Closing an already-reset peer is normal. The transport is closed
        # regardless, and a pending request is marked failed by the observer.
        writer.transport.abort()


@asynccontextmanager
async def observation_proxy(
    observer: FixtureObserver,
    backend_port: int,
) -> AsyncIterator[int]:
    """Yield an ephemeral loopback port that observes real TCP socket lifetimes.

    `observer` receives local accept/close events and an in-memory ASGI peer
    association. `backend_port` must be a generated local TLS fixture listener;
    the target address is not configurable. No TLS keys or plaintext enter
    this bridge. Each of at most 512 connections uses bounded stream buffers.
    On exit, stop admission and close/cancel all owned connections. Unexpected
    observer failures propagate through the task group, invalidating capture.
    """
    if not 1 <= backend_port <= 65535:
        raise ValueError("invalid local fixture port")
    tasks: set[asyncio.Task[None]] = set()
    writers: set[asyncio.StreamWriter] = set()
    rejected = False
    async with asyncio.TaskGroup() as group:

        async def bridge(
            identifier: int,
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            downstream: asyncio.StreamWriter | None = None
            copies: list[asyncio.Task[None]] = []
            try:
                async with asyncio.timeout(2):
                    backend_reader, downstream = await asyncio.open_connection(
                        "127.0.0.1",
                        backend_port,
                        limit=65536,
                    )
                peer = cast(tuple[str, int], downstream.get_extra_info("sockname"))
                # Bind before forwarding even the TLS ClientHello, so Hypercorn
                # can never dispatch a request without its TCP association.
                observer.bind_peer(identifier, peer)
                copies = [
                    asyncio.create_task(_copy(reader, downstream)),
                    asyncio.create_task(_copy(backend_reader, writer)),
                ]
                done, _ = await asyncio.wait(
                    copies, return_when=asyncio.FIRST_COMPLETED
                )
                for completed in done:
                    completed.result()
            except (ConnectionError, TimeoutError):
                # Fault scenarios include resets and unreachable synthetic
                # gateways. The socket close fails any incomplete requests.
                pass
            finally:
                for task in copies:
                    task.cancel()
                await asyncio.gather(*copies, return_exceptions=True)
                await _close(writer)
                writers.discard(writer)
                if downstream is not None:
                    await _close(downstream)
                observer.closed(identifier)

        async def reject() -> None:
            # Invalidate the entire observation, not a silent partial sample.
            observer.invalidate()

        def connected(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            nonlocal rejected
            if len(tasks) >= 512:
                writer.transport.abort()
                # Use one fatal task; do not create unbounded reject tasks.
                if not rejected:
                    rejected = True
                    task = group.create_task(reject())
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                return
            try:
                # Record synchronously at accept, before another ready stdin
                # task can incorrectly declare the fixture idle or change flow.
                identifier = observer.accepted()
            except FixtureObservationError:
                writer.transport.abort()
                if not rejected:
                    rejected = True
                    task = group.create_task(reject())
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                return
            writers.add(writer)
            task = group.create_task(bridge(identifier, reader, writer))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        server = await asyncio.start_server(connected, "127.0.0.1", 0, limit=65536)
        try:
            yield cast(tuple[str, int], server.sockets[0].getsockname())[1]
        finally:
            server.close()
            await server.wait_closed()
            # A task cancelled before its first coroutine step cannot execute
            # its finally block. Close admitted transports independently too.
            for writer in tuple(writers):
                writer.transport.abort()
            for task in tuple(tasks):
                task.cancel()
