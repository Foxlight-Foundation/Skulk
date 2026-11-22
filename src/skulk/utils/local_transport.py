"""Local stream transport between Skulk and the processes it supervises.

Every stream between the node, its plugin manager and installed plugin owners
goes through this module: where an endpoint lives, whether it is private to
this account, how to listen on it and how to connect to it. Callers keep their
own deadlines, read limits, framing and authentication.

On POSIX an endpoint is a Unix socket in a per-account directory under
``/tmp``, short enough for the platform's socket-address limit. On Windows it
is a named pipe private to the account
(:mod:`skulk.utils._local_transport_windows`). No other module may open or
bind a local socket or pipe directly (``tests/test_local_transport.py``
enforces this). There is one transport per platform and no fallback.
"""

import asyncio
import os
import stat
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol

# What ``asyncio`` itself accepts for a stream server: a coroutine function or
# a plain function that starts its own task.
type ClientConnected = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None] | None
]


class LocalServer(Protocol):
    """A served endpoint: what a caller may do with what :func:`serve` returns."""

    def close(self) -> None:
        """Stop accepting clients."""
        ...

    async def wait_closed(self) -> None:
        """Return once the server has stopped."""
        ...


if sys.platform == "win32":
    from skulk.utils._local_transport_windows import (
        connect as connect,
    )
    from skulk.utils._local_transport_windows import (
        discard as discard,
    )
    from skulk.utils._local_transport_windows import (
        endpoint_directory as endpoint_directory,
    )
    from skulk.utils._local_transport_windows import (
        endpoint_is_private as endpoint_is_private,
    )
    from skulk.utils._local_transport_windows import (
        serve as serve,
    )
else:

    def endpoint_directory(kind: Literal["control", "manager"], digest: str) -> Path:
        """Name the per-account directory that holds one root's endpoints.

        The directory is named, not created or checked: each caller applies its
        own rule for that. The account's user ID is part of the name, so two
        accounts never share a directory.

        Args:
            kind: Which side owns the endpoints: an installed plugin owner
                (``control``) or the plugin manager (``manager``).
            digest: The caller's digest of the root the endpoints belong to.

        Returns:
            The directory path.
        """
        return Path("/tmp") / f"skulk-{kind}-{os.getuid()}-{digest}"

    def endpoint_is_private(path: Path) -> bool:
        """Whether ``path`` is a listening endpoint only this account can reach.

        The path itself is examined, never a link's target.

        Args:
            path: The endpoint address.

        Returns:
            True when it is a socket owned by this account with no access for its
            group or for others.

        Raises:
            OSError: When the path cannot be examined, for example because nothing
                is listening there yet.
        """
        info = path.lstat()
        return (
            stat.S_ISSOCK(info.st_mode)
            and info.st_uid == os.getuid()
            and not info.st_mode & 0o077
        )

    async def connect(
        path: Path, *, limit: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a stream to a local endpoint.

        This does not check the endpoint: a caller that requires a private one
        checks :func:`endpoint_is_private` first.

        Args:
            path: The endpoint address.
            limit: The reader's buffer limit in bytes.

        Returns:
            The stream's reader and writer.
        """
        return await asyncio.open_unix_connection(path, limit=limit)

    async def serve(
        path: Path, client_connected: ClientConnected, *, limit: int
    ) -> LocalServer:
        """Listen on a local endpoint that only this account can reach.

        Anything left at the address by an earlier listener is removed first. The
        caller closes the returned server and then calls :func:`discard`.

        Args:
            path: The endpoint address, inside a directory private to this account.
            client_connected: Called with a reader and a writer for each client.
            limit: Each client reader's buffer limit in bytes.

        Returns:
            The listening server.
        """
        path.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(
            client_connected, path=path, limit=limit
        )
        try:
            path.chmod(0o600)
        except BaseException:
            server.close()
            raise
        return server

    def discard(path: Path) -> None:
        """Remove a closed endpoint's address so a later listener starts clean."""
        path.unlink(missing_ok=True)
