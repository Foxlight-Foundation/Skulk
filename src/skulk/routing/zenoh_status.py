"""Data-plane Zenoh connectivity sampling for cluster health.

A node whose Zenoh session holds zero peer transports while other live nodes
advertise Zenoh is data-plane isolated: every remote stream to or from it dies
with transport errors while the libp2p control plane (events, telemetry,
election) still looks healthy. The canonical shape is a zero-config remote
member (for example a node joined over an overlay network) that Zenoh
multicast scouting cannot reach. This module samples the router's session so
the condition is advertised on ``NodeResources`` and named by cluster health,
instead of failing silently one stream at a time.
"""

import time
from collections.abc import Awaitable, Callable
from typing import final

# How long after startup a still-unconnected session advertises "unknown"
# rather than 0. Mesh formation after a cold boot takes seconds to tens of
# seconds (scouting, dial retries), and cluster health must not flap an
# isolation error on every restart. A genuinely isolated node converges to a
# loud 0 once this window passes.
STARTUP_GRACE_SECONDS: float = 90.0


@final
class ZenohPeerSampler:
    """Samples live Zenoh peer-transport counts for advertisement.

    Wraps the router's ``zenoh_connected_peer_count`` with the one piece of
    state advertisement needs: a startup grace window. While the session has
    never connected a peer and the window is still open, the sampler reports
    ``None`` (unknown) so health does not fire during normal mesh formation.
    After the window, or after the first successful connection, the raw count
    is reported and a sustained 0 becomes positive evidence of isolation.
    """

    def __init__(
        self,
        count_peers: Callable[[], Awaitable[int | None]],
        *,
        grace_seconds: float = STARTUP_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a sampler over the router's peer-count coroutine.

        Args:
            count_peers: Coroutine returning the session's live peer-transport
                count, or ``None`` when DATA rides gossipsub.
            grace_seconds: Startup window during which a never-connected
                session advertises ``None`` instead of 0.
            clock: Monotonic clock, injectable for tests.
        """
        self._count_peers = count_peers
        self._grace_seconds = grace_seconds
        self._clock = clock
        self._started_at = clock()
        self._ever_connected = False

    async def advertised_count(self) -> int | None:
        """Return the count to advertise on ``NodeResources``.

        ``None`` when DATA rides gossipsub, when the underlying sample fails
        (advertising unknown beats fabricating evidence), or while the startup
        grace window shields a never-connected session; otherwise the live
        peer-transport count, where a 0 is trustworthy isolation evidence.
        """
        try:
            count = await self._count_peers()
        except Exception:
            # A failed introspection must not take down the telemetry loop
            # that called us, and must not masquerade as isolation.
            return None
        if count is None:
            return None
        if count > 0:
            self._ever_connected = True
            return count
        if (
            not self._ever_connected
            and (self._clock() - self._started_at) < self._grace_seconds
        ):
            return None
        return 0
