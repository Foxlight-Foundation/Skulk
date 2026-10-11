"""Fetch another node's restart recovery record over its API."""

from collections.abc import Awaitable, Callable

import httpx
from loguru import logger
from pydantic import ValidationError

from skulk.recovery.record import ClusterIntentRecord, RestartRecoveryRecordResponse
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeNetworkInfo
from skulk.shared.types.restart_recovery import RestartRecoveryAdvert

RECORD_PATH = "/v1/internal/restart-recovery/record"
"""API route that serves a node's restart recovery record."""

FETCH_TIMEOUT_SECONDS = 5.0

RestartRecordFetcher = Callable[
    [NodeId, RestartRecoveryAdvert, NodeNetworkInfo | None],
    Awaitable[ClusterIntentRecord | None],
]
"""How a cold-started master fetches a peer's record; injected so tests can
supply records without a network."""


async def fetch_peer_record(
    node_id: NodeId,
    advert: RestartRecoveryAdvert,
    network: NodeNetworkInfo | None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ClusterIntentRecord | None:
    """Fetch ``node_id``'s record from its API, trying each of its addresses.

    The advert names the port the node's API runs on, and every response says
    which node answered, so a shared address (two nodes on one machine, or a
    different service on the default port) can never supply the wrong record.

    Args:
        node_id: The node whose record is wanted.
        advert: Its latest advert, which carries its API port.
        network: Its interfaces from replicated state, or None if unknown.
        transport: HTTP transport override, for tests.

    Returns:
        The record, or None when no address answered with that node's record.
    """
    if advert.api_port is None or network is None:
        return None
    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT_SECONDS, transport=transport
    ) as client:
        for interface in network.interfaces:
            address = interface.ip_address
            host = f"[{address}]" if ":" in address else address
            url = f"http://{host}:{advert.api_port}{RECORD_PATH}"
            try:
                response = await client.get(url)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                served = RestartRecoveryRecordResponse.model_validate_json(response.content)
            except ValidationError:
                logger.warning(f"Restart recovery: {url} returned an unreadable record")
                continue
            if served.node_id != node_id:
                continue
            return served.record
    return None
