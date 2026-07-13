import ipaddress
from collections import defaultdict
from collections.abc import AsyncGenerator, Mapping
from contextlib import suppress

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeNetworkInfo
from skulk.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3
# Full-fleet sweeps (dashboard observability fan-out) probe every advertised
# interface of every peer; a dead address must fail fast, not retry on the
# targeted-lookup policy above (3 attempts x (5s timeout + 1s sleep) = ~18s of
# stall per unroutable address, which read as "observability is broken", #558).
SWEEP_ATTEMPTS = 1
SWEEP_TIMEOUT_SECONDS = 2.0


def _should_probe_remote_ip(target_ip: str) -> bool:
    """Return whether a remote reachability probe should target this address.

    Remote-node probing should ignore loopback and unspecified addresses such as
    ``127.0.0.1`` or ``::1`` because they resolve back to the local node on the
    probing machine and create misleading identity-mismatch logs.
    """

    candidate = target_ip.strip()
    if not candidate:
        return False

    zone_delimiter = candidate.find("%")
    if zone_delimiter != -1:
        candidate = candidate[:zone_delimiter]

    try:
        parsed = ipaddress.ip_address(candidate)
    except ValueError:
        return candidate not in {"localhost"}

    return not (parsed.is_loopback or parsed.is_unspecified)


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
    attempts: int = REACHABILITY_ATTEMPTS,
) -> None:
    """Check if a node is reachable at the given IP and verify its identity.

    ``attempts`` selects the retry budget: targeted lookups keep the patient
    default, fleet-wide sweeps pass ``SWEEP_ATTEMPTS`` so one dead address
    cannot stall an interactive caller.
    """
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:52415/node_id"
    else:
        url = f"http://{target_ip}:52415/node_id"

    remote_node_id = None
    last_error = None

    for _ in range(attempts):
        try:
            r = await client.get(url)
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            break

        # expected failure cases
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            await anyio.sleep(1)

        # other failures should be logged on last attempt
        except httpx.HTTPError as e:
            last_error = e
            await anyio.sleep(1)

    if last_error is not None:
        logger.warning(
            f"connect error {type(last_error).__name__} from {target_ip} after {REACHABILITY_ATTEMPTS} attempts; treating as down"
        )

    if remote_node_id is None:
        return

    if remote_node_id != expected_node_id:
        logger.debug(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return

    if remote_node_id not in out:
        out[remote_node_id] = set()
    out[remote_node_id].add(target_ip)


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> AsyncGenerator[tuple[str, NodeId], None]:
    """Yield (ip, node_id) pairs as reachability probes complete."""

    send, recv = channel[tuple[str, NodeId]]()

    # Sweep policy: short timeout, single attempt per address (see
    # SWEEP_ATTEMPTS). This generator backs interactive surfaces.
    timeout = httpx.Timeout(timeout=SWEEP_TIMEOUT_SECONDS)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe(
        target_ip: str,
        expected_node_id: NodeId,
        client: httpx.AsyncClient,
        send: Sender[tuple[str, NodeId]],
    ) -> None:
        async with send:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            await check_reachability(
                target_ip, expected_node_id, out, client, attempts=SWEEP_ATTEMPTS
            )
            if expected_node_id in out:
                await send.send((target_ip, expected_node_id))

    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id not in node_network:
                continue
            if node_id == self_node_id:
                continue
            for iface in node_network[node_id].interfaces:
                if not _should_probe_remote_ip(iface.ip_address):
                    continue
                tg.start_soon(_probe, iface.ip_address, node_id, client, send.clone())
        send.close()

        with recv:
            async for item in recv:
                yield item


async def first_reachable_ip(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    target_node_id: NodeId,
) -> str | None:
    """Return the first verified API address for one target node.

    Unlike :func:`check_reachable`, this helper owns and closes its probe task
    group before returning. Callers can therefore stop at the first hit without
    finalizing an async generator whose AnyIO cancel scope is still active.
    """

    if (
        target_node_id == self_node_id
        or target_node_id not in topology.list_nodes()
        or target_node_id not in node_network
    ):
        return None
    target_ips = [
        interface.ip_address
        for interface in node_network[target_node_id].interfaces
        if _should_probe_remote_ip(interface.ip_address)
    ]
    if not target_ips:
        return None

    send, recv = channel[str]()
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe_first(
        target_ip: str,
        client: httpx.AsyncClient,
        probe_sender: Sender[str],
    ) -> None:
        async with probe_sender:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            await check_reachability(target_ip, target_node_id, out, client)
            if target_node_id in out:
                # Another interface may win and close the receiver before this
                # probe publishes. That late result is intentionally disposable.
                with suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                    await probe_sender.send(target_ip)

    result: str | None = None
    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as task_group,
    ):
        for target_ip in target_ips:
            task_group.start_soon(_probe_first, target_ip, client, send.clone())
        send.close()
        with recv, suppress(anyio.EndOfStream):
            result = await recv.receive()
        if result is not None:
            task_group.cancel_scope.cancel()
    return result
