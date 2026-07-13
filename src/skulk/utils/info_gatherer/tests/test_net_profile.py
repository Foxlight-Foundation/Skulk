from collections.abc import AsyncIterator

import anyio
import httpx
import pytest

from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from skulk.utils.channels import Sender
from skulk.utils.info_gatherer import net_profile


async def _collect_reachable_targets(
    topology: Topology,
    self_node_id: NodeId,
    node_network: dict[NodeId, NodeNetworkInfo],
) -> list[tuple[str, NodeId]]:
    reachable_targets: list[tuple[str, NodeId]] = []
    reachable_iter: AsyncIterator[tuple[str, NodeId]] = net_profile.check_reachable(
        topology=topology,
        self_node_id=self_node_id,
        node_network=node_network,
    )
    async for reachable in reachable_iter:
        reachable_targets.append(reachable)
    return reachable_targets


@pytest.mark.anyio
async def test_check_reachable_skips_loopback_and_unspecified_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_node_id = NodeId("self")
    remote_node_id = NodeId("remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)
    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="lo0", ip_address="127.0.0.1"),
                NetworkInterfaceInfo(name="lo0", ip_address="::1"),
                NetworkInterfaceInfo(name="lo0", ip_address="0.0.0.0"),
                NetworkInterfaceInfo(name="lo0", ip_address="::"),
                NetworkInterfaceInfo(name="lo0", ip_address="localhost"),
                NetworkInterfaceInfo(name="en0", ip_address="192.168.0.117"),
                NetworkInterfaceInfo(
                    name="en7", ip_address="fe80::20:315a:c2e5:286b%en0"
                ),
            ]
        )
    }
    probed_targets: list[str] = []

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        _client: object,
        attempts: int = net_profile.REACHABILITY_ATTEMPTS,
    ) -> None:
        del attempts
        probed_targets.append(target_ip)
        out[expected_node_id].add(target_ip)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)

    reachable_targets = await _collect_reachable_targets(
        topology=topology,
        self_node_id=self_node_id,
        node_network=node_network,
    )

    assert probed_targets == ["192.168.0.117", "fe80::20:315a:c2e5:286b%en0"]
    assert reachable_targets == [
        ("192.168.0.117", remote_node_id),
        ("fe80::20:315a:c2e5:286b%en0", remote_node_id),
    ]


@pytest.mark.anyio
async def test_first_reachable_ip_returns_before_slower_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_node_id = NodeId("self")
    remote_node_id = NodeId("remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)
    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="slow", ip_address="192.168.0.117"),
                NetworkInterfaceInfo(name="fast", ip_address="192.168.0.118"),
            ]
        )
    }

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        _client: object,
    ) -> None:
        if target_ip.endswith("117"):
            await anyio.sleep(10)
        out.setdefault(expected_node_id, set()).add(target_ip)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)

    with anyio.fail_after(1.0):
        result = await net_profile.first_reachable_ip(
            topology,
            self_node_id,
            node_network,
            remote_node_id,
        )

    assert result == "192.168.0.118"


@pytest.mark.anyio
async def test_first_reachable_ip_ignores_discarded_interface_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_node_id = NodeId("self")
    remote_node_id = NodeId("remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)
    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="first", ip_address="192.168.0.117"),
                NetworkInterfaceInfo(name="second", ip_address="192.168.0.118"),
            ]
        )
    }

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        _client: object,
    ) -> None:
        out.setdefault(expected_node_id, set()).add(target_ip)

    send_count = 0

    async def fail_discarded_send(sender: Sender[str], item: str) -> None:
        nonlocal send_count
        await anyio.sleep(0)
        send_count += 1
        if send_count > 1:
            raise anyio.BrokenResourceError
        sender.send_nowait(item)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)
    monkeypatch.setattr(Sender, "send", fail_discarded_send)

    result = await net_profile.first_reachable_ip(
        topology,
        self_node_id,
        node_network,
        remote_node_id,
    )

    assert result == "192.168.0.117"
    assert send_count == 2


@pytest.mark.anyio
async def test_first_reachable_ip_returns_none_when_target_has_no_valid_address(
) -> None:
    self_node_id = NodeId("self")
    remote_node_id = NodeId("remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)

    result = await net_profile.first_reachable_ip(
        topology,
        self_node_id,
        {
            remote_node_id: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(name="lo0", ip_address="127.0.0.1")
                ]
            )
        },
        remote_node_id,
    )

    assert result is None


@pytest.mark.anyio
async def test_check_reachable_passes_caller_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep budget is the caller's choice: interactive surfaces pass the
    fail-fast SWEEP_ATTEMPTS (#558) while the worker's connection maintenance
    keeps the patient default (one dead address must not stall a dashboard,
    but link upkeep deliberately retries)."""

    self_node_id = NodeId("self")
    remote_node_id = NodeId("remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)
    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="192.168.0.117")]
        )
    }
    seen_attempts: list[int] = []

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        _client: object,
        attempts: int = net_profile.REACHABILITY_ATTEMPTS,
    ) -> None:
        seen_attempts.append(attempts)
        out[expected_node_id].add(target_ip)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)

    async for _ in net_profile.check_reachable(
        topology=topology,
        self_node_id=self_node_id,
        node_network=node_network,
        attempts=net_profile.SWEEP_ATTEMPTS,
    ):
        pass
    assert seen_attempts == [net_profile.SWEEP_ATTEMPTS]

    seen_attempts.clear()
    await _collect_reachable_targets(
        topology=topology, self_node_id=self_node_id, node_network=node_network
    )
    assert seen_attempts == [net_profile.REACHABILITY_ATTEMPTS]


@pytest.mark.anyio
async def test_check_reachability_single_attempt_probes_once() -> None:
    """attempts=1 must mean exactly one HTTP try against a dead address."""

    calls = 0

    class _DeadClient:
        async def get(self, url: str) -> object:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError(f"connect timeout stand-in for {url}")

    out: dict[NodeId, set[str]] = {}
    await net_profile.check_reachability(
        "203.0.113.9", NodeId("remote"), out, _DeadClient(), attempts=1  # pyright: ignore[reportArgumentType]
    )

    assert calls == 1
    assert out == {}


@pytest.mark.anyio
async def test_check_reachability_no_backoff_after_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 1s inter-retry backoff must not run after the last attempt: for
    attempts=1 sweeps it would silently reintroduce the stall the fail-fast
    budget exists to remove."""

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(net_profile.anyio, "sleep", fake_sleep)

    class _DeadClient:
        async def get(self, url: str) -> object:
            raise httpx.ConnectError(f"dead: {url}")

    out: dict[NodeId, set[str]] = {}
    await net_profile.check_reachability(
        "203.0.113.9", NodeId("remote"), out, _DeadClient(), attempts=1  # pyright: ignore[reportArgumentType]
    )
    assert sleeps == []

    await net_profile.check_reachability(
        "203.0.113.9", NodeId("remote"), out, _DeadClient(), attempts=3  # pyright: ignore[reportArgumentType]
    )
    assert sleeps == [1, 1]


@pytest.mark.anyio
async def test_check_reachability_no_warning_on_eventual_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry that eventually succeeds must not log 'treating as down':
    last_error is cleared on success so only genuinely-down addresses warn."""

    warnings: list[str] = []

    def _record_warning(msg: object, *_args: object, **_kwargs: object) -> None:
        warnings.append(str(msg))

    monkeypatch.setattr(net_profile.logger, "warning", _record_warning)

    calls = 0

    class _FlakyClient:
        async def get(self, url: str) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError(f"transient for {url}")
            return httpx.Response(
                200, text='"remote"', request=httpx.Request("GET", url)
            )

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(net_profile.anyio, "sleep", _no_sleep)
    out: dict[NodeId, set[str]] = {}
    await net_profile.check_reachability(
        "203.0.113.9", NodeId("remote"), out, _FlakyClient(), attempts=3  # pyright: ignore[reportArgumentType]
    )

    assert calls == 2
    assert out == {NodeId("remote"): {"203.0.113.9"}}
    assert warnings == []


@pytest.mark.anyio
async def test_check_reachable_bounds_probe_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wide fan-out must not run more than MAX_CONCURRENT_PROBES probes at
    once, so a large fleet or interface-heavy peers cannot exhaust file
    descriptors (#558)."""

    self_node_id = NodeId("self")
    topology = Topology()
    topology.add_node(self_node_id)
    node_network: dict[NodeId, NodeNetworkInfo] = {}
    total = net_profile.MAX_CONCURRENT_PROBES * 3
    for i in range(total):
        node = NodeId(f"remote-{i}")
        topology.add_node(node)
        node_network[node] = NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address=f"10.0.0.{i % 250}")]
        )

    in_flight = 0
    peak = 0

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        _client: object,
        attempts: int = net_profile.REACHABILITY_ATTEMPTS,
    ) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await anyio.sleep(0)
        in_flight -= 1
        out[expected_node_id].add(target_ip)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)
    results = await _collect_reachable_targets(
        topology=topology, self_node_id=self_node_id, node_network=node_network
    )

    assert len(results) == total
    assert peak <= net_profile.MAX_CONCURRENT_PROBES
