# pyright: reportPrivateUsage=false
"""Real preparation barriers and ordinary compatibility; native effects are injected."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from fastapi import HTTPException

from skulk.api.demucs_preparation import DemucsMountPreparer
from skulk.master.tests.test_separation_placement import card
from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PrepareDemucs
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import DemucsPreparationCompleted
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

NODE = NodeId("worker")
OWNER = NodeId("api")
BUILD = DEMUCS_PACKAGES[1].build_identity


def fixture(
    send: Callable[[PrepareDemucs], Awaitable[None]],
    *,
    clock: Callable[[], float] | None = None,
) -> tuple[DemucsMountPreparer, TelemetryView]:
    """Use actual topology, memory reservation and card compatibility functions."""
    topology = Topology()
    topology.add_node(NODE)
    snapshot = State(topology=topology)
    view = TelemetryView()
    view.node_resources[NODE] = NodeResources(
        backends=frozenset(), installable_engine_builds={"demucs-cpu": BUILD}
    )
    view.node_memory[NODE] = MemoryUsage.from_bytes(
        ram_total=64 * 1024**3,
        ram_available=60 * 1024**3,
        swap_total=0,
        swap_available=0,
    )
    preparer = DemucsMountPreparer(
        OWNER,
        lambda: snapshot,
        view,
        send,
        **({"clock": clock} if clock is not None else {}),
    )
    return preparer, view


def completion(
    request: PrepareDemucs, *, build: str = BUILD
) -> DemucsPreparationCompleted:
    """Supply a worker-observed library identity, never loaded-model readiness."""
    return DemucsPreparationCompleted(
        request_id=request.command_id,
        target_node=NODE,
        owner_node=OWNER,
        success=True,
        resources=NodeResources(
            backends=frozenset({"demucs-cpu"}), engine_builds={"demucs-cpu": build}
        ),
    )


async def test_installable_package_is_not_placement_readiness_until_correlated_completion() -> (
    None
):
    """The request uses fixed bytes and waits for the chosen worker's exact observation."""
    requests: list[PrepareDemucs] = []

    async def send(request: PrepareDemucs) -> None:
        requests.append(request)

    preparer, view = fixture(send)
    pending = asyncio.create_task(preparer.prepare(card(), set()))
    try:
        async with asyncio.timeout(3):
            while not requests:
                await asyncio.sleep(0)
        request = requests[0]
        assert (
            request.package_sha256 == DEMUCS_PACKAGES[1].sha256
            and request.device == "cpu"
        )
        assert not pending.done() and not view.node_resources[NODE].backends
        good = completion(request)
        preparer.record(good.model_copy(update={"target_node": NodeId("other")}))
        preparer.record(good.model_copy(update={"owner_node": NodeId("other-api")}))
        assert not pending.done()
        preparer.record(good)
        assert await pending == (NODE, good.resources)
        assert not preparer._pending
        # An old completion cannot recreate a retained waiter after its owner ends.
        preparer.record(good)
        assert not preparer._pending
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("build", ["", "demucs@sha256:" + "0" * 64])
async def test_wrong_or_unobserved_build_cannot_expand_placement(build: str) -> None:
    """A plausible backend name is insufficient without the selected package identity."""

    async def send(request: PrepareDemucs) -> None:
        preparer.record(completion(request, build=build))

    preparer, _ = fixture(send)
    with pytest.raises(HTTPException, match="compatible exact-build") as failure:
        await preparer.prepare(card(), set())
    assert failure.value.status_code == 503 and not preparer._pending


async def test_preparation_failure_is_actionable_without_rewriting_existing_resources() -> (
    None
):
    """Failure settles this mount and keeps the worker's other capability observations."""

    async def send(request: PrepareDemucs) -> None:
        preparer.record(
            DemucsPreparationCompleted(
                request_id=request.command_id,
                target_node=NODE,
                owner_node=OWNER,
                success=False,
                error="offline cache is missing the pinned package",
            )
        )

    preparer, view = fixture(send)
    before = view.node_resources[NODE]
    with pytest.raises(HTTPException, match="offline cache"):
        await preparer.prepare(card(), set())
    assert view.node_resources[NODE] == before and not preparer._pending


async def test_memory_is_reassessed_after_package_preparation() -> None:
    """A different admitted workload may consume memory while a package is installing."""

    async def send(request: PrepareDemucs) -> None:
        view.node_memory[NODE] = MemoryUsage.from_bytes(
            ram_total=64 * 1024**3, ram_available=1, swap_total=0, swap_available=0
        )
        preparer.record(completion(request))

    preparer, view = fixture(send)
    with pytest.raises(HTTPException, match="memory changed"):
        await preparer.prepare(card(), set())


@pytest.mark.parametrize(
    "selection", ["excluded", "different_required", "unimplemented_lane"]
)
async def test_unavailable_or_unimplemented_target_has_no_preparation_effect(
    selection: str,
) -> None:
    """Explicit choices cannot silently fall back to another worker or compute lane."""
    requests: list[PrepareDemucs] = []

    async def send(request: PrepareDemucs) -> None:
        requests.append(request)

    preparer, _ = fixture(send)
    with pytest.raises(HTTPException):
        await preparer.prepare(
            card(),
            {NODE} if selection == "excluded" else set(),
            required_nodes={NodeId("different")}
            if selection == "different_required"
            else None,
            requested_backend="demucs-cuda"
            if selection == "unimplemented_lane"
            else None,
        )
    assert not requests and not preparer._pending


async def test_four_pending_mounts_bound_admission_and_cancel_without_waiter_leaks() -> (
    None
):
    """A stalled worker cannot create unbounded retained preparation waiters."""
    requests: list[PrepareDemucs] = []

    async def send(request: PrepareDemucs) -> None:
        requests.append(request)

    preparer, _ = fixture(send)
    pending = [asyncio.create_task(preparer.prepare(card(), set())) for _ in range(4)]
    try:
        async with asyncio.timeout(3):
            while len(requests) != 4:
                await asyncio.sleep(0)
        with pytest.raises(HTTPException, match="capacity is full"):
            await preparer.prepare(card(), set())
        assert len(preparer._pending) == 4
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    assert not preparer._pending


async def test_total_preparation_budget_includes_waiting_for_send() -> None:
    """A delayed command send cannot grant another full wait budget afterwards."""
    now = [100.0]

    async def send(_: PrepareDemucs) -> None:
        now[0] += 1801.0

    preparer, _ = fixture(send, clock=lambda: now[0])
    with pytest.raises(HTTPException, match="timed out"):
        await preparer.prepare(card(), set())
    assert not preparer._pending
