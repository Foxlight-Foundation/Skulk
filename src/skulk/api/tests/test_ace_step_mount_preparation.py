# pyright: reportPrivateUsage=false
"""Real API barrier and shared phase admission; no library/model inference claims."""

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import HTTPException

import skulk.master.placement as placement_module
from skulk.api.ace_step_preparation import AceStepMountPreparer
from skulk.api.main import API
from skulk.api.types.api import CreateInstanceParams, PlaceInstanceParams
from skulk.master.main import ace_step_preparation_event
from skulk.master.placement import place_instance
from skulk.provisioning.ace_step_inventory import ACE_STEP_PACKAGES
from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.shared.models.tests.test_native_music_admission import (
    instance as native_instance,
)
from skulk.shared.models.tests.test_native_music_runtime import card_with_layouts, probe
from skulk.shared.topology import Topology
from skulk.shared.types.commands import (
    Command,
    CreateInstance,
    PlaceInstance,
    PrepareAceStep,
)
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.events import (
    AceStepPreparationCompleted,
    AceStepPreparationRequested,
)
from skulk.shared.types.native_music import NativeMusicDevice, NativeMusicMemoryCapacity
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

NODE = NodeId("node")
OWNER = NodeId("api")
GIB = 1024**3
MAC = ACE_STEP_PACKAGES[0]


def backend(device: NativeMusicDevice) -> str:
    """Use public compute vocabulary, with Metal's private torch device explicit."""
    return "ace_step-metal" if device == "mps" else "ace_step-" + device


def candidate(package: AceStepWheel, device: NativeMusicDevice) -> NodeResources:
    """Actual host ABI/generic vendor facts and installable pins, without readiness."""
    mac = package.platform_tag.startswith("macosx_")
    return NodeResources(
        backends=frozenset(),
        architecture="arm64"
        if mac
        else "aarch64"
        if package.platform_tag.endswith("aarch64")
        else "x86_64",
        hardware_classes=frozenset(
            {
                "platform:darwin" if mac else "platform:linux",
                "apple"
                if mac
                else "nvidia"
                if device == "cuda"
                else "amd"
                if device == "rocm"
                else "cpu",
            }
        ),
        installable_engine_builds={backend(device): package.build_identity},
    )


def ready(
    package: AceStepWheel, device: NativeMusicDevice, *, free: int = 64 * GIB
) -> NodeResources:
    """Provide valid controlled actual-device observations for the original package."""
    shared = device in {"mps", "rocm"}
    observed = probe(device).model_copy(
        update={
            "payload_sha256": package.payload_sha256,
            "memory_capacity": NativeMusicMemoryCapacity(
                sampled_at_unix_ms=100_000,
                host_total_bytes=64 * GIB,
                host_available_bytes=free,
                device_index=None if device == "cpu" else 0,
                device_memory_kind=None
                if device == "cpu"
                else "unified"
                if shared
                else "discrete",
                device_total_bytes=None if device == "cpu" else 64 * GIB,
                device_available_bytes=None if device == "cpu" else free,
                shared_total_bytes=64 * GIB if shared else None,
                shared_available_bytes=free if shared else None,
            ),
        }
    )
    return candidate(package, device).model_copy(
        update={
            "backends": frozenset({backend(device)}),
            "engine_builds": {backend(device): package.build_identity},
            "native_music_probes": {backend(device): observed},
        }
    )


def fixture(
    send: Callable[[PrepareAceStep], Awaitable[None]],
    package: AceStepWheel = MAC,
    *,
    device: NativeMusicDevice = "cpu",
    clock: Callable[[], float] = lambda: 100.0,
) -> tuple[AceStepMountPreparer, State, TelemetryView]:
    """Keep actual membership, catalogue identity and phase admission in the barrier."""
    topology = Topology()
    topology.add_node(NODE)
    state = State(topology=topology)
    view = TelemetryView()
    view.node_resources[NODE] = candidate(package, device)
    return (
        AceStepMountPreparer(
            OWNER, lambda: state, view, send, clock=clock, wall_clock=clock
        ),
        state,
        view,
    )


def completion(
    request: PrepareAceStep, package: AceStepWheel = MAC, *, free: int = 64 * GIB
) -> AceStepPreparationCompleted:
    """Return exact correlated metadata, not a mounted or qualified music model."""
    return AceStepPreparationCompleted(
        request_id=request.command_id,
        target_node=request.target_node,
        owner_node=request.owner_node,
        package_sha256=request.package_sha256,
        device=request.device,
        success=True,
        resources=ready(package, request.device, free=free),
    )


@pytest.mark.parametrize("package", ACE_STEP_PACKAGES)
async def test_all_six_fixed_profiles_prepare_then_use_actual_shared_admission(
    package: AceStepWheel,
) -> None:
    """Every OS/CPU package reaches the same barrier without empirical permission claims."""
    requests: list[PrepareAceStep] = []

    async def send(request: PrepareAceStep) -> None:
        requests.append(request)
        ordered = ace_step_preparation_event(state, request, now_unix=100.0)
        assert isinstance(ordered, AceStepPreparationRequested)
        assert ordered.expires_at == request.expires_at == 1900.0
        preparer.record(completion(request, package))

    preparer, state, view = fixture(send, package, device=package.variant)
    before = view.node_resources[NODE]
    result = await preparer.prepare(card_with_layouts(), set())
    assert result == (NODE, ready(package, package.variant))
    assert (
        requests[0].package_sha256 == package.sha256
        and requests[0].device == package.variant
    )
    assert view.node_resources[NODE] is before and not before.backends
    assert not preparer._pending and not state.instances and not state.tasks


async def test_metal_uses_cpu_profile_without_loading_an_audio_cpp_model() -> None:
    """The Mac CPU package explicitly implements MPS; it is not a Vulkan fallback."""

    async def send(request: PrepareAceStep) -> None:
        assert request.device == "mps" and request.package_sha256 == MAC.sha256
        preparer.record(completion(request))

    preparer, _, _ = fixture(send, device="mps")
    assert await preparer.prepare(
        card_with_layouts(), set(), requested_backend="ace_step-metal"
    ) == (NODE, ready(MAC, "mps"))


async def test_only_fully_correlated_ordered_completion_releases_the_waiter() -> None:
    """Wrong owner, worker, digest, request or device cannot settle another mount."""
    requests: list[PrepareAceStep] = []

    async def send(request: PrepareAceStep) -> None:
        requests.append(request)

    preparer, _, view = fixture(send)
    pending = asyncio.create_task(preparer.prepare(card_with_layouts(), set()))
    try:
        async with asyncio.timeout(3):
            while not requests:
                await asyncio.sleep(0)
        event = completion(requests[0])
        for changes in (
            {"owner_node": NodeId("other")},
            {"target_node": NodeId("other")},
            {"package_sha256": "a" * 64},
            {"request_id": CommandId("other")},
            {"device": "cuda"},
        ):
            preparer.record(event.model_copy(update=changes))
            assert not pending.done()
        assert not view.node_resources[NODE].backends
        preparer.record(event)
        assert await pending == (NODE, event.resources)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert not preparer._pending


@pytest.mark.parametrize(
    "fault",
    [
        "unknown-build",
        "wrong-abi",
        "missing-vendor",
        "management",
        "excluded",
        "other-node",
        "vulkan",
    ],
)
async def test_incompatible_candidate_has_no_preparation_effect(fault: str) -> None:
    """Genuine package/hardware/operator constraints apply before network effects."""
    requests: list[PrepareAceStep] = []
    package = next(item for item in ACE_STEP_PACKAGES if item.variant == "cuda")

    async def send(request: PrepareAceStep) -> None:
        requests.append(request)

    preparer, _, view = fixture(send, package, device="cuda")
    observed = view.node_resources[NODE]
    changes: dict[str, object] = {}
    if fault == "unknown-build":
        changes["installable_engine_builds"] = {"ace_step-cuda": "unknown"}
    if fault == "wrong-abi":
        changes["architecture"] = "unknown-cpu"
    if fault == "missing-vendor":
        changes["hardware_classes"] = frozenset({"platform:linux"})
    if fault == "management":
        changes["participation"] = "management"
    view.node_resources[NODE] = observed.model_copy(update=changes)
    with pytest.raises(HTTPException):
        await preparer.prepare(
            card_with_layouts(),
            {NODE} if fault == "excluded" else set(),
            required_nodes={NodeId("other")} if fault == "other-node" else None,
            requested_backend="ace_step-vulkan" if fault == "vulkan" else None,
        )
    assert not requests and not preparer._pending


@pytest.mark.parametrize(
    "fault", ["offline", "wrong-build", "on-demand", "low-memory", "departed"]
)
async def test_failed_verification_or_changed_capacity_does_not_advertise_readiness(
    fault: str,
) -> None:
    """Failure is actionable and leaves ordinary health/resource observations intact."""

    async def send(request: PrepareAceStep) -> None:
        event = completion(request, free=1 if fault == "low-memory" else 64 * GIB)
        assert event.resources is not None
        fresh = event.resources
        if fault == "offline":
            event = event.model_copy(
                update={
                    "success": False,
                    "resources": None,
                    "error": "Offline cache does not contain the verified package",
                }
            )
        if fault == "wrong-build":
            event = event.model_copy(
                update={
                    "resources": fresh.model_copy(
                        update={"engine_builds": {"ace_step-cpu": "unknown"}}
                    )
                }
            )
        if fault == "on-demand":
            event = event.model_copy(
                update={
                    "resources": fresh.model_copy(
                        update={"on_demand_backends": frozenset({"ace_step-cpu"})}
                    )
                }
            )
        if fault == "departed":
            state.topology.remove_node(NODE)
        preparer.record(event)

    preparer, state, view = fixture(send)
    before = view.node_resources[NODE]
    with pytest.raises(HTTPException):
        await preparer.prepare(card_with_layouts(), set())
    assert view.node_resources[NODE] is before and not preparer._pending


async def test_public_storage_reason_survives_barrier_and_fresh_retry() -> None:
    """Storage refusal reaches the caller unchanged and releases the API's mount slot."""
    requests: list[PrepareAceStep] = []
    reason = "Native music preparation needs more free disk space on the selected worker; free storage and retry."

    async def send(request: PrepareAceStep) -> None:
        requests.append(request)
        event = completion(request)
        if len(requests) == 1:
            event = event.model_copy(
                update={
                    "success": False,
                    "resources": None,
                    "error": reason,
                }
            )
        preparer.record(event)

    preparer, state, view = fixture(send)
    before = view.node_resources[NODE]
    with pytest.raises(HTTPException) as failed:
        await preparer.prepare(card_with_layouts(), set())
    assert failed.value.status_code == 503 and reason in failed.value.detail
    assert not preparer._pending and not state.instances and not state.tasks
    assert view.node_resources[NODE] is before and not before.backends

    assert await preparer.prepare(card_with_layouts(), set()) == (
        NODE,
        ready(MAC, "cpu"),
    )
    assert len(requests) == 2 and requests[0].command_id != requests[1].command_id
    assert not preparer._pending and view.node_resources[NODE] is before


async def test_all_candidate_attempts_keep_original_absolute_deadline() -> None:
    """A second candidate receives remaining time, including first-attempt queue/work."""
    now = [100.0]
    requests: list[PrepareAceStep] = []

    async def send(request: PrepareAceStep) -> None:
        requests.append(request)
        if len(requests) == 1:
            now[0] += 120.0
            preparer.record(
                AceStepPreparationCompleted(
                    request_id=request.command_id,
                    target_node=request.target_node,
                    owner_node=request.owner_node,
                    package_sha256=request.package_sha256,
                    device=request.device,
                    success=False,
                    error="Controlled first candidate verification refusal",
                )
            )
        else:
            event = completion(request)
            assert event.resources is not None
            observed = event.resources.native_music_probes["ace_step-cpu"]
            assert observed.memory_capacity is not None
            refreshed = observed.model_copy(
                update={
                    "memory_capacity": observed.memory_capacity.model_copy(
                        update={"sampled_at_unix_ms": int(now[0] * 1000)}
                    )
                }
            )
            preparer.record(
                event.model_copy(
                    update={
                        "resources": event.resources.model_copy(
                            update={"native_music_probes": {"ace_step-cpu": refreshed}}
                        )
                    }
                )
            )

    preparer, state, view = fixture(send, clock=lambda: now[0])
    other = NodeId("other")
    state.topology.add_node(other)
    view.node_resources[other] = candidate(MAC, "cpu")
    result = await preparer.prepare(card_with_layouts(), set())
    assert result is not None and len(requests) == 2
    assert requests[0].command_id != requests[1].command_id
    assert requests[0].expires_at == requests[1].expires_at == 1900.0


async def test_send_wait_cannot_reset_total_budget() -> None:
    """Completion arriving after the deadline cannot gain another wait period."""
    now = [100.0]

    async def send(request: PrepareAceStep) -> None:
        now[0] += 1801.0
        preparer.record(completion(request))

    preparer, _, _ = fixture(send, clock=lambda: now[0])
    with pytest.raises(HTTPException, match="complete deadline"):
        await preparer.prepare(card_with_layouts(), set())
    assert not preparer._pending


@pytest.mark.parametrize(
    "failure", [anyio.BrokenResourceError, anyio.ClosedResourceError]
)
async def test_lost_command_transport_is_actionable_and_releases_waiter(
    failure: type[Exception],
) -> None:
    """Channel loss cannot leave a mount waiting or escape as an unrelated 500."""

    async def send(_request: PrepareAceStep) -> None:
        raise failure()

    preparer, _, view = fixture(send)
    before = view.node_resources[NODE]
    with pytest.raises(HTTPException, match="lost its command channel") as raised:
        await preparer.prepare(card_with_layouts(), set())
    assert raised.value.status_code == 503
    assert not preparer._pending and view.node_resources[NODE] is before


@pytest.mark.parametrize(
    "fault", ["expired", "unknown-package", "wrong-device", "departed"]
)
def test_actual_master_orders_only_live_fixed_requests_with_remaining_budget(
    fault: str,
) -> None:
    """The command processor's actual decision cannot reset expiry or install arbitrary bytes."""
    topology = Topology()
    if fault != "departed":
        topology.add_node(NODE)
    state = State(topology=topology)
    command = PrepareAceStep(
        target_node=NODE,
        owner_node=OWNER,
        package_sha256="f" * 64 if fault == "unknown-package" else MAC.sha256,
        device="cuda" if fault == "wrong-device" else "cpu",
        expires_at=99.0 if fault == "expired" else 2000.0,
    )
    refused = ace_step_preparation_event(state, command, now_unix=100.0)
    assert isinstance(refused, AceStepPreparationCompleted) and not refused.success
    assert refused.request_id == command.command_id and refused.error is not None
    assert refused.resources is None


def test_actual_master_caps_excessive_request_duration_and_preserves_earlier_deadline() -> (
    None
):
    """The original absolute budget survives queueing; even direct commands are bounded."""
    topology = Topology()
    topology.add_node(NODE)
    state = State(topology=topology)
    command = PrepareAceStep(
        target_node=NODE,
        owner_node=OWNER,
        package_sha256=MAC.sha256,
        device="cpu",
        expires_at=9000.0,
    )
    bounded = ace_step_preparation_event(state, command, now_unix=100.0)
    assert (
        isinstance(bounded, AceStepPreparationRequested)
        and bounded.expires_at == 1900.0
    )
    earlier = ace_step_preparation_event(
        state, command.model_copy(update={"expires_at": 700.0}), now_unix=200.0
    )
    assert (
        isinstance(earlier, AceStepPreparationRequested) and earlier.expires_at == 700.0
    )


async def test_four_pending_mounts_are_bounded_and_api_cancellation_cleans_waiters() -> (
    None
):
    """Unresponsive workers cannot accumulate unbounded API waiters."""
    requests: list[PrepareAceStep] = []

    async def send(request: PrepareAceStep) -> None:
        requests.append(request)

    preparer, _, _ = fixture(send)
    active = [
        asyncio.create_task(preparer.prepare(card_with_layouts(), set()))
        for _ in range(4)
    ]
    try:
        async with asyncio.timeout(3):
            while len(requests) != 4:
                await asyncio.sleep(0)
        with pytest.raises(HTTPException, match="capacity is full"):
            await preparer.prepare(card_with_layouts(), set())
    finally:
        for request in active:
            request.cancel()
        await asyncio.gather(*active, return_exceptions=True)
    assert not preparer._pending


async def test_public_exact_instance_uses_real_native_barrier_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public method cannot dispatch an exact native instance ahead of preparation."""
    forwarded: list[Command] = []
    api = object.__new__(API)
    api.node_id = OWNER
    topology = Topology()
    topology.add_node(NODE)
    api.state = State(topology=topology, last_event_applied_idx=0)
    api._telemetry_view = TelemetryView()
    api._telemetry_view.node_resources[NODE] = candidate(MAC, "cpu")

    async def send(
        command: Command, *, expected_session: SystemId | None = None
    ) -> None:
        forwarded.append(command)
        if isinstance(command, PrepareAceStep):
            api._ace_step_mount_preparer.record(completion(command))

    api._ace_step_mount_preparer = AceStepMountPreparer(
        OWNER,
        lambda: api.state,
        api._telemetry_view,
        send,
        clock=lambda: 100.0,
        wall_clock=lambda: 100.0,
    )
    monkeypatch.setattr(api, "_send", send)
    monkeypatch.setattr(
        API, "_load_authorized_model_card", AsyncMock(return_value=card_with_layouts())
    )
    owner = native_instance(ready(MAC, "cpu"))
    response = await api.create_instance(CreateInstanceParams(instance=owner))
    assert len(forwarded) == 2 and isinstance(forwarded[0], PrepareAceStep)
    assert isinstance(forwarded[1], CreateInstance)
    assert forwarded[1].prepared_node_resources == {NODE: ready(MAC, "cpu")}
    assert response.instance_id == owner.instance_id
    assert not api._telemetry_view.node_resources[NODE].backends


async def test_public_ordinary_mount_uses_actual_platform_and_placement_without_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real API preparation and placer agree despite an older installable-only packet.

    Only catalog lookup and the transport/native completion are controlled;
    public card/request types, platform rules, fixed pins and memory admission
    are actual. This is source integration, not an installed fresh-user journey.
    """
    forwarded: list[Command] = []
    api = object.__new__(API)
    api.node_id = OWNER
    api.paused = False
    topology = Topology()
    topology.add_node(NODE)
    api.state = State(topology=topology, last_event_applied_idx=0)
    api._telemetry_view = TelemetryView()
    api._telemetry_view.node_resources[NODE] = candidate(MAC, "cpu")
    api._telemetry_view.node_memory[NODE] = MemoryUsage.from_bytes(
        ram_total=64 * GIB,
        ram_available=64 * GIB,
        swap_total=0,
        swap_available=0,
    )

    async def send(
        command: Command, *, expected_session: SystemId | None = None
    ) -> None:
        forwarded.append(command)
        if isinstance(command, PrepareAceStep):
            ordered = ace_step_preparation_event(api.state, command, now_unix=100.0)
            assert isinstance(ordered, AceStepPreparationRequested)
            api._ace_step_mount_preparer.record(completion(command))

    api._ace_step_mount_preparer = AceStepMountPreparer(
        OWNER,
        lambda: api.state,
        api._telemetry_view,
        send,
        clock=lambda: 100.0,
        wall_clock=lambda: 100.0,
    )
    monkeypatch.setattr(api, "_send", send)
    monkeypatch.setattr(
        API, "_load_authorized_model_card", AsyncMock(return_value=card_with_layouts())
    )
    monkeypatch.setattr(placement_module.time, "time", lambda: 100.0)
    response = await api.place_instance(
        PlaceInstanceParams(model_id=card_with_layouts().model_id)
    )
    assert len(forwarded) == 2 and isinstance(forwarded[0], PrepareAceStep)
    request = forwarded[1]
    assert isinstance(request, PlaceInstance)
    assert request.prepared_node_resources == {NODE: ready(MAC, "cpu")}
    # The authoritative placement path also uses the actual returned snapshot,
    # rather than interpreting an old installable packet as engine readiness.
    placed = place_instance(
        request,
        api.state.topology,
        api.state.instances,
        api._telemetry_view.node_memory,
        api.state.node_network,
        node_resources=request.prepared_node_resources,
    )
    shard = next(
        iter(placed[response.instance_id].shard_assignments.runner_to_shard.values())
    )
    assert (
        shard.resolved_backend == "ace_step-cpu"
        and shard.native_music_policy is not None
    )
    assert shard.resolved_engine_build == MAC.build_identity
    assert response.engine_install is None
    assert not api._telemetry_view.node_resources[NODE].backends
