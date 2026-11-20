"""Bounded native ACE preparation before public single-host music placement.

Installability is only an ABI/dependency candidate. Actual package verification,
executed device math and timestamped memory must return through an ordered
completion before the API forwards ordinary placement. No model is loaded here.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final, final

import anyio
from fastapi import HTTPException

from skulk.provisioning.ace_step_inventory import ACE_STEP_PACKAGES
from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.shared.backends import (
    ACE_STEP_COMPUTE_BACKENDS,
    expand_engine_backends,
    resolve_node_backend,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    model_engine_requirements,
    registry_compatible_backends_for_node,
)
from skulk.shared.models.native_music_admission import assess_native_music_placement
from skulk.shared.types.commands import PrepareAceStep
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import AceStepPreparationCompleted
from skulk.shared.types.native_music import NativeMusicDevice
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

_DEVICES: Final[dict[str, NativeMusicDevice]] = {
    "ace_step-cpu": "cpu",
    "ace_step-metal": "mps",
    "ace_step-cuda": "cuda",
    "ace_step-rocm": "rocm",
}


@dataclass
class _Pending:
    request: PrepareAceStep
    ready: anyio.Event = field(default_factory=anyio.Event)
    result: AceStepPreparationCompleted | None = None


def _eligible_host(
    package: AceStepWheel, device: NativeMusicDevice, resources: NodeResources
) -> bool:
    """Check actual ABI/generic hardware, never a tested-model/GPU allowlist."""
    architecture = (resources.architecture or "").lower()
    classes = resources.hardware_classes
    mac = package.platform_tag.startswith("macosx_")
    if mac:
        host = "platform:darwin" in classes and architecture in {"arm64", "aarch64"}
    else:
        arm = package.platform_tag.endswith("aarch64")
        host = "platform:linux" in classes and architecture in (
            {"arm64", "aarch64"} if arm else {"x86_64", "amd64"}
        )
    if not host:
        return False
    if device == "cpu":
        return package.variant == "cpu"
    if device == "mps":
        return mac and package.variant == "cpu" and "apple" in classes
    return package.variant == device and (
        "nvidia" in classes if device == "cuda" else "amd" in classes
    )


@final
class AceStepMountPreparer:
    """Own at most four API mount barriers under one total thirty-minute budget."""

    def __init__(
        self,
        owner: NodeId,
        state: Callable[[], State],
        telemetry: TelemetryView,
        send: Callable[[PrepareAceStep], Awaitable[None]],
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """Bind cluster owners without starting native, file or network effects."""
        self.owner, self.state, self.telemetry, self.send = (
            owner,
            state,
            telemetry,
            send,
        )
        self.clock, self.wall_clock = clock, wall_clock
        self._pending: dict[CommandId, _Pending] = {}

    def record(self, event: AceStepPreparationCompleted) -> None:
        """Accept one ordered result matching every identity of the live request."""
        pending = self._pending.get(event.request_id)
        if pending is None or pending.result is not None:
            return
        request = pending.request
        if (
            event.owner_node != self.owner
            or event.owner_node != request.owner_node
            or event.target_node != request.target_node
            or event.package_sha256 != request.package_sha256
            or event.device != request.device
        ):
            return
        pending.result = event
        pending.ready.set()

    async def prepare(
        self,
        card: ModelCard,
        excluded: set[NodeId],
        *,
        required_nodes: set[NodeId] | None = None,
        requested_backend: str | None = None,
    ) -> tuple[NodeId, NodeResources] | None:
        """Prepare fixed bytes/device, then assess actual complete native memory.

        A declaration and generic hardware make a node a candidate, never ready.
        All attempts share one absolute deadline, including send/queue time.
        Worker offline policy only allows independently verified cached bytes.
        Returned facts are request-scoped; Master admission independently checks
        authoritative pending commitments and its current platform limitations.
        """
        if card.music is None or card.music.native is None:
            return None
        if (
            requested_backend is not None
            and requested_backend not in ACE_STEP_COMPUTE_BACKENDS
        ):
            raise HTTPException(
                400, "No native ACE package implements the requested compute lane"
            )
        if required_nodes is not None and len(required_nodes) != 1:
            raise HTTPException(
                400, "Native ACE mounting requires exactly one specified node"
            )
        if len(self._pending) >= 4:
            raise HTTPException(
                503,
                "Native music preparation capacity is full; retry after a mount settles",
            )
        permitted = (
            expand_engine_backends(model_engine_requirements(card))
            | registry_compatible_backends_for_node(
                card, node_backends=ACE_STEP_COMPUTE_BACKENDS
            )
        ) & ACE_STEP_COMPUTE_BACKENDS
        if requested_backend is not None:
            permitted &= {requested_backend}
        order: dict[str, int] = {}
        remaining_backends = permitted
        while remaining_backends:
            selected = resolve_node_backend(
                remaining_backends,
                card.placement.backend_preference,
                remaining_backends,
            )
            assert selected is not None
            order[selected] = len(order)
            remaining_backends -= {selected}
        packages = {package.build_identity: package for package in ACE_STEP_PACKAGES}
        candidates: list[tuple[int, bool, int, str, NodeId, str, AceStepWheel]] = []
        for node in self.state().topology.list_nodes():
            if (
                node in excluded
                or required_nodes is not None
                and node not in required_nodes
            ):
                continue
            resources = self.telemetry.node_resources.get(node)
            if resources is None or resources.participation != "full":
                continue
            for backend in permitted:
                build = resources.installable_engine_builds.get(
                    backend
                ) or resources.engine_builds.get(backend)
                package = packages.get(build or "")
                if package is None or not _eligible_host(
                    package, _DEVICES[backend], resources
                ):
                    continue
                memory = self.telemetry.node_memory.get(node)
                # General telemetry only ranks candidates. It cannot establish
                # native dtype, device-zero capacity or complete phase fit.
                free = memory.ram_available.in_bytes if memory is not None else 0
                ready = (
                    backend in resources.backends
                    and resources.engine_builds.get(backend) == package.build_identity
                )
                candidates.append(
                    (
                        order[backend],
                        not ready,
                        -free,
                        str(node),
                        node,
                        backend,
                        package,
                    )
                )
        candidates.sort(key=lambda candidate: candidate[:4])
        if not candidates:
            raise HTTPException(
                503,
                "No participating worker advertises a compatible fixed native ACE package for its OS, CPU and requested device; refresh node resources or choose another backend",
            )
        deadline = self.clock() + 30 * 60
        absolute_deadline = self.wall_clock() + 30 * 60
        errors: list[str] = []
        for _, _, _, _, node, backend, package in candidates:
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            request = PrepareAceStep(
                target_node=node,
                owner_node=self.owner,
                package_sha256=package.sha256,
                device=_DEVICES[backend],
                expires_at=min(absolute_deadline, self.wall_clock() + remaining),
            )
            pending = _Pending(request)
            self._pending[request.command_id] = pending
            try:
                with anyio.move_on_after(remaining):
                    try:
                        await self.send(request)
                    except (
                        anyio.BrokenResourceError,
                        anyio.ClosedResourceError,
                    ) as error:
                        raise HTTPException(
                            503,
                            "Native music preparation lost its command channel; retry after cluster connectivity recovers",
                        ) from error
                    if self.clock() < deadline:
                        await pending.ready.wait()
                result = pending.result if self.clock() < deadline else None
                if result is None:
                    raise HTTPException(
                        503,
                        "Native music preparation reached its complete deadline; inspect worker cache/offline/device diagnostics and retry",
                    )
                fresh = result.resources
                if not result.success or fresh is None:
                    errors.append(
                        result.error
                        or "Worker could not verify the native package/device"
                    )
                    continue
                if (
                    node not in self.state().topology.list_nodes()
                    or fresh.participation != "full"
                ):
                    errors.append(
                        "Prepared worker is no longer a participating cluster member"
                    )
                    continue
                if (
                    fresh.engine_builds.get(backend) != package.build_identity
                    or backend in fresh.on_demand_backends
                ):
                    errors.append("Worker returned no ready exact native package build")
                    continue
                try:
                    assessed = assess_native_music_placement(
                        card,
                        backend,
                        node,
                        fresh,
                        self.state().instances,
                        now_unix_ms=int(self.wall_clock() * 1000),
                    )
                except ValueError as error:
                    errors.append(str(error))
                    continue
                if assessed.refusal is not None:
                    errors.append(assessed.refusal)
                    continue
                return node, fresh
            finally:
                self._pending.pop(request.command_id, None)
                errors = list(dict.fromkeys(errors[-8:]))
        raise HTTPException(
            503, "Native music could not be prepared: " + "; ".join(errors)[:2048]
        )
