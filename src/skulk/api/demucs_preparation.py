"""Exact on-demand native-library preparation before ordinary model placement.

Candidate package metadata never authorizes placement. A worker must verify the
pinned package and explicit device, then publish its ordered resource snapshot.
Loaded-model readiness is established later by the normal supervised runner.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import final

import anyio
from fastapi import HTTPException

from skulk.master.placement import model_platform_backends
from skulk.master.placement_utils import (
    reserve_system_ram_usage,
    unified_memory_gpu_node_ids,
    usable_vram_by_node,
)
from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES
from skulk.provisioning.demucs_runtime import DemucsWheel
from skulk.shared.backends import expand_engine_backends
from skulk.shared.models.memory_estimate import (
    backend_offloads_to_vram,
    estimate_shard_footprint,
    gpu_working_set_ceiling,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    model_engine_requirements,
    registry_compatible_backends_for_node,
)
from skulk.shared.types.commands import PrepareDemucs
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import DemucsPreparationCompleted
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView


@dataclass
class _Pending:
    node: NodeId
    ready: anyio.Event = field(default_factory=anyio.Event)
    result: DemucsPreparationCompleted | None = None


@final
class DemucsMountPreparer:
    """Own at most four bounded mount preparations on the accepting API node."""

    def __init__(
        self,
        owner: NodeId,
        state: Callable[[], State],
        telemetry: TelemetryView,
        send: Callable[[PrepareDemucs], Awaitable[None]],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind existing cluster owners without files, network or package effects."""
        self.owner, self.state, self.telemetry, self.send, self.clock = (
            owner,
            state,
            telemetry,
            send,
            clock,
        )
        self._pending: dict[CommandId, _Pending] = {}

    def record(self, event: DemucsPreparationCompleted) -> None:
        """Accept one ordered completion only for its current owner and worker."""
        pending = self._pending.get(event.request_id)
        if (
            pending is None
            or pending.result is not None
            or event.owner_node != self.owner
            or event.target_node != pending.node
        ):
            return
        pending.result = event
        pending.ready.set()

    def _available(
        self,
        card: ModelCard,
        node: NodeId,
        backend: str,
        fresh: NodeResources | None = None,
    ) -> int:
        snapshot = self.state()
        resources = (
            self.telemetry.node_resources
            if fresh is None
            else {**self.telemetry.node_resources, node: fresh}
        )
        video_memory = usable_vram_by_node(
            self.telemetry.node_system,
            resources,
            node_memory=self.telemetry.node_memory,
            current_instances=snapshot.instances,
        )
        memory = reserve_system_ram_usage(
            self.telemetry.node_memory,
            snapshot.instances,
            video_memory,
            unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                self.telemetry.node_system,
                resources,
                node_memory=self.telemetry.node_memory,
            ),
            node_resources=resources,
        ).get(node)
        if memory is None:
            return 0
        pool = (
            video_memory.get(node)
            if backend_offloads_to_vram(backend)
            else (
                memory.ram_available
                if backend == "demucs-cpu"
                else min(
                    memory.ram_available, gpu_working_set_ceiling(memory.ram_total)
                )
            )
        )
        if (
            pool is None
            or estimate_shard_footprint(card, 1.0, resolved_backend=backend) > pool
        ):
            return 0
        return pool.in_bytes

    async def prepare(
        self,
        card: ModelCard,
        excluded: set[NodeId],
        *,
        required_nodes: set[NodeId] | None = None,
        requested_backend: str | None = None,
    ) -> tuple[NodeId, NodeResources] | None:
        """Prepare an eligible fixed package and return verified placement facts.

        A complete thirty-minute budget covers all candidate attempts. Offline
        workers may reuse only independently verified cached packages. Failure
        leaves unrelated engines and the worker's ordinary health untouched.
        """
        if ModelTask.AudioSeparation not in card.tasks:
            return None
        if requested_backend is not None and requested_backend not in {
            "demucs-cpu",
            "demucs-metal",
        }:
            raise HTTPException(
                400, "No native separation package implements that compute lane"
            )
        if len(self._pending) >= 4:
            raise HTTPException(
                503,
                "Native separation preparation capacity is full; retry after a mount settles",
            )
        permitted = expand_engine_backends(model_engine_requirements(card))
        packages = {package.build_identity: package for package in DEMUCS_PACKAGES}
        candidates: list[tuple[int, bool, int, NodeId, str, DemucsWheel]] = []
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
            for backend in ("demucs-metal", "demucs-cpu"):
                if requested_backend is not None and requested_backend != backend:
                    continue
                if (
                    backend not in permitted
                    and backend
                    not in registry_compatible_backends_for_node(
                        card, node_backends=frozenset({backend})
                    )
                ):
                    continue
                build = resources.installable_engine_builds.get(
                    backend
                ) or resources.engine_builds.get(backend)
                package = packages.get(build or "")
                capacity = self._available(card, node, backend)
                if package is not None and capacity:
                    candidates.append(
                        (
                            2 if backend == "demucs-metal" else 1,
                            resources.engine_builds.get(backend) == build,
                            capacity,
                            node,
                            backend,
                            package,
                        )
                    )
        candidates.sort(
            key=lambda value: (value[0], value[1], value[2], str(value[3])),
            reverse=True,
        )
        if not candidates:
            raise HTTPException(
                503,
                "No participating worker has a compatible native separation package and enough available memory",
            )
        errors: list[str] = []
        deadline = self.clock() + 30 * 60
        for _, _, _, node, backend, package in candidates:
            if self.clock() >= deadline:
                break
            request = PrepareDemucs(
                target_node=node,
                owner_node=self.owner,
                package_sha256=package.sha256,
                device="mps" if backend == "demucs-metal" else "cpu",
            )
            pending = _Pending(node)
            self._pending[request.command_id] = pending
            try:
                remaining = deadline - self.clock()
                with anyio.move_on_after(max(0.0, remaining)):
                    await self.send(request)
                    if self.clock() < deadline:
                        await pending.ready.wait()
                result = pending.result if self.clock() < deadline else None
                if result is None:
                    errors.append(
                        "Worker preparation timed out or its API session reset"
                    )
                    continue
                resources = result.resources
                if not result.success or resources is None:
                    errors.append(
                        result.error
                        or "Worker could not verify the native package/device"
                    )
                    continue
                if resources.engine_builds.get(
                    backend
                ) != package.build_identity or backend not in model_platform_backends(
                    card, resources
                ):
                    errors.append(
                        "Worker returned no compatible exact-build separation lane"
                    )
                    continue
                if not self._available(card, node, backend, resources):
                    errors.append(
                        "Available memory changed during preparation; retry placement"
                    )
                    continue
                return node, resources
            finally:
                self._pending.pop(request.command_id, None)
        raise HTTPException(
            503, "Native separation could not be prepared: " + "; ".join(errors)
        )
