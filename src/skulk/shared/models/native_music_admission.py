"""Pure native admission from actual lane observations and complete commitments.

No model is installed, loaded, qualified or placed by these helpers. Each caller
provides deduplicated authoritative and pending placements, plus the actual
selected runtime's observations. Existing allocations are unattributed here:
loaded status alone never earns a physical-memory credit.
"""

from collections.abc import Mapping
from typing import final

from pydantic import Field

from skulk.provisioning.ace_step_inventory import ACE_STEP_PACKAGES
from skulk.shared.models.memory_estimate import (
    backend_offloads_to_vram,
    estimate_shard_footprint,
    shard_admission_context_tokens,
    shard_fraction_of_model,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.native_music_memory import (
    NativeMusicMemoryEstimate,
    estimate_native_music_peak,
    native_music_memory_error,
    reserve_native_music_memory,
)
from skulk.shared.models.native_music_runtime import (
    NativeMusicModelPolicy,
    resolve_native_music_policy,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.native_music import (
    NativeMusicDevice,
    NativeMusicLibraryProbe,
    NativeMusicMemoryBudget,
    NativeMusicMemoryCapacity,
    native_music_admission_capacity,
)
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.instances import Instance, InstanceId, LlamaRpcInstance
from skulk.utils.pydantic_ext import FrozenModel


@final
class NativeMusicPlacementAssessment(FrozenModel):
    """Same exact data/math/peak policy for API, Master and Worker fit checks."""

    policy: NativeMusicModelPolicy = Field(
        description="Actual probed model math policy."
    )
    estimate: NativeMusicMemoryEstimate = Field(
        description="Complete target phase costs."
    )
    capacity: NativeMusicMemoryCapacity = Field(
        description="Raw actual observation; physical totals for the live process guard."
    )
    remaining: NativeMusicMemoryCapacity = Field(
        description="Pressure headroom normalized once, then all commitments reserved."
    )
    refusal: str | None = Field(
        max_length=1024,
        description="Shared fit diagnostic, or None when every pool fits.",
    )


def _gpu_backend(backend: str | None) -> bool | None:
    if backend is None:
        return None
    if backend.endswith("-cpu") or backend == "test_video":
        return False
    if (
        backend_offloads_to_vram(backend)
        or backend.endswith(("-metal", "-cuda", "-rocm", "-vulkan"))
        or backend in {"mlx", "mlx_audio"}
    ):
        return True
    return None


def native_music_instance_budget(
    instance: Instance, node: NodeId, *, unified: bool
) -> NativeMusicMemoryBudget:
    """Validate one complete native owner and return its full future phase budget.

    No loaded/idle status earns allocation credit. Unknown GPU topology must be
    passed as unified so a missing observation cannot lend physical host pages.
    Invalid restored owners raise ValueError; admission callers must refuse the
    affected pools rather than fall back to the much smaller generic footprint.
    """
    assignments = instance.shard_assignments
    runner = assignments.node_to_runner.get(node)
    shard = assignments.runner_to_shard.get(runner) if runner is not None else None
    if shard is None or isinstance(instance, LlamaRpcInstance):
        raise ValueError("Native music needs a complete single-host owner")
    policy = shard.native_music_policy
    card = shard.model_card
    if (
        policy is None
        or shard.resolved_backend != policy.backend
        or shard.resolved_engine_build != "ace_step@sha256:" + policy.payload_sha256
        or len(assignments.node_to_runner) != 1
        or len(assignments.runner_to_shard) != 1
        or shard.world_size != 1
        or shard.device_rank != 0
        or shard.start_layer != 0
        or shard.end_layer != card.n_layers
        or shard.n_layers != card.n_layers
    ):
        raise ValueError(
            "Native music cannot size an unstamped or partial native commitment"
        )
    policy = NativeMusicModelPolicy.model_validate_json(policy.model_dump_json())
    cost = estimate_native_music_peak(card, policy).process_budget()
    if policy.execution.device == "cpu" or not unified:
        return cost
    return cost.model_copy(update={"host_peak_bytes": cost.shared_peak_bytes})


def native_music_placement_commitments(
    node: NodeId,
    resources: NodeResources,
    instances: Mapping[InstanceId, Instance],
    *,
    target_device: NativeMusicDevice,
    node_is_unified: bool | None,
    excluded_instance: Instance | None = None,
) -> tuple[NativeMusicMemoryBudget, ...]:
    """Project pending and idle work onto this actual target's physical pools.

    Native work holds its complete phase policy, including idle future workspace.
    Other engines retain their existing concrete-shard footprint policy. Unknown
    GPU topology/ownership reserves host memory conservatively. CPU admission
    includes an existing shared GPU's simultaneous physical peak. Every GPU
    commitment is conservatively held against selected device zero; no free
    memory is borrowed from another GPU. RPC partitions cannot be fabricated.

    ``excluded_instance`` is only the exact authoritative instance currently
    being created by its Worker. A changed or missing owner cannot remove an
    unrelated commitment. API/Master new-placement calls exclude nothing.
    """
    if (
        excluded_instance is not None
        and instances.get(excluded_instance.instance_id) != excluded_instance
    ):
        raise ValueError("Native music admission ownership changed; refresh placement")
    if excluded_instance is not None:
        # Self-exclusion avoids charging the admitted owner twice; it cannot
        # turn a partial or corrupt restored owner into a valid native runner.
        native_music_instance_budget(
            excluded_instance, node, unified=node_is_unified is not False
        )
    budgets: list[NativeMusicMemoryBudget] = []
    for instance_id, instance in instances.items():
        if instance.instance_id != instance_id:
            raise ValueError(
                "Native music needs uniquely identified placement commitments"
            )
        if (
            excluded_instance is not None
            and instance_id == excluded_instance.instance_id
        ):
            continue
        assignments = instance.shard_assignments
        runner = assignments.node_to_runner.get(node)
        if runner is None:
            continue
        if isinstance(instance, LlamaRpcInstance):
            raise ValueError(
                "Native music cannot reserve unknown RPC partitions; wait for that placement to retire"
            )
        shard = assignments.runner_to_shard.get(runner)
        if shard is None:
            raise ValueError("Native music needs complete committed shard metadata")
        card = shard.model_card
        if card.music is not None and card.music.native is not None:
            policy = shard.native_music_policy
            if policy is None:
                raise ValueError(
                    "Native music cannot size an unstamped or partial native commitment"
                )
            existing_gpu = policy.execution.device != "cpu"
            probe = resources.native_music_probes.get(policy.backend)
            observation = (
                probe.memory_capacity
                if probe is not None and probe.payload_sha256 == policy.payload_sha256
                else None
            )
            # Unknown topology earns no host-memory saving. Metal is always
            # shared; a real matching observation may establish a discrete GPU.
            existing_unified = existing_gpu and (
                policy.execution.device == "mps"
                or observation is None
                or observation.device_memory_kind != "discrete"
            )
            cost = native_music_instance_budget(
                instance, node, unified=existing_unified
            )
            host = cost.host_peak_bytes
            device = cost.device_peak_bytes if target_device != "cpu" else 0
            shared = (
                cost.shared_peak_bytes
                if existing_unified
                else cost.host_peak_bytes
                if target_device == "cpu" or node_is_unified is True
                else host + device
            )
            # Unknown/mixed physical devices must not expose extra capacity.
            # The shared pool may conservatively cover that device commitment
            # too, but can never describe fewer pages than either domain view.
            shared = max(shared, host, device)
        else:
            fraction = shard_fraction_of_model(shard)
            if fraction is None:
                raise ValueError(
                    "Native music cannot size an unknown committed shard partition"
                )
            footprint = estimate_shard_footprint(
                card,
                fraction,
                resolved_backend=shard.resolved_backend,
                llama_server_settings=shard.llama_server_settings,
                memory_settings=shard.memory_settings,
                cache_layer_count=shard.end_layer - shard.start_layer,
                context_budget=shard_admission_context_tokens(
                    shard, instance.context_token_limit
                ),
            ).in_bytes
            gpu = _gpu_backend(shard.resolved_backend)
            shared_gpu = (
                node_is_unified is not False
                or (shard.resolved_backend or "").endswith("-metal")
                or shard.resolved_backend in {"mlx", "mlx_audio"}
            )
            host = footprint if gpu is not True or shared_gpu else 0
            device = footprint if gpu is not False and target_device != "cpu" else 0
            shared = max(host, device)
        budgets.append(
            NativeMusicMemoryBudget(
                host_peak_bytes=host, device_peak_bytes=device, shared_peak_bytes=shared
            )
        )
    return tuple(budgets)


def assess_native_music_placement(
    card: ModelCard,
    backend: str,
    node: NodeId,
    resources: NodeResources,
    instances: Mapping[InstanceId, Instance],
    *,
    now_unix_ms: int,
    node_is_unified: bool | None = None,
    excluded_instance: Instance | None = None,
) -> NativeMusicPlacementAssessment:
    """Bind a live fixed library and assess all three pools without qualification gates.

    The caller independently enforces model/engine compatibility and authoritative
    placement ownership. This function prepares nothing and supplies no lease or
    reservation. A fresh call is required at each actual admission boundary.
    """
    observed = resources.native_music_probes.get(backend)
    if observed is None or backend not in resources.backends:
        raise ValueError("Prepare the native music library before memory admission")
    probe = NativeMusicLibraryProbe.model_validate_json(observed.model_dump_json())
    build = "ace_step@sha256:" + probe.payload_sha256
    package = next(
        (item for item in ACE_STEP_PACKAGES if item.build_identity == build), None
    )
    if (
        resources.engine_builds.get(backend) != build
        or package is None
        or not (
            package.variant == probe.device
            or package.variant == "cpu"
            and probe.device == "mps"
            and package.platform_tag.startswith("macosx_")
        )
    ):
        raise ValueError("Native music needs the current exact verified package build")
    capacity = probe.memory_capacity
    if capacity is None:
        raise ValueError(
            "Native music needs its actual current host/device memory sample"
        )
    policy = resolve_native_music_policy(card, backend, probe)
    estimate = estimate_native_music_peak(card, policy)
    unified = capacity.device_memory_kind == "unified" or node_is_unified
    commitments = native_music_placement_commitments(
        node,
        resources,
        instances,
        target_device=probe.device,
        node_is_unified=unified,
        excluded_instance=excluded_instance,
    )
    remaining = reserve_native_music_memory(
        native_music_admission_capacity(capacity), (), additional_budgets=commitments
    )
    return NativeMusicPlacementAssessment(
        policy=policy,
        estimate=estimate,
        capacity=capacity,
        remaining=remaining,
        refusal=native_music_memory_error(
            estimate, remaining, device=probe.device, now_unix_ms=now_unix_ms
        ),
    )
