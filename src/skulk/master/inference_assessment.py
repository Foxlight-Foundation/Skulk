"""Assess workloads with production admission and isolated candidate snapshots."""

import math
from collections.abc import Mapping
from datetime import datetime, timezone

from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    model_platform_backends,
    place_instance,
)
from skulk.master.placement_utils import (
    reserve_system_ram_usage,
    unified_memory_gpu_node_ids,
    usable_vram_by_node,
)
from skulk.shared.backends import engine_of
from skulk.shared.models.inference_assessment import (
    AssessedReplica,
    AssessedShard,
    ExternalConfiguration,
    InferenceAssessment,
    InferenceWorkload,
)
from skulk.shared.models.llama_server_settings import LlamaServerSettings
from skulk.shared.models.memory_estimate import (
    GPU_VRAM_WORKING_SET_FRACTION,
    GPU_WORKING_SET_FRACTION,
    backend_offloads_to_vram,
    estimate_shard_footprint,
    gpu_working_set_ceiling,
    per_token_kv_bytes,
    shard_fraction_of_model,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    authorized_model_card_digest,
    get_model_engine_support,
    same_authorized_model_card,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage, NodeNetworkInfo, NodeResources
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import Sharding, TensorShardMetadata


def _plan_replica(
    card: ModelCard,
    workload: InferenceWorkload,
    topology: Topology,
    instances: Mapping[InstanceId, Instance],
    memory: Mapping[NodeId, MemoryUsage],
    network: Mapping[NodeId, NodeNetworkInfo],
    resources: Mapping[NodeId, NodeResources],
    vram: Mapping[NodeId, Memory],
    unified: frozenset[NodeId],
    excluded: set[NodeId],
    request_count: int = 1,
) -> tuple[Instance | None, tuple[str, ...], bool]:
    failures: list[str] = []
    pending = False
    for sharding in (Sharding.Pipeline, Sharding.Tensor):
        if sharding == Sharding.Tensor and not card.supports_tensor:
            continue
        for meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
            try:
                planned = place_instance(
                    PlaceInstance(
                        model_card=card,
                        sharding=sharding,
                        instance_meta=meta,
                        min_nodes=1,
                        requested_context_tokens=workload.context_tokens
                        * request_count,
                    ),
                    topology=topology,
                    current_instances=instances,
                    node_memory=memory,
                    node_network=network,
                    node_resources=resources,
                    node_vram=vram,
                    unified_memory_gpu_nodes=unified,
                    excluded_nodes=excluded,
                )
                instance = next(
                    value for key, value in planned.items() if key not in instances
                )
                if any(
                    shard.resolved_backend is None
                    or engine_of(shard.resolved_backend) is None
                    for shard in instance.shard_assignments.runner_to_shard.values()
                ):
                    failures.append(
                        "The candidate backend has no implemented core inference engine."
                    )
                    continue
                if any(
                    backend_offloads_to_vram(
                        instance.shard_assignments.runner_to_shard[
                            runner
                        ].resolved_backend
                    )
                    and node not in vram
                    for node, runner in instance.shard_assignments.node_to_runner.items()
                ):
                    pending = True
                    failures.append(
                        "GPU serving requires observed usable GPU memory; system RAM cannot substitute for missing VRAM evidence."
                    )
                    continue
                if _replica_slots(instance) < request_count:
                    failures.append(
                        "This candidate does not advertise enough simultaneous serving slots."
                    )
                    continue
                # RPC chooses its actual weight partition at runtime. One replica
                # can use ordinary RPC admission, but a second cannot reserve
                # concrete per-node bytes from that nominal driver shard.
                if workload.concurrent_requests > 1 and any(
                    shard_fraction_of_model(shard) is None
                    for shard in instance.shard_assignments.runner_to_shard.values()
                ):
                    pending = True
                    failures.append(
                        "Concurrent RPC replica reservations lack a concrete memory partition."
                    )
                    continue
                return instance, (), False
            except PlacementInfoPendingError as error:
                pending = True
                failures.append(str(error))
            except PlacementError as error:
                failures.append(str(error))
    return None, tuple(dict.fromkeys(failures)), pending


def _replica_slots(instance: Instance) -> int:
    """Observed llama-server slot contract; other runners use serial replicas."""
    slots: list[int] = []
    for shard in instance.shard_assignments.runner_to_shard.values():
        if (
            shard.resolved_backend is None
            or not shard.resolved_backend.startswith("llama_server")
            or shard.llama_server_settings is None
        ):
            return 1
        runtime = shard.model_card.runtime
        slots.append(
            shard.llama_server_settings.effective_slots(
                speculative_vision=shard.model_card.vision is not None
                and runtime is not None
                and runtime.served_spec_type not in (None, "none"),
            )
        )
    return min(slots, default=1)


def _external_configurations(
    card: ModelCard,
    workload: InferenceWorkload,
) -> tuple[ExternalConfiguration, ...]:
    """Dry-run each evidenced engine on a separate full-model host specification."""
    configs: list[ExternalConfiguration] = []
    claims = get_model_engine_support(card)
    backends = set(card.placement.compatible_backends)
    backends.update(claim.engine for claim in claims if claim.status == "supported")
    for backend in sorted(backends):
        if engine_of(backend) is None:
            continue
        matching = [
            claim
            for claim in claims
            if claim.status == "supported" and claim.engine == backend
        ]
        # Never combine build and hardware evidence from different support rows.
        variants = matching or [None]
        for claim in variants:
            settings = LlamaServerSettings()
            requests = (
                min(
                    workload.concurrent_requests,
                    settings.parallel_slots,
                    max(1, card.context_length // workload.context_tokens),
                )
                if backend.startswith("llama_server")
                else 1
            )
            context_pool = workload.context_tokens * requests
            required = estimate_shard_footprint(
                card,
                1.0,
                context_pool,
                resolved_backend=backend,
                llama_server_settings=settings,
            ).in_bytes
            gpu = backend_offloads_to_vram(backend)
            fraction = (
                GPU_VRAM_WORKING_SET_FRACTION if gpu else GPU_WORKING_SET_FRACTION
            )
            total = math.ceil(required / fraction) + 1024**2
            node = NodeId("assessment-candidate")
            topology = Topology()
            topology.add_node(node)
            resources = NodeResources(
                backends=frozenset({backend}),
                engine_builds={backend: claim.engine_build}
                if claim is not None
                else {},
                hardware_classes=frozenset(claim.hardware_classes)
                if claim is not None
                else frozenset(),
                llama_server_settings=settings,
            )
            instance, _, _ = _plan_replica(
                card,
                workload,
                topology,
                {},
                {
                    node: MemoryUsage.from_bytes(
                        ram_total=total,
                        ram_available=total,
                        swap_total=0,
                        swap_available=0,
                    )
                },
                {node: NodeNetworkInfo()},
                {node: resources},
                {node: Memory.from_bytes(int(total * fraction))} if gpu else {},
                frozenset(),
                set(),
                request_count=requests,
            )
            if instance is None:
                continue
            projector = (
                card.vision.projector_size or 0 if card.vision is not None else 0
            )
            configs.append(
                ExternalConfiguration(
                    backend=backend,
                    engine_build=claim.engine_build if claim is not None else None,
                    hardware_classes=tuple(claim.hardware_classes)
                    if claim is not None
                    else (),
                    replica_count=math.ceil(workload.concurrent_requests / requests),
                    requests_per_replica=requests,
                    context_pool_tokens=context_pool,
                    gpus_per_node=1 if gpu else 0,
                    memory_pool="vram" if gpu else "ram",
                    estimated_memory_bytes_per_replica=required,
                    minimum_memory_bytes_per_node=total,
                    minimum_weight_disk_bytes_per_node=card.storage_size.in_bytes
                    + projector,
                    host_ram_bytes=None if gpu else total,
                )
            )
    return tuple(configs)


def assess_inference(
    card: ModelCard,
    workload: InferenceWorkload,
    *,
    topology: Topology,
    instances: Mapping[InstanceId, Instance],
    network: Mapping[NodeId, NodeNetworkInfo],
    telemetry: TelemetryView,
    runners: Mapping[RunnerId, RunnerStatus] | None = None,
    excluded_nodes: set[NodeId] | None = None,
    now: datetime | None = None,
) -> InferenceAssessment:
    """Return estimated local fit and complete external alternatives without effects.

    Uses authorized artifact metadata, current reservations and ordinary placer
    compatibility/sharding checks. Concurrency is conservatively represented by
    independent replicas, not unmeasured batching or latency guarantees. Candidate
    hosts exist only inside the dry-run. No membership, state or telemetry changes.
    """
    read_at = now or datetime.now(timezone.utc)
    assumptions = (
        "Observed llama-server slots share one aggregate context pool; other engines use conservative serial replicas.",
        "Context includes input and output tokens for each request.",
        "Weights use this card's selected artifact size, never all repository quantizations.",
        "Runtime overhead, cache and usable-memory margins use the shared estimator.",
        "External configurations host the entire workload; no local/remote hybrid sharding is assumed.",
    )
    uncertainty = (
        "Memory feasibility is estimated; model load and successful serving have not been observed.",
        "No latency, throughput, startup-time, availability or price guarantee is inferred.",
        "Disk figures cover model weights only; staging and engine installation require additional space.",
        "Telemetry receipt time is not a measurement of payload freshness.",
    )
    if card.gguf_cache_geometry is None and card.attention_head_dim is None:
        uncertainty += (
            "Cache geometry uses the declared KV heads and an assumed 128-element head dimension.",
        )
    base = InferenceAssessment(
        model_id=str(card.model_id),
        card_digest=authorized_model_card_digest(card),
        source_repository=str(card.artifact_repository),
        source_revision=card.source_revision,
        selected_file=card.gguf_file,
        quantization=card.quantization,
        workload=workload,
        status="insufficient_evidence",
        read_at=read_at.isoformat(),
        reasons=(),
        assumptions=assumptions,
        uncertainty=uncertainty,
        weight_bytes=card.storage_size.in_bytes,
    )
    incomplete = [
        claim.capability_id
        for claim in card.registry_capability_claims
        if claim.scope == "artifact" and claim.status == "incomplete"
    ]
    if incomplete:
        return base.model_copy(
            update={
                "status": "blocked",
                "reasons": (
                    "Selected artifact has incomplete capability components: "
                    + ", ".join(incomplete),
                ),
            }
        )
    if ModelTask.TextGeneration not in card.tasks:
        return base.model_copy(
            update={
                "status": "blocked",
                "reasons": ("This assessment supports text-generation workloads.",),
            }
        )
    if card.context_length > 0 and workload.context_tokens > card.context_length:
        return base.model_copy(
            update={
                "status": "blocked",
                "reasons": (
                    "Requested context exceeds the artifact's declared context limit.",
                ),
            }
        )
    if (
        card.source_revision is None
        or card.storage_size.in_bytes <= 0
        or per_token_kv_bytes(card) <= 0
    ):
        return base.model_copy(
            update={
                "reasons": (
                    "Planning needs an immutable artifact revision, positive selected-weight size and cache geometry or KV-head metadata.",
                )
            }
        )
    external = _external_configurations(card, workload)
    excluded = excluded_nodes or set()
    # Receipt clocks belong to the local telemetry view. A heartbeat must not
    # refresh a stale memory reading or a stale backend advertisement.
    vram = usable_vram_by_node(
        telemetry.node_system,
        telemetry.node_resources,
        telemetry.node_memory,
        instances,
    )
    stale = {
        node
        for node in topology.list_nodes()
        if node not in excluded
        and any(
            receipt is None
            or not 0 <= (read_at - receipt).total_seconds() <= maximum_age
            for receipt, maximum_age in (
                (telemetry.node_memory_received_at.get(node), 60),
                (telemetry.node_resources_received_at.get(node), 120),
                *(
                    ((telemetry.node_system_received_at.get(node), 60),)
                    if node in vram
                    else ()
                ),
            )
        )
    }
    unified = frozenset(
        unified_memory_gpu_node_ids(
            telemetry.node_system, telemetry.node_resources, telemetry.node_memory
        )
    )
    memory = reserve_system_ram_usage(
        telemetry.node_memory, instances, vram, unified_memory_gpu_nodes=unified
    )
    planned = dict(instances)
    replicas: list[AssessedReplica] = []
    failures: tuple[str, ...] = ()
    pending = False
    remaining = workload.concurrent_requests
    concurrency_unknown = False
    occupied_matching_instance = False
    for instance in instances.values():
        assignments = instance.shard_assignments
        if (
            instance.system_role is not None
            or not assignments.node_to_runner
            or any(
                node in excluded | stale or not topology.contains_node(node)
                for node in assignments.node_to_runner
            )
        ):
            continue
        if any(
            not same_authorized_model_card(shard.model_card, card)
            for shard in assignments.runner_to_shard.values()
        ):
            continue
        if any(
            isinstance((runners or {}).get(runner), RunnerRunning)
            for runner in assignments.runner_to_shard
        ):
            occupied_matching_instance = True
            continue
        if any(
            not isinstance((runners or {}).get(runner), RunnerReady)
            for runner in assignments.runner_to_shard
        ):
            continue
        if any(
            assignments.runner_to_shard[runner].resolved_backend
            not in model_platform_backends(card, telemetry.node_resources[node])
            for node, runner in assignments.node_to_runner.items()
        ):
            continue
        requests = min(
            remaining,
            _replica_slots(instance),
            (instance.context_token_limit or 0) // workload.context_tokens,
        )
        if requests <= 0:
            continue
        concurrency_unknown |= any(
            shard.resolved_backend is None
            or not shard.resolved_backend.startswith("llama_server")
            for shard in assignments.runner_to_shard.values()
        )
        replicas.append(
            AssessedReplica(
                sharding="existing",
                shards=(),
                concurrent_requests=requests,
                context_pool_tokens=workload.context_tokens * requests,
                existing_instance_id=str(instance.instance_id),
            )
        )
        remaining -= requests
        if remaining == 0:
            break
    while remaining > 0:
        instance = None
        requests = remaining
        # A shared pool can fit where N separately loaded copies cannot. Try
        # it first using the observed engine's slot contract, then smaller groups.
        for requests in range(remaining, 0, -1):
            if workload.context_tokens * requests > 1_048_576:
                continue
            instance, failures, pending = _plan_replica(
                card,
                workload,
                topology,
                planned,
                memory,
                network,
                telemetry.node_resources,
                vram,
                unified,
                excluded | stale,
                request_count=requests,
            )
            if instance is not None:
                break
        if instance is None:
            break
        shards: list[AssessedShard] = []
        for node, runner in instance.shard_assignments.node_to_runner.items():
            shard = instance.shard_assignments.runner_to_shard[runner]
            fraction = shard_fraction_of_model(shard)
            footprint = (
                estimate_shard_footprint(
                    card,
                    fraction,
                    workload.context_tokens * requests,
                    resolved_backend=shard.resolved_backend,
                    llama_server_settings=shard.llama_server_settings,
                )
                if fraction is not None
                else None
            )
            gpu = node in vram and backend_offloads_to_vram(shard.resolved_backend)
            usable = (
                vram[node]
                if gpu
                else min(
                    memory[node].ram_available,
                    gpu_working_set_ceiling(memory[node].ram_total),
                )
            )
            shards.append(
                AssessedShard(
                    node_id=str(node),
                    backend=shard.resolved_backend,
                    estimated_memory_bytes=footprint.in_bytes
                    if footprint is not None
                    else None,
                    usable_memory_bytes=usable.in_bytes,
                    memory_pool="vram" if gpu else "ram",
                )
            )
            if footprint is not None:
                # Only new hypothetical replicas are debited. Existing committed
                # allocations were already bounded against live free memory above.
                remainder = Memory.from_bytes(
                    max(0, usable.in_bytes - footprint.in_bytes)
                )
                if gpu:
                    vram[node] = remainder
                if not gpu or node in unified:
                    memory[node] = memory[node].model_copy(
                        update={"ram_available": remainder}
                    )
        first_shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
        concurrency_unknown |= any(
            shard.resolved_backend is None
            or not shard.resolved_backend.startswith("llama_server")
            for shard in instance.shard_assignments.runner_to_shard.values()
        )
        replicas.append(
            AssessedReplica(
                sharding="Tensor"
                if isinstance(first_shard, TensorShardMetadata)
                else "Pipeline",
                shards=tuple(shards),
                concurrent_requests=requests,
                context_pool_tokens=workload.context_tokens * requests,
            )
        )
        planned[instance.instance_id] = instance
        remaining -= requests
    if remaining == 0:
        return base.model_copy(
            update={
                "status": "local",
                "readiness": "loaded_not_workload_verified"
                if replicas
                and all(
                    replica.existing_instance_id is not None for replica in replicas
                )
                else "not_observed",
                "reasons": (
                    "Production admission estimates a local configuration for the requested workload.",
                ),
                "local_replicas": tuple(replicas),
            }
        )
    if stale or pending:
        return base.model_copy(
            update={
                "reasons": (
                    "Local capacity evidence is missing, stale or incomplete; refresh memory, backend and topology observations.",
                )
                + failures[:2],
                "external_configurations": external,
            }
        )
    if occupied_matching_instance:
        return base.model_copy(
            update={
                "reasons": (
                    "This artifact is already serving locally, but its free concurrent slots are unknown. Wait for idle capacity or inspect current request reservations before deciding to burst.",
                ),
                "external_configurations": external,
            }
        )
    if replicas and concurrency_unknown:
        return base.model_copy(
            update={
                "reasons": (
                    "Serial replicas do not fit, but this engine's concurrent capacity is not advertised; external necessity is uncertain.",
                ),
                "external_configurations": external,
            }
        )
    return base.model_copy(
        update={
            "status": "external_required" if external else "blocked",
            "reasons": (
                (
                    "The exact artifact and requested workload have no admitted local configuration."
                    if external
                    else "No supported external engine configuration is evidenced for this artifact."
                ),
            )
            + failures[:2],
            "external_configurations": external,
        }
    )
