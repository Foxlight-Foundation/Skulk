"""Pure complete-placement native admission, not installed/hardware qualification."""

import pytest

from skulk.provisioning.ace_step_inventory import select_ace_step_package
from skulk.shared.models.memory_estimate import estimate_shard_footprint
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.models.native_music_admission import (
    assess_native_music_placement,
    native_music_placement_commitments,
)
from skulk.shared.models.native_music_memory import estimate_native_music_peak
from skulk.shared.models.native_music_runtime import resolve_native_music_policy
from skulk.shared.models.tests.test_native_music_runtime import card_with_layouts, probe
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.native_music import NativeMusicDevice, NativeMusicMemoryCapacity
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata

GIB = 1024**3
NODE = NodeId("node")


def resources(
    device: NativeMusicDevice = "cpu", *, free: int = 64 * GIB
) -> NodeResources:
    """Use valid synthetic current-package observations, never a physical-device run."""
    platforms = (
        {"macosx_14_0_arm64"}
        if device in {"cpu", "mps"}
        else {"manylinux_2_28_x86_64", "linux_x86_64"}
    )
    package = select_ace_step_package(platforms, device=device)
    backend = "ace_step-metal" if device == "mps" else f"ace_step-{device}"
    observed = probe(device).model_copy(
        update={
            "payload_sha256": package.payload_sha256,
            "memory_capacity": NativeMusicMemoryCapacity(
                sampled_at_unix_ms=1000,
                host_total_bytes=64 * GIB,
                host_available_bytes=free,
                device_index=None if device == "cpu" else 0,
                device_memory_kind=None
                if device == "cpu"
                else "unified"
                if device in {"mps", "rocm"}
                else "discrete",
                device_total_bytes=None if device == "cpu" else 64 * GIB,
                device_available_bytes=None if device == "cpu" else free,
                shared_total_bytes=64 * GIB if device in {"mps", "rocm"} else None,
                shared_available_bytes=free if device in {"mps", "rocm"} else None,
            ),
        }
    )
    return NodeResources(
        backends=frozenset({backend}),
        engine_builds={backend: package.build_identity},
        native_music_probes={backend: observed},
    )


def instance(observed: NodeResources, identifier: str = "one") -> MlxRingInstance:
    """Create one complete immutable native placement with its full stamped policy."""
    card = card_with_layouts()
    backend = next(iter(observed.backends))
    policy = resolve_native_music_policy(
        card, backend, observed.native_music_probes[backend]
    )
    runner = RunnerId(identifier + "-runner")
    return MlxRingInstance(
        instance_id=InstanceId(identifier),
        hosts_by_node={},
        ephemeral_port=24000,
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={NODE: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    world_size=1,
                    device_rank=0,
                    start_layer=0,
                    end_layer=card.n_layers,
                    n_layers=card.n_layers,
                    resolved_backend=backend,
                    resolved_engine_build=observed.engine_builds[backend],
                    native_music_policy=policy,
                )
            },
        ),
    )


def ordinary_instance(backend: str) -> MlxRingInstance:
    """Hold a concrete ordinary GGUF shard under the existing footprint contract."""
    card = ModelCard(
        model_id=ModelId("test/ordinary"),
        storage_size=Memory.from_bytes(8 * GIB),
        n_layers=1,
        hidden_size=1000,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model.gguf",
    )
    runner = RunnerId("ordinary-runner")
    return MlxRingInstance(
        instance_id=InstanceId("ordinary"),
        hosts_by_node={},
        ephemeral_port=24001,
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={NODE: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    world_size=1,
                    device_rank=0,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                    resolved_backend=backend,
                )
            },
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "rocm"])
def test_native_assessment_uses_current_probed_math_and_one_pressure_floor(
    device: NativeMusicDevice,
) -> None:
    """The actual selected-device contract drives fit; no qualification record is needed."""
    observed = resources(device)
    backend = next(iter(observed.backends))
    assessed = assess_native_music_placement(
        card_with_layouts(),
        backend,
        NODE,
        observed,
        {},
        now_unix_ms=1000,
    )
    assert assessed.refusal is None and assessed.policy.execution.device == device
    assert assessed.capacity.host_total_bytes == 64 * GIB
    assert assessed.remaining.host_available_bytes == 64 * GIB - 512 * 1024**2
    assert (
        assessed.remaining.host_total_bytes == assessed.remaining.host_available_bytes
    )


def test_idle_native_model_still_holds_its_entire_future_workspace() -> None:
    """An idle loaded model cannot lend its unallocated future budget to another mount."""
    observed = resources(free=50 * GIB)
    current = instance(observed)
    assessed = assess_native_music_placement(
        card_with_layouts(),
        "ace_step-cpu",
        NODE,
        observed,
        {current.instance_id: current},
        now_unix_ms=1000,
    )
    assert assessed.refusal is not None and "host memory" in assessed.refusal
    assert assessed.remaining.host_available_bytes == (
        50 * GIB - 512 * 1024**2 - assessed.estimate.host_peak_bytes
    )


def test_worker_can_exclude_only_its_exact_unchanged_creation_owner() -> None:
    """The same placement is not charged twice, but a stale exclusion cannot release it."""
    observed = resources(free=35 * GIB)
    current = instance(observed)
    placements = {current.instance_id: current}
    assessed = assess_native_music_placement(
        card_with_layouts(),
        "ace_step-cpu",
        NODE,
        observed,
        placements,
        now_unix_ms=1000,
        excluded_instance=current,
    )
    assert assessed.refusal is None
    with pytest.raises(ValueError, match="ownership changed"):
        assess_native_music_placement(
            card_with_layouts(),
            "ace_step-cpu",
            NODE,
            observed,
            placements,
            now_unix_ms=1000,
            excluded_instance=current.model_copy(update={"ephemeral_port": 24001}),
        )


def test_worker_self_exclusion_cannot_authorize_a_partial_restored_owner() -> None:
    """Exact identity alone cannot skip completeness validation at Worker creation."""
    observed = resources()
    current = instance(observed)
    assignments = current.shard_assignments
    runner, shard = next(iter(assignments.runner_to_shard.items()))
    current = current.model_copy(
        update={
            "shard_assignments": assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner: shard.model_copy(
                            update={"n_layers": shard.n_layers + 1}
                        )
                    },
                }
            )
        }
    )
    with pytest.raises(ValueError, match="partial native"):
        assess_native_music_placement(
            card_with_layouts(),
            "ace_step-cpu",
            NODE,
            observed,
            {current.instance_id: current},
            now_unix_ms=1000,
            excluded_instance=current,
        )


def test_cpu_target_accounts_for_shared_gpu_physical_peak() -> None:
    """A CPU-only observation cannot hide another native model's shared GPU storage."""
    metal = resources("mps")
    current = instance(metal)
    shard = next(iter(current.shard_assignments.runner_to_shard.values()))
    assert shard.native_music_policy is not None
    expected = estimate_native_music_peak(shard.model_card, shard.native_music_policy)
    budgets = native_music_placement_commitments(
        NODE,
        metal,
        {current.instance_id: current},
        target_device="cpu",
        node_is_unified=True,
    )
    assert len(budgets) == 1
    assert budgets[0].host_peak_bytes == expected.unified_peak_bytes
    assert budgets[0].device_peak_bytes == 0


def test_old_memory_sample_and_foreign_runtime_build_cannot_admit() -> None:
    """Valid-looking data does not refresh a clock or authorize an unpinned binary."""
    observed = resources()
    assessed = assess_native_music_placement(
        card_with_layouts(),
        "ace_step-cpu",
        NODE,
        observed,
        {},
        now_unix_ms=122000,
    )
    assert assessed.refusal is not None and "stale" in assessed.refusal
    with pytest.raises(ValueError, match="exact verified package"):
        assess_native_music_placement(
            card_with_layouts(),
            "ace_step-cpu",
            NODE,
            observed.model_copy(
                update={"engine_builds": {"ace_step-cpu": "different"}}
            ),
            {},
            now_unix_ms=1000,
        )


def test_incomplete_restored_native_commitment_cannot_expose_free_capacity() -> None:
    """A lost policy fails the new admission instead of silently charging legacy weights."""
    observed = resources()
    current = instance(observed)
    runner, shard = next(iter(current.shard_assignments.runner_to_shard.items()))
    broken = current.model_copy(
        update={
            "shard_assignments": current.shard_assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner: shard.model_copy(update={"native_music_policy": None})
                    }
                }
            )
        }
    )
    with pytest.raises(ValueError, match="unstamped or partial"):
        assess_native_music_placement(
            card_with_layouts(),
            "ace_step-cpu",
            NODE,
            observed,
            {broken.instance_id: broken},
            now_unix_ms=1000,
        )


def test_ordinary_cpu_work_reduces_native_host_admission() -> None:
    """Spare GPU capacity cannot fund a native CPU mount beside committed CPU work."""
    observed = resources(free=36 * GIB)
    assert (
        assess_native_music_placement(
            card_with_layouts(), "ace_step-cpu", NODE, observed, {}, now_unix_ms=1000
        ).refusal
        is None
    )
    ordinary = ordinary_instance("llama_cpp-cpu")
    assessed = assess_native_music_placement(
        card_with_layouts(),
        "ace_step-cpu",
        NODE,
        observed,
        {ordinary.instance_id: ordinary},
        now_unix_ms=1000,
    )
    assert assessed.refusal is not None and "host memory" in assessed.refusal


def test_ordinary_shared_gpu_cost_is_one_physical_reservation() -> None:
    """RAM and GPU views cannot double the same existing shared tensor storage."""
    observed = resources("rocm")
    ordinary = ordinary_instance("llama_cpp-vulkan")
    shard = next(iter(ordinary.shard_assignments.runner_to_shard.values()))
    expected = estimate_shard_footprint(
        shard.model_card,
        1.0,
        resolved_backend=shard.resolved_backend,
        cache_layer_count=1,
    ).in_bytes
    costs = native_music_placement_commitments(
        NODE,
        observed,
        {ordinary.instance_id: ordinary},
        target_device="rocm",
        node_is_unified=True,
    )
    assert len(costs) == 1
    assert costs[0].host_peak_bytes == costs[0].device_peak_bytes == expected
    assert costs[0].shared_peak_bytes == expected


def test_discrete_gpu_commitment_cannot_borrow_spare_host_memory() -> None:
    """A native GPU mount must fit selected device zero after ordinary GPU costs."""
    observed = resources("cuda")
    current_probe = observed.native_music_probes["ace_step-cuda"]
    assert current_probe.memory_capacity is not None
    changed = current_probe.model_copy(
        update={
            "memory_capacity": current_probe.memory_capacity.model_copy(
                update={"device_available_bytes": 20 * GIB}
            )
        }
    )
    observed = observed.model_copy(
        update={"native_music_probes": {"ace_step-cuda": changed}}
    )
    ordinary = ordinary_instance("llama_cpp-cuda")
    assessed = assess_native_music_placement(
        card_with_layouts(),
        "ace_step-cuda",
        NODE,
        observed,
        {ordinary.instance_id: ordinary},
        now_unix_ms=1000,
        node_is_unified=False,
    )
    assert assessed.remaining.host_available_bytes == 64 * GIB - 512 * 1024**2
    assert assessed.refusal is not None and "selected device zero" in assessed.refusal


def test_cpu_payload_cannot_masquerade_as_a_cuda_library() -> None:
    """An inventory digest from another dependency profile supplies no math lane."""
    cpu, cuda = resources(), resources("cuda")
    cpu_probe = cpu.native_music_probes["ace_step-cpu"]
    forged = cuda.native_music_probes["ace_step-cuda"].model_copy(
        update={"payload_sha256": cpu_probe.payload_sha256}
    )
    cuda = cuda.model_copy(
        update={
            "engine_builds": {"ace_step-cuda": cpu.engine_builds["ace_step-cpu"]},
            "native_music_probes": {"ace_step-cuda": forged},
        }
    )
    with pytest.raises(ValueError, match="exact verified package"):
        assess_native_music_placement(
            card_with_layouts(), "ace_step-cuda", NODE, cuda, {}, now_unix_ms=1000
        )
