"""Admission and unreflected reservations use every actual native memory domain."""

import pytest
from pydantic import ValidationError

from skulk.shared.models.native_music_memory import (
    NativeMusicMemoryCapacity,
    NativeMusicMemoryEstimate,
    native_music_memory_error,
    reserve_native_music_memory,
)
from skulk.shared.types.native_music import NativeMusicDevice


def estimate(*, host: int, device: int = 0) -> NativeMusicMemoryEstimate:
    """Use simple independent allocations so domain ownership is visible."""
    return NativeMusicMemoryEstimate(
        projected_tensor_bytes=1,
        host_load_bytes=host,
        host_forward_bytes=host,
        device_load_bytes=device,
        device_forward_bytes=device,
        workspace_bytes=1,
        assumptions=("workspace_allowance_unqualified",),
    )


def capacity(
    *, host: int = 100, device: int | None = None, shared: int | None = None
) -> NativeMusicMemoryCapacity:
    """Supply explicit actual-device readings, never an inferred GPU sum."""
    return NativeMusicMemoryCapacity(
        sampled_at_unix_ms=1,
        host_total_bytes=100,
        host_available_bytes=host,
        device_index=0 if device is not None else None,
        device_memory_kind=("unified" if shared is not None else "discrete")
        if device is not None
        else None,
        device_total_bytes=100 if device is not None else None,
        device_available_bytes=device,
        shared_total_bytes=100 if shared is not None else None,
        shared_available_bytes=shared,
    )


@pytest.mark.parametrize(
    (
        "device",
        "host_free",
        "gpu_free",
        "shared_free",
        "needed_host",
        "needed_gpu",
        "reason",
    ),
    [
        ("cpu", 20, 100, None, 30, 0, "host memory"),
        ("cpu", 30, None, None, 30, 0, None),
        ("cuda", 20, 100, None, 30, 40, "host memory"),
        ("cuda", 100, None, None, 30, 40, "selected device zero"),
        ("cuda", 100, 39, None, 30, 40, "selected device zero"),
        ("cuda", 30, 40, None, 30, 40, None),
        ("mps", 100, 100, 60, 30, 40, "unified physical memory"),
        ("rocm", 100, 100, 70, 30, 40, None),
        ("cpu", 100, 100, 29, 30, 0, "unified physical memory"),
    ],
)
def test_every_required_memory_domain_must_fit(
    device: NativeMusicDevice,
    host_free: int,
    gpu_free: int | None,
    shared_free: int | None,
    needed_host: int,
    needed_gpu: int,
    reason: str | None,
) -> None:
    """Spare VRAM cannot fund host staging; separate free values cannot be summed."""
    failure = native_music_memory_error(
        estimate(host=needed_host, device=needed_gpu),
        capacity(host=host_free, device=gpu_free, shared=shared_free),
        device=device,
        now_unix_ms=1,
    )
    assert (
        failure is None if reason is None else failure is not None and reason in failure
    )


def test_loaded_work_is_not_subtracted_from_observed_free_memory_twice() -> None:
    """A loaded allocation affects the total ceiling, while only pending work nets free."""
    loaded, pending = estimate(host=20, device=30), estimate(host=10, device=20)
    observed = capacity(host=50, device=50)
    reserved = reserve_native_music_memory(
        observed, [loaded, pending], pending=[pending]
    )
    assert reserved.host_available_bytes == 40
    assert reserved.device_available_bytes == 30
    assert observed.host_available_bytes == observed.device_available_bytes == 50


def test_stale_free_telemetry_is_bounded_by_all_committed_capacity() -> None:
    """Two pending allocations cannot both spend an unchanged pre-load snapshot."""
    first, second = estimate(host=40, device=30), estimate(host=40, device=30)
    reserved = reserve_native_music_memory(
        capacity(device=100), [first, second], pending=[first, second]
    )
    assert reserved.host_available_bytes == 20
    assert reserved.device_available_bytes == 40
    assert (
        native_music_memory_error(
            estimate(host=30, device=20), reserved, device="cuda", now_unix_ms=1
        )
        is not None
    )


def test_unified_pending_gpu_work_also_reserves_physical_capacity_for_cpu_work() -> (
    None
):
    """A CPU candidate cannot reuse host pages promised to an unstarted shared GPU."""
    pending_gpu = estimate(host=10, device=60)
    reserved = reserve_native_music_memory(
        capacity(device=100, shared=100), [pending_gpu], pending=[pending_gpu]
    )
    assert reserved.host_available_bytes == 90
    assert reserved.device_available_bytes == 40
    assert reserved.shared_available_bytes == 30
    error = native_music_memory_error(
        estimate(host=40), reserved, device="cpu", now_unix_ms=1
    )
    assert error is not None and "unified physical memory" in error


def test_overcommit_clamps_all_domains_to_zero_without_mutating_observation() -> None:
    """Restored reservations beyond capacity produce a clear refusal, not negatives."""
    held = estimate(host=150, device=150)
    reserved = reserve_native_music_memory(
        capacity(device=100, shared=100), [held], pending=[held]
    )
    assert (
        reserved.host_available_bytes
        == reserved.device_available_bytes
        == reserved.shared_available_bytes
        == 0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_and_gpu_estimates_cannot_be_interchanged(
    device: NativeMusicDevice,
) -> None:
    """A changed compute policy must obtain its matching estimate before admission."""
    projected = estimate(host=10, device=10 if device == "cpu" else 0)
    error = native_music_memory_error(
        projected, capacity(device=100), device=device, now_unix_ms=1
    )
    assert error is not None and "execution estimate" in error


def test_unknown_topology_does_not_silently_become_discrete_capacity() -> None:
    """Missing UMA facts cannot grant two independent copies of the same RAM."""
    observed = capacity(device=100).model_copy(update={"device_memory_kind": None})
    error = native_music_memory_error(
        estimate(host=30, device=40), observed, device="cuda", now_unix_ms=1
    )
    assert error is not None and "memory topology" in error


def test_metal_cannot_claim_a_discrete_only_pool() -> None:
    """Apple GPU and host allocations must satisfy a single physical budget too."""
    error = native_music_memory_error(
        estimate(host=30, device=40), capacity(device=100), device="mps", now_unix_ms=1
    )
    assert error is not None and "unified physical-memory" in error


@pytest.mark.parametrize(
    ("sampled", "now", "allowed"),
    [
        (0, 1, False),
        (1, 0, False),
        (1000, 121000, True),
        (1000, 121001, False),
        (31000, 1000, True),
        (31001, 1000, False),
    ],
)
def test_preparation_memory_requires_a_recent_actual_observation(
    sampled: int,
    now: int,
    allowed: bool,
) -> None:
    """A ready library is not permanently ready memory or a fresh sample after restart."""
    observed = capacity().model_copy(update={"sampled_at_unix_ms": sampled})
    error = native_music_memory_error(
        estimate(host=10), observed, device="cpu", now_unix_ms=now
    )
    assert (error is None) is allowed


@pytest.mark.parametrize(
    "fault",
    ["host", "ordinal", "device_total", "device_free", "shared_total", "shared_free"],
)
def test_incomplete_or_impossible_observations_are_not_capacity(fault: str) -> None:
    """No missing-device inference, foreign ordinal or over-total free reading."""
    body = capacity(device=100, shared=100).model_dump(mode="json")
    changes = {
        "host": {"host_available_bytes": 101},
        "ordinal": {"device_index": 1},
        "device_total": {"device_total_bytes": None},
        "device_free": {"device_available_bytes": 101},
        "shared_total": {"shared_total_bytes": None},
        "shared_free": {"shared_available_bytes": 101},
    }
    body.update(changes[fault])
    with pytest.raises(ValidationError):
        NativeMusicMemoryCapacity.model_validate(body)
