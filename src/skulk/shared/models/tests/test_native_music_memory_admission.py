"""Admission and unreflected reservations use every actual native memory domain."""

from typing import Literal

import pytest
from pydantic import ValidationError

from skulk.shared.models.native_music_memory import (
    NativeMusicMemoryCapacity,
    NativeMusicMemoryEstimate,
    NativeMusicMemoryReservation,
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
    """Only verified allocations in this exact snapshot avoid a second charge."""
    loaded, pending = estimate(host=20, device=30), estimate(host=10, device=20)
    observed = capacity(host=50, device=50)
    reserved = reserve_native_music_memory(
        observed,
        [
            NativeMusicMemoryReservation(
                estimate=loaded,
                reflected_at_unix_ms=observed.sampled_at_unix_ms,
                reflected_host_bytes=20,
                reflected_device_bytes=30,
            ),
            NativeMusicMemoryReservation(estimate=pending),
        ],
    )
    assert reserved.host_available_bytes == 40
    assert reserved.device_available_bytes == 30
    assert observed.host_available_bytes == observed.device_available_bytes == 50


@pytest.mark.parametrize("domain", ["host", "device", "shared"])
def test_idle_model_retains_its_unallocated_forward_peak(
    domain: Literal["host", "device", "shared"],
) -> None:
    """External use plus resident weights cannot consume a future workspace twice."""
    device: NativeMusicDevice
    if domain == "host":
        observed = capacity(host=60)
        reservation = NativeMusicMemoryReservation(
            estimate=estimate(host=60),
            reflected_at_unix_ms=1,
            reflected_host_bytes=20,
        )
        candidate, device = estimate(host=30), "cpu"
    elif domain == "device":
        observed = capacity(device=60)
        reservation = NativeMusicMemoryReservation(
            estimate=estimate(host=0, device=60),
            reflected_at_unix_ms=1,
            reflected_device_bytes=20,
        )
        candidate, device = estimate(host=0, device=30), "cuda"
    else:
        observed = capacity(device=100, shared=60)
        reservation = NativeMusicMemoryReservation(
            estimate=estimate(host=10, device=50),
            reflected_at_unix_ms=1,
            reflected_host_bytes=5,
            reflected_device_bytes=15,
            reflected_shared_bytes=20,
        )
        candidate, device = estimate(host=0, device=30), "mps"
    reserved = reserve_native_music_memory(observed, [reservation])
    assert getattr(reserved, domain + "_available_bytes") == 20
    assert (
        native_music_memory_error(candidate, reserved, device=device, now_unix_ms=1)
        is not None
    )


@pytest.mark.parametrize("sampled", [0, 2, 3])
def test_unknown_or_other_snapshot_allocations_grant_no_credit(sampled: int) -> None:
    """A loaded transition or a stale allocation receipt cannot credit current free."""
    held = NativeMusicMemoryReservation(
        estimate=estimate(host=30, device=30),
        reflected_at_unix_ms=sampled,
        reflected_host_bytes=20,
        reflected_device_bytes=20,
        reflected_shared_bytes=40,
    )
    reserved = reserve_native_music_memory(
        capacity(host=60, device=60, shared=60), [held]
    )
    assert reserved.host_available_bytes == 30
    assert reserved.device_available_bytes == 30
    assert reserved.shared_available_bytes == 0


def test_unsampled_capacity_cannot_credit_matching_zero_timestamp() -> None:
    """Two unknown timestamps do not establish observed allocations."""
    held = NativeMusicMemoryReservation(
        estimate=estimate(host=30), reflected_host_bytes=30
    )
    observed = capacity(host=60).model_copy(update={"sampled_at_unix_ms": 0})
    assert reserve_native_music_memory(observed, [held]).host_available_bytes == 30


def test_allocation_credit_never_exceeds_the_reserved_domain_peak() -> None:
    """A grown allocation cannot manufacture capacity for another instance."""
    held = NativeMusicMemoryReservation(
        estimate=estimate(host=20, device=30),
        reflected_at_unix_ms=1,
        reflected_host_bytes=200,
        reflected_device_bytes=300,
        reflected_shared_bytes=500,
    )
    reserved = reserve_native_music_memory(
        capacity(host=40, device=40, shared=40), [held]
    )
    assert reserved.host_available_bytes == 40
    assert reserved.device_available_bytes == 40
    assert reserved.shared_available_bytes == 40


def test_shared_credit_uses_unique_physical_storage_and_simultaneous_phase_peak() -> (
    None
):
    """Independent host/device peaks and their credits cannot be added for UMA."""
    projected = estimate(host=60, device=10).model_copy(
        update={"host_forward_bytes": 10, "device_forward_bytes": 60}
    )
    held = NativeMusicMemoryReservation(
        estimate=projected,
        reflected_at_unix_ms=1,
        reflected_host_bytes=20,
        reflected_device_bytes=20,
        reflected_shared_bytes=25,
    )
    reserved = reserve_native_music_memory(
        capacity(host=80, device=80, shared=60), [held]
    )
    assert reserved.shared_available_bytes == 15


def test_stale_free_telemetry_is_bounded_by_all_committed_capacity() -> None:
    """Two pending allocations cannot both spend an unchanged pre-load snapshot."""
    first, second = estimate(host=40, device=30), estimate(host=40, device=30)
    reserved = reserve_native_music_memory(
        capacity(device=100),
        [NativeMusicMemoryReservation(estimate=item) for item in (first, second)],
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
        capacity(device=100, shared=100),
        [NativeMusicMemoryReservation(estimate=pending_gpu)],
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
        capacity(device=100, shared=100),
        [NativeMusicMemoryReservation(estimate=held)],
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
