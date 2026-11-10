"""Actual-pressure policy differs from re-admitting an already resident peak."""

from typing import Literal

import pytest
from pydantic import TypeAdapter, ValidationError

from skulk.shared.types.native_music import (
    NativeMusicDevice,
    NativeMusicFrame,
    NativeMusicMemoryBudget,
    NativeMusicMemoryCapacity,
    NativeMusicMemoryFrame,
    NativeMusicMemoryObservation,
    native_music_admission_capacity,
    native_music_live_memory_error,
)


def observation(device: NativeMusicDevice = "cpu") -> NativeMusicMemoryObservation:
    """Use distinct physical domains and an explicitly sampled own process."""
    capacity = NativeMusicMemoryCapacity(
        sampled_at_unix_ms=1000,
        host_total_bytes=1000,
        host_available_bytes=500,
        device_index=0 if device != "cpu" else None,
        device_memory_kind="unified"
        if device == "mps"
        else "discrete"
        if device != "cpu"
        else None,
        device_total_bytes=1000 if device != "cpu" else None,
        device_available_bytes=500 if device != "cpu" else None,
        shared_total_bytes=1000 if device == "mps" else None,
        shared_available_bytes=500 if device == "mps" else None,
    )
    return NativeMusicMemoryObservation(
        process_id=123,
        payload_sha256="a" * 64,
        device=device,
        capacity=capacity,
        host_resident_bytes=200,
        device_allocated_bytes=200 if device != "cpu" else 0,
    )


def budget(device: NativeMusicDevice = "cpu") -> NativeMusicMemoryBudget:
    """Retain simultaneous physical peaks independently of observed RSS aliases."""
    return NativeMusicMemoryBudget(
        host_peak_bytes=300,
        device_peak_bytes=400 if device != "cpu" else 0,
        shared_peak_bytes=600 if device != "cpu" else 300,
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "rocm"])
def test_same_live_policy_accepts_each_correct_actual_device(
    device: NativeMusicDevice,
) -> None:
    """No architecture permission list is inferred from qualification history."""
    assert (
        native_music_live_memory_error(
            budget(device), observation(device), now_unix_ms=1000
        )
        is None
    )


@pytest.mark.parametrize(
    ("now", "allowed"), [(6000, True), (6001, False), (0, False), (1000, True)]
)
def test_stopped_or_unsampled_observation_cannot_remain_a_live_guard(
    now: int, allowed: bool
) -> None:
    """The same process must continue producing recent actually collected facts."""
    assert (
        native_music_live_memory_error(budget(), observation(), now_unix_ms=now) is None
    ) is allowed


@pytest.mark.parametrize("domain", ["host", "device", "shared"])
def test_physical_pressure_refuses_even_when_current_allocations_fit(
    domain: Literal["host", "device", "shared"],
) -> None:
    """Other processes may exhaust a pool without changing this model's RSS."""
    actual = observation("mps")
    actual = actual.model_copy(
        update={
            "capacity": actual.capacity.model_copy(
                update={domain + "_available_bytes": 49}
            )
        }
    )
    error = native_music_live_memory_error(budget("mps"), actual, now_unix_ms=1000)
    assert error is not None and "pressure" in error


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_process_ceiling_remains_the_admitted_host_or_shared_phase(
    device: NativeMusicDevice,
) -> None:
    """Unified RSS may alias device storage but cannot grow beyond the shared peak."""
    actual = observation(device).model_copy(
        update={"host_resident_bytes": 601 if device == "mps" else 301}
    )
    assert (
        native_music_live_memory_error(budget(device), actual, now_unix_ms=1000)
        is not None
    )


def test_gpu_allocator_ceiling_cannot_borrow_spare_host_ram() -> None:
    """The actual selected device remains bounded even on a unified-memory host."""
    actual = observation("mps").model_copy(update={"device_allocated_bytes": 401})
    error = native_music_live_memory_error(budget("mps"), actual, now_unix_ms=1000)
    assert error is not None and "device allocation" in error
    assert "allocated=401 bytes" in error and "budget=400 bytes" in error


def test_rss_aliases_are_not_added_or_used_as_free_memory_credits() -> None:
    """RSS plus driver allocation is not a measurement of unique physical pages."""
    actual = observation("mps").model_copy(
        update={"host_resident_bytes": 500, "device_allocated_bytes": 350}
    )
    assert (
        native_music_live_memory_error(budget("mps"), actual, now_unix_ms=1000) is None
    )
    normalized = native_music_admission_capacity(actual.capacity)
    assert (
        normalized.host_available_bytes
        == normalized.shared_available_bytes
        == normalized.device_available_bytes
        == 450
    )
    assert actual.capacity.host_available_bytes == 500


def test_admission_reserves_the_same_live_pressure_floors_once() -> None:
    """Pending reservations spend the reduced physical ceilings, not OS headroom."""
    actual = observation("mps").capacity
    normalized = native_music_admission_capacity(actual)
    assert (
        normalized.host_total_bytes
        == normalized.device_total_bytes
        == normalized.shared_total_bytes
        == 950
    )
    assert (
        NativeMusicMemoryCapacity.model_validate_json(normalized.model_dump_json())
        == normalized
    )


def test_memory_frame_roundtrips_without_becoming_a_generation_terminal() -> None:
    """The bounded private control union has a distinct live-fact discriminator."""
    frame = NativeMusicMemoryFrame(sequence=1, observation=observation())
    assert TypeAdapter(NativeMusicFrame).validate_json(frame.model_dump_json()) == frame
    with pytest.raises(ValidationError):
        NativeMusicMemoryObservation.model_validate(
            observation().model_dump() | {"device_allocated_bytes": 1}
        )


def test_contradictory_simultaneous_budget_is_not_a_process_limit() -> None:
    """Independent host/device peaks cannot invent an impossible shared phase."""
    with pytest.raises(ValidationError):
        NativeMusicMemoryBudget(
            host_peak_bytes=300, device_peak_bytes=400, shared_peak_bytes=800
        )
