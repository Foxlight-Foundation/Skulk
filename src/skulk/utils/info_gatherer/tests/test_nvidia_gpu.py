"""Unit tests for the NVML telemetry collector. No GPU, no pynvml needed."""

from __future__ import annotations

from typing import final

import pytest

from skulk.utils.info_gatherer.nvidia_gpu import (
    has_nvidia_gpu,
    read_accelerator_metrics,
    read_system_profile,
)


@final
class _Memory:
    total = 24 * 1024**3
    used = 7 * 1024**3


@final
class _Rates:
    gpu = 62


@final
class _FakeNvml:
    """Scriptable NvmlLike: per-query failures via the `broken` set."""

    def __init__(
        self,
        broken: set[str] | None = None,
        name: str | bytes = "NVIDIA GeForce RTX 4090",
        cc: tuple[int, int] = (8, 9),  # RTX 4090 is Ada sm89
    ) -> None:
        self.broken = broken or set()
        self._name = name
        self._cc = cc

    def _maybe_break(self, query: str) -> None:
        if query in self.broken:
            raise RuntimeError(f"{query} unsupported on this driver")

    def nvmlInit(self) -> None:  # noqa: N802
        return None

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802
        self._maybe_break("count")
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:  # noqa: N802
        self._maybe_break("handle")
        return object()

    def nvmlDeviceGetName(self, handle: object) -> str | bytes:  # noqa: N802
        self._maybe_break("name")
        return self._name

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> object:  # noqa: N802
        self._maybe_break("utilization")
        return _Rates()

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> object:  # noqa: N802
        self._maybe_break("memory")
        return _Memory()

    def nvmlDeviceGetPowerUsage(self, handle: object) -> int:  # noqa: N802
        self._maybe_break("power")
        return 285_000  # milliwatts

    def nvmlDeviceGetTemperature(self, handle: object, sensor: int) -> int:  # noqa: N802
        self._maybe_break("temperature")
        return 63

    def nvmlDeviceGetClockInfo(self, handle: object, clock: int) -> int:  # noqa: N802
        self._maybe_break("clock")
        return 2520

    def nvmlDeviceGetCudaComputeCapability(  # noqa: N802
        self, handle: object
    ) -> tuple[int, int]:
        self._maybe_break("compute_capability")
        return self._cc


def test_full_metrics_normalize() -> None:
    metrics = read_accelerator_metrics(_FakeNvml())
    assert metrics.vendor == "nvidia"
    assert metrics.name == "NVIDIA GeForce RTX 4090"
    assert metrics.utilization_ratio == 0.62
    assert metrics.vram_total_bytes == 24 * 1024**3
    assert metrics.vram_used_bytes == 7 * 1024**3
    assert metrics.power_watts == 285.0
    assert metrics.temperature_celsius == 63.0
    assert metrics.clock_mhz == 2520
    # RTX 4090 (Ada sm89): FP8 native, FP4 not.
    assert metrics.compute_capability == "8.9"
    assert metrics.native_fp8 is True
    assert metrics.native_fp4 is False


@pytest.mark.parametrize(
    ("cc", "capability", "fp4", "fp8"),
    [
        ((8, 0), "8.0", False, False),  # A100 Ampere: no native FP4 or FP8
        ((8, 9), "8.9", False, True),  # Ada: FP8
        ((9, 0), "9.0", False, True),  # H100 Hopper: FP8
        ((10, 0), "10.0", True, True),  # B100/B200 Blackwell: FP4
        ((12, 0), "12.0", True, True),  # RTX 50 Blackwell: FP4
    ],
)
def test_compute_capability_derives_native_formats(
    cc: tuple[int, int], capability: str, fp4: bool, fp8: bool
) -> None:
    metrics = read_accelerator_metrics(_FakeNvml(cc=cc))
    assert metrics.compute_capability == capability
    assert metrics.native_fp4 is fp4
    assert metrics.native_fp8 is fp8


def test_compute_capability_degrades_to_none() -> None:
    metrics = read_accelerator_metrics(_FakeNvml(broken={"compute_capability"}))
    assert metrics.compute_capability is None
    assert metrics.native_fp4 is None
    assert metrics.native_fp8 is None
    # ...while the rest still report.
    assert metrics.clock_mhz == 2520


def test_bytes_name_decodes() -> None:
    metrics = read_accelerator_metrics(_FakeNvml(name=b"NVIDIA A5000"))
    assert metrics.name == "NVIDIA A5000"


def test_per_field_degradation_never_blanks_the_rest() -> None:
    metrics = read_accelerator_metrics(
        _FakeNvml(broken={"power", "temperature", "clock"})
    )
    # Failing queries degrade to None...
    assert metrics.power_watts is None
    assert metrics.temperature_celsius is None
    assert metrics.clock_mhz is None
    # ...while the rest still report.
    assert metrics.vendor == "nvidia"
    assert metrics.utilization_ratio == 0.62
    assert metrics.vram_total_bytes == 24 * 1024**3


def test_all_queries_failing_still_yields_a_vendor_stamped_profile() -> None:
    metrics = read_accelerator_metrics(
        _FakeNvml(broken={"name", "utilization", "memory", "power", "temperature", "clock"})
    )
    assert metrics.vendor == "nvidia"
    assert metrics.name == "Unknown"
    assert metrics.utilization_ratio is None
    assert metrics.vram_total_bytes is None


def test_system_profile_fills_legacy_scalars_in_codebase_units() -> None:
    profile = read_system_profile(_FakeNvml())
    assert profile.accelerator is not None
    # gpu_usage is a 0-100 PERCENT across the codebase (mactop, linux_gpu).
    assert profile.gpu_usage == 62.0
    assert profile.temp == 63.0
    assert profile.sys_power == 285.0
    degraded = read_system_profile(
        _FakeNvml(broken={"utilization", "temperature", "power"})
    )
    assert degraded.gpu_usage == 0.0
    assert degraded.temp == 0.0
    assert degraded.sys_power == 0.0


def test_handle_failure_degrades_to_vendor_stamped_profile() -> None:
    metrics = read_accelerator_metrics(_FakeNvml(broken={"handle"}))
    assert metrics.vendor == "nvidia"
    assert metrics.name == "Unknown"
    assert metrics.vram_total_bytes is None


def test_has_nvidia_gpu() -> None:
    assert has_nvidia_gpu(_FakeNvml()) is True
    assert has_nvidia_gpu(_FakeNvml(broken={"count"})) is False
