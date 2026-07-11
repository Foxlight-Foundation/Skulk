"""NVIDIA GPU telemetry via NVML, normalized at the collector boundary
(:class:`~skulk.shared.types.profiling.AcceleratorMetrics`).

Sibling of the AMD sysfs collector (``linux_gpu.py``): passive reads only,
one normalization point, and the node simply reports no accelerator when
NVML is unavailable. NVML queries are the vendor-supported passive
interface (the same one ``nvidia-smi`` uses); they never collide with
compute work the way the historical macOS IOGPUFamily poller did.

The ``nvidia-ml-py`` binding (imported as ``pynvml``) is intentionally NOT a
Skulk dependency: it is inert without an NVIDIA driver, and CUDA nodes are
the exception, not the rule. The CUDA install recipe
(``deployment/cuda/install-deps.sh``) installs it; the import is guarded
here so every other node pays nothing. All NVML access goes through the
small :class:`NvmlLike` surface so tests inject a fake and never need a GPU.
"""

from __future__ import annotations

from functools import cache
from typing import Protocol, cast, final

from loguru import logger

from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    SystemPerformanceProfile,
)


class NvmlLike(Protocol):
    """The slice of NVML this collector touches (see nvidia-ml-py)."""

    def nvmlInit(self) -> None: ...  # noqa: N802 - NVML's own naming
    def nvmlDeviceGetCount(self) -> int: ...  # noqa: N802
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...  # noqa: N802
    def nvmlDeviceGetName(self, handle: object) -> str | bytes: ...  # noqa: N802
    def nvmlDeviceGetUtilizationRates(self, handle: object) -> object: ...  # noqa: N802
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> object: ...  # noqa: N802
    def nvmlDeviceGetPowerUsage(self, handle: object) -> int: ...  # noqa: N802
    def nvmlDeviceGetTemperature(self, handle: object, sensor: int) -> int: ...  # noqa: N802
    def nvmlDeviceGetClockInfo(self, handle: object, clock: int) -> int: ...  # noqa: N802


#: NVML constants used below (values fixed by the NVML ABI; duplicated so the
#: fake in tests does not need the real module).
_NVML_TEMPERATURE_GPU = 0
_NVML_CLOCK_SM = 1


def _as_nvml(module: object) -> NvmlLike:
    """Treat the stub-less pynvml module as the NvmlLike protocol surface."""
    return cast(NvmlLike, module)


@cache
def load_nvml() -> NvmlLike | None:
    """Import and initialize NVML; ``None`` when absent or driverless.

    Memoized (including the ``None`` case): callers like the worker shard-fit
    guard run per runner spawn, and NVML should initialize at most once per
    process. Installing pynvml later requires a process restart to notice,
    the same contract as any import-time capability.

    Absence is the normal case on every non-NVIDIA node and is logged at
    debug only; an installed-but-failing NVML (driver mismatch) logs a
    warning once so a CUDA operator notices missing telemetry.
    """
    try:
        import pynvml  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.debug("pynvml not installed; no NVIDIA telemetry on this node")
        return None
    nvml = _as_nvml(pynvml)
    try:
        nvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - any NVML failure means "no telemetry"
        logger.warning(f"NVML present but failed to initialize: {exc}")
        return None
    return nvml


def has_nvidia_gpu(nvml: NvmlLike) -> bool:
    """True when at least one NVML device exists."""
    try:
        return nvml.nvmlDeviceGetCount() > 0
    except Exception:  # noqa: BLE001 - treat NVML errors as "no GPU"
        return False


def _name_of(nvml: NvmlLike, handle: object) -> str:
    raw = nvml.nvmlDeviceGetName(handle)
    return raw.decode() if isinstance(raw, bytes) else raw


@final
class _MemoryInfoLike:
    total: int
    used: int


@final
class _UtilizationLike:
    gpu: int


def read_accelerator_metrics(nvml: NvmlLike) -> AcceleratorMetrics:
    """Read normalized metrics from NVML device 0.

    Every field degrades independently to its unmeasured form: one failing
    query (common across driver generations) must not blank the rest.
    Multi-GPU pods report device 0 for now, matching the single-accelerator
    shape of the AMD collector. A failing handle acquisition degrades to a
    vendor-stamped all-unmeasured profile instead of raising, keeping the
    per-field promise at the device level too.
    """
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:  # noqa: BLE001 - device-level degradation
        logger.debug(f"NVML handle acquisition failed: {exc}")
        return AcceleratorMetrics(vendor="nvidia")

    name = "Unknown"
    try:
        name = _name_of(nvml, handle)
    except Exception as exc:  # noqa: BLE001 - per-field degradation
        logger.debug(f"NVML name query failed: {exc}")

    utilization_ratio: float | None = None
    try:
        rates = cast(_UtilizationLike, nvml.nvmlDeviceGetUtilizationRates(handle))
        utilization_ratio = max(0.0, min(1.0, rates.gpu / 100.0))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NVML utilization query failed: {exc}")

    vram_total: int | None = None
    vram_used: int | None = None
    try:
        memory = cast(_MemoryInfoLike, nvml.nvmlDeviceGetMemoryInfo(handle))
        vram_total = int(memory.total)
        vram_used = int(memory.used)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NVML memory query failed: {exc}")

    power_watts: float | None = None
    try:
        power_watts = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NVML power query failed: {exc}")

    temperature_celsius: float | None = None
    try:
        temperature_celsius = float(
            nvml.nvmlDeviceGetTemperature(handle, _NVML_TEMPERATURE_GPU)
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NVML temperature query failed: {exc}")

    clock_mhz: int | None = None
    try:
        clock_mhz = int(nvml.nvmlDeviceGetClockInfo(handle, _NVML_CLOCK_SM))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NVML clock query failed: {exc}")

    return AcceleratorMetrics(
        vendor="nvidia",
        name=name,
        utilization_ratio=utilization_ratio,
        vram_total_bytes=vram_total,
        vram_used_bytes=vram_used,
        power_watts=power_watts,
        temperature_celsius=temperature_celsius,
        clock_mhz=clock_mhz,
    )


def read_system_profile(nvml: NvmlLike) -> SystemPerformanceProfile:
    """Build the node's system profile from NVML device 0.

    Fills BOTH the normalized ``accelerator`` block and the legacy scalar
    fields (``gpu_usage`` PERCENT 0-100, ``temp``, ``sys_power``) from the
    same readings, mirroring the AMD collector, so Mac-shaped readers (the
    topology GPU bar, the power sampler) show real values. CPU fields stay
    zero: the generic system collector owns them.
    """
    accelerator = read_accelerator_metrics(nvml)
    return SystemPerformanceProfile(
        gpu_usage=(accelerator.utilization_ratio * 100)
        if accelerator.utilization_ratio is not None
        else 0.0,
        temp=accelerator.temperature_celsius
        if accelerator.temperature_celsius is not None
        else 0.0,
        sys_power=accelerator.power_watts
        if accelerator.power_watts is not None
        else 0.0,
        accelerator=accelerator,
    )
