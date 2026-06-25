"""AMD/Linux GPU telemetry from passive sysfs reads.

The non-Mac collector for the normalized accelerator expression
(:class:`~skulk.shared.types.profiling.AcceleratorMetrics`). It reads the amdgpu
driver's sysfs nodes under ``/sys/class/drm/card*/device`` rather than polling
the GPU through a vendor library: the reads are passive and out-of-process, so
they cannot collide with in-flight GPU work the way macmon's IOGPUFamily polling
crashed MLX (see ``mactop.py`` / exo#2088). Every value is best-effort; a missing
or unparseable node yields ``None`` rather than a fabricated zero.
"""

from pathlib import Path

from skulk.shared.types.profiling import AcceleratorMetrics, SystemPerformanceProfile
from skulk.utils.pydantic_ext import TaggedModel

_DRM_ROOT = Path("/sys/class/drm")


class LinuxGpuMetrics(TaggedModel):
    """Normalized GPU readings from a Linux node, for the telemetry plane.

    Carries only ``system_profile`` (with its ``accelerator`` block filled);
    node memory comes from the separate psutil memory monitor, so this never
    competes with it for the ``node_memory`` slot.
    """

    system_profile: SystemPerformanceProfile


def _read_int(path: Path) -> int | None:
    """Read a single integer from a sysfs file, or ``None`` if unavailable."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_current_sclk_mhz(device: Path) -> int | None:
    """Parse the active shader clock from ``pp_dpm_sclk`` (the starred line).

    The file lists DPM states like ``0: 600Mhz *`` with the current one starred.
    Returns the starred state's MHz, or ``None`` if the file is absent/unparsed.
    """
    try:
        lines = (device / "pp_dpm_sclk").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if "*" not in line:
            continue
        # e.g. "0: 600Mhz *"
        for token in line.replace(":", " ").split():
            lowered = token.lower()
            if lowered.endswith("mhz"):
                try:
                    return int(lowered[:-3])
                except ValueError:
                    return None
    return None


def find_amd_gpu_device(drm_root: Path = _DRM_ROOT) -> Path | None:
    """Return the amdgpu render device dir (the one exposing utilization).

    Scans ``/sys/class/drm/card*/device`` for the first entry with a
    ``gpu_busy_percent`` node (the connector subdirs like ``card1-DP-1`` do not
    have one). Returns ``None`` on a node with no such device.
    """
    try:
        candidates = sorted(drm_root.glob("card[0-9]*/device"))
    except OSError:
        return None
    for device in candidates:
        if (device / "gpu_busy_percent").is_file():
            return device
    return None


def read_accelerator_metrics(device: Path) -> AcceleratorMetrics:
    """Read normalized :class:`AcceleratorMetrics` from an amdgpu sysfs device.

    Normalizes at this boundary to the shared units: ``gpu_busy_percent`` (0..100)
    becomes a 0..1 ``utilization_ratio``; ``hwmon`` ``power1_average`` microwatts
    become watts; ``temp1_input`` millidegrees become Celsius. Any node that is
    missing yields ``None`` for that field.
    """
    busy = _read_int(device / "gpu_busy_percent")
    power_uw: int | None = None
    temp_mc: int | None = None
    hwmon_root = device / "hwmon"
    if hwmon_root.is_dir():
        hwmons = sorted(hwmon_root.glob("hwmon*"))
        if hwmons:
            power_uw = _read_int(hwmons[0] / "power1_average")
            temp_mc = _read_int(hwmons[0] / "temp1_input")
    return AcceleratorMetrics(
        vendor="amd",
        name="AMD GPU",
        # Clamp to the 0..1 contract defensively; the kernel reports 0..100 so
        # this is belt-and-suspenders against a bad reading rather than letting
        # an out-of-range value reach the planner/dashboard.
        utilization_ratio=(min(max(busy / 100, 0.0), 1.0))
        if busy is not None
        else None,
        vram_total_bytes=_read_int(device / "mem_info_vram_total"),
        vram_used_bytes=_read_int(device / "mem_info_vram_used"),
        # GTT aperture: host RAM the GPU can map beyond the VRAM carve-out. On a
        # Strix Halo APU this is ~the unified pool, which is what lets a model
        # larger than the BIOS VRAM carve-out run on the GPU.
        gtt_total_bytes=_read_int(device / "mem_info_gtt_total"),
        power_watts=(power_uw / 1_000_000) if power_uw is not None else None,
        temperature_celsius=(temp_mc / 1000) if temp_mc is not None else None,
        clock_mhz=_read_current_sclk_mhz(device),
    )


def read_system_profile(device: Path) -> SystemPerformanceProfile:
    """Build the node's system profile from an amdgpu sysfs device.

    Fills BOTH the normalized ``accelerator`` block and the legacy scalar fields
    (``gpu_usage`` percent, ``temp``, ``sys_power``) from the same readings, so
    existing Mac-shaped readers (the topology GPU bar, the power sampler) show an
    AMD node's real values instead of a misleading default 0%/0C/0W.
    """
    acc = read_accelerator_metrics(device)
    return SystemPerformanceProfile(
        gpu_usage=(acc.utilization_ratio * 100)
        if acc.utilization_ratio is not None
        else 0.0,
        temp=acc.temperature_celsius if acc.temperature_celsius is not None else 0.0,
        sys_power=acc.power_watts if acc.power_watts is not None else 0.0,
        accelerator=acc,
    )
