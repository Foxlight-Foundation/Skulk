"""AMD/Linux GPU telemetry from passive sysfs reads.

The non-Mac collector for the normalized accelerator expression
(:class:`~skulk.shared.types.profiling.AcceleratorMetrics`). It reads the amdgpu
driver's sysfs nodes under ``/sys/class/drm/card*/device`` rather than polling
the GPU through a vendor library: the reads are passive and out-of-process, so
they cannot collide with in-flight GPU work the way macmon's IOGPUFamily polling
crashed MLX (see ``mactop.py`` / exo#2088). Every value is best-effort; a missing
or unparseable node yields ``None`` rather than a fabricated zero.
"""

import os
from pathlib import Path

from skulk.shared.types.profiling import AcceleratorMetrics, SystemPerformanceProfile
from skulk.utils.pydantic_ext import TaggedModel

_DRM_ROOT = Path("/sys/class/drm")
_KFD_NODES_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
_TTM_PAGES_LIMIT = Path("/sys/module/ttm/parameters/pages_limit")


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


def _render_minor(device: Path) -> int | None:
    """Return the DRM render-node minor (``renderD<minor>``) of an amdgpu device."""
    try:
        names = sorted(entry.name for entry in (device / "drm").iterdir())
    except OSError:
        return None
    for name in names:
        if name.startswith("renderD"):
            try:
                return int(name.removeprefix("renderD"))
            except ValueError:
                return None
    return None


def _kfd_properties(path: Path) -> dict[str, int]:
    """Parse a KFD ``properties`` file into integers; empty when unreadable."""
    try:
        text = path.read_text()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1], 0)
        except ValueError:
            continue
    return values


def kfd_memory_bank_bytes(device: Path, kfd_root: Path = _KFD_NODES_ROOT) -> int | None:
    """Return the total of the KFD memory banks of the GPU behind ``device``.

    The kernel sizes a discrete GPU's bank at its VRAM (``real_vram_size``, the
    same figure as ``mem_info_vram_total``) and an APU's bank, in its prefer-GTT
    mode, at the TTM page limit (``amdgpu_amdkfd_get_local_mem_info``). The KFD
    node is matched by DRM render minor, which names exactly one device even
    when several identical cards are installed; a chip-ID match could not.

    Args:
        device: The amdgpu sysfs device directory.
        kfd_root: The KFD topology nodes directory (injectable for tests).

    Returns:
        The summed bank size in bytes, or ``None`` when KFD is absent, the
        device has no matching node, or the node reports no bank.
    """
    minor = _render_minor(device)
    if minor is None:
        return None
    try:
        nodes = sorted(kfd_root.glob("[0-9]*"))
    except OSError:
        return None
    for node in nodes:
        properties = _kfd_properties(node / "properties")
        if (
            properties.get("drm_render_minor") != minor
            or properties.get("gfx_target_version", 0) <= 0
        ):
            continue
        try:
            banks = sorted((node / "mem_banks").glob("[0-9]*"))
        except OSError:
            return None
        sizes = [
            size
            for bank in banks
            if (size := _kfd_properties(bank / "properties").get("size_in_bytes"))
            is not None
            and size > 0
        ]
        return sum(sizes) if sizes else None
    return None


def ttm_addressable_bytes(
    pages_limit: Path = _TTM_PAGES_LIMIT, *, page_size: int | None = None
) -> int | None:
    """Return the host memory the GPU driver may map: TTM page limit x page size.

    This is the figure ``amd-ttm`` reports. When nobody sets
    ``ttm.pages_limit``, the kernel stores its computed default (half of RAM) in
    the parameter at load, so the read is meaningful on an untuned system too.

    Args:
        pages_limit: The TTM ``pages_limit`` module parameter (injectable).
        page_size: Kernel page size in bytes; the running system's by default.

    Returns:
        The limit in bytes, or ``None`` when the parameter is unreadable.
    """
    pages = _read_int(pages_limit)
    if pages is None or pages <= 0:
        return None
    size = page_size if page_size is not None else os.sysconf("SC_PAGE_SIZE")
    return pages * size if size > 0 else None


def classify_unified_memory(
    *,
    vram_total: int | None,
    kfd_bank: int | None,
    ttm_addressable: int | None,
    has_uma_interface: bool,
) -> tuple[bool | None, int | None]:
    """Decide whether an amdgpu device is a unified-memory APU, from kernel facts.

    Mirrors the kernel's own rule rather than comparing sizes that depend on
    optional tuning (``amdgpu.gttsize``, ``ttm.pages_limit``, the BIOS carve):

    * KFD bank equal to the VRAM total: the kernel reports dedicated memory, so
      the GPU is discrete (or an APU whose carve exceeds GTT, which the kernel
      also serves from its carve).
    * KFD bank equal to the TTM page limit but not the VRAM total: an APU in the
      kernel's prefer-GTT mode, whose memory is the host RAM it may map.
    * No KFD report at all, but the APU-only ``uma`` sysfs interface exists
      (newer kernels): an APU, sized by the TTM limit.
    * A KFD bank matching neither (a partitioned accelerator, or a TTM limit
      changed at runtime after the driver sized the bank), or no kernel facts:
      unknown, so placement keeps its size heuristic.

    Args:
        vram_total: ``mem_info_vram_total`` in bytes.
        kfd_bank: :func:`kfd_memory_bank_bytes` for the device.
        ttm_addressable: :func:`ttm_addressable_bytes`.
        has_uma_interface: Whether the device exposes the ``uma`` directory.

    Returns:
        ``(unified_memory, addressable_memory_bytes)``; the second item is set
        only for a unified-memory GPU and may still be ``None`` when the TTM
        limit is unreadable.
    """
    if kfd_bank is not None and vram_total is not None:
        if kfd_bank == vram_total:
            return False, None
        if ttm_addressable is not None and kfd_bank == ttm_addressable:
            return True, kfd_bank
        # A KFD report matching neither (a partitioned device, or a TTM limit
        # changed after the driver sized the bank) contradicts both rules, so
        # it stays unknown rather than falling through to the uma heuristic.
        return None, None
    if has_uma_interface:
        return True, ttm_addressable
    return None, None


def read_accelerator_metrics(
    device: Path,
    *,
    kfd_root: Path = _KFD_NODES_ROOT,
    ttm_pages_limit: Path = _TTM_PAGES_LIMIT,
    page_size: int | None = None,
) -> AcceleratorMetrics:
    """Read normalized :class:`AcceleratorMetrics` from an amdgpu sysfs device.

    Normalizes at this boundary to the shared units: ``gpu_busy_percent`` (0..100)
    becomes a 0..1 ``utilization_ratio``; ``hwmon`` ``power1_average`` microwatts
    become watts; ``temp1_input`` millidegrees become Celsius. Any node that is
    missing yields ``None`` for that field. ``unified_memory`` and
    ``addressable_memory_bytes`` come from :func:`classify_unified_memory`.

    Args:
        device: The amdgpu sysfs device directory.
        kfd_root: The KFD topology nodes directory (injectable for tests).
        ttm_pages_limit: The TTM ``pages_limit`` parameter (injectable).
        page_size: Kernel page size override (tests); the system's by default.
    """
    busy = _read_int(device / "gpu_busy_percent")
    vram_total = _read_int(device / "mem_info_vram_total")
    unified_memory, addressable_memory = classify_unified_memory(
        vram_total=vram_total,
        kfd_bank=kfd_memory_bank_bytes(device, kfd_root),
        ttm_addressable=ttm_addressable_bytes(ttm_pages_limit, page_size=page_size),
        has_uma_interface=(device / "uma").is_dir(),
    )
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
        vram_total_bytes=vram_total,
        vram_used_bytes=_read_int(device / "mem_info_vram_used"),
        # GTT aperture: host RAM the GPU can map beyond the VRAM carve-out. On a
        # Strix Halo APU this is ~the unified pool, which is what lets a model
        # larger than the BIOS VRAM carve-out run on the GPU.
        gtt_total_bytes=_read_int(device / "mem_info_gtt_total"),
        unified_memory=unified_memory,
        addressable_memory_bytes=addressable_memory,
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
