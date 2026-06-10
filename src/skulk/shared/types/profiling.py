import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self, final

import psutil
from pydantic import BaseModel

from skulk.shared.types.memory import Memory
from skulk.shared.types.thunderbolt import ThunderboltIdentifier
from skulk.utils.pydantic_ext import CamelCaseModel


class MemoryUsage(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )


@final
class MachMemoryCategories(BaseModel, frozen=True, strict=True):
    """One consistent snapshot of macOS Mach page-category counters.

    Sourced from ``vm_stat`` (one ``host_statistics64`` snapshot), which is the
    only stock interface exposing all three of wired, anonymous, and compressor
    occupancy together — psutil lacks compressor occupancy and the ``vm.*``
    sysctls lack wired/compressor, and mixing sources tears the snapshot.
    """

    wired_bytes: int
    """Unpageable memory (kernel + GPU-wired). Never reclaimable."""

    anonymous_bytes: int
    """Resident anonymous (non-file-backed) pages — process heaps. Disjoint
    from wired in Mach accounting; reclaimable only via compression/swap."""

    compressor_bytes: int
    """Physical pages holding compressed memory. Resident until decompressed
    or swapped; counting them avoids overstating availability on a box
    already under memory pressure."""


_VM_STAT_PAGE_SIZE_PATTERN = re.compile(r"page size of (\d+) bytes")
_VM_STAT_COUNTER_PATTERN = re.compile(r"^(.+?):\s+(\d+)\.?\s*$", re.MULTILINE)


def parse_vm_stat_output(text: str) -> MachMemoryCategories | None:
    """Parse ``vm_stat`` output into a :class:`MachMemoryCategories` snapshot.

    Pure function (the subprocess lives at the caller). Returns ``None`` when
    the expected header or counters are missing, so a changed/foreign format
    degrades to "no snapshot" rather than a wrong number.
    """
    page_size_match = _VM_STAT_PAGE_SIZE_PATTERN.search(text)
    if page_size_match is None:
        return None
    page_size = int(page_size_match.group(1))
    counters = {
        match.group(1).strip(): int(match.group(2))
        for match in _VM_STAT_COUNTER_PATTERN.finditer(text)
    }
    try:
        return MachMemoryCategories(
            wired_bytes=counters["Pages wired down"] * page_size,
            anonymous_bytes=counters["Anonymous pages"] * page_size,
            compressor_bytes=counters["Pages occupied by compressor"] * page_size,
        )
    except KeyError:
        return None


def gpu_wireable_memory_bytes(
    ram_total_bytes: int, categories: MachMemoryCategories
) -> int:
    """Memory the GPU could wire without fighting resident working sets.

    ``total − wired − anonymous − compressor``: everything else (free,
    file-backed cache, purgeable) is reclaimed by macOS the moment Metal wires
    pages. The naive ``free + inactive + speculative`` figure that mactop (and
    macmon before it) reports counts reclaimable file cache as *used* — after a
    model download, ~weights-sized cache deflates it by the model's full size
    and placement refuses fits that run comfortably (observed on a 24 GB node:
    11.6 GB of just-downloaded weights in cache dropped "available" to 12 GB
    while 14.6 GB was genuinely wireable). Deliberately does NOT credit
    compression of idle anonymous memory — that would re-introduce the
    oversized-placement OOM class that the 1.30 overhead factor guards against.
    """
    return max(
        0,
        ram_total_bytes
        - categories.wired_bytes
        - categories.anonymous_bytes
        - categories.compressor_bytes,
    )


def read_wired_memory_bytes() -> int | None:
    """OS-level wired (unpageable) memory in use, or None where unavailable.

    Kept OFF the gossiped ``MemoryUsage`` (which rides ``NodeGatheredInfo``
    events under ``extra=forbid`` — adding a field there breaks old nodes in
    a mixed-version rollout). Read locally on the diagnostics path only, where
    it powers leaked-wired detection (Skulk#239). psutil exposes ``wired`` on
    macOS only.
    """
    wired: object = getattr(psutil.virtual_memory(), "wired", None)
    return int(wired) if isinstance(wired, (int, float)) else None


class DiskUsage(CamelCaseModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(CamelCaseModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"
    skulk_version: str = "Unknown"
    skulk_commit: str = "Unknown"


class NodeNetworkInfo(CamelCaseModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(CamelCaseModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(CamelCaseModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool
    interfaces_present: bool = True


class ThunderboltBridgeStatus(CamelCaseModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
