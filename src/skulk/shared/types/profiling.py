import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Self, cast, final

import psutil
from pydantic import BaseModel, field_serializer, field_validator

from skulk.shared.backends import probe_node_backends
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

    @classmethod
    def from_local_gpu_wireable(cls) -> Self:
        """Local snapshot with ``ram_available`` as the GPU-wireable figure.

        ``total − wired − anonymous − compressor`` from a vm_stat snapshot —
        the same metric the telemetry path gossips for placement admission, so
        the master's check and the worker's local pre-spawn guard agree on
        what "available" means. psutil's ``available`` (free + inactive)
        counts reclaimable file cache as used and is kept only as the fallback
        when vm_stat fails.
        """
        categories = read_mach_memory_categories()
        return cls.from_psutil(
            override_memory=None
            if categories is None
            else gpu_wireable_memory_bytes(
                int(psutil.virtual_memory().total), categories
            )
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


def read_mach_memory_categories() -> MachMemoryCategories | None:
    """One synchronous ``vm_stat`` snapshot, or ``None`` on any failure.

    For rare, latency-tolerant call sites (the worker's pre-spawn fit guard);
    the telemetry loop has its own anyio-based reader so the 1 Hz sample
    cadence never blocks the event loop.
    """
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return parse_vm_stat_output(result.stdout.decode("utf-8", errors="replace"))


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


AcceleratorVendor = Literal["apple", "amd", "nvidia", "intel", "cpu", "unknown"]


class AcceleratorMetrics(CamelCaseModel):
    """One accelerator's live readings, normalized across collectors.

    The collector-agnostic GPU/accelerator expression: any platform's collector
    (mactop on Apple Silicon, rocm-smi/sysfs on AMD, nvidia-smi on CUDA) fills
    the same shape, so the planner and dashboard reason about a heterogeneous
    fleet uniformly. A field a given collector cannot measure stays ``None``
    (never a fake zero), so a reader can tell "0%" apart from "not reported".
    Units are fixed here so collectors normalize at their boundary:
    ``utilization_ratio`` is a 0..1 fraction, power is watts, temperature is
    degrees Celsius.
    """

    vendor: AcceleratorVendor = "unknown"
    name: str = "Unknown"
    utilization_ratio: float | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    power_watts: float | None = None
    temperature_celsius: float | None = None
    clock_mhz: int | None = None


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    # Collector-agnostic accelerator readings (None when unreported, e.g. a
    # management or CPU-only node, or a collector that cannot measure them).
    # The scalars above stay for back-compat with existing Mac-only readers;
    # cross-vendor readers use this block.
    accelerator: AcceleratorMetrics | None = None


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


NodeParticipation = Literal["full", "management", "ffn_only"]
"""How deeply a node participates in inference (Axis 1 of the heterogeneous-
participation model, #149/#286):

- ``full``: attention + FFN; an ordinary inference rank (today's default).
- ``management``: control plane only; sees the whole cluster and serves the
  API/dashboard, but the planner never assigns it an inference shard. The
  declared form of the ``excluded_nodes`` workaround (e.g. a remote node on a
  high-latency link).
- ``ffn_only``: reserved for LARQL slice placement (FFN/expert but not
  attention); not yet honored by the planner.
"""


class NodeResources(CamelCaseModel):
    """Inference-relevant capability and policy a node advertises to the planner.

    Mixes probed capability (``backends``) with operator-declared policy
    (``participation``); both ride the same node-info gossip path. The planner
    reads this to hard-filter placement candidates. Defaults describe a normal
    Apple-Silicon full-participation node so pre-upgrade gossip and missing
    entries stay non-breaking.
    """

    backends: frozenset[str] = frozenset({"mlx"})
    participation: NodeParticipation = "full"

    @field_validator("backends", mode="before")
    @classmethod
    def _coerce_backends(cls, v: object) -> object:
        # Strict mode rejects a list where a frozenset is declared, but the
        # wire path (model_dump(mode="json") -> array -> model_validate) and
        # any list-shaped input arrive as a list. Coerce iterables to a
        # frozenset before strict validation so node_resources actually
        # populates over gossip (without this the feature is inert).
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(cast("Iterable[str]", v))
        return v

    @field_serializer("backends")
    def _serialize_backends(self, value: frozenset[str]) -> list[str]:
        # Emit a sorted list in both json and python dump modes so JSON wire
        # encoding and TOML serialization (tomlkit cannot encode a frozenset)
        # both succeed and round-trip deterministically.
        return sorted(value)

    @classmethod
    async def gather(cls) -> "NodeResources":
        """Probe backends and read the declared participation role at startup."""
        backends = probe_node_backends()
        declared = os.environ.get("SKULK_NODE_PARTICIPATION", "full").strip().lower()
        participation: NodeParticipation = (
            declared if declared in ("full", "management", "ffn_only") else "full"
        )
        return cls(backends=backends, participation=participation)


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
