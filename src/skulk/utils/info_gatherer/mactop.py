from typing import Self

from pydantic import BaseModel

from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    MachMemoryCategories,
    MemoryUsage,
    SystemPerformanceProfile,
    gpu_wireable_memory_bytes,
)
from skulk.utils.pydantic_ext import TaggedModel


class _SocMetrics(BaseModel, extra="ignore"):
    """SoC power/thermal fields from mactop's ``soc_metrics`` block."""

    system_power: float
    gpu_power: float
    gpu_temp: float


class _MemoryMetrics(BaseModel, extra="ignore"):
    """Memory block from mactop.

    ``available`` is ``free + inactive + speculative`` (empirically
    ``total - used``, the same figure macmon derived). That counts reclaimable
    file cache as used, so after a model download "available" is deflated by
    roughly the model's size and placement refuses fits that run comfortably.
    :meth:`MactopMetrics.from_raw` therefore prefers a GPU-wireable figure
    derived from Mach page categories when the caller supplies a snapshot,
    keeping this raw field only as the fallback.
    """

    total: int
    available: int
    swap_total: int
    swap_used: int


class RawMactopMetrics(BaseModel, extra="ignore"):
    """One sample of ``mactop --headless --format json`` (newline-delimited).

    mactop reads Apple's IOReport / SMC counters — NOT the IOGPUFamily
    command-buffer/notification interface that macmon used. macmon's IOGPUFamily
    polling collided with MLX's in-flight Metal command buffers, throwing inside
    the completion-dispatch block (``mlx::core::gpu::check_error``) and either
    aborting (SIGABRT) or hanging the GPU — which on macOS starved WindowServer
    into a watchdog reboot (exo-explore/exo#2088, #1823). mactop's IOReport path
    is safe to sample concurrently with active inference. Unknown fields are
    ignored for forward-compatibility.
    """

    soc_metrics: _SocMetrics
    memory: _MemoryMetrics
    gpu_usage: float
    ecpu_usage: tuple[int, float]  # (freq mhz, usage %)
    pcpu_usage: tuple[int, float]  # (freq mhz, usage %)


class MactopMetrics(TaggedModel):
    """Node hardware metrics sourced from mactop, normalized onto Skulk's
    profiling shapes.

    Replaces the former ``MacmonMetrics``; macmon was removed because its
    IOGPUFamily GPU polling crashed/hung MLX (see ``RawMactopMetrics``).
    """

    system_profile: SystemPerformanceProfile
    memory: MemoryUsage

    @classmethod
    def from_raw(
        cls,
        raw: RawMactopMetrics,
        mach_categories: MachMemoryCategories | None = None,
    ) -> Self:
        """Normalize a raw mactop sample, correcting ``ram_available``.

        When ``mach_categories`` (a vm_stat snapshot taken alongside the
        sample) is provided, ``ram_available`` becomes the GPU-wireable figure
        ``total − wired − anonymous − compressor`` instead of mactop's
        cache-deflated ``available`` — a value-only change, so the gossiped
        ``MemoryUsage`` shape (``extra=forbid`` on the wire) is untouched and
        mixed-version clusters keep interoperating. Without a snapshot (vm_stat
        missing/unparseable) the raw mactop figure is kept.
        """
        ram_available = (
            raw.memory.available
            if mach_categories is None
            else gpu_wireable_memory_bytes(raw.memory.total, mach_categories)
        )
        return cls(
            system_profile=SystemPerformanceProfile(
                gpu_usage=raw.gpu_usage,
                temp=raw.soc_metrics.gpu_temp,
                sys_power=raw.soc_metrics.system_power,
                pcpu_usage=raw.pcpu_usage[1],
                ecpu_usage=raw.ecpu_usage[1],
                # Also fill the collector-agnostic block so cross-vendor readers
                # see Apple nodes uniformly. mactop's gpu_usage is a percentage,
                # so divide to the 0..1 ratio convention. power_watts is the GPU
                # power (gpu_power), matching the AMD collector's GPU-power figure,
                # not whole-SoC system_power. Apple is unified memory with no
                # distinct VRAM pool, so vram_* stay None.
                accelerator=AcceleratorMetrics(
                    vendor="apple",
                    name="Apple GPU",
                    utilization_ratio=min(max(raw.gpu_usage / 100, 0.0), 1.0),
                    power_watts=raw.soc_metrics.gpu_power,
                    temperature_celsius=raw.soc_metrics.gpu_temp,
                ),
            ),
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.total,
                ram_available=ram_available,
                swap_total=raw.memory.swap_total,
                swap_available=(raw.memory.swap_total - raw.memory.swap_used),
            ),
        )

    @classmethod
    def from_raw_json(
        cls,
        json: str,
        mach_categories: MachMemoryCategories | None = None,
    ) -> Self:
        """Parse one mactop JSON line; see :meth:`from_raw` for the
        ``mach_categories`` availability correction."""
        return cls.from_raw(RawMactopMetrics.model_validate_json(json), mach_categories)


class MacmonMetrics(TaggedModel):
    """Read-only decode shim for the former macmon telemetry event.

    macmon no longer runs (its IOGPUFamily polling crashed MLX — see
    ``RawMactopMetrics``), but ``NodeGatheredInfo.info`` is gossiped over the
    wire and replayed from the retained session tail. During a rolling upgrade,
    macOS workers still on the old build keep publishing ``{"MacmonMetrics": …}``
    events; without a matching union member a newly-upgraded master/consumer
    would reject them and silently lose those nodes' memory/system telemetry
    until every node restarts together. The normalized on-wire shape is
    identical to :class:`MactopMetrics` (``system_profile`` + ``memory``), so we
    keep this tag decodable and apply it on the same path. Safe to delete once
    no node can still be running a macmon-era build.
    """

    system_profile: SystemPerformanceProfile
    memory: MemoryUsage
