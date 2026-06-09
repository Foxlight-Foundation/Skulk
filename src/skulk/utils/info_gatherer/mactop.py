from typing import Self

from pydantic import BaseModel

from skulk.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from skulk.utils.pydantic_ext import TaggedModel


class _SocMetrics(BaseModel, extra="ignore"):
    """SoC power/thermal fields from mactop's ``soc_metrics`` block."""

    system_power: float
    gpu_temp: float


class _MemoryMetrics(BaseModel, extra="ignore"):
    """Memory block from mactop.

    ``available`` is reported directly; empirically it equals ``total - used``
    (the same figure macmon derived), so placement fit margins are unchanged by
    the swap.
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
    def from_raw(cls, raw: RawMactopMetrics) -> Self:
        return cls(
            system_profile=SystemPerformanceProfile(
                gpu_usage=raw.gpu_usage,
                temp=raw.soc_metrics.gpu_temp,
                sys_power=raw.soc_metrics.system_power,
                pcpu_usage=raw.pcpu_usage[1],
                ecpu_usage=raw.ecpu_usage[1],
            ),
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.total,
                ram_available=raw.memory.available,
                swap_total=raw.memory.swap_total,
                swap_available=(raw.memory.swap_total - raw.memory.swap_used),
            ),
        )

    @classmethod
    def from_raw_json(cls, json: str) -> Self:
        return cls.from_raw(RawMactopMetrics.model_validate_json(json))
