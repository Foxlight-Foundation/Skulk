---
id: telemetry-normalization
title: Collector-Agnostic Telemetry Normalization
sidebar_position: 97
---

<!-- Copyright 2025 Foxlight Foundation -->

# Collector-Agnostic Telemetry Normalization

This is a design record for expressing node metrics the same way regardless of
which collector produced them, so a heterogeneous fleet (Apple Silicon, AMD,
future CUDA) renders and is reasoned about uniformly.

## Problem

Node metrics today are Mac-shaped and Mac-only:

- The normalized carrier `SystemPerformanceProfile`
  (`src/skulk/shared/types/profiling.py`) has `gpu_usage`, `temp`, `sys_power`,
  `pcpu_usage`, `ecpu_usage`. The CPU split is Apple P-core / E-core specific,
  and there is no VRAM, no GPU-distinct power/temperature, no accelerator
  identity.
- Only the Darwin path fills it: `InfoGatherer` runs the `mactop` collector when
  `sys.platform == "darwin"`, else falls back to psutil for memory only
  (`src/skulk/utils/info_gatherer/info_gatherer.py`). An AMD/Linux node (kite4)
  sends memory but **no `system_profile` at all** (no `node_system` entry),
  rather than a profile of zeros: it is a GPU telemetry blind spot.

The collector pattern is already right, though: the active Darwin collector is
`mactop`, which produces the normalized `memory` + `system_profile` shape in its
own `MactopMetrics.from_raw*` constructors (`src/skulk/utils/info_gatherer/mactop.py`);
`TelemetryView.apply()` then just coalesces those already-normalized readings.
(`MacmonMetrics` is not an active collector: it is a decode-only rolling-upgrade
shim with the same normalized shape.) We extend that precedent, normalizing in
the collector, rather than invent a new mechanism.

## Principle

Normalize at the collector boundary; express identically downstream. A collector
(`mactop` on Mac, `rocm-smi`/sysfs on AMD, `nvidia-smi` on CUDA) is the only code
that knows a vendor-specific format. Everything past it (telemetry plane,
planner, dashboard) sees one normalized shape.

## Proposed normalized expression

A new collector-agnostic accelerator block, carried on the existing telemetry
plane. Every field a given collector cannot measure is `None`, never a fake
zero, so the dashboard can distinguish "0%" from "not reported".

```python
class AcceleratorMetrics(CamelCaseModel):
    """One accelerator's live readings, normalized across collectors."""
    vendor: Literal["apple", "amd", "nvidia", "intel", "cpu", "unknown"] = "unknown"
    name: str = "Unknown"                 # "Apple M4", "Radeon 8060S", "RTX 4090"
    utilization_ratio: float | None = None  # 0..1 GPU-busy fraction
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    power_watts: float | None = None        # accelerator/package power draw
    temperature_celsius: float | None = None
    clock_mhz: int | None = None
```

`SystemPerformanceProfile` gains one optional field:

```python
    accelerator: AcceleratorMetrics | None = None
```

Units convention: `utilization_ratio` is a 0..1 fraction, so each collector
divides its native percentage. mactop's `gpu_usage` is already a percentage
(e.g. `8.66`), so the Mac mapping is `utilization_ratio = gpu_usage / 100`;
AMD sysfs `gpu_busy_percent` (0..100) divides likewise. Power is watts and
temperature is degrees Celsius, so sysfs values in microwatts / millidegrees
are scaled at the collector.

The existing Mac-specific scalars (`gpu_usage`, `pcpu_usage`, `ecpu_usage`,
`temp`, `sys_power`) stay for back-compat with the current dashboard and power
sampler; the Mac collector also fills `accelerator` (vendor=`apple`,
`utilization_ratio = gpu_usage / 100`, `power_watts = sys_power`, etc.) so new
readers use the normalized block uniformly. New cross-vendor readers (dashboard GPU card,
any capacity/energy aggregate) read `accelerator` only.

Why a nested block rather than more flat fields: it carries its own vendor/name
identity, it is obviously optional as a unit (a management node or a CPU-only
node reports `None`), and it leaves room to become a list later (multi-GPU nodes)
without another flat-field migration.

## Collector boundary

- **Mac** (`mactop`): existing collector additionally maps its sample into
  `AcceleratorMetrics(vendor="apple", ...)`. No new process.
- **AMD/Linux** (new): an `InfoGatherer` monitor gated on Linux + an available
  source. Order of preference: `amd-smi`/`rocm-smi` if present (gives VRAM,
  power, temp, utilization), else sysfs (`/sys/class/drm/card*/device/`:
  `gpu_busy_percent`, `mem_info_vram_*`, `hwmon` power/temp) so a node with the
  driver but no SMI CLI still reports. Cadence matches the Mac sampler.
  - Hard lesson honored: do **not** add a poller that collides with the GPU the
    way macmon's IOGPUFamily polling crashed MLX (#249). `rocm-smi`/sysfs reads
    are passive and out-of-process; keep them so.
- **CUDA** (future): `nvidia-smi` collector, same normalized output.

## Plane wiring

Rides the telemetry plane unchanged (LWW gossip, #279): the new Linux collector
emits a `GatheredInfo` variant that `TelemetryView.apply()` normalizes into
`node_system` (and `node_memory`) exactly as `MactopMetrics` does today. Add the
variant to `TELEMETRY_PLANE_INFO`. No control-plane or event-log involvement.

## Dashboard

One GPU/accelerator card driven by `node_system[node].accelerator`, rendering the
same fields for every node and showing "not reported" for `None`. Replaces any
Mac-only assumptions in the current node view.

## Sequencing

1. **Schema** (this record): add `AcceleratorMetrics` + the optional field; Mac
   collector fills it; add the new Linux `GatheredInfo` variant to the telemetry
   plane. Wire types only, no behavior change for existing Mac readers.
2. **AMD collector**: implement the Linux monitor (amd-smi/rocm-smi/sysfs);
   validate on kite4 (real VRAM/power/temp/util in the view).
3. **Dashboard**: the uniform accelerator card.
4. **Docs**: architecture + architecture-reference entries; the env/telemetry
   reference notes the new reading.

Each step is a coordinated whole-fleet upgrade (no mixed-version clusters).
