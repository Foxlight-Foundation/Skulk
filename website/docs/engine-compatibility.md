---
id: engine-compatibility
title: Model and Engine Compatibility
sidebar_position: 5
---

<!-- Copyright 2026 Foxlight Foundation -->

**Compatible untested hardware remains eligible. We test to ensure, not to exclude.**

A model card identifies an exact artifact, its components and intrinsic behavior,
and compatible engine runtimes. Engine requirements and live resources determine
where a workload can run. Qualification records supply evidence about tested
configurations; they do not create a list of permitted GPU models.

## Compatibility and evidence

Engine packages own actual operating-system/CPU ABI requirements, helper
instructions, compiled accelerator targets, driver/runtime requirements and
implemented model operations. An engine can be available through installation
on first use. Node facts and runtime probes establish the machine's actual
capabilities; a missing test result does not establish incompatibility.

Qualification preserves the exact tested artifact, build, hardware, operation,
workload and outcome. Missing, stale, differently scoped or failed qualification
does not revoke independent compatibility. A failure can reveal a technical defect
that needs correction, but its scope must be explained. An editing test result
alone cannot exclude an otherwise compatible generation operation.

Accurate, complete and structurally valid cards can publish independently of
physical qualification. Artifact integrity, repository-code authorization and
genuinely missing executable components remain real requirements. Measurements
may calibrate estimates and inform confidence or preferences without becoming
hardware permission lists.

## Memory belongs to a serving configuration

A card supplies model facts, not a universal required-memory number. Memory fit
uses three inputs:

| Input | Responsibility |
| --- | --- |
| Exact artifacts and model geometry | Card: selected weight sizes, components, attention dimensions/windows and recurrent state |
| Effective engine policy | Runtime: resident weight representation, actual cache precision/compression/allocation, speculation, supported offloading and sharding |
| Workload and live capacity | Requested context/concurrency or media size; current usable RAM/VRAM and reservations |

Stored weight quantization and cache quantization are independent. Q4 weights do
not imply a Q4 cache. Artifact byte size contributes the selected stored weight
cost, but a runtime can cast/expand weights or allocate temporary copies. Cache
accounting must include full-precision layers or residual buffers, quantization
metadata, allocation blocks and retained prefixes. Reuse avoids computation but
does not make retained cache free. Budget peak load/prefill/generation workspace
and account for allocations in their actual CPU/GPU or shared memory pools.

The placer, reservations, context sizing, assessments and worker load guard must
use the same resolved actual engine and cache settings. Savings apply only when
the selected runner actually enables the optimization for this model. Bind those
effective settings to the placement and reassess a changed policy before load;
an estimator must not assume compressed cache while the runner allocates FP16.

A refusal must describe the assessed configuration, assumptions and uncertainty.
Feasible alternatives can change context, concurrency, cache precision or offload
within the user's permitted constraints. Do not silently change those choices or
turn a conservative estimate into permanent model/hardware incompatibility. A
published reference memory figure identifies its engine, settings and workload.

## Shared estimator and uncertainty

`EngineMemorySettings` resolves the actual inherited runtime controls. Nodes
advertise those controls in `NodeResources.engine_memory_settings`; placements
capture `ShardMetadata.memory_settings`. The worker checks for settings drift
before load, and the MLX runner verifies its own imported settings before setup.
Placement, reservations, context sizing and preview estimates share
`src/skulk/shared/models/memory_estimate.py`.

Known standard MLX attention layouts account for affine packed storage and group
metadata, or TurboQuant byte codes and FP32 norms. Compressed-cache estimates also
include allocation blocks, temporary full-precision workspace and each pipeline
rank's adaptive edge layers. TurboQuant bit widths control codebook precision;
its current byte storage does not provide ideal bit-packed memory savings.
Stored artifact bytes receive an overhead margin chosen for the actual engine.
Recurrent state and served speculation/slot costs remain separate.

Unknown layouts, Optiq's model-specific fallback, or unsupported affine geometry
retain an explicit conservative FP16 cache budget. Missing geometry stays unknown,
not proven zero cost. Placement previews expose `memory_estimate_notes` and the
instance carries its effective policy; these estimates are not measurements.
Retained prefixes consume live memory and receive no assumed reuse discount.
The managed llama-server pins F16 K/V, in-process llama.cpp uses its F16 binding defaults, and vLLM pins its default dtype policy;
MLX cache precision remains independent of weight precision. Resource guards
remain active against a demonstrated capacity shortfall, without requiring
physical qualification.

See [Model Cards](model-cards.md), [KV Cache Backends](kv-cache-backends.md) and
[Architecture Reference](architecture-reference.md).
