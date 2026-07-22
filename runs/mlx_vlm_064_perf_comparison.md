# MLX-VLM 0.6.4 Harness Performance Comparison

Generated from local `skulk-test-harness/runs/*/report.json` artifacts on
2026-07-07.

## Scope

- Scanned 297 saved harness reports.
- Treated the final MLX-VLM 0.6.4 validation battery as:
  - `20260707-033838-dense-singles-chat-tests`
  - `20260707-034004-moe-chat-tests`
  - `20260707-035534-multinode-large-chat-tests`
  - `20260707-041324-tensor-sharding-chat-tests`
  - `20260707-041119-smoke-cancellation`
  - `20260707-041227-context-admission-context-admission`
  - `20260707-041237-embeddings-embeddings`
  - `20260707-041242-vision-vision`
- Compared each latest result by `(model_id, test_name, repetition)` against
  the median of up to three previous clean matching runs with the same
  `model_set`, `test_set`, sharding, and `min_nodes`.
- Positive TTFT means slower. Positive TPS means faster.

## Summary

The latest MLX-VLM 0.6.4 battery is materially better for stability than the
debugging runs that preceded it, but the saved timing data does not show a
broad performance win. Dense single-model runs are effectively flat. MoE,
multinode-large, tensor-sharding, and vision are slower on median versus prior
matching saved runs.

This is not a controlled benchmark result. The harness timings are real
cluster E2E measurements, but prior runs may differ in cache warmth, placement,
cluster contention, staged-model state, and code branch. The reports also do
not record the installed `mlx-vlm` version, so version attribution is inferred
from run timing and the active worktree context rather than encoded in the
artifact itself.

## Clean-Baseline Comparison

| Latest suite | Baseline runs | Comparable results | Median TTFT change | Median wall TPS change |
|---|---:|---:|---:|---:|
| dense-singles/chat-tests | `20260707-032417`, `20260707-030203`, `20260706-005924` | 15 | +0.6% | -0.1% |
| moe/chat-tests | `20260703-191818`, `20260702-083353`, `20260630-203421` | 15 | +5.9% | -11.6% |
| multinode-large/chat-tests | `20260706-031215`, `20260705-141808`, `20260705-071032` | 12 | +2.8% | -8.6% |
| tensor-sharding/chat-tests | `20260706-014355`, `20260705-143259`, `20260705-072747` | 3 | +11.1% | -15.4% |
| smoke/cancellation | `20260706-014427`, `20260705-143417`, `20260705-072905` | 1 | -1.4% | -1.9% |
| vision/vision | `20260706-014448`, `20260705-143435`, `20260705-072923` | 1 | +29.5% | -20.3% |

`context-admission` and `embeddings` passed but do not emit TTFT/TPS metrics in
the same way, so they are excluded from timing deltas.

## Long-Output-Only View

The largest wall-TPS swings come from very short `concise-factual-answer`
results, where a 5-character output makes wall TPS noisy. Filtering to outputs
with at least 50 characters gives:

| Latest suite | Comparable long-output results | Median TTFT change | Median wall TPS change |
|---|---:|---:|---:|
| dense-singles/chat-tests | 10 | -0.3% | +0.4% |
| moe/chat-tests | 10 | +12.3% | -11.2% |
| multinode-large/chat-tests | 8 | +2.2% | -5.1% |
| tensor-sharding/chat-tests | 2 | +11.6% | -14.1% |

## Sensitivity Check

For MoE, the nearest July 5/6 matching runs had all result rows passing but
also recorded harness issues. Including those issue-marked recent runs instead
of requiring clean baseline runs does not change the conclusion:

| Latest suite | Recent baseline including issue-marked runs | Median TTFT change | Median wall TPS change |
|---|---|---:|---:|
| moe/chat-tests | `20260706-010655` issues=2, `20260705-135619` issues=2, `20260705-064848` issues=1 | +11.6% | -12.1% |
| multinode-large/chat-tests | `20260706-031215` issues=0, `20260706-012848` issues=1, `20260705-141808` issues=0 | +2.2% | -6.4% |

## Notable Changes

Largest useful-looking gain:
- `mlx-community/GLM-4.7-Flash-4bit`, `concise-factual-answer`: wall TPS
  +165.8%, TTFT -4.1%. This is a 5-character output, so treat it as noisy.

Stable/flat area:
- `dense-singles/chat-tests` is basically flat overall. Long-output median is
  TTFT -0.3%, wall TPS +0.4%.

Most concerning timing changes:
- `tensor-sharding/chat-tests`: median TTFT +11.1%, wall TPS -15.4%.
- `vision/vision`: TTFT +29.5%, wall TPS -20.3%, but only one comparable
  result.
- `moe/chat-tests`: long-output median TTFT +12.3%, wall TPS -11.2%.

## Read

The upgrade/fix branch looks good on stability, not speed. If we want a real
performance claim, the next step should be a controlled benchmark run with:

- dependency version recorded in each report,
- cold/warm cache labels,
- fixed model placement/node selection,
- repeated samples per model/test,
- separate summaries for short-output latency and longer-output throughput.
