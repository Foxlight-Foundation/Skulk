import type { TraceEventResponse } from './traces';

/**
 * Common shape consumed by the observability waterfall renderer.
 *
 * Two upstream sources collapse onto this type today:
 *
 *  - **Saved traces** (`TraceEventResponse` from `/v1/traces/{taskId}` or the cluster
 *    proxy variant). Adapter: `traceEventToWaterfall`.
 *  - **Live cluster timeline** (`ClusterTimelineEntry` from `/v1/diagnostics/cluster/timeline`).
 *    The Phase 3 Live tab (#120) will produce these via a separate adapter; the type
 *    here keeps the contract honest so the waterfall component does not need to know
 *    which source produced its props.
 *
 * `TraceEventLike` is intentionally a *render* contract — it is not a domain object.
 * Pre-compute lane assignments and labels in the adapters so the renderer stays a
 * pure function of its props.
 */
export interface TraceEventLike {
  /** Stable React key. Adapters synthesize this from rank/start/name. */
  id: string;
  /** Lane bucket. Events with the same `laneKey` share a horizontal track. */
  laneKey: string;
  /** Human-readable lane label shown on the left axis. */
  laneLabel: string;
  /** Microseconds from the trace's t0 origin (adapters normalize). */
  startUs: number;
  /** Bar width in microseconds. Zero-duration events render as ticks. */
  durationUs: number;
  /** Top-level category, used for color-bucketing. */
  category: string;
  /** Bar label / hover title. */
  name: string;
  /** Free-form attributes; surfaced in the detail panel on selection. */
  attrs?: Record<string, string | number | boolean | string[]>;
  /** Operator-supplied tags (e.g. `"vision"`, `"prefill-warmup"`). */
  tags?: string[];
  /** Originating node id when the upstream payload supplies it. */
  sourceNodeId?: string | null;
}

/**
 * Convert a saved-trace API payload to renderer-ready events.
 *
 * Lane assignment: one lane per `rank`. The rank order is preserved as the
 * order of first appearance in the input — typically rank 0 first, which keeps
 * the visually expected ordering without an explicit sort.
 */
export function traceEventToWaterfall(events: readonly TraceEventResponse[]): TraceEventLike[] {
  return events.map((event) => ({
    id: `${event.rank}:${event.startUs}:${event.name}`,
    laneKey: `rank-${event.rank}`,
    laneLabel: `Rank ${event.rank}`,
    startUs: event.startUs,
    durationUs: event.durationUs,
    category: topLevelCategory(event.category),
    name: event.name,
    attrs: event.attrs,
    tags: event.tags,
    sourceNodeId: event.nodeId ?? null,
  }));
}

function topLevelCategory(category: string): string {
  // Backend currently emits dotted/slashed sub-categories like "decode/sample"
  // and "compute/forward". Bucket them by the segment before the first
  // separator so the color palette is per-category, not per-subcategory.
  const slash = category.indexOf('/');
  return slash >= 0 ? category.slice(0, slash) : category;
}
