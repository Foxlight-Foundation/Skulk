/**
 * Types for the observe-only performance-envelope diagnostics (adaptive
 * concurrency, Phase 0). Mirrors the API's `PerformanceEnvelopeReport` /
 * `ClusterPerformanceEnvelopes` (camelCase over the wire). Data only; no
 * behavior is driven from these yet.
 */

/** Aggregated performance at one in-flight-concurrency level. */
export interface ConcurrencyBucketSummary {
  /** Total in-flight requests INCLUDING this one at admission (a lone request is 1). */
  concurrency: number;
  /** Observations folded into this bucket. */
  requestCount: number;
  /** Observations that finished cleanly. */
  successCount: number;
  /** Observations that ended in a generation error. */
  errorCount: number;
  /** Median time-to-first-token, seconds (null until sampled). */
  ttftSecondsP50: number | null;
  /** 90th-percentile time-to-first-token, seconds. */
  ttftSecondsP90: number | null;
  /** Mean steady-state decode tokens/second. */
  decodeTpsMean: number | null;
  /** Median steady-state decode tokens/second. */
  decodeTpsP50: number | null;
  /** `concurrency * decodeTpsMean` — total useful throughput at this level. */
  aggregateDecodeTps: number | null;
}

/** One model+engine+hardware envelope: its per-concurrency curve and knee. */
export interface PerformanceEnvelopeSummary {
  /** Canonical class of the serving node(s), e.g. `nvidia-a100-80gb`. */
  hardwareClass: string;
  /** The served model identifier. */
  modelId: string;
  /** Resolved engine+backend tag, e.g. `vllm-cuda`, `mlx`. */
  backend: string;
  /** Model quantization label, or empty when unquantized/unknown. */
  quantization: string;
  /** Per-concurrency summaries, ascending by concurrency. */
  buckets: ConcurrencyBucketSummary[];
  /** Concurrency past which aggregate throughput stops rising, or null. */
  kneeConcurrency: number | null;
  /** Total observations across all buckets. */
  observationCount: number;
}

/** Read-only snapshot of one node's performance envelopes. */
export interface PerformanceEnvelopeReport {
  /** UTC ISO-8601 time the snapshot was computed. */
  generatedAt: string;
  /** One summary per observed (hardware × model × engine × quant). */
  envelopes: PerformanceEnvelopeSummary[];
}

/** One cluster member's envelope report, or the reason it is unavailable. */
export interface NodePerformanceEnvelopes {
  nodeId: string;
  url: string | null;
  ok: boolean;
  report?: PerformanceEnvelopeReport | null;
  error?: string | null;
}

/** Performance envelopes gathered from every reachable cluster member. */
export interface ClusterPerformanceEnvelopes {
  generatedAt: string;
  nodes: NodePerformanceEnvelopes[];
}
