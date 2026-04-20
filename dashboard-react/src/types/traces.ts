/** Lightweight trace list item returned by the trace index API. */
export interface TraceListItem {
  taskId: string;
  createdAt: string;
  fileSize: number;
}

/** Aggregated stats for one trace category. */
export interface TraceCategoryStats {
  totalUs: number;
  count: number;
  minUs: number;
  maxUs: number;
  avgUs: number;
}

/** Per-rank trace stats keyed by category. */
export interface TraceRankStats {
  byCategory: Record<string, TraceCategoryStats>;
}

/** Full stats payload for one trace task. */
export interface TraceStatsResponse {
  taskId: string;
  totalWallTimeUs: number;
  byCategory: Record<string, TraceCategoryStats>;
  byRank: Record<string, TraceRankStats>;
}
