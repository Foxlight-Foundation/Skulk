export type TraceTaskKind = 'image' | 'text' | 'embedding';

export interface TraceSourceNode {
  nodeId: string;
  friendlyName?: string | null;
}

/** Lightweight trace list item returned by the trace index API. */
export interface TraceListItem {
  taskId: string;
  createdAt: string;
  fileSize: number;
  modelId?: string | null;
  taskKind?: TraceTaskKind | null;
  categories: string[];
  tags: string[];
  hasToolActivity: boolean;
  sourceNodes: TraceSourceNode[];
}

export interface TraceEventResponse {
  name: string;
  startUs: number;
  durationUs: number;
  rank: number;
  category: string;
  nodeId?: string | null;
  modelId?: string | null;
  taskKind?: TraceTaskKind | null;
  tags: string[];
  attrs: Record<string, string | number | boolean | string[]>;
}

export interface TraceResponse {
  taskId: string;
  traces: TraceEventResponse[];
  sourceNodes: TraceSourceNode[];
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
  sourceNodes: TraceSourceNode[];
}

export interface TraceListResponse {
  traces: TraceListItem[];
}

export interface TracingStateResponse {
  enabled: boolean;
}
