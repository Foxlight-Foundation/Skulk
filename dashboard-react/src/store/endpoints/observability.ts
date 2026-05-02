import { apiSlice } from '../api';
import type {
  ClusterTimeline,
  DiagnosticCaptureResponse,
  NodeDiagnostics,
} from '../../types/diagnostics';
import type {
  TraceListResponse,
  TraceResponse,
  TracingStateResponse,
} from '../../types/traces';

export type TraceScope = 'cluster' | 'local';

interface CancelRunnerTaskArgs {
  nodeId: string;
  runnerId: string;
  taskId: string;
}

interface CaptureRunnerBundleArgs {
  nodeId: string;
  runnerId: string;
  taskId?: string | null;
  includeProcessSamples?: boolean;
  sampleDurationSeconds?: number;
}

interface CancelRunnerTaskResponse {
  message?: string;
  detail?: string;
}

function tracesBasePath(scope: TraceScope): string {
  return scope === 'cluster' ? '/v1/traces/cluster' : '/v1/traces';
}

/**
 * Observability surface (Live tab cluster health, Traces tab, Node tab
 * diagnostics) endpoints injected into the root API slice. Polling cadence
 * is set per-call via the auto-generated hook's options object so the panel
 * can pause polling when closed (`{ skip: !panelOpen }`).
 */
export const observabilityApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    /* ── Cluster timeline ──────────────────────────────── */

    getClusterTimeline: build.query<ClusterTimeline, void>({
      query: () => '/v1/diagnostics/cluster/timeline',
      providesTags: ['ClusterTimeline'],
    }),

    /* ── Tracing toggle ─────────────────────────────────── */

    getTracingState: build.query<TracingStateResponse, void>({
      query: () => '/v1/tracing',
      providesTags: ['TracingState'],
    }),
    setTracingState: build.mutation<TracingStateResponse, boolean>({
      query: (enabled) => ({
        url: '/v1/tracing',
        method: 'PUT',
        body: { enabled },
      }),
      invalidatesTags: ['TracingState'],
    }),

    /* ── Traces ─────────────────────────────────────────── */

    getTracesList: build.query<TraceListResponse, TraceScope>({
      query: (scope) => tracesBasePath(scope),
      // Scope-specific cache; tagged so a future bulk-delete mutation can
      // invalidate the right list.
      providesTags: (_result, _err, scope) => [{ type: 'TraceList', id: scope }],
    }),

    getTrace: build.query<TraceResponse, { scope: TraceScope; taskId: string }>({
      query: ({ scope, taskId }) => `${tracesBasePath(scope)}/${encodeURIComponent(taskId)}`,
      providesTags: (_result, _err, arg) => [{ type: 'Trace', id: `${arg.scope}:${arg.taskId}` }],
    }),

    /* ── Node diagnostics ───────────────────────────────── */

    getNodeDiagnostics: build.query<NodeDiagnostics, string>({
      query: (nodeId) => `/v1/diagnostics/cluster/${encodeURIComponent(nodeId)}`,
      providesTags: (_result, _err, nodeId) => [{ type: 'NodeDiagnostics', id: nodeId }],
    }),

    cancelRunnerTask: build.mutation<CancelRunnerTaskResponse, CancelRunnerTaskArgs>({
      query: ({ nodeId, runnerId, taskId }) => ({
        url: `/v1/diagnostics/cluster/${encodeURIComponent(nodeId)}/runners/${encodeURIComponent(runnerId)}/cancel`,
        method: 'POST',
        body: { taskId },
      }),
      // Cancelling a task changes runner status; invalidate the node entry
      // so a re-fetch surfaces the new state.
      invalidatesTags: (_result, _err, arg) => [{ type: 'NodeDiagnostics', id: arg.nodeId }],
    }),

    captureRunnerBundle: build.mutation<DiagnosticCaptureResponse, CaptureRunnerBundleArgs>({
      query: ({ nodeId, runnerId, taskId, includeProcessSamples, sampleDurationSeconds }) => ({
        url: `/v1/diagnostics/cluster/${encodeURIComponent(nodeId)}/capture`,
        method: 'POST',
        body: {
          runnerId,
          taskId: taskId ?? undefined,
          includeProcessSamples: includeProcessSamples ?? true,
          sampleDurationSeconds: sampleDurationSeconds ?? 3,
        },
      }),
      invalidatesTags: (_result, _err, arg) => [{ type: 'NodeDiagnostics', id: arg.nodeId }],
    }),
  }),
});

export const {
  useGetClusterTimelineQuery,
  useGetTracingStateQuery,
  useSetTracingStateMutation,
  useGetTracesListQuery,
  useGetTraceQuery,
  useGetNodeDiagnosticsQuery,
  useCancelRunnerTaskMutation,
  useCaptureRunnerBundleMutation,
} = observabilityApi;
