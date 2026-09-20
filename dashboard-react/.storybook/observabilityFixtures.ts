import type { ClusterTimeline } from '../src/types/diagnostics';
import type { ClusterPerformanceEnvelopes } from '../src/types/performanceEnvelope';
import type { TraceListResponse, TraceResponse } from '../src/types/traces';

const generatedAt = '2026-09-18T12:00:00Z';
const sourceNodes = [{ nodeId: 'example-workstation', friendlyName: 'Example workstation' }];
const timeline: ClusterTimeline = {
  generatedAt, localNodeId: 'example-workstation', masterNodeId: 'example-workstation', unreachableNodes: [],
  runners: [{ nodeId: 'example-workstation', runnerId: 'example-runner', instanceId: 'example-instance', modelId: 'example/Chat-8B', deviceRank: 0, worldSize: 1, processAlive: true, statusKind: 'RunnerReady', phase: 'idle', secondsInPhase: 12 }],
  timeline: [{ at: generatedAt, nodeId: 'example-workstation', runnerId: 'example-runner', deviceRank: 0, worldSize: 1, phase: 'idle', event: 'request_completed', detail: 'Fictional request completed', attrs: {} }],
};
const performance: ClusterPerformanceEnvelopes = {
  generatedAt, nodes: [{ nodeId: 'example-workstation', url: null, ok: true, report: { generatedAt, envelopes: [{ hardwareClass: 'example-workstation', modelId: 'example/Chat-8B', backend: 'mlx', quantization: '4bit', kneeConcurrency: null, batches: false, observationCount: 12, buckets: [{ concurrency: 1, requestCount: 12, successCount: 12, errorCount: 0, ttftSecondsP50: .412, ttftSecondsP90: .6, decodeTpsMean: 48.2, decodeTpsP50: 48, aggregateDecodeTps: 48.2 }] }] } }],
};
const traces: TraceListResponse = { traces: [{ taskId: 'example-task', createdAt: generatedAt, fileSize: 48000, modelId: 'example/Chat-8B', taskKind: 'text', categories: ['prefill', 'decode'], tags: ['fixture'], hasToolActivity: false, sourceNodes }] };
const trace: TraceResponse = { taskId: 'example-task', sourceNodes, traces: [
  { name: 'prefill', startUs: 0, durationUs: 412000, rank: 0, category: 'prefill', tags: [], attrs: {} },
  { name: 'decode', startUs: 412000, durationUs: 1820000, rank: 0, category: 'decode', tags: [], attrs: {} },
] };

/** Fictional, read-only gallery observations; requests never reach a live cluster. */
export const observabilityFixtures: Record<string, unknown> = {
  '/v1/diagnostics/cluster/timeline': timeline,
  '/v1/diagnostics/performance-envelopes/cluster': performance,
  '/v1/traces': traces,
  '/v1/traces/cluster': traces,
  '/v1/traces/example-task': trace,
  '/v1/traces/cluster/example-task': trace,
};
