import { current } from '@reduxjs/toolkit';
import { apiSlice } from '../api';
import type { EngineChange, EngineChangeRequest, EnginePackage, EnginesReport } from '../../types/engines';

/**
 * Put a machine's answer to a change into the report, so the panel shows what
 * the machine holds now without waiting for its next report.
 *
 * A null package means the machine no longer holds or is offered the engine,
 * so its entry leaves the report, and an engine no machine reports leaves too.
 */
export function applyEngineChange(report: EnginesReport, change: EngineChange): EnginesReport {
  const same = (entry: { nodeId: string; package: EnginePackage }) =>
    entry.nodeId === change.nodeId && entry.package.lane === change.lane;
  const known = report.engines.some(engine => engine.engineId === change.engineId);
  const engines = report.engines.map(engine => {
    if (engine.engineId !== change.engineId) return engine;
    const others = engine.nodes.filter(entry => !same(entry));
    const nodes = change.package ? [...others, { nodeId: change.nodeId, package: change.package }] : others;
    nodes.sort((left, right) => left.nodeId.localeCompare(right.nodeId) || left.package.lane.localeCompare(right.package.lane));
    return { ...engine, nodes };
  }).filter(engine => engine.nodes.length > 0);
  if (!known && change.package) {
    engines.push({
      engineId: change.engineId,
      name: change.engineId.split('.').slice(1).join('.') || change.engineId,
      nodes: [{ nodeId: change.nodeId, package: change.package }],
    });
    engines.sort((left, right) => left.engineId.localeCompare(right.engineId));
  }
  return { ...report, engines };
}

/** Read the sentence a node sent with a refused or unanswered change. */
export function engineChangeErrorDetail(error: unknown): string | null {
  if (typeof error !== 'object' || error === null || !('data' in error)) return null;
  const data = (error as { data: unknown }).data;
  if (typeof data !== 'object' || data === null) return null;
  const body = data as { detail?: unknown; error?: unknown };
  if (typeof body.detail === 'string' && body.detail) return body.detail;
  if (typeof body.error === 'object' && body.error !== null) {
    const message = (body.error as { message?: unknown }).message;
    if (typeof message === 'string' && message) return message;
  }
  return null;
}

/**
 * Add-on engines: the cluster's report and the three changes a person may
 * make to an engine a machine holds.
 *
 * A change takes effect on one machine and answers with what that machine
 * holds straight afterwards. The report itself is assembled from what each
 * machine last advertised, which can trail the change by a moment, so the
 * answer is written into the cached report instead of asking again at once.
 */
const enginesApi = apiSlice.injectEndpoints({
  endpoints: build => ({
    getEngines: build.query<EnginesReport, void>({
      query: () => '/v1/engines',
      providesTags: ['Engines'],
    }),
    changeEngine: build.mutation<EngineChange, EngineChangeRequest>({
      query: ({ engineId, action, nodeId, lane, build: engineBuild }) => ({
        url: `/v1/engines/${encodeURIComponent(engineId)}/${action}`,
        method: 'POST',
        body: { nodeId, lane, ...(engineBuild ? { build: engineBuild } : {}) },
      }),
      async onQueryStarted(_request, { dispatch, queryFulfilled }) {
        try {
          const { data } = await queryFulfilled;
          dispatch(enginesApi.util.updateQueryData('getEngines', undefined, report => applyEngineChange(current(report), data)));
        } catch {
          // The caller shows the refusal; a change that was sent and not
          // confirmed may still have been made, so read the report again.
          dispatch(enginesApi.util.invalidateTags(['Engines']));
        }
      },
    }),
  }),
});

export const { useGetEnginesQuery, useChangeEngineMutation } = enginesApi;
