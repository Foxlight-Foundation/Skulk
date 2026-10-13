import { expect, it } from 'vitest';
import { setupRunKey, setupRunOpen, setupRunsActions, setupRunsSliceReducer, type SetupRunsState } from './setupRunsSlice';

const key = setupRunKey({ pluginId: 'managed.runpod', nodeId: 'gpu' }, 'limit');

function started(operationId: string): SetupRunsState {
  return setupRunsSliceReducer(
    {},
    setupRunsActions.started({ key, operationId, values: { budget: 20 }, submission: { state: 'sent' } }),
  );
}

it('keeps runs per provider node and run name', () => {
  expect(key).toBe('managed.runpod/gpu/limit');
  expect(setupRunKey({ pluginId: 'managed.runpod', nodeId: 'gpu' }, 'startOver')).not.toBe(key);
});

it('treats a sent or unconfirmed run as open, and a settled one as finished', () => {
  let state = started('a');
  expect(setupRunOpen(state[key])).toBe(true);
  state = setupRunsSliceReducer(
    state,
    setupRunsActions.requestEnded({ key, operationId: 'a', submission: { state: 'unconfirmed', since: 1 } }),
  );
  expect(state[key]?.submission).toEqual({ state: 'unconfirmed', since: 1, gaveUp: false });
  expect(setupRunOpen(state[key])).toBe(true);
  state = setupRunsSliceReducer(state, setupRunsActions.settled({ key, operationId: 'a', outcome: 'complete', detail: null }));
  expect(setupRunOpen(state[key])).toBe(false);
});

it('never lets a failed reply rewrite an outcome already observed', () => {
  let state = started('a');
  state = setupRunsSliceReducer(state, setupRunsActions.settled({ key, operationId: 'a', outcome: 'complete', detail: null }));
  state = setupRunsSliceReducer(
    state,
    setupRunsActions.requestEnded({ key, operationId: 'a', submission: { state: 'unconfirmed', since: 1 } }),
  );
  expect(state[key]?.submission).toEqual({ state: 'settled', outcome: 'complete', detail: null });
});

it('ignores changes about an earlier run under the same key', () => {
  let state = started('a');
  state = setupRunsSliceReducer(state, setupRunsActions.settled({ key, operationId: 'a', outcome: 'failed', detail: null }));
  state = setupRunsSliceReducer(
    state,
    setupRunsActions.started({ key, operationId: 'b', values: {}, submission: { state: 'sent' } }),
  );
  state = setupRunsSliceReducer(
    state,
    setupRunsActions.requestEnded({ key, operationId: 'a', submission: { state: 'refused', detail: 'late' } }),
  );
  state = setupRunsSliceReducer(state, setupRunsActions.forgotten({ key, operationId: 'a' }));
  expect(state[key]).toEqual({ operationId: 'b', values: {}, submission: { state: 'sent' } });
});
