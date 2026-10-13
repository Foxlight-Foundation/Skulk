import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { NodeAddress } from '../endpoints/plugins';

/**
 * How the request that started a setup action run ended, as far as the
 * dashboard saw it. `settled` records an outcome observed among the
 * provider's operations, so it stays known after the provider stops listing
 * that operation.
 */
export type SetupRunSubmission =
  | { state: 'sent' }
  | { state: 'refused'; detail: string | null }
  | { state: 'unconfirmed'; since: number; gaveUp: boolean }
  | { state: 'missing' }
  | { state: 'settled'; outcome: 'complete' | 'failed'; detail: string | null };

/** One retained setup action run: the operation it submitted, with what values, and how that went. */
export interface SetupRunRecord {
  operationId: string;
  /** The ordinary form values sent, so a returning view can say what is being changed. */
  values: Record<string, unknown>;
  submission: SetupRunSubmission;
}

/**
 * Setup action runs, keyed by provider address and the caller's run name.
 *
 * The runs live here rather than in the view that starts them because that
 * view unmounts when its drawer closes. A run whose request is still in
 * flight, or whose reply was lost, must stay visible when the drawer opens
 * again, or a second operation could be submitted for a change that already
 * landed. The slice is not persisted: a record kept in one browser cannot
 * tell a request that never arrived from one still being worked on, and would
 * block the action there for good. Past a reload, the provider's own list of
 * unfinished setup work blocks a second submit instead (`useSetupActionRun`).
 */
export type SetupRunsState = Record<string, SetupRunRecord>;

const initialState: SetupRunsState = {};

interface RunTarget { key: string; operationId: string }

/** The key one named run of one provider node is kept under. */
export function setupRunKey(address: NodeAddress, runName: string): string {
  return `${address.pluginId}/${address.nodeId}/${runName}`;
}

/** True while a run's request is in flight or its reply was never seen. */
export function setupRunOpen(record: SetupRunRecord | undefined): boolean {
  return record?.submission.state === 'sent' || record?.submission.state === 'unconfirmed';
}

// Every change names the operation it is about, so a late reply to one run
// can never rewrite a newer run kept under the same key.
const current = (state: SetupRunsState, target: RunTarget) => {
  const record = state[target.key];
  return record?.operationId === target.operationId ? record : undefined;
};

const setupRunsSlice = createSlice({
  name: 'setupRuns',
  initialState,
  reducers: {
    /** A run began with a fresh operation ID, replacing any earlier run under its key. */
    started(
      state,
      action: PayloadAction<RunTarget & { values: Record<string, unknown>; submission: { state: 'sent' } | { state: 'missing' } }>,
    ) {
      const { key, operationId, values, submission } = action.payload;
      state[key] = { operationId, values, submission };
    },
    /** The run's request failed: refused by the host, or with no decided reply. */
    requestEnded(state, action: PayloadAction<RunTarget & { submission: { state: 'refused'; detail: string | null } | { state: 'unconfirmed'; since: number } }>) {
      const record = current(state, action.payload);
      // An outcome already observed is more certain than a failed reply.
      if (record?.submission.state !== 'sent') return;
      const { submission } = action.payload;
      record.submission = submission.state === 'refused' ? submission : { ...submission, gaveUp: false };
    },
    /** The run's own operation was seen to finish. */
    settled(state, action: PayloadAction<RunTarget & { outcome: 'complete' | 'failed'; detail: string | null }>) {
      const record = current(state, action.payload);
      if (!record || !setupRunOpen(record)) return;
      record.submission = { state: 'settled', outcome: action.payload.outcome, detail: action.payload.detail };
    },
    /** A run with no decided reply waited long enough to say its outcome is unconfirmed. */
    gaveUp(state, action: PayloadAction<RunTarget>) {
      const record = current(state, action.payload);
      if (record?.submission.state === 'unconfirmed') record.submission = { ...record.submission, gaveUp: true };
    },
    /** A finished run's outcome was dismissed. */
    forgotten(state, action: PayloadAction<RunTarget>) {
      if (current(state, action.payload)) delete state[action.payload.key];
    },
  },
});

export const setupRunsActions = setupRunsSlice.actions;
export const setupRunsSliceReducer = setupRunsSlice.reducer;
