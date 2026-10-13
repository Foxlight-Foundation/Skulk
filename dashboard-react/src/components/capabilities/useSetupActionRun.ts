import { useEffect, useRef } from 'react';
import { apiSlice } from '../../store/api';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import {
  pluginRefusalDetail,
  pluginRequestRefused,
  useGetNodeSetupActionsQuery,
  useStartNodeSetupMutation,
  type NodeAddress,
  type SetupAction,
} from '../../store/endpoints/plugins';
import { setupRunKey, setupRunOpen, setupRunsActions } from '../../store/slices/setupRunsSlice';
import { randomHex32 } from '../../utils/randomIds';
import { UNCONFIRMED_POLL_MS, UNCONFIRMED_WAIT_MS } from './setupRunTiming';

/**
 * Where one run of a provider's setup action stands. `uncertain` means the
 * request may have landed but its outcome was never seen: the run keeps
 * looking for it and cannot be submitted again.
 */
export type SetupActionRunPhase = 'idle' | 'running' | 'complete' | 'failed' | 'uncertain';

/** What a setup action run reports to its view. */
export interface SetupActionRun {
  phase: SetupActionRunPhase;
  /** The provider's own next step, or the refusal's text, when it failed. */
  detail: string | null;
  /** True once the provider's current setup forms are known, so a run can start. */
  ready: boolean;
  /** The values the current run sent, or null with no run. */
  values: Record<string, unknown> | null;
  /** The provider's current form for one action, once known, or null. */
  action: (actionId: string) => SetupAction | null;
  /** Runs the named action with these values, bound to the forms just read. */
  start: (actionId: string, values: Record<string, unknown>) => Promise<void>;
  /**
   * Forgets the last outcome. A run still in flight, or one whose outcome was
   * never confirmed, is kept, so it is never submitted a second time.
   */
  reset: () => void;
}

/**
 * Run one of a burst provider's setup actions, such as changing its spending
 * limit or starting the count over, and follow it to completion.
 *
 * The provider's setup forms are read while `active`, and a run binds to the
 * exact revisions it read, so a change made elsewhere meanwhile is refused
 * rather than overwritten. A refusal is reported in the provider's own words.
 * A reply that never arrives is never retried: the run watches for its own
 * operation, and after a short wait reports the outcome as unconfirmed while
 * it keeps looking, so the same change is never submitted twice. A completed
 * or unconfirmed run refreshes the provider's spending limit.
 *
 * The run is kept in the store under the provider and `runName`, so a view
 * that unmounts mid-run, as a closed drawer does, finds it again on return
 * instead of offering a second submit.
 */
export function useSetupActionRun(address: NodeAddress, runName: string, active: boolean): SetupActionRun {
  const dispatch = useAppDispatch();
  const key = setupRunKey(address, runName);
  const record = useAppSelector((state) => state.setupRuns[key]);
  const operationId = record?.operationId ?? null;
  const submission = record?.submission ?? null;
  const open = setupRunOpen(record);
  const sending = useRef(false);
  const [startSetup] = useStartNodeSetupMutation();
  const forms = useGetNodeSetupActionsQuery(address, {
    skip: !active && !open,
    refetchOnMountOrArgChange: true,
  });
  const snapshot = forms.currentData;
  const operation = open && operationId ? snapshot?.operations.find((item) => item.operationId === operationId) : undefined;

  let phase: SetupActionRunPhase = 'running';
  let detail: string | null = null;
  if (submission === null) phase = 'idle';
  else if (submission.state === 'refused') {
    phase = 'failed';
    detail = submission.detail;
  } else if (submission.state === 'missing') phase = 'failed';
  else if (submission.state === 'settled') {
    phase = submission.outcome;
    detail = submission.detail;
  } else if (operation?.phase === 'complete') phase = 'complete';
  else if (operation?.phase === 'failed') {
    phase = 'failed';
    detail = operation.correctiveAction;
  } else if (submission.state === 'unconfirmed' && submission.gaveUp) phase = 'uncertain';

  // A second subscription to the same cached forms polls while the outcome is
  // still open, so a settled outcome stops the reads and a late operation is
  // still seen.
  useGetNodeSetupActionsQuery(address, {
    skip: phase !== 'running' && phase !== 'uncertain',
    pollingInterval: phase === 'uncertain' ? UNCONFIRMED_POLL_MS : 2000,
  });

  // The observed outcome is recorded, so it is still known once the provider
  // stops listing the operation.
  const observedPhase = operation?.phase === 'complete' || operation?.phase === 'failed' ? operation.phase : null;
  const observedDetail = operation?.phase === 'failed' ? operation.correctiveAction : null;
  useEffect(() => {
    if (observedPhase === null || operationId === null) return;
    dispatch(setupRunsActions.settled({ key, operationId, outcome: observedPhase, detail: observedDetail }));
  }, [observedPhase, observedDetail, operationId, key, dispatch]);

  useEffect(() => {
    // An unconfirmed change may have landed, so the limit is read again too.
    if (phase === 'complete' || phase === 'uncertain') dispatch(apiSlice.util.invalidateTags(['BurstBudget']));
  }, [phase, dispatch]);

  // The wait is measured from the failed reply, so a view that returns
  // mid-wait does not start it over.
  const unconfirmedSince =
    submission?.state === 'unconfirmed' && !submission.gaveUp && operation === undefined ? submission.since : null;
  useEffect(() => {
    if (unconfirmedSince === null || operationId === null) return;
    const remaining = Math.max(0, UNCONFIRMED_WAIT_MS - (Date.now() - unconfirmedSince));
    const timer = window.setTimeout(() => dispatch(setupRunsActions.gaveUp({ key, operationId })), remaining);
    return () => window.clearTimeout(timer);
  }, [unconfirmedSince, operationId, key, dispatch]);

  const start = async (actionId: string, values: Record<string, unknown>) => {
    if (sending.current || !snapshot || phase === 'running' || phase === 'uncertain') return;
    const identity = randomHex32();
    const action = snapshot.actions.find((item) => item.actionId === actionId);
    if (!action) {
      dispatch(setupRunsActions.started({ key, operationId: identity, values, submission: { state: 'missing' } }));
      return;
    }
    sending.current = true;
    dispatch(setupRunsActions.started({ key, operationId: identity, values, submission: { state: 'sent' } }));
    try {
      await startSetup({
        ...address,
        mutation: {
          operationId: identity,
          actionId: action.actionId,
          expectedRevision: snapshot.revision,
          expectedSchemaDigest: snapshot.schemaDigest,
          expectedCredentialRevision: snapshot.credentialRevision,
          expectedCredentialSchemaDigest: snapshot.credentialSchemaDigest,
          expectedActionSchemaDigest: action.schemaDigest,
          expectedRequiresApproval: action.requiresApproval === true,
          values,
        },
      }).unwrap();
    } catch (error: unknown) {
      // A 4xx means the host decided and nothing changed; anything else may
      // still have landed, so the run watches for its own operation. The
      // store records the reply even if the view closed meanwhile.
      dispatch(setupRunsActions.requestEnded({
        key,
        operationId: identity,
        submission: pluginRequestRefused(error) !== null
          ? { state: 'refused', detail: pluginRefusalDetail(error) }
          : { state: 'unconfirmed', since: Date.now() },
      }));
    } finally {
      sending.current = false;
    }
  };

  const reset = () => {
    // A run in flight, or one that may have landed, stays observable, so the
    // same change is never submitted a second time.
    if (operationId === null || phase === 'running' || phase === 'uncertain') return;
    dispatch(setupRunsActions.forgotten({ key, operationId }));
  };

  const action = (actionId: string) => snapshot?.actions.find((item) => item.actionId === actionId) ?? null;

  return { phase, detail, ready: snapshot !== undefined, values: record?.values ?? null, action, start, reset };
}
