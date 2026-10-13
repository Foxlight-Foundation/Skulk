import { useEffect, useRef, useState } from 'react';
import { apiSlice } from '../../store/api';
import { useAppDispatch } from '../../store/hooks';
import {
  pluginRefusalDetail,
  pluginRequestRefused,
  useGetNodeSetupActionsQuery,
  useStartNodeSetupMutation,
  type NodeAddress,
  type SetupAction,
} from '../../store/endpoints/plugins';
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

/** How the request that started a run ended, before its operation is observed. */
type Submission =
  | { state: 'sent' }
  | { state: 'refused'; detail: string | null }
  | { state: 'unconfirmed'; gaveUp: boolean }
  | { state: 'missing' };

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
 */
export function useSetupActionRun(address: NodeAddress, active: boolean): SetupActionRun {
  const dispatch = useAppDispatch();
  const [operationId, setOperationId] = useState<string | null>(null);
  const [submission, setSubmission] = useState<Submission>({ state: 'sent' });
  const sending = useRef(false);
  const [startSetup] = useStartNodeSetupMutation();
  const forms = useGetNodeSetupActionsQuery(address, {
    skip: !active && operationId === null,
    refetchOnMountOrArgChange: true,
  });
  const snapshot = forms.currentData;
  const operation = operationId ? snapshot?.operations.find((item) => item.operationId === operationId) : undefined;

  let phase: SetupActionRunPhase = 'running';
  let detail: string | null = null;
  if (operationId === null) phase = 'idle';
  else if (submission.state === 'refused') {
    phase = 'failed';
    detail = submission.detail;
  } else if (submission.state === 'missing') phase = 'failed';
  else if (operation?.phase === 'complete') phase = 'complete';
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

  useEffect(() => {
    // An unconfirmed change may have landed, so the limit is read again too.
    if (phase === 'complete' || phase === 'uncertain') dispatch(apiSlice.util.invalidateTags(['BurstBudget']));
  }, [phase, dispatch]);

  const waitingUnconfirmed = submission.state === 'unconfirmed' && !submission.gaveUp && operation === undefined;
  useEffect(() => {
    if (!waitingUnconfirmed) return;
    const timer = window.setTimeout(() => setSubmission({ state: 'unconfirmed', gaveUp: true }), UNCONFIRMED_WAIT_MS);
    return () => window.clearTimeout(timer);
  }, [waitingUnconfirmed]);

  const start = async (actionId: string, values: Record<string, unknown>) => {
    if (sending.current || !snapshot || phase === 'running' || phase === 'uncertain') return;
    const identity = randomHex32();
    const action = snapshot.actions.find((item) => item.actionId === actionId);
    setOperationId(identity);
    if (!action) {
      setSubmission({ state: 'missing' });
      return;
    }
    sending.current = true;
    setSubmission({ state: 'sent' });
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
      // still have landed, so the run watches for its own operation.
      setSubmission(
        pluginRequestRefused(error) !== null
          ? { state: 'refused', detail: pluginRefusalDetail(error) }
          : { state: 'unconfirmed', gaveUp: false },
      );
    } finally {
      sending.current = false;
    }
  };

  const reset = () => {
    // A run in flight, or one that may have landed, stays observable, so the
    // same change is never submitted a second time.
    if (phase === 'running' || phase === 'uncertain') return;
    setOperationId(null);
    setSubmission({ state: 'sent' });
  };

  const action = (actionId: string) => snapshot?.actions.find((item) => item.actionId === actionId) ?? null;

  return { phase, detail, ready: snapshot !== undefined, action, start, reset };
}
