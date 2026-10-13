import { skipToken } from '@reduxjs/toolkit/query';
import { Fragment, useRef, useState } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';
import {
  useApproveNodeProposalMutation, useGetNodeProposalOperationQuery,
  useGetNodeProposalReviewQuery, useGetNodeProposalsQuery, useResumeNodeProposalMutation,
  type NodeAddress, type ProposalOperation, type ProposalReconciliation, type ProposalReference, type ProposalSummary,
} from '../../store/endpoints/plugins';
import { Button } from '../common/Button';
import { randomHex32 } from '../../utils/randomIds';

const Panel = styled.section`margin-top: 16px; overflow-wrap: anywhere;`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0;`;
const Facts = styled.dl`dt { font-weight: 600; } dd { margin: 4px 0 12px; }`;

function retainedOperation(key: string): string {
  try { const value = localStorage.getItem(key); return value && /^[a-f0-9]{32}$/.test(value) ? value : ''; }
  catch { return ''; }
}

/** A moment in the viewer's own time zone and language. */
const moment = (seconds: number) => new Date(seconds * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

function proposalState(t: SkulkTranslate, state: ProposalSummary['state']): string {
  switch (state) {
    case 'pending_approval': return t('plugins.proposalStatePending', 'Waiting for your approval');
    case 'approved': return t('plugins.proposalStateApproved', 'Approved');
    case 'execution_recorded': return t('plugins.proposalStateStarted', 'Started');
    case 'expired': return t('plugins.proposalStateExpired', 'Expired, nothing was done');
    case 'unavailable': return t('plugins.proposalStateUnavailable', 'Cannot be read right now');
  }
}

function operationPhase(t: SkulkTranslate, phase: ProposalOperation['phase']): string {
  switch (phase) {
    case 'accepted': return t('plugins.proposalPhaseAccepted', 'Approval received');
    case 'approving': return t('plugins.proposalPhaseApproving', 'Approving');
    case 'approved': return t('plugins.proposalPhaseApproved', 'Approved');
    case 'dispatching': return t('plugins.proposalPhaseDispatching', 'Starting');
    case 'acknowledged': return t('plugins.proposalPhaseAcknowledged', 'Started');
    case 'succeeded': return t('plugins.proposalPhaseSucceeded', 'Done');
    case 'refused': return t('plugins.proposalPhaseRefused', 'Not done: the plugin refused it');
    case 'approval_interrupted': return t('plugins.proposalPhaseInterrupted', 'The approval was interrupted');
    case 'uncertain': return t('plugins.proposalPhaseUncertain', 'Not yet known whether it started');
  }
}

function cleanupState(t: SkulkTranslate, state: ProposalReconciliation['state']): string {
  switch (state) {
    case 'pending': return t('plugins.proposalCleanupPending', 'The cleanup service has not seen it yet');
    case 'active': return t('plugins.proposalCleanupActive', 'Running');
    case 'releasing': return t('plugins.proposalCleanupReleasing', 'Being released');
    case 'absent': return t('plugins.proposalAbsent', 'Confirmed gone');
    case 'attention': return t('plugins.proposalCleanupAttention', 'Needs attention');
    case 'unknown': return t('plugins.proposalCleanupUnknown', 'Not known yet');
  }
}

/**
 * Review immutable provider terms and explicitly approve once; reconnect only
 * reads. Exported so a burst provider's Rentals view shows the same review.
 * The default view keeps to plain words; `troubleshooting` adds what an
 * operator needs to trace one approval: the proposal digest, the operation
 * ID, result codes, and lookup by operation ID.
 */
export function ProposalControls({ pluginId, nodeId, actionsAvailable, troubleshooting = false }: NodeAddress & { actionsAvailable: boolean; troubleshooting?: boolean }) {
  const { t } = useSkulkTranslation();
  const storageKey = `skulk-proposal-operation-v1:${encodeURIComponent(pluginId)}:${encodeURIComponent(nodeId)}`;
  const [operationId, setOperationId] = useState(() => retainedOperation(storageKey));
  const [lookup, setLookup] = useState('');
  const [offset, setOffset] = useState(0);
  const [reference, setReference] = useState<ProposalReference>();
  const [notice, setNotice] = useState('');
  // Without storage the operation ID is the only way back to this approval
  // after a reload, so it is shown even outside troubleshooting.
  const [storageFailed, setStorageFailed] = useState(false);
  const submitting = useRef(false);
  const [approve, approving] = useApproveNodeProposalMutation();
  const [resume, resuming] = useResumeNodeProposalMutation();
  const listing = useGetNodeProposalsQuery({ pluginId, nodeId, offset }, { refetchOnMountOrArgChange: true });
  const review = useGetNodeProposalReviewQuery(reference ?? skipToken, { refetchOnMountOrArgChange: true });
  const operation = useGetNodeProposalOperationQuery(operationId ? { pluginId, nodeId, operationId } : skipToken,
    { pollingInterval: 3000, skipPollingIfUnfocused: true, refetchOnMountOrArgChange: true });
  const current = review.currentData;
  const observed = operation.currentData;
  const busy = approving.isLoading || resuming.isLoading;
  const remember = (identity: string) => {
    // Persist only a lookup ID, before the POST. No proposal contents, approval
    // material or credentials are stored; reload always performs a fresh GET.
    try { if (identity) localStorage.setItem(storageKey, identity); else localStorage.removeItem(storageKey); }
    catch {
      setStorageFailed(true);
      setNotice(t('plugins.proposalKeepId', 'This browser cannot remember the approval. Note the operation ID below to check its progress after a reload.'));
    }
    setOperationId(identity);
  };
  const approveReviewed = async () => {
    if (submitting.current || operationId || !current?.approvalRevision || review.isFetching || review.isError || !actionsAvailable) return;
    submitting.current = true;
    setNotice('');
    const identity = randomHex32();
    remember(identity);
    try {
      await approve({ operationId: identity, reference: current.proposal.reference, reviewRevision: current.approvalRevision }).unwrap();
    } catch {
      setNotice(t('plugins.proposalReplyLost', 'The approval was sent but not confirmed. Its progress shows here; do not approve it again.'));
    } finally { submitting.current = false; }
  };
  const resumeOriginal = async () => {
    if (submitting.current || observed?.phase !== 'approval_interrupted' || operation.isError || !actionsAvailable) return;
    submitting.current = true;
    setNotice('');
    try { await resume({ pluginId, nodeId, operationId }).unwrap(); }
    catch { setNotice(t('plugins.proposalResumeFailed', 'Continuing was not confirmed. Check progress before doing anything else.')); }
    finally { submitting.current = false; void operation.refetch(); }
  };
  const proposals = listing.currentData?.proposals;
  return <Panel>
    {troubleshooting ? <h4>{t('plugins.proposalReviewTitle', 'Proposal review')}</h4> : null}
    <p>{t('plugins.proposalReviewIntro', 'Read the terms before you approve. Approving can start work that costs money.')}</p>
    <Actions>
      <Button type="button" disabled={listing.isFetching || busy} onClick={() => { setReference(undefined); void listing.refetch(); }}>{t('plugins.proposalRefresh', 'Refresh')}</Button>
      {offset > 0 ? <Button type="button" onClick={() => { setReference(undefined); setOffset(0); }}>{t('plugins.proposalFirst', 'First page')}</Button> : null}
      {listing.currentData?.nextOffset != null ? <Button type="button" onClick={() => { setReference(undefined); setOffset(listing.currentData!.nextOffset!); }}>{t('plugins.proposalNext', 'Next page')}</Button> : null}
    </Actions>
    {listing.isError ? <p role="alert">{t('plugins.proposalsUnavailable', 'Proposals cannot be read right now. Check the plugin in Plugins, then refresh.')}</p> : null}
    {proposals?.length === 0 ? <p>{t('plugins.proposalNone', 'Nothing is waiting for your approval.')}</p> : null}
    {proposals?.map((item) => <p key={`${item.reference.proposalId}:${item.reference.proposalDigest}`}>
      {item.summary} · {proposalState(t, item.state)}
      {item.state === 'pending_approval' ? ` · ${t('plugins.proposalApproveBy', 'approve by {time}', { time: moment(item.expiresAt) })}` : ''}{' '}
      <Button type="button" disabled={busy || listing.isFetching || listing.isError} onClick={() => { setNotice(''); setReference(item.reference); }}>{t('plugins.proposalReview', 'Review')}</Button>
    </p>)}
    {review.isError ? <p role="alert">{t('plugins.proposalReviewUnavailable', 'This proposal cannot be read right now. Refresh before approving.')}</p> : null}
    {reference && current ? <section aria-label={t('plugins.proposalTerms', 'Terms')}>
      <h5>{current.proposal.summary}</h5>
      <Facts>{current.fields.map((field, index) => <Fragment key={index}><dt>{field.label}</dt><dd>{field.value}</dd></Fragment>)}</Facts>
      <p>{current.proposal.state === 'pending_approval'
        ? t('plugins.proposalExpiry', 'Approve by {time}. After that it expires and nothing is done.', { time: moment(current.proposal.expiresAt) })
        : proposalState(t, current.proposal.state)}</p>
      {troubleshooting ? <>
        <p>{t('plugins.proposalDigest', 'Proposal digest')}: {current.proposal.reference.proposalDigest}</p>
        <p>{t('plugins.proposalObserved', 'Reviewed at')}: {moment(current.observedAt)}</p>
      </> : null}
      <Button type="button" disabled={!actionsAvailable || busy || !!operationId || review.isFetching || review.isError || !current.approvalRevision || current.proposal.state !== 'pending_approval' || current.proposal.expiresAt * 1000 <= Date.now()}
        onClick={() => void approveReviewed()}>{t('plugins.proposalApprove', 'Approve')}</Button>
      {!current.approvalRevision ? <p>{t('plugins.proposalApprovalUnavailable', 'Approval is not set up on this computer yet. Open Plugins and finish this plugin\'s setup.')}</p> : null}
    </section> : null}
    {operationId ? <section aria-label={t('plugins.proposalProgress', 'Progress')}>
      {troubleshooting || storageFailed ? <p>{t('plugins.proposalOperationId', 'Operation ID')}: {operationId}</p> : null}
      {observed ? <p role="status">{operationPhase(t, observed.phase)}{troubleshooting && observed.code ? ` · ${observed.code}` : ''}{observed.correctiveAction ? ` · ${observed.correctiveAction}` : ''}</p> : null}
      {observed?.reconciliation ? <section aria-label={t('plugins.proposalCleanup', 'Cleanup service')}>
        <p>{cleanupState(t, observed.reconciliation.state)}</p>
        {observed.reconciliation.observedAt != null ? <p>{t('plugins.proposalCleanupObserved', 'Cleanup service last checked')}: {moment(observed.reconciliation.observedAt)}</p> : null}
        {observed.reconciliation.stale ? <p role="alert">{t('plugins.proposalCleanupStale', 'The cleanup service could not be reached, so this may be out of date. Refresh later; do not approve again.')}</p> : null}
        {troubleshooting ? <p>{t('plugins.proposalCleanupHistory', 'Cleanup evidence does not change the submission result or prove inference readiness.')}</p> : null}
      </section> : null}
      {operation.isError ? <p role="alert">{t('plugins.proposalStatusUnavailable', 'Progress cannot be read right now. Refresh later; do not approve again.')}</p> : null}
      <Actions>
        <Button type="button" disabled={operation.isFetching || busy} onClick={() => void operation.refetch()}>{t('plugins.proposalRefreshOperation', 'Check progress')}</Button>
        {observed?.phase === 'approval_interrupted' ? <Button type="button" disabled={!actionsAvailable || busy || operation.isError || operation.isFetching} onClick={() => void resumeOriginal()}>{t('plugins.proposalResume', 'Continue the approval')}</Button> : null}
        <Button type="button" disabled={busy} onClick={() => { remember(''); setReference(undefined); }}>{t('plugins.proposalAnother', 'Back to the list')}</Button>
      </Actions>
    </section> : null}
    {troubleshooting ? <form onSubmit={(event) => { event.preventDefault(); if (/^[a-f0-9]{32}$/.test(lookup) && !busy) remember(lookup); }}>
      <label>{t('plugins.proposalLookup', 'Existing operation ID')}<input value={lookup} maxLength={32} onChange={(event) => setLookup(event.target.value)} /></label>
      <Button type="submit" disabled={busy || !/^[a-f0-9]{32}$/.test(lookup)}>{t('plugins.proposalObserve', 'Observe operation')}</Button>
    </form> : null}
    {notice ? <p role="status">{notice}</p> : null}
  </Panel>;
}

/** Load proposal metadata only after the operator opens this node's review panel. */
export function NodeProposalsPanel({ pluginId, nodeId, actionsAvailable }: NodeAddress & { actionsAvailable: boolean }) {
  const { t } = useSkulkTranslation();
  const [open, setOpen] = useState(false);
  return <Panel><Button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? t('plugins.proposalsClose', 'Close proposals') : t('plugins.proposalsOpen', 'Review proposals')}</Button>
    {open ? <ProposalControls key={`${pluginId}/${nodeId}`} pluginId={pluginId} nodeId={nodeId} actionsAvailable={actionsAvailable} troubleshooting /> : null}
  </Panel>;
}
