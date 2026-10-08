import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { FiCheck, FiX } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { Surface } from '../common/Surfaces';
import {
  pluginRefusalDetail, pluginRequestRefused, useActivateRuntimeReleaseMutation, useLazyGetManagedOperationQuery, useLazyGetManagedRuntimesQuery,
  useLazyGetRuntimeInstallationQuery, type ManagedOperation, type ManagedRuntime, type RuntimeInstallation,
} from '../../store/endpoints/plugins';
import { randomHex32 } from '../../utils/randomIds';
import { clearJourney, formatMegabytes, isBound, markInterrupted, readJourneys, saveJourney, type BoundJourney, type InstallStartRefusal } from './catalogJourney';

/** What the setup checklist needs once the release is running. */
export interface SetupTarget { pluginId: string; title: string }

/** Props for the install progress view. */
export interface CatalogInstallProgressProps {
  title: string;
  publisher: string;
  sequence: number;
  transferBytes: number;
  updating: boolean;
  /** The saved journey once the consent click has bound the release and asked for the download. */
  journey: BoundJourney | null;
  /** Why starting stopped, when it did. */
  refusal: InstallStartRefusal | null;
  onDone: (target: SetupTarget) => void;
  onBack: () => void;
  /**
   * Retry an install the host reports stopped, with the consent this
   * browser saved for its release. Offered only when the install stopped or a
   * retry could not start.
   */
  onRetry?: () => void;
  /** Interval between status reads, in milliseconds; tests shorten it. */
  pollMs?: number;
}

type StepState = 'waiting' | 'active' | 'done' | 'failed';
interface Steps { verified: StepState; downloaded: StepState; prepared: StepState; activated: StepState }
type Failure =
  | { kind: 'detail'; step: keyof Steps; detail: string }
  | { kind: 'stopped'; step: 'downloaded' | 'prepared'; cause: 'download' | 'prepare' | 'interrupted' }
  | { kind: 'code'; step: keyof Steps; code: 'download-unconfirmed' | 'inventory' | 'activation-unconfirmed' | 'start-slow' | 'unreadable' }
  | { kind: 'code-with-value'; step: keyof Steps; code: 'activation-failed' | 'activation-refused' | 'start-failed'; value: string };

const POLL_MS = 2000;
const CONFIRM_ATTEMPTS = 15;
const START_ATTEMPTS = 90;
// Consecutive failed status reads before the page stops following. A failed
// read says nothing about the operation, so the journey is kept for later.
const UNREADABLE_ATTEMPTS = 30;
// Activation is submitted at most once per installation across remounts.
const activationInFlight = new Set<string>();

function sleep(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, milliseconds);
    signal.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('aborted', 'AbortError')); }, { once: true });
  });
}

/**
 * Follow one consented release from download to a running installation.
 *
 * The consent click already bound the release and asked for the download.
 * This view reads the host's retained operations and submits activation at
 * most once, recording its identity before sending; a lost response or a
 * page opened again reads that operation back and never sends it twice.
 */
export function CatalogInstallProgress({ title, publisher, sequence, transferBytes, updating, journey, refusal, onDone, onBack, onRetry, pollMs = POLL_MS }: CatalogInstallProgressProps) {
  const { t } = useSkulkTranslation();
  const [activate] = useActivateRuntimeReleaseMutation();
  const [readInstall] = useLazyGetRuntimeInstallationQuery();
  const [readOperation] = useLazyGetManagedOperationQuery();
  const [readRuntimes] = useLazyGetManagedRuntimesQuery();
  const [steps, setSteps] = useState<Steps>({ verified: 'active', downloaded: 'waiting', prepared: 'waiting', activated: 'waiting' });
  const [downloaded, setDownloaded] = useState(0);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [target, setTarget] = useState<SetupTarget | null>(null);
  // A retry follows the same install operation again, so the start time
  // tells one attempt from the next.
  const journeyKey = journey ? `${journey.pluginId}/${journey.installOperationId}/${journey.startedAt}` : null;

  useEffect(() => {
    if (!refusal) return;
    // A retry that could not start reads like a refused download: its
    // release was verified when it was first installed.
    setSteps((current) => refusal.stage === 'bind' ? { ...current, verified: 'failed' } : { ...current, verified: 'done', downloaded: 'failed' });
  }, [refusal]);

  useEffect(() => {
    if (!journey) return;
    const pluginId = journey.pluginId;
    const controller = new AbortController();
    const signal = controller.signal;
    const mark = (patch: Partial<Steps>) => { if (!signal.aborted) setSteps((current) => ({ ...current, ...patch })); };
    const fail = (value: Failure) => { if (!signal.aborted) { mark({ [value.step]: 'failed' }); setFailure(value); } };
    // Storage is the record of what was sent; read it at every decision.
    const saved = (): BoundJourney | null => readJourneys().find((item): item is BoundJourney => item.pluginId === pluginId && isBound(item)) ?? null;
    const findRuntime = async (): Promise<ManagedRuntime | null> =>
      (await readRuntimes(undefined, false).unwrap()).installations.find((runtime) => runtime.plugin_id === pluginId) ?? null;

    const follow = async () => {
      mark({ verified: 'done', downloaded: 'active' });
      let current = saved() ?? journey;
      if (current.activationOperationId === null) {
        let unconfirmed = 0;
        let unreadable = 0;
        for (;;) {
          let operation: RuntimeInstallation | null = null;
          // Only the host's answer can show the operation is missing: a 404
          // means the installation itself is gone, any other failure is unknown.
          let answered = false;
          try { operation = (await readInstall(pluginId, false).unwrap()).operation; answered = true; } catch (error) { answered = pluginRequestRefused(error) === 404; }
          if (signal.aborted) return;
          unreadable = answered ? 0 : unreadable + 1;
          if (operation && operation.request.operation_id === current.installOperationId) {
            setDownloaded(operation.downloaded_bytes);
            if (operation.state === 'staged') break;
            if (operation.state === 'recovery_required') {
              // The host never repeats a stopped install by itself. The journey
              // stays as the consent a retry of this release carries.
              markInterrupted(pluginId);
              const prepared = operation.error_code === 'installation_failed';
              if (prepared) mark({ downloaded: 'done' });
              fail({ kind: 'stopped', step: prepared ? 'prepared' : 'downloaded', cause: prepared ? 'prepare' : operation.error_code === 'download_failed' ? 'download' : 'interrupted' });
              return;
            }
            if (operation.state === 'staging') mark({ downloaded: 'done', prepared: 'active' });
          } else if (answered && ++unconfirmed >= CONFIRM_ATTEMPTS) {
            clearJourney(pluginId); fail({ kind: 'code', step: 'downloaded', code: 'download-unconfirmed' }); return;
          } else if (unreadable >= UNREADABLE_ATTEMPTS) {
            fail({ kind: 'code', step: 'downloaded', code: 'unreadable' }); return;
          }
          try { await sleep(pollMs, signal); } catch { return; }
        }
      }
      mark({ downloaded: 'done', prepared: 'done', activated: 'active' });
      let runtime: ManagedRuntime | null = null;
      let answered = false;
      for (let attempt = 0; attempt < UNREADABLE_ATTEMPTS && !answered; attempt += 1) {
        try { runtime = await findRuntime(); answered = true; } catch {
          try { await sleep(pollMs, signal); } catch { return; }
        }
      }
      if (signal.aborted) return;
      if (!answered) { fail({ kind: 'code', step: 'activated', code: 'unreadable' }); return; }
      // The host answered without this installation, so nothing is left to follow.
      if (!runtime) { clearJourney(pluginId); fail({ kind: 'code', step: 'activated', code: 'inventory' }); return; }
      const alreadyActive = runtime.enabled && runtime.selected_digest === current.runtimeDigest;
      current = saved() ?? current;
      if (!alreadyActive && current.activationOperationId === null && !activationInFlight.has(pluginId)) {
        activationInFlight.add(pluginId);
        current = { ...current, activationOperationId: randomHex32() };
        saveJourney(current);
        try {
          await activate({ pluginId, operationId: current.activationOperationId!, expectedRevision: runtime.selection_revision, runtimeDigest: current.runtimeDigest, rollback: false }).unwrap();
        } catch (error) {
          // Only a 4xx is a decided refusal; a 5xx, timeout or lost reply may
          // have landed, so the operation is read back below.
          const status = pluginRequestRefused(error);
          if (status !== null) {
            const detail = pluginRefusalDetail(error);
            clearJourney(pluginId); activationInFlight.delete(pluginId);
            fail(detail ? { kind: 'detail', step: 'activated', detail } : { kind: 'code-with-value', step: 'activated', code: 'activation-refused', value: String(status) });
            return;
          }
        } finally {
          activationInFlight.delete(pluginId);
        }
      }
      if (!alreadyActive && current.activationOperationId !== null) {
        let unconfirmed = 0;
        let unreadable = 0;
        for (;;) {
          let operation: ManagedOperation | null = null;
          let answered = false;
          try { operation = await readOperation({ pluginId, operationId: current.activationOperationId }, false).unwrap(); answered = true; } catch (error) { answered = pluginRequestRefused(error) === 404; }
          if (signal.aborted) return;
          unreadable = answered ? 0 : unreadable + 1;
          if (operation?.state === 'complete') break;
          if (operation && (operation.state === 'failed' || operation.state === 'recovery_required' || operation.state === 'superseded')) {
            clearJourney(pluginId); fail({ kind: 'code-with-value', step: 'activated', code: 'activation-failed', value: operation.error_code ?? operation.state }); return;
          }
          if (!operation && answered && ++unconfirmed >= CONFIRM_ATTEMPTS) { clearJourney(pluginId); fail({ kind: 'code', step: 'activated', code: 'activation-unconfirmed' }); return; }
          if (unreadable >= UNREADABLE_ATTEMPTS) { fail({ kind: 'code', step: 'activated', code: 'unreadable' }); return; }
          try { await sleep(pollMs, signal); } catch { return; }
        }
      }
      // A completed activation starts the plugin; wait until it is observed running.
      for (let attempt = 0; attempt < START_ATTEMPTS; attempt += 1) {
        try { runtime = await findRuntime(); } catch { runtime = null; }
        if (signal.aborted) return;
        const service = runtime?.service;
        if (service?.state === 'running' && service.active_digest === current.runtimeDigest) {
          clearJourney(pluginId);
          mark({ activated: 'done' });
          setTarget({ pluginId, title: current.title });
          return;
        }
        if (service?.state === 'failed' && service.error_code) {
          clearJourney(pluginId); fail({ kind: 'code-with-value', step: 'activated', code: 'start-failed', value: service.error_code }); return;
        }
        try { await sleep(pollMs, signal); } catch { return; }
      }
      // Activation completed; starting is the plugin's own business now, and
      // Installed follows it. Keeping the journey would hold this release.
      clearJourney(pluginId);
      fail({ kind: 'code', step: 'activated', code: 'start-slow' });
    };
    void follow();
    return () => controller.abort();
    // One follower per saved journey; the RTK triggers are stable and the
    // journey object is read from storage at every decision.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [journeyKey]);

  const failureText = (value: Failure): string => {
    if (value.kind === 'detail') return value.detail;
    if (value.kind === 'stopped') {
      return {
        download: t('plugins.catalog.stoppedDownload', 'The download stopped before it finished. Choose Retry to download the same release again and finish installing it. Its card under Installed offers the same Retry.'),
        prepare: t('plugins.catalog.stoppedPrepare', 'The download finished, but preparing its runtime failed. Choose Retry to prepare the same release again and finish installing it. Its card under Installed offers the same Retry.'),
        interrupted: t('plugins.catalog.stoppedInterrupted', 'The install was interrupted before it finished. Choose Retry to pick up the same release and finish installing it. Its card under Installed offers the same Retry.'),
      }[value.cause];
    }
    if (value.kind === 'code-with-value') {
      if (value.code === 'activation-refused') return t('plugins.catalog.activationRefused', 'The host refused the activation (HTTP {status}). The release stays downloaded; activate it from Installed.', { status: value.value });
      return value.code === 'activation-failed'
        ? t('plugins.catalog.activationFailed', 'Activation did not complete ({code}). Open Installed to see the retained operation.', { code: value.value })
        : t('plugins.catalog.startFailed', 'It was activated but did not start ({code}). Open Installed for the reason.', { code: value.value });
    }
    return {
      'download-unconfirmed': t('plugins.catalog.downloadUnconfirmed', 'The host never confirmed the download. Nothing was repeated. Review the release again to start over.'),
      inventory: t('plugins.catalog.inventoryUnavailable', 'The installation is not in the host’s inventory. Open Installed to check it.'),
      'activation-unconfirmed': t('plugins.catalog.activationUnconfirmed', 'The host never confirmed the activation. Nothing was repeated. Open Installed to check it.'),
      'start-slow': t('plugins.catalog.startSlow', 'It was activated but has not reported running yet. Open Installed to follow it.'),
      unreadable: t('plugins.catalog.statusUnreadable', 'This page lost contact with the host and stopped following. The install may still be running there: come back to Browse to pick it up.'),
    }[value.code];
  };
  const retryRefusalText = (value: InstallStartRefusal): string => value.code ? {
    'nothing-to-retry': t('plugins.catalog.retryNothing', 'The host has no stopped install for this plugin, so nothing was retried. Open Installed to see where it stands.'),
    'source-unavailable': t('plugins.catalog.retrySourceUnavailable', 'This plugin’s release source or its credential is not ready on the host, so nothing was retried. Restore it from the plugin’s details under Installed, then retry.'),
    unreadable: t('plugins.catalog.retryUnreadable', 'The host could not be read, so nothing was retried. Check the connection, then retry.'),
  }[value.code] : t('plugins.catalog.retryRefused', 'The host did not accept the retry. Nothing was repeated.');
  const refusalText = refusal?.unconfirmed ? t('plugins.catalog.bindUnconfirmed', 'The host did not confirm this release, so it may already be bound. Review and install it again to continue: the same installation is used, so nothing is duplicated.')
    : refusal ? (refusal.detail ?? (refusal.stage === 'retry' ? retryRefusalText(refusal) : refusal.stage === 'bind'
    ? t('plugins.catalog.bindRefused', 'The host did not accept this release. Nothing was installed. Read the catalog again and retry.')
    : t('plugins.catalog.downloadRefused', 'The host refused the download. Nothing was installed.'))
    + (refusal.status !== undefined && !refusal.detail ? ` (HTTP ${refusal.status})` : '')) : null;
  // A stopped install, or a retry that could not start, can be retried again
  // from here with the consent already given; an install the host does not
  // have cannot.
  const retryable = !!onRetry && (failure?.kind === 'stopped' || (refusal?.stage === 'retry' && refusal.code !== 'nothing-to-retry'));
  const message = refusalText ?? (failure ? failureText(failure) : null);
  const rows: { key: keyof Steps; label: string; detail: string }[] = [
    { key: 'verified', label: t('plugins.catalog.stepVerified', 'Signature and compatibility verified'), detail: publisher ? `${publisher} · ${t('plugins.catalog.release', 'release {sequence}', { sequence })}` : '' },
    { key: 'downloaded', label: t('plugins.catalog.stepDownloaded', 'Downloaded'), detail: steps.downloaded === 'done' ? formatMegabytes(transferBytes) : `${formatMegabytes(downloaded)} / ${formatMegabytes(transferBytes)}` },
    { key: 'prepared', label: t('plugins.catalog.stepPrepared', 'Runtime prepared'), detail: t('plugins.catalog.isolated', 'isolated environment') },
    { key: 'activated', label: t('plugins.catalog.stepActivated', 'Activated'), detail: steps.activated === 'done' ? t('plugins.catalog.running', 'running') : t('plugins.catalog.starting', 'starting') },
  ];
  return <Panel aria-labelledby="catalog-install-title">
    <h2 id="catalog-install-title">{updating ? t('plugins.catalog.updatingTitle', 'Updating {title}', { title }) : t('plugins.catalog.installingTitle', 'Installing {title}', { title })}</h2>
    <Lead>{target ? t('plugins.catalog.installedLead', '{title} is installed and running.', { title }) : t('plugins.catalog.leaveLead', 'You can leave this page. The download continues on the host, and this page picks up where it left off.')}</Lead>
    <StepList aria-live="polite">
      {rows.map((row) => <Step key={row.key} $state={steps[row.key]}>
        <Icon aria-hidden="true" $state={steps[row.key]}>{steps[row.key] === 'done' ? <FiCheck /> : steps[row.key] === 'failed' ? <FiX /> : steps[row.key] === 'active' ? <Spinner size={16} /> : null}</Icon>
        <span>{row.label}</span>
        <Detail>{steps[row.key] === 'waiting' ? '' : row.detail}</Detail>
      </Step>)}
    </StepList>
    {message ? <FailureBox role="alert">{message}</FailureBox> : null}
    <Actions>
      {target ? <Button variant="primary" onClick={() => onDone(target)}>{t('plugins.catalog.setItUp', 'Set it up')}</Button> : null}
      {retryable ? <Button variant="primary" onClick={onRetry}>{t('plugins.catalog.retryInstall', 'Retry')}</Button> : null}
      <Button variant="ghost" onClick={onBack}>{t('plugins.catalog.backToBrowse', 'Back to Browse')}</Button>
    </Actions>
  </Panel>;
}

const Panel = styled(Surface)`
  max-width: 640px; padding: 28px; display: flex; flex-direction: column; gap: 14px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
`;
const Lead = styled.p`margin: -6px 0 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14px; line-height: 1.5;`;
const StepList = styled.ol`list-style: none; margin: 4px 0 0; padding: 0; display: flex; flex-direction: column;`;
const Step = styled.li<{ $state: StepState }>`
  display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; align-items: center; gap: 12px; padding: 12px 4px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  > span { font-size: 14.5px; color: ${({ theme, $state }) => $state === 'waiting' ? theme.colors.textMuted : theme.colors.text}; }
`;
const Icon = styled.span<{ $state: StepState }>`
  display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 50%;
  border: 1px solid ${({ theme, $state }) => $state === 'done' ? theme.colors.borderHealthy : $state === 'failed' ? theme.colors.borderDanger : theme.colors.border};
  color: ${({ theme, $state }) => $state === 'done' ? theme.colors.healthy : $state === 'failed' ? theme.colors.error : theme.colors.textSecondary};
`;
const Detail = styled.span`font: 12px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.metadataText}; text-align: right;`;
const FailureBox = styled.p`
  margin: 0; padding: 12px 14px; border-radius: 10px; font-size: 13.5px; line-height: 1.5;
  border: 1px solid ${({ theme }) => theme.colors.borderDanger}; background: ${({ theme }) => theme.colors.errorBg}; color: ${({ theme }) => theme.colors.text};
`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px;`;
