import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { FiCheck, FiX } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { Surface } from '../common/Surfaces';
import {
  pluginRefusalDetail, useActivateRuntimeReleaseMutation, useLazyGetManagedOperationQuery, useLazyGetManagedRuntimesQuery,
  useLazyGetRuntimeInstallationQuery, type ManagedRuntime,
} from '../../store/endpoints/plugins';
import { randomHex32 } from '../../utils/randomIds';
import { clearJourney, formatMegabytes, readJourneys, saveJourney, type InstallJourney, type InstallStartRefusal } from './catalogJourney';

/** What the setup checklist needs once the release is running. */
export interface SetupTarget { pluginId: string; title: string; descriptors: string[] }

/** Props for the install progress view. */
export interface CatalogInstallProgressProps {
  title: string;
  publisher: string;
  sequence: number;
  transferBytes: number;
  updating: boolean;
  /** The saved journey once the consent click has bound the release and asked for the download. */
  journey: InstallJourney | null;
  /** Why starting stopped, when it did. */
  refusal: InstallStartRefusal | null;
  onDone: (target: SetupTarget) => void;
  onBack: () => void;
}

type StepState = 'waiting' | 'active' | 'done' | 'failed';
interface Steps { verified: StepState; downloaded: StepState; prepared: StepState; activated: StepState }
type Failure =
  | { kind: 'detail'; step: keyof Steps; detail: string }
  | { kind: 'code'; step: keyof Steps; code: 'recovery' | 'download-unconfirmed' | 'inventory' | 'activation-unconfirmed' | 'start-slow' }
  | { kind: 'code-with-value'; step: keyof Steps; code: 'activation-failed' | 'start-failed'; value: string };

const POLL_MS = 2000;
const CONFIRM_ATTEMPTS = 15;
const START_ATTEMPTS = 90;
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
export function CatalogInstallProgress({ title, publisher, sequence, transferBytes, updating, journey, refusal, onDone, onBack }: CatalogInstallProgressProps) {
  const { t } = useSkulkTranslation();
  const [activate] = useActivateRuntimeReleaseMutation();
  const [readInstall] = useLazyGetRuntimeInstallationQuery();
  const [readOperation] = useLazyGetManagedOperationQuery();
  const [readRuntimes] = useLazyGetManagedRuntimesQuery();
  const [steps, setSteps] = useState<Steps>({ verified: 'active', downloaded: 'waiting', prepared: 'waiting', activated: 'waiting' });
  const [downloaded, setDownloaded] = useState(0);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [target, setTarget] = useState<SetupTarget | null>(null);
  const journeyKey = journey ? `${journey.pluginId}/${journey.installOperationId}` : null;

  useEffect(() => {
    if (!refusal) return;
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
    const saved = (): InstallJourney | null => readJourneys().find((item) => item.pluginId === pluginId) ?? null;
    const findRuntime = async (): Promise<ManagedRuntime | null> =>
      (await readRuntimes(undefined, false).unwrap()).installations.find((runtime) => runtime.plugin_id === pluginId) ?? null;

    const follow = async () => {
      mark({ verified: 'done', downloaded: 'active' });
      let current = saved() ?? journey;
      if (current.activationOperationId === null) {
        let unconfirmed = 0;
        for (;;) {
          let operation;
          try { operation = (await readInstall(pluginId, false).unwrap()).operation; } catch { operation = null; }
          if (signal.aborted) return;
          if (operation && operation.request.operation_id === current.installOperationId) {
            setDownloaded(operation.downloaded_bytes);
            if (operation.state === 'staged') break;
            if (operation.state === 'recovery_required') { clearJourney(pluginId); fail({ kind: 'code', step: 'downloaded', code: 'recovery' }); return; }
            if (operation.state === 'staging') mark({ downloaded: 'done', prepared: 'active' });
          } else if (++unconfirmed >= CONFIRM_ATTEMPTS) {
            clearJourney(pluginId); fail({ kind: 'code', step: 'downloaded', code: 'download-unconfirmed' }); return;
          }
          try { await sleep(POLL_MS, signal); } catch { return; }
        }
      }
      mark({ downloaded: 'done', prepared: 'done', activated: 'active' });
      let runtime: ManagedRuntime | null;
      try { runtime = await findRuntime(); } catch { runtime = null; }
      if (signal.aborted) return;
      if (!runtime) { fail({ kind: 'code', step: 'activated', code: 'inventory' }); return; }
      const alreadyActive = runtime.enabled && runtime.selected_digest === current.runtimeDigest;
      current = saved() ?? current;
      if (!alreadyActive && current.activationOperationId === null && !activationInFlight.has(pluginId)) {
        activationInFlight.add(pluginId);
        current = { ...current, activationOperationId: randomHex32() };
        saveJourney(current);
        try {
          await activate({ pluginId, operationId: current.activationOperationId!, expectedRevision: runtime.selection_revision, runtimeDigest: current.runtimeDigest, rollback: false }).unwrap();
        } catch (error) {
          const detail = pluginRefusalDetail(error);
          if (detail) { clearJourney(pluginId); activationInFlight.delete(pluginId); fail({ kind: 'detail', step: 'activated', detail }); return; }
        } finally {
          activationInFlight.delete(pluginId);
        }
      }
      if (!alreadyActive && current.activationOperationId !== null) {
        let unconfirmed = 0;
        for (;;) {
          let operation;
          try { operation = await readOperation({ pluginId, operationId: current.activationOperationId }, false).unwrap(); } catch { operation = null; }
          if (signal.aborted) return;
          if (operation?.state === 'complete') break;
          if (operation && (operation.state === 'failed' || operation.state === 'recovery_required' || operation.state === 'superseded')) {
            clearJourney(pluginId); fail({ kind: 'code-with-value', step: 'activated', code: 'activation-failed', value: operation.error_code ?? operation.state }); return;
          }
          if (!operation && ++unconfirmed >= CONFIRM_ATTEMPTS) { clearJourney(pluginId); fail({ kind: 'code', step: 'activated', code: 'activation-unconfirmed' }); return; }
          try { await sleep(POLL_MS, signal); } catch { return; }
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
          setTarget({ pluginId, title: current.title, descriptors: current.descriptors });
          return;
        }
        if (service?.state === 'failed' && service.error_code) {
          clearJourney(pluginId); fail({ kind: 'code-with-value', step: 'activated', code: 'start-failed', value: service.error_code }); return;
        }
        try { await sleep(POLL_MS, signal); } catch { return; }
      }
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
    if (value.kind === 'code-with-value') {
      return value.code === 'activation-failed'
        ? t('plugins.catalog.activationFailed', 'Activation did not complete ({code}). Open Installed to see the retained operation.', { code: value.value })
        : t('plugins.catalog.startFailed', 'It was activated but did not start ({code}). Open Installed for the reason.', { code: value.value });
    }
    return {
      recovery: t('plugins.catalog.downloadRecovery', 'The download stopped and needs recovery. Open Installed to retry it; nothing is repeated automatically.'),
      'download-unconfirmed': t('plugins.catalog.downloadUnconfirmed', 'The host never confirmed the download. Nothing was repeated. Review the release again to start over.'),
      inventory: t('plugins.catalog.inventoryUnavailable', 'The installation is not in the host’s inventory. Open Installed to check it.'),
      'activation-unconfirmed': t('plugins.catalog.activationUnconfirmed', 'The host never confirmed the activation. Nothing was repeated. Open Installed to check it.'),
      'start-slow': t('plugins.catalog.startSlow', 'It was activated but has not reported running yet. Open Installed to follow it.'),
    }[value.code];
  };
  const refusalText = refusal ? (refusal.detail ?? (refusal.stage === 'bind'
    ? t('plugins.catalog.bindRefused', 'The host did not accept this release. Nothing was installed. Read the catalog again and retry.')
    : t('plugins.catalog.downloadRefused', 'The host refused the download. Nothing was installed.'))) : null;
  const message = refusalText ?? (failure ? failureText(failure) : null);
  const rows: { key: keyof Steps; label: string; detail: string }[] = [
    { key: 'verified', label: t('plugins.catalog.stepVerified', 'Signature and compatibility verified'), detail: `${publisher} · ${t('plugins.catalog.release', 'release {sequence}', { sequence })}` },
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
