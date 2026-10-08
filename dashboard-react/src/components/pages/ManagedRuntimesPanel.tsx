import { FiChevronRight, FiPlus } from 'react-icons/fi';
import { derivePluginHealth, installNeedsRetry, recordedServiceFailure, type PluginFilter } from './pluginHealth';
import type { PluginNodes } from '../../store/endpoints/plugins';
import { useMemo, useState, type ReactNode } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  installOperationIds, pluginRefusalDetail, pluginRequestRefused, useGetManagedRuntimesQuery, useGetManagedOperationQuery, useGetInstallOperationsQuery,
  useWithdrawManagedRuntimeMutation, useRecoverManagedOperationMutation, useActivateRuntimeReleaseMutation,
  useRegisterManagedRuntimeMutation, usePurgeManagedRuntimeMutation, useGetCatalogSourceQuery, useGetPluginCatalogQuery, useGetRuntimeSourceStatusQuery,
  type ManagedRuntime, type RuntimeInstallation,
} from '../../store/endpoints/plugins';
import { RightDrawer } from '../common/RightDrawer';
import { PluginSummaryCard } from '../common/PluginSummaryCard';
import { Button } from '../common/Button';
import { RuntimeReleasePanel } from './RuntimeReleasePanel';
import { RuntimeSourceForm } from './RuntimeSourceForm';
import { randomHex32 } from '../../utils/randomIds';
import { catalogOffers, clearJourney, readJourneys, unfinishedInstalls, type RetryRequest } from './catalogJourney';
import { StatusPill, type StatusTone } from '../common/Surfaces';
import { ActionRow, CardHeader, Meta, Muted, Notice, PluginCard, Section } from './pluginCardStyles';

const DrawerBody = styled.div`padding: 20px 24px 32px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; @media (max-width: 600px) { padding: 16px; }`;
// Labels the node cards below the runtime card, in the cards' own section-label style.
const GroupLabel = styled.h3`
  margin: 8px 0 -4px; font: 600 11px ${({ theme }) => theme.fonts.mono}; letter-spacing: .14em; text-transform: uppercase;
  color: ${({ theme }) => theme.colors.subtleText};
`;
const DetailsToggle = styled(Button)<{ $open: boolean }>`
  align-self: flex-start;
  svg { transition: transform 120ms; transform: rotate(${({ $open }) => $open ? '90deg' : '0deg'}); }
`;
const Facts = styled.dl`
  margin: 0; display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 10px 16px; align-items: center; font-size: 13px;
  dt { color: ${({ theme }) => theme.colors.textSecondary}; }
  dd { margin: 0; min-width: 0; color: ${({ theme }) => theme.colors.text}; }
  @media (max-width: 480px) { grid-template-columns: minmax(0, 1fr); dt { margin-top: 4px; } }
`;
const FactValue = styled.span`
  display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; min-width: 0;
  code { font: 12.5px ${({ theme }) => theme.fonts.mono}; overflow-wrap: anywhere; }
`;

/** A fingerprint or identifier: shown short, in full on hover, and copyable. */
function Fingerprint({ value, label, full = false }: { value: string; label: string; full?: boolean }) {
  const { t } = useSkulkTranslation();
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try { void navigator.clipboard?.writeText(value).then(() => setCopied(true), () => undefined); } catch { /* copying is a convenience */ }
  };
  return <FactValue>
    <code title={value}>{full ? value : value.slice(0, 12)}</code>
    <Button type="button" variant="outline" size="sm" aria-label={t('plugins.runtime.copyValue', 'Copy {label}', { label })} onClick={copy}>
      {copied ? t('plugins.runtime.copied', 'Copied') : t('plugins.runtime.copy', 'Copy')}
    </Button>
  </FactValue>;
}

/** Read server-retained operation references; reconnect never submits another mutation. */
function RuntimeControls({ runtime, installations, unavailable, nodes, details, nodeEvidence, filter, installOperation = null, onRetryInstall, onUpdate }: {
  nodeEvidence?: PluginNodes; filter: PluginFilter; runtime: ManagedRuntime; unavailable: boolean; nodes: string[]; details?: ReactNode;
  /** Every installation on this host, so an update is offered exactly as Browse offers it. */
  installations: ManagedRuntime[];
  /** Review the newer release of this bundle in Browse; without it no update is offered here. */
  onUpdate?: (bundleId: string) => void;
  /** The installation's retained install, when it was read. */
  installOperation?: RuntimeInstallation | null;
  /** Retry a stopped install through to running; without it the card opens the manual release controls. */
  onRetryInstall?: (request: RetryRequest) => void;
}) {
  const { t } = useSkulkTranslation();
  const [expanded, setExpanded] = useState(false);
  const [width, setWidth] = useState(640);
  const [notice, setNotice] = useState('');
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [releaseOpen, setReleaseOpen] = useState(false);
  // Drawer dismissal must not erase a request fence before server observation.
  const [releaseSubmitted, setReleaseSubmitted] = useState<string | null>(null);
  const [activationSubmitted, setActivationSubmitted] = useState<string | null>(null);
  // Terminal confirmation releases the fence even when the drawer is closed.
  if (activationSubmitted && runtime.operation_id === activationSubmitted &&
    ['complete', 'failed', 'superseded'].includes(runtime.operation_state ?? '')) {
    setActivationSubmitted(null);
  }
  const [detailsOpen, setDetailsOpen] = useState(false);
  // A decided refusal to start is shown on its own: no operation was created,
  // so the last completed one must not hide it.
  const [startRefusal, setStartRefusal] = useState('');
  const [withdraw, withdrawing] = useWithdrawManagedRuntimeMutation();
  const [activate, activating] = useActivateRuntimeReleaseMutation();
  const [recover, recovering] = useRecoverManagedOperationMutation();
  const [purge, purging] = usePurgeManagedRuntimeMutation();
  // Removal is armed from the menu and confirmed in the drawer: it is the
  // explicit end of an uninstall, and the retained state does not come back.
  const [removalArmed, setRemovalArmed] = useState(false);
  // Dismissing the drawer disarms it: the next opening asks again.
  const closeDrawer = () => { setExpanded(false); setRemovalArmed(false); };
  // A refused removal is said on its own: the uninstall it follows is complete,
  // and the lifecycle notice is hidden once an operation completes.
  const [purgeNotice, setPurgeNotice] = useState('');
  const operationId = submitted ?? runtime.operation_id;
  const operation = useGetManagedOperationQuery({ pluginId: runtime.plugin_id, operationId: operationId ?? '' }, {
    skip: !operationId,
    pollingInterval: 2000,
    skipPollingIfUnfocused: true,
  });
  const state = operation.currentData?.state ?? (operationId === runtime.operation_id ? runtime.operation_state : null);
  const stateLabel = state ? {
    accepted: t('plugins.operationAccepted', 'Accepted'),
    applying: t('plugins.operationApplying', 'Applying'),
    complete: t('plugins.operationComplete', 'Complete'),
    failed: t('plugins.operationFailed', 'Failed'),
    recovery_required: t('plugins.operationRecovery', 'Recovery needed'),
    superseded: t('plugins.operationSuperseded', 'Withdrawn by a later operation'),
  }[state] : t('plugins.runtimeStatusUnknown', 'Status unavailable');
  const pending = state === 'accepted' || state === 'applying';
  const withdrawable = state === 'recovery_required' && !!operation.currentData && !['disable', 'uninstall'].includes(operation.currentData.request.action);
  const confirmed = state === 'complete' || state === 'failed' || state === 'superseded' || withdrawable;
  const busy = withdrawing.isLoading || recovering.isLoading || purging.isLoading || activating.isLoading;
  const purgeRuntime = async () => {
    setPurgeNotice('');
    try {
      await purge(runtime.plugin_id).unwrap();
      // Nothing is left to retry, so the consent this browser kept for it goes too.
      clearJourney(runtime.plugin_id);
    } catch {
      setPurgeNotice(t('plugins.purgeRefused', 'Removal was refused. The plugin must be uninstalled, with no operation or release work under way.'));
    } finally {
      setRemovalArmed(false);
    }
  };
  const withdrawRuntime = async (action: 'disable' | 'uninstall') => {
    const id = randomHex32();
    setSubmitted(id);
    setNotice('');
    try {
      await withdraw({ action, pluginId: runtime.plugin_id, operationId: id, expectedRevision: runtime.selection_revision }).unwrap();
    } catch {
      setNotice(t('plugins.runtimeUncertain', 'The request did not return a confirmed result. Refresh operation status before taking another action.'));
    }
  };
  // Starting runs the release already selected again, with the permissions it
  // already holds, under the same operation fence as a withdrawal.
  const startRuntime = async () => {
    if (!runtime.selected_digest) return;
    const id = randomHex32();
    setSubmitted(id);
    setNotice('');
    setStartRefusal('');
    try {
      await activate({ pluginId: runtime.plugin_id, operationId: id, expectedRevision: runtime.selection_revision, runtimeDigest: runtime.selected_digest, rollback: false, acceptPermissions: false }).unwrap();
    } catch (error) {
      // Only a 4xx is decided: nothing was started, so the fence is released.
      if (pluginRequestRefused(error) === null) {
        setNotice(t('plugins.runtimeUncertain', 'The request did not return a confirmed result. Refresh operation status before taking another action.'));
        return;
      }
      setSubmitted(null);
      const detail = pluginRefusalDetail(error);
      setStartRefusal(detail ? `${t('plugins.runtime.startRefused', 'It was not started.')} ${t('plugins.hostSaid', 'The host said: {reason}', { reason: detail })}`
        : t('plugins.runtime.startRefusedGeneric', 'It was not started. Check your access, then refresh its status.'));
    }
  };
  const recoverOperation = async () => {
    if (!operationId) return;
    setNotice('');
    try {
      await recover({ pluginId: runtime.plugin_id, operationId }).unwrap();
    } catch {
      setNotice(t('plugins.runtimeRecoveryFailed', 'Recovery was refused. Check service health and your plugin permissions.'));
    }
  };
  const disableBlocked = runtime.uninstalled || (!runtime.enabled && !withdrawable) || unavailable || busy || pending || (state === 'recovery_required' && !withdrawable) || (!!submitted && !confirmed);
  const uninstallBlocked = (runtime.uninstalled && !withdrawable) || (!runtime.selected_digest && !withdrawable) || unavailable || busy || pending || (state === 'recovery_required' && !withdrawable) || (!!submitted && !confirmed);
  const bundleNames = [...new Set(nodeEvidence?.nodes.map(node => node.bundleId) ?? [])];
  // The signed title names the plugin even when nothing runs to report it.
  const release = runtime.release ?? null;
  // A first install that stopped has no selected release yet, so its retained
  // install names it: the catalog title this browser saw when it was
  // installed, else the signed bundle id. Never only the raw installation id.
  const retryNeeded = installNeedsRetry(runtime, installOperation);
  const installing = installOperation?.review ?? null;
  const bundleId = release?.bundle_id ?? installing?.bundle_id ?? null;
  const catalogTitle = bundleId ? readJourneys().find((journey) => journey.pluginId === runtime.plugin_id && journey.bundleId === bundleId)?.title ?? null : null;
  const name = release?.title ?? catalogTitle ?? (bundleNames.length === 1 ? bundleNames[0] : null) ?? release?.bundle_id ?? installing?.bundle_id ?? runtime.plugin_id;
  const retryInstall = () => {
    if (!retryNeeded) return;
    if (!onRetryInstall) { setReleaseOpen(true); setExpanded(true); return; }
    const review = installOperation.review;
    onRetryInstall({ pluginId: runtime.plugin_id, title: name, publisher: review.publisher, sequence: review.sequence, transferBytes: review.artifact_bytes, updating: runtime.selected_digest !== null });
  };
  // A first install that stopped never selected a release, so the host
  // removes it outright, as it does an uninstalled one: the way to drop a
  // stray installation instead of retrying it.
  const neverInstalled = retryNeeded && runtime.selected_digest === null && !runtime.enabled;
  const removable = !!runtime.uninstalled || neverInstalled;
  const retryReason = retryNeeded ? ({
    download_failed: t('plugins.retryDownloadFailed', 'Installing release {sequence} stopped while it downloaded. Retry downloads it again and finishes the install.', { sequence: installOperation.review.sequence }),
    installation_failed: t('plugins.retryPrepareFailed', 'Release {sequence} downloaded, but preparing its runtime failed. Retry prepares it again and finishes the install.', { sequence: installOperation.review.sequence }),
  } as Record<string, string>)[installOperation.error_code ?? ''] ?? t('plugins.retryInterrupted', 'Installing release {sequence} was interrupted before it finished. Retry picks it up and finishes the install.', { sequence: installOperation.review.sequence }) : null;
  const failure = runtime.uninstalled || !runtime.enabled ? null : recordedServiceFailure(runtime);
  const failureReason = failure === null ? null : ({
    verification_failed: t('plugins.failureVerification', 'This release was built for a different Skulk build, platform or dependency set, so it cannot run here. Install a release built for this host.'),
    owner_exited: t('plugins.failureOwnerExited', 'The plugin process stopped. Check its setup, then refresh.'),
    ownership_busy: t('plugins.failureOwnershipBusy', 'Another process is using this plugin. Wait a moment, then refresh.'),
    service_failed: t('plugins.failureService', 'The plugin service could not start.'),
  } as Record<string, string>)[failure] ?? t('plugins.failureOther', 'The plugin stopped with {code}.', { code: failure });
  const releaseNote = retryNeeded && !release ? t('plugins.notInstalledYet', 'Not installed yet') : runtime.uninstalled ? t('plugins.cleanupRetained', 'Cleanup state retained') : failure !== null ? t('plugins.notRunning', 'Not running') : runtime.stale || unavailable ? t('plugins.releaseStatusUnavailable', 'Release status unavailable')
    : runtime.service?.active_digest && runtime.service.active_digest === runtime.selected_digest ? t('plugins.releaseActive', 'Active')
    : runtime.service?.active_digest ? t('plugins.differentActiveRelease', 'Different release active')
    : t('plugins.noActiveRelease', 'None active');
  const openDetails = () => setExpanded(true);
  const openRelease = () => { setReleaseOpen(true); setDetailsOpen(true); setExpanded(true); };
  const category = derivePluginHealth(runtime, nodeEvidence, unavailable || !!operation.error, state, installOperation);
  const health = retryNeeded && !unavailable ? t('plugins.installNeedsRetry', 'Install needs a retry') : {
    healthy: t('plugins.healthy', 'Healthy'), attention: t('plugins.needsAttention', 'Needs attention'),
    uninstalled: t('plugins.uninstalled', 'Uninstalled'), unknown: t('plugins.healthUnknown', 'Status unavailable'),
    updating: t('plugins.updating', 'Updating'), disabled: t('plugins.disabled', 'Disabled'),
  }[category];
  // The newer fitting release Browse would offer as an update, read only while the drawer is open.
  const wantsUpdate = expanded && !!onUpdate && !!release && !runtime.uninstalled;
  const catalogSource = useGetCatalogSourceQuery(undefined, { skip: !wantsUpdate });
  const catalog = useGetPluginCatalogQuery(undefined, { skip: !wantsUpdate || !catalogSource.data?.configured });
  const update = useMemo(() => {
    if (!wantsUpdate || !catalog.data) return null;
    const offer = catalogOffers(catalog.data, installations.length > 0 ? installations : [runtime], unfinishedInstalls([runtime], { [runtime.plugin_id]: installOperation }))
      .find((item) => item.installed?.plugin_id === runtime.plugin_id);
    return offer?.state === 'update' ? offer.entry : null;
  }, [wantsUpdate, catalog.data, installations, runtime, installOperation]);
  const source = useGetRuntimeSourceStatusQuery(runtime.plugin_id, { skip: !expanded });
  // The release shown: the selected one, else the one a stopped first install was installing.
  const shownRelease = release ? { version: release.bundle_version, publisher: release.publisher, sequence: release.sequence }
    : installing ? { version: installing.version, publisher: installing.publisher, sequence: installing.sequence } : null;
  // Only where it came from: a publisher's release number is an identifier, kept under Details.
  const metaLine = [
    ...(source.currentData?.follows_store ? [t('plugins.runtime.fromStore', 'From the Foxlight capability store')]
      : source.currentData?.configured ? [t('plugins.runtime.fromPrivate', 'From a private source')] : []),
  ];
  const runtimeStatus: { label: string; tone: StatusTone } = runtime.uninstalled ? { label: t('plugins.runtime.uninstalled', 'Uninstalled'), tone: 'neutral' }
    : retryNeeded && !release ? { label: t('plugins.runtime.installStopped', 'Install stopped'), tone: 'live' }
      : pending ? { label: t('plugins.runtime.updating', 'Updating'), tone: 'live' }
        : failure !== null ? { label: t('plugins.runtime.failed', 'Failed'), tone: 'danger' }
          : !runtime.enabled ? { label: t('plugins.runtime.stopped', 'Stopped'), tone: 'neutral' }
            : runtime.stale || unavailable ? { label: t('plugins.runtime.unknown', 'Status unknown'), tone: 'neutral' }
              : runtime.service?.state === 'running' && runtime.service.active_digest === runtime.selected_digest ? { label: t('plugins.runtime.running', 'Running'), tone: 'healthy' }
                : { label: t('plugins.runtime.starting', 'Starting'), tone: 'live' };
  const startable = !runtime.uninstalled && !runtime.enabled && !withdrawable && !!runtime.selected_digest;
  const startBlocked = !startable || unavailable || busy || pending || state === 'recovery_required' || (!!submitted && !confirmed);
  const reinstallBlocked = !runtime.uninstalled || !runtime.selected_digest || unavailable || busy || pending || state === 'recovery_required' || (!!submitted && !confirmed);
  // One primary at a time: an update, else a retry, else starting what is stopped.
  const primaryTaken = (!!update && !!onUpdate) || retryNeeded;
  const explanation = runtime.uninstalled
    ? (runtime.selected_digest
      ? t('plugins.runtime.uninstalledHelp', 'Reinstall starts release {version} again with what it kept. Remove everything deletes what it kept; after that you can install it fresh from Browse.', { version: shownRelease?.version ?? '' })
      : t('plugins.runtime.uninstalledRemoveHelp', 'Remove everything deletes what it kept; after that you can install it fresh from Browse.'))
    : runtime.enabled ? t('plugins.runtime.stopHelp', 'Stop plugin stops all of its capabilities until you start it again. Its settings and data stay.')
      : startable ? t('plugins.runtime.startHelp', 'Start plugin runs release {version} again with its settings.', { version: shownRelease?.version ?? '' }) : null;
  return <>
    <div hidden={filter !== 'all' && filter !== category}>
    <PluginSummaryCard name={name} pluginId={runtime.plugin_id}
      description={retryReason ?? failureReason ?? (nodeEvidence?.nodes.length ? t('plugins.nodeCount', 'Installed capability nodes: {count}', { count: nodeEvidence.nodes.length }) : undefined)}
      health={health} tone={category === 'healthy' ? 'healthy' : category === 'attention' ? 'live' : category === 'updating' ? 'live' : 'neutral'}
      release={release ? t('plugins.releaseIdentity', '{version} ({sequence}) from {publisher}', { version: release.bundle_version, sequence: release.sequence, publisher: release.publisher })
        : retryNeeded ? t('plugins.releaseIdentity', '{version} ({sequence}) from {publisher}', { version: installOperation.review.version, sequence: installOperation.review.sequence, publisher: installOperation.review.publisher })
        : runtime.selected_digest?.slice(0, 12) ?? t('plugins.noRelease', 'None selected')} releaseNote={releaseNote} nodes={nodes}
      muted={runtime.uninstalled}
      // A stopped install's one next step is its retry, so the card leads with it.
      onOpen={retryNeeded ? retryInstall : openDetails} primaryLabel={retryNeeded ? t('plugins.retryInstall', 'Retry') : undefined} primaryVariant={retryNeeded ? 'primary' : 'outline'}
      actions={[
        ...(retryNeeded ? [{ id: 'retry', label: t('plugins.retryInstallMenu', 'Retry install'), disabled: unavailable, onSelect: retryInstall }] : []),
        { id: 'configure', label: t('plugins.configureSettings', 'Configure settings'), onSelect: openDetails },
        { id: 'release', label: t('plugins.runtime.chooseReleaseMenu', 'Choose another release…'), onSelect: openRelease },
        { id: 'refresh', label: t('plugins.runtime.refreshStatus', 'Refresh status'), disabled: !operationId || operation.isFetching || busy, onSelect: () => { void operation.refetch(); } },
        { id: 'disable', label: t('plugins.runtime.stop', 'Stop plugin'), separatorBefore: true, disabled: disableBlocked, onSelect: () => { setExpanded(true); void withdrawRuntime('disable'); } },
        { id: 'uninstall', label: t('plugins.uninstallMenu', 'Uninstall plugin…'), danger: true, disabled: uninstallBlocked, onSelect: () => { setExpanded(true); void withdrawRuntime('uninstall'); } },
        { id: 'purge', label: neverInstalled ? t('plugins.removeStoppedMenu', 'Remove this installation…') : t('plugins.purgeMenu', 'Remove uninstalled plugin…'), danger: true, disabled: !removable || pending || busy || unavailable, onSelect: () => { setRemovalArmed(true); setExpanded(true); } },
      ]} />
    </div>
    <RightDrawer open={expanded} onClose={closeDrawer} title={name} ariaLabel={t('plugins.runtimeDetails', 'Runtime details')}
      width={width} minWidth={360} maxWidth={900} onWidthChange={setWidth} closeLabel={t('common.close', 'Close')} resizeLabel={t('plugins.resize', 'Resize plugin details')}>
    <DrawerBody>
    <PluginCard aria-label={t('plugins.runtime.label', 'Plugin release and runtime')}>
      <CardHeader>
        <div>
          <h3>{shownRelease ? t('plugins.runtime.release', 'Release {version} from {publisher}', { version: shownRelease.version, publisher: shownRelease.publisher }) : t('plugins.runtime.noRelease', 'No release installed')}</h3>
          {metaLine.length > 0 ? <Meta>{metaLine.join(' · ')}</Meta> : null}
        </div>
        <StatusPill tone={runtimeStatus.tone}>{runtimeStatus.label}</StatusPill>
      </CardHeader>
      <Section>
        {runtime.uninstalled ? <Muted>{t('plugins.runtime.uninstalledLead', 'It is uninstalled. Its settings, credentials and records are kept, so it can come back as it was.')}</Muted> : null}
        {failureReason ? <Notice $problem role="status">{failureReason}</Notice> : null}
        {retryReason ? <Notice role="status">{retryReason}</Notice> : null}
        {/* A stopped or uninstalled plugin has no service to observe, so staleness only matters while it should run. */}
        {(unavailable || (runtime.stale && runtime.enabled)) && !failureReason && !runtime.uninstalled ? <Notice role="status">{t('plugins.runtimeStale', 'Service health is stale or unavailable.')}</Notice> : null}
        {runtime.error_code ? <Notice $problem role="status">{t('plugins.runtimeNeedsAttention', 'The local service needs attention.')}</Notice> : null}
        {withdrawable ? <Notice role="status">{t('plugins.runtime.withdrawPending', 'Stop plugin or Uninstall withdraws this pending change without running its release. Its history and cleanup records are kept.')}</Notice> : null}
        {operation.error ? <Notice role="status">{t('plugins.runtimeReadFailed', 'Operation status could not be read. The original request has not been resubmitted.')}</Notice> : null}
        <ActionRow>
          {update && onUpdate && !retryNeeded ? <Button type="button" variant="solid" disabled={busy || pending} onClick={() => onUpdate(update.bundle_id)}>{t('plugins.runtime.updateTo', 'Update to {version}', { version: update.bundle_version })}</Button> : null}
          {retryNeeded ? <Button type="button" variant="solid" disabled={unavailable} onClick={retryInstall}>{t('plugins.retryInstallMenu', 'Retry install')}</Button> : null}
          {runtime.uninstalled && runtime.selected_digest ? <Button type="button" variant="solid" disabled={reinstallBlocked} onClick={() => void startRuntime()}>{t('plugins.runtime.reinstall', 'Reinstall')}</Button> : null}
          {!runtime.uninstalled && (runtime.enabled || withdrawable) ? <Button type="button" variant="outline" disabled={disableBlocked} onClick={() => void withdrawRuntime('disable')}>{t('plugins.runtime.stop', 'Stop plugin')}</Button> : null}
          {startable ? <Button type="button" variant={primaryTaken ? 'outline' : 'solid'} disabled={startBlocked} onClick={() => void startRuntime()}>{t('plugins.runtime.start', 'Start plugin')}</Button> : null}
          {(!runtime.uninstalled && runtime.selected_digest) || withdrawable ? <Button type="button" variant="danger" disabled={uninstallBlocked} onClick={() => void withdrawRuntime('uninstall')}>{t('plugins.runtime.uninstall', 'Uninstall')}</Button> : null}
          {state === 'recovery_required' ? <Button type="button" variant="outline" disabled={unavailable || busy} onClick={() => void recoverOperation()}>{t('plugins.recoverRuntime', 'Recover local operation')}</Button> : null}
          {removable ? <Button type="button" variant="danger" disabled={pending || busy || unavailable} onClick={() => { if (removalArmed) { void purgeRuntime(); } else { setRemovalArmed(true); } }}>
            {removalArmed ? t('plugins.purgeConfirm', 'Remove now') : neverInstalled ? t('plugins.removeStopped', 'Remove this installation') : t('plugins.runtime.removeEverything', 'Remove everything')}
          </Button> : null}
        </ActionRow>
        {explanation ? <Muted>{explanation}</Muted> : null}
        {removable && removalArmed ? <Notice $problem role="status">{neverInstalled
          ? t('plugins.removeStoppedWarning', 'Removing deletes this installation and everything its stopped install kept: the download, records and source settings. Nothing was ever activated. Install it again from Browse if you want it later.')
          : t('plugins.purgeWarning', 'Removing deletes everything this uninstalled plugin retained: records, staged releases, credentials and cleanup state. It cannot be reinstalled from this installation afterwards.')}</Notice> : null}
        {purgeNotice ? <Notice $problem role="status">{purgeNotice}</Notice> : null}
        {notice && state !== 'complete' ? <Notice role="status">{notice}</Notice> : null}
        {startRefusal ? <Notice $problem role="status">{startRefusal}</Notice> : null}
      </Section>
      <Section>
        <DetailsToggle type="button" variant="outline" size="sm" $open={detailsOpen} aria-expanded={detailsOpen} aria-controls={`${runtime.plugin_id}-details`} onClick={() => setDetailsOpen(!detailsOpen)}>
          <FiChevronRight aria-hidden="true" />{t('plugins.runtime.details', 'Details')}
        </DetailsToggle>
        {detailsOpen ? <div id={`${runtime.plugin_id}-details`} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Facts>
            {shownRelease ? <>
              <dt>{t('plugins.runtime.releaseNumber', 'Release number')}</dt>
              <dd><FactValue><code>{shownRelease.sequence}</code></FactValue></dd>
            </> : null}
            <dt>{t('plugins.runtime.installation', 'Installation')}</dt>
            <dd><Fingerprint value={runtime.plugin_id} label={t('plugins.runtime.installation', 'Installation')} full /></dd>
            <dt>{t('plugins.selectedRelease', 'Selected release')}</dt>
            <dd>{runtime.selected_digest ? <Fingerprint value={runtime.selected_digest} label={t('plugins.selectedRelease', 'Selected release')} /> : t('plugins.noRelease', 'None selected')}</dd>
            <dt>{t('plugins.activeRelease', 'Active release')}</dt>
            <dd>{runtime.service?.active_digest ? <Fingerprint value={runtime.service.active_digest} label={t('plugins.activeRelease', 'Active release')} /> : t('plugins.noActiveRelease', 'None active')}</dd>
            <dt>{t('plugins.runtimeOperation', 'Local operation')}</dt>
            <dd><FactValue>
              <span role="status">{operationId ? stateLabel : t('plugins.runtime.noOperation', 'None')}</span>
              {operationId ? <Button type="button" variant="outline" size="sm" disabled={operation.isFetching || busy} onClick={() => void operation.refetch()}>{t('plugins.runtime.refreshStatus', 'Refresh status')}</Button> : null}
            </FactValue></dd>
          </Facts>
          <Muted>{t('plugins.runtime.retainedData', 'Stop plugin and Uninstall both keep its settings, credentials, records and recovery files, so starting or reinstalling it later picks up where it left off. Uninstalling is not a data purge: to delete everything it kept, uninstall it, then choose Remove everything.')}</Muted>
          <Button type="button" variant="outline" style={{ alignSelf: 'flex-start' }} aria-expanded={releaseOpen} onClick={() => setReleaseOpen(!releaseOpen)}>
            {releaseOpen ? t('plugins.runtime.hideReleaseChoice', 'Hide release choice') : t('plugins.runtime.chooseRelease', 'Choose another release')}
          </Button>
          {releaseOpen ? <RuntimeReleasePanel runtime={runtime} ownership={{ submitted: releaseSubmitted, setSubmitted: setReleaseSubmitted, activationSubmitted, setActivationSubmitted }} /> : null}
        </div> : null}
      </Section>
    </PluginCard>
    {details ? <><GroupLabel>{t('plugins.capabilityNodes', 'Capability nodes')}</GroupLabel>{details}</> : null}
  </DrawerBody></RightDrawer></>;
}

/** Observe independently supervised runtimes even when no plugin child is available. */
export function ManagedRuntimesPanel({ nodeNames = () => [], renderDetails, nodeEvidence, renderHeader, filter = 'all', onRetryInstall, onUpdate }: {
  /** Retry a stopped install through to running, as Browse does for a catalog install. */
  onRetryInstall?: (request: RetryRequest) => void;
  /** Review a newer release of an installed bundle in Browse; without it no update is offered in the drawer. */
  onUpdate?: (bundleId: string) => void;
  /** Page composition may place the existing registration action in its header. */
  renderHeader?: (registrationAction: ReactNode) => ReactNode;
  /** Fresh node evidence for derived card health. */
  nodeEvidence?: (pluginId: string) => PluginNodes | undefined;
  /** Hide cards while retaining their request ownership. */
  filter?: PluginFilter;
  /** Installed node labels associated with one runtime. */
  nodeNames?: (pluginId: string) => string[];
  /** Existing fenced node workflows to compose inside runtime details. */
  renderDetails?: (pluginId: string) => ReactNode;
} = {}) {
  const { t } = useSkulkTranslation();
  const query = useGetManagedRuntimesQuery(undefined, { pollingInterval: 5000, skipPollingIfUnfocused: true });
  // The inventory does not say whether an install stopped; each installation's
  // retained install does, read at the same cadence.
  const operationIds = installOperationIds(query.data?.installations);
  const installs = useGetInstallOperationsQuery(operationIds, { skip: operationIds.length === 0, pollingInterval: 5000, skipPollingIfUnfocused: true });
  const [register, registering] = useRegisterManagedRuntimeMutation();
  const [setupId, setSetupId] = useState<string | null>(null);
  const [registrationUncertain, setRegistrationUncertain] = useState(false);
  const addPlugin = async () => {
    const id = setupId ?? `managed.${randomHex32()}`;
    setSetupId(id);
    setRegistrationUncertain(false);
    try { await register(id).unwrap(); }
    catch { setRegistrationUncertain(true); }
  };
  const registrationAction = <Button type="button" disabled={registering.isLoading || !!query.error || query.isLoading || !!setupId} onClick={() => void addPlugin()}><FiPlus aria-hidden />{t('plugins.addManagedPlugin', 'Add plugin')}</Button>;
  return <section aria-label={t('plugins.managedRuntimes', 'Managed runtimes')}>
    {renderHeader ? renderHeader(registrationAction) : <><h2>{t('plugins.managedRuntimes', 'Managed runtimes')}</h2>{registrationAction}</>}
    {registrationUncertain ? <p role="status">{t('plugins.registrationUncertain', 'Registration was not confirmed. Refresh source status to check the retained installation before continuing.')}</p> : null}
    {registrationUncertain ? <Button type="button" disabled={registering.isLoading} onClick={() => void addPlugin()}>{t('plugins.retryRegistration', 'Retry the same registration')}</Button> : null}
    {setupId && !registering.isLoading ? <>
      <RuntimeSourceForm pluginId={setupId} onSaved={() => { setSetupId(null); setRegistrationUncertain(false); }} />
      <Button type="button" onClick={() => { setSetupId(null); setRegistrationUncertain(false); }}>{t('plugins.closeSourceSetup', 'Close source setup')}</Button>
    </> : null}
    {query.isLoading ? <p>{t('plugins.loadingRuntimes', 'Loading local services…')}</p> : null}
    {query.error ? <InventoryNotice role="status"><h2>{t('plugins.inventoryUnavailable', 'Plugin inventory unavailable')}</h2><p>{t('plugins.managerUnavailable', 'Local runtime management is unavailable. Check local service setup and your plugin permissions.')}</p><Button disabled={query.isFetching} onClick={() => void query.refetch()}>{t('plugins.retryInventory', 'Retry inventory')}</Button></InventoryNotice> : null}
    {!query.error && query.data?.installations.length === 0 ? <p>{t('plugins.noManagedRuntimes', 'No managed runtimes are installed.')}</p> : null}
    {query.data?.installations.map((runtime) => <RuntimeControls key={runtime.plugin_id} runtime={runtime} filter={filter} nodeEvidence={nodeEvidence?.(runtime.plugin_id)} unavailable={!!query.error} nodes={nodeNames(runtime.plugin_id)} details={renderDetails?.(runtime.plugin_id)}
      installOperation={installs.data?.[runtime.plugin_id] ?? null} onRetryInstall={onRetryInstall} installations={query.data?.installations ?? []} onUpdate={onUpdate} />)}
  </section>;
}

const InventoryNotice = styled.div`
  padding: 24px; margin: 10px 0 20px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px; background: ${({ theme }) => theme.colors.surface};
  h2 { font-size: 16px; margin-bottom: 8px; } p { font-size: 14px; line-height: 1.6; color: ${({ theme }) => theme.colors.textSecondary}; margin-bottom: 16px; }
`;
