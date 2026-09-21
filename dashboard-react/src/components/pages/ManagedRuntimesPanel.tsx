import { FiPlus } from 'react-icons/fi';
import { derivePluginHealth, type PluginFilter } from './pluginHealth';
import type { PluginNodes } from '../../store/endpoints/plugins';
import { useState, type ReactNode } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  useGetManagedRuntimesQuery, useGetManagedOperationQuery,
  useWithdrawManagedRuntimeMutation, useRecoverManagedOperationMutation,
  useRegisterManagedRuntimeMutation, usePurgeManagedRuntimeMutation,
  type ManagedRuntime,
} from '../../store/endpoints/plugins';
import { RightDrawer } from '../common/RightDrawer';
import { PluginSummaryCard } from '../common/PluginSummaryCard';
import { Button } from '../common/Button';
import { RuntimeReleasePanel } from './RuntimeReleasePanel';
import { RuntimeSourceForm } from './RuntimeSourceForm';

const RuntimeCard = styled.article`
  margin: 0; padding: 24px; border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md}; background: ${({ theme }) => theme.colors.surface};
  overflow-wrap: anywhere;
  border: 0; background: transparent;
  > h3 { font-size: 16px; margin-bottom: 8px; }
  > p { font-size: 13px; line-height: 1.55; color: ${({ theme }) => theme.colors.textSecondary}; margin: 8px 0; }
  > button { margin-top: 16px; }
  > article { margin: 20px 0; }
  @media(max-width: 600px) { padding: 20px 16px; }
`;
const RuntimeIdentity = styled.p`font-family: ${({ theme }) => theme.fonts.mono};`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;`;

/** Read server-retained operation references; reconnect never submits another mutation. */
function RuntimeControls({ runtime, unavailable, nodes, details, nodeEvidence, filter }: { nodeEvidence?: PluginNodes; filter: PluginFilter; runtime: ManagedRuntime; unavailable: boolean; nodes: string[]; details?: ReactNode }) {
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
  const [withdraw, withdrawing] = useWithdrawManagedRuntimeMutation();
  const [recover, recovering] = useRecoverManagedOperationMutation();
  const [purge, purging] = usePurgeManagedRuntimeMutation();
  // Removal is armed from the menu and confirmed in the drawer: it is the
  // explicit end of an uninstall, and the retained state does not come back.
  const [removalArmed, setRemovalArmed] = useState(false);
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
  const busy = withdrawing.isLoading || recovering.isLoading || purging.isLoading;
  const purgeRuntime = async () => {
    setNotice('');
    try {
      await purge(runtime.plugin_id).unwrap();
    } catch {
      setNotice(t('plugins.purgeRefused', 'Removal was refused. The plugin must be uninstalled, with no operation or release download under way.'));
    } finally {
      setRemovalArmed(false);
    }
  };
  const withdrawRuntime = async (action: 'disable' | 'uninstall') => {
    const id = crypto.randomUUID().replaceAll('-', '');
    setSubmitted(id);
    setNotice('');
    try {
      await withdraw({ action, pluginId: runtime.plugin_id, operationId: id, expectedRevision: runtime.selection_revision }).unwrap();
    } catch {
      setNotice(t('plugins.runtimeUncertain', 'The request did not return a confirmed result. Refresh operation status before taking another action.'));
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
  const name = bundleNames.length === 1 ? bundleNames[0] : runtime.plugin_id;
  const releaseNote = runtime.uninstalled ? t('plugins.cleanupRetained', 'Cleanup state retained') : runtime.stale || unavailable ? t('plugins.releaseUnavailable', 'Release status unavailable')
    : runtime.service?.active_digest && runtime.service.active_digest === runtime.selected_digest ? t('plugins.releaseActive', 'Active')
    : runtime.service?.active_digest ? t('plugins.differentActiveRelease', 'Different release active')
    : t('plugins.noActiveRelease', 'None active');
  const openDetails = () => setExpanded(true);
  const openRelease = () => { setReleaseOpen(true); setExpanded(true); };
  const category = derivePluginHealth(runtime, nodeEvidence, unavailable || !!operation.error, state);
  const health = {
    healthy: t('plugins.healthy', 'Healthy'), attention: t('plugins.needsAttention', 'Needs attention'),
    uninstalled: t('plugins.uninstalled', 'Uninstalled'), unknown: t('plugins.healthUnknown', 'Status unavailable'),
    updating: t('plugins.updating', 'Updating'), disabled: t('plugins.disabled', 'Disabled'),
  }[category];
  return <>
    <div hidden={filter !== 'all' && filter !== category}>
    <PluginSummaryCard name={name} pluginId={runtime.plugin_id}
      description={nodeEvidence?.nodes.length ? t('plugins.nodeCount', 'Installed capability nodes: {count}', { count: nodeEvidence.nodes.length }) : undefined}
      health={health} tone={category === 'healthy' ? 'healthy' : category === 'attention' ? 'live' : category === 'updating' ? 'live' : 'neutral'}
      release={runtime.selected_digest?.slice(0, 12) ?? t('plugins.noRelease', 'None selected')} releaseNote={releaseNote} nodes={nodes}
      muted={runtime.uninstalled} onOpen={openDetails} actions={[
        { id: 'configure', label: t('plugins.configureSettings', 'Configure settings'), onSelect: openDetails },
        { id: 'release', label: t('plugins.installReleaseMenu', 'Install a release…'), onSelect: openRelease },
        { id: 'refresh', label: t('plugins.refreshOperation', 'Refresh operation status'), disabled: !operationId || operation.isFetching || busy, onSelect: () => { void operation.refetch(); } },
        { id: 'disable', label: t('plugins.disableRuntime', 'Disable runtime'), separatorBefore: true, disabled: disableBlocked, onSelect: () => { setExpanded(true); void withdrawRuntime('disable'); } },
        { id: 'uninstall', label: t('plugins.uninstallMenu', 'Uninstall plugin…'), danger: true, disabled: uninstallBlocked, onSelect: () => { setExpanded(true); void withdrawRuntime('uninstall'); } },
        { id: 'purge', label: t('plugins.purgeMenu', 'Remove uninstalled plugin…'), danger: true, disabled: !runtime.uninstalled || pending || busy || unavailable, onSelect: () => { setRemovalArmed(true); setExpanded(true); } },
      ]} />
    </div>
    <RightDrawer open={expanded} onClose={() => setExpanded(false)} title={name} ariaLabel={t('plugins.runtimeDetails', 'Runtime details')}
      width={width} minWidth={360} maxWidth={900} onWidthChange={setWidth} closeLabel={t('common.close', 'Close')} resizeLabel={t('plugins.resize', 'Resize plugin details')}>
    <div style={{ overflowY: 'auto' }}><RuntimeCard aria-label={runtime.plugin_id}>
    <h3>{t('plugins.runtimeOverview', 'Runtime overview')}</h3>
    <RuntimeIdentity>{runtime.plugin_id}</RuntimeIdentity>
    <p>{runtime.uninstalled ? t('plugins.runtimeUninstalled', 'Plugin uninstalled; cleanup state retained') : runtime.enabled ? t('plugins.runtimeEnabled', 'Runtime enabled') : t('plugins.runtimeDisabled', 'Runtime disabled')}</p>
    <p>{t('plugins.selectedRelease', 'Selected release')}: {runtime.selected_digest?.slice(0, 12) ?? t('plugins.noRelease', 'None selected')}</p>
    <p>{t('plugins.activeRelease', 'Active release')}: {runtime.service?.active_digest?.slice(0, 12) ?? t('plugins.noActiveRelease', 'None active')}</p>
    {runtime.stale || unavailable ? <p role="status">{t('plugins.runtimeStale', 'Service health is stale or unavailable.')}</p> : null}
    {runtime.error_code ? <p role="status">{t('plugins.runtimeNeedsAttention', 'The local service needs attention.')}</p> : null}
    {operationId ? <p role="status">{t('plugins.runtimeOperation', 'Local operation')}: {stateLabel}</p> : null}
    {operation.error ? <p role="status">{t('plugins.runtimeReadFailed', 'Operation status could not be read. The original request has not been resubmitted.')}</p> : null}
    <Actions>
      <Button type="button" disabled={disableBlocked} onClick={() => void withdrawRuntime('disable')}>{t('plugins.disableRuntime', 'Disable runtime')}</Button>
      <Button type="button" disabled={uninstallBlocked} onClick={() => void withdrawRuntime('uninstall')}>{t('plugins.uninstallRuntime', 'Uninstall plugin')}</Button>
      {state === 'recovery_required' ? <Button type="button" disabled={unavailable || busy} onClick={() => void recoverOperation()}>{t('plugins.recoverRuntime', 'Recover local operation')}</Button> : null}
      {operationId ? <Button type="button" disabled={operation.isFetching || busy} onClick={() => void operation.refetch()}>{t('plugins.refreshOperation', 'Refresh operation status')}</Button> : null}
    </Actions>
    {withdrawable ? <p>{t('plugins.withdrawInterruptedRuntime', 'Disable or uninstall withdraws this pending local change without running its release. Installation history and cleanup records are retained.')}</p> : null}
    <p>{t('plugins.runtimeCleanup', 'Disable and uninstall stop future capability work. Cleanup supervision, credentials, records and recovery artifacts are retained. Uninstall is not a data purge. Select or activate a verified release to reinstall, or remove the uninstalled plugin to purge what it retained.')}</p>
    {runtime.uninstalled && removalArmed ? <p role="status">{t('plugins.purgeWarning', 'Removing deletes everything this uninstalled plugin retained: records, staged releases, credentials and cleanup state. It cannot be reinstalled from this installation afterwards.')}</p> : null}
    {runtime.uninstalled ? <Button type="button" disabled={pending || busy || unavailable} onClick={() => { if (removalArmed) { void purgeRuntime(); } else { setRemovalArmed(true); } }}>{removalArmed ? t('plugins.purgeConfirm', 'Remove now') : t('plugins.purgeRuntime', 'Remove uninstalled plugin')}</Button> : null}
    {notice && state !== 'complete' ? <p role="status">{notice}</p> : null}
    <Button type="button" onClick={() => setReleaseOpen(!releaseOpen)}>{releaseOpen ? t('plugins.closeReleaseInstallation', 'Close release installation') : t('plugins.openReleaseInstallation', 'Install a release')}</Button>
    {releaseOpen ? <RuntimeReleasePanel runtime={runtime} ownership={{ submitted: releaseSubmitted, setSubmitted: setReleaseSubmitted, activationSubmitted, setActivationSubmitted }} /> : null}
    {details && <section style={{ marginTop: 28 }}><h3>{t('plugins.capabilityNodes', 'Capability nodes')}</h3>{details}</section>}
  </RuntimeCard></div></RightDrawer></>;
}

/** Observe independently supervised runtimes even when no plugin child is available. */
export function ManagedRuntimesPanel({ nodeNames = () => [], renderDetails, nodeEvidence, renderHeader, filter = 'all' }: {
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
  const [register, registering] = useRegisterManagedRuntimeMutation();
  const [setupId, setSetupId] = useState<string | null>(null);
  const [registrationUncertain, setRegistrationUncertain] = useState(false);
  const addPlugin = async () => {
    const id = setupId ?? `managed.${crypto.randomUUID().replaceAll('-', '')}`;
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
    {query.data?.installations.map((runtime) => <RuntimeControls key={runtime.plugin_id} runtime={runtime} filter={filter} nodeEvidence={nodeEvidence?.(runtime.plugin_id)} unavailable={!!query.error} nodes={nodeNames(runtime.plugin_id)} details={renderDetails?.(runtime.plugin_id)} />)}
  </section>;
}

const InventoryNotice = styled.div`
  padding: 24px; margin: 10px 0 20px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px; background: ${({ theme }) => theme.colors.surface};
  h2 { font-size: 16px; margin-bottom: 8px; } p { font-size: 14px; line-height: 1.6; color: ${({ theme }) => theme.colors.textSecondary}; margin-bottom: 16px; }
`;
