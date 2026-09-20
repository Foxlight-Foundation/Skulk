import { derivePluginHealth, type PluginFilter } from './pluginHealth';
import { useState, useSyncExternalStore } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useGetManagedRuntimesQuery, useGetPluginNodesQuery, useGetNodeConfigurationQuery, useConfigurePluginNodeMutation, type ConfigurableNode, type NodeConfiguration } from '../../store/endpoints/plugins';
import { RightDrawer } from '../common/RightDrawer';
import { PluginSummaryCard } from '../common/PluginSummaryCard';
import { Button } from '../common/Button';
import { ManagedRuntimesPanel } from './ManagedRuntimesPanel';
import { PluginConfigurationFields, supportedConfigurationSchema } from './PluginConfigurationFields';
import { NodeCredentialsPanel } from './NodeCredentialsPanel';
import { NodePreflightPanel } from './NodePreflightPanel';
import { NodeSetupPanel } from './NodeSetupPanel';
import { NodeSetupActionsPanel } from './NodeSetupActionsPanel';
import { NodeProposalsPanel } from './NodeProposalsPanel';
import { OperatorAccessPanel } from './OperatorAccessPanel';
import { operatorSession } from '../../auth/operatorSession';

const Page = styled.section`padding: 32px; @media (max-width: 600px) { padding: 16px; } width: 100%; max-width: 1044px; margin: 0 auto; box-sizing: border-box; container-type: inline-size;`;
const Card = styled.article`
  margin: 16px 0; padding: 20px; border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md}; background: ${({ theme }) => theme.colors.surface};
`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;`;

/** A node draft retains its observed revision until the owner explicitly saves/reloads. */
function NodeEditor({ pluginId, configuration, reload }: { pluginId: string; configuration: NodeConfiguration; reload: () => Promise<NodeConfiguration> }) {
  const { t } = useSkulkTranslation();
  const [baseline, setBaseline] = useState(configuration);
  const [values, setValues] = useState(configuration.values);
  const [notice, setNotice] = useState('');
  const [reloading, setReloading] = useState(false);
  const [mutate, { isLoading }] = useConfigurePluginNodeMutation();
  const busy = isLoading || reloading;
  const changedElsewhere = configuration.revision !== baseline.revision || configuration.schemaDigest !== baseline.schemaDigest;
  const dirty = JSON.stringify(values) !== JSON.stringify(baseline.values);
  const supported = supportedConfigurationSchema(baseline.configurationSchema);
  const reloadSettings = async () => {
    setReloading(true);
    try {
      const fresh = await reload();
      setBaseline(fresh);
      setValues(fresh.values);
      setNotice('');
    } catch {
      setNotice(t('plugins.reloadFailed', 'Settings could not be reloaded. Your draft is preserved.'));
    } finally {
      setReloading(false);
    }
  };
  const act = async (operation: 'validate' | 'edit' | 'enable' | 'disable') => {
    setNotice('');
    try {
      const result = await mutate({ pluginId, nodeId: baseline.nodeId, operation,
        expectedRevision: baseline.revision, expectedSchemaDigest: baseline.schemaDigest,
        ...(operation === 'validate' || operation === 'edit' ? { values } : {}),
      }).unwrap();
      if (operation !== 'validate') setBaseline(result.configuration);
      if (operation === 'edit') setValues(result.configuration.values);
      setNotice(operation === 'validate' ? t('plugins.valid', 'Settings are valid.') : t('plugins.saved', 'Settings updated.'));
    } catch {
      setNotice(t('plugins.refused', 'The change was refused. Check your access and settings; reload if another operator changed this node.'));
    }
  };
  return <form onSubmit={(event) => { event.preventDefault(); void act('edit'); }}>
    <p>{baseline.enabled ? t('plugins.enabled', 'Enabled') : t('plugins.disabled', 'Disabled')}</p>
    {changedElsewhere ? <p role="status">{t('plugins.changed', 'This node changed elsewhere. Your draft is preserved; reload before saving.')}</p> : null}
    {supported ? <PluginConfigurationFields schema={baseline.configurationSchema} values={values} onChange={setValues} disabled={busy} /> : <p>{t('plugins.unsupported', 'This settings schema is not yet supported by the dashboard. Use the plugin’s configuration tool.')}</p>}
    <Actions>
      <Button type="button" disabled={!supported || busy || changedElsewhere} onClick={() => void act('validate')}>{t('plugins.validate', 'Validate')}</Button>
      <Button type="submit" disabled={!supported || !dirty || busy || changedElsewhere}>{t('plugins.save', 'Save settings')}</Button>
      <Button type="button" disabled={busy || changedElsewhere || (!baseline.enabled && (!supported || dirty))} onClick={() => void act(baseline.enabled ? 'disable' : 'enable')}>{baseline.enabled ? t('plugins.disable', 'Disable') : t('plugins.enable', 'Enable')}</Button>
      <Button type="button" disabled={busy} onClick={() => void reloadSettings()}>{t('plugins.reload', 'Reload settings')}</Button>
    </Actions>
    {notice ? <p role="status">{notice}</p> : null}
  </form>;
}

function NodeCard({ pluginId, node }: { pluginId: string; node: ConfigurableNode }) {
  const { t } = useSkulkTranslation();
  const [expanded, setExpanded] = useState(false);
  const [credentialsOpen, setCredentialsOpen] = useState(false);
  const query = useGetNodeConfigurationQuery({ pluginId, nodeId: node.nodeId }, { skip: !expanded || !node.configurable });
  return <Card>
    <h3>{node.bundleId} · {node.version}</h3><p>{node.status}</p>
    {node.configurable ? <Button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? t('plugins.close', 'Close settings') : t('plugins.configure', 'Configure')}</Button> : <p>{t('plugins.noSettings', 'This node has no configurable settings.')}</p>}
    {expanded && query.isLoading ? <p>{t('plugins.loadingSettings', 'Loading settings…')}</p> : null}
    {expanded && query.error ? <p role="alert">{t('plugins.settingsUnavailable', 'Settings are unavailable. Check node health and your plugin permissions.')}</p> : null}
    {expanded && query.data ? <NodeEditor pluginId={pluginId} configuration={query.data} reload={() => query.refetch().unwrap()} /> : null}
    {node.preflightAvailable ? <NodePreflightPanel pluginId={pluginId} nodeId={node.nodeId} /> : null}
    {node.proposalsAvailable ? <NodeProposalsPanel pluginId={pluginId} nodeId={node.nodeId} actionsAvailable={node.proposalActionsAvailable === true} /> : null}
    {node.setupActionsAvailable ? <NodeSetupActionsPanel pluginId={pluginId} nodeId={node.nodeId} /> : null}
    {node.setupAvailable ? <NodeSetupPanel pluginId={pluginId} nodeId={node.nodeId} /> : null}
    {node.credentialsConfigurable ? <Button type="button" aria-expanded={credentialsOpen} onClick={() => setCredentialsOpen(!credentialsOpen)}>{credentialsOpen ? t('plugins.closeCredentials', 'Close credentials') : t('plugins.manageCredentials', 'Manage credentials')}</Button> : null}
    {credentialsOpen ? <NodeCredentialsPanel pluginId={pluginId} nodeId={node.nodeId} /> : null}
  </Card>;
}

/** Render configuration declared by installed plugins, without provider-specific UI. */
function PluginInventory() {
  const runtimes = useGetManagedRuntimesQuery();
  const session = useSyncExternalStore(operatorSession.subscribe, operatorSession.snapshot);
  const [accessOpen, setAccessOpen] = useState(false);
  const [filter, setFilter] = useState<PluginFilter>('all');
  const [selected, setSelected] = useState<string | null>(null);
  const [width, setWidth] = useState(640);
  const { t } = useSkulkTranslation();
  const query = useGetPluginNodesQuery(undefined, { pollingInterval: 5000, skipPollingIfUnfocused: true });
  const counts: Record<PluginFilter, number> = { all: 0, healthy: 0, attention: 0, uninstalled: 0 };
  for (const runtime of runtimes.data?.installations ?? []) {
    counts.all += 1;
    const category = derivePluginHealth(runtime, query.data?.find(plugin => plugin.pluginId === runtime.plugin_id), !!runtimes.error || !!query.error);
    if (category === 'healthy' || category === 'attention' || category === 'uninstalled') counts[category] += 1;
  }
  counts.all += query.data?.filter(plugin => !runtimes.data?.installations.some(runtime => runtime.plugin_id === plugin.pluginId)).length ?? 0;
  const inventoryKnown = !!runtimes.data && !runtimes.error && !!query.data && !query.error;
  return <>
    <ManagedRuntimesPanel renderHeader={registrationAction => <>
      <PageHeading><div>    <h1>{t('plugins.title', 'Plugins')}</h1>
    <p>{t('plugins.introReference', 'Capability runtimes installed on this host.')} {' '}
      {inventoryKnown ? t('plugins.inventorySummary', '{count} installed · {healthy} healthy.', { count: counts.all - counts.uninstalled, healthy: counts.healthy })
        : runtimes.isLoading || query.isLoading ? t('plugins.loadingInventory', 'Loading inventory…') : t('plugins.inventoryUnknown', 'Inventory unavailable.')}
    </p>
</div><HeaderActions>
        <AccessButton variant="ghost" size="sm" onClick={() => setAccessOpen(true)}><AccessDot $direct={session.mode === 'direct'} aria-hidden />{session.mode === 'direct' ? t('operator.direct', 'Direct host access') : t('operator.browserAccess', 'Browser access')}</AccessButton>
        {registrationAction}
      </HeaderActions></PageHeading>
      <Filters aria-label={t('plugins.filters', 'Filter plugins')}>
        {([{ value: 'all', label: t('plugins.all', 'All') }, { value: 'healthy', label: t('plugins.healthy', 'Healthy') }, { value: 'attention', label: t('plugins.needsAttention', 'Needs attention') }, { value: 'uninstalled', label: t('plugins.uninstalled', 'Uninstalled') }] as const).map(option => <FilterButton key={option.value} type="button" $active={filter === option.value} aria-pressed={filter === option.value} disabled={!inventoryKnown} onClick={() => setFilter(option.value)}>{option.label}{inventoryKnown ? ` · ${counts[option.value]}` : ''}</FilterButton>)}
      </Filters>
    </>} filter={filter} nodeEvidence={pluginId => query.error ? undefined : query.data?.find(plugin => plugin.pluginId === pluginId)} nodeNames={pluginId => query.data?.find(plugin => plugin.pluginId === pluginId)?.nodes.map(node => node.nodeId) ?? []}
      renderDetails={pluginId => query.data?.find(plugin => plugin.pluginId === pluginId)?.nodes.map(node => <NodeCard key={node.nodeId} pluginId={pluginId} node={node} />)} />
    <Button type="button" variant="ghost" size="sm" disabled={query.isFetching || runtimes.isFetching} onClick={() => { void query.refetch(); void runtimes.refetch(); }}>{t('plugins.refresh', 'Refresh')}</Button>
    {query.isLoading ? <p>{t('plugins.loading', 'Loading plugins…')}</p> : null}
    {query.error ? <p role="alert">{t('plugins.accessRequired', 'Plugin management is unavailable. Open the host dashboard through localhost or Tailscale, or use a paired operator with plugin access.')}</p> : null}

    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 16, marginTop: 20 }}>
    {query.data?.filter(plugin => filter === 'all' && !runtimes.data?.installations.some(runtime => runtime.plugin_id === plugin.pluginId)).map(plugin => <PluginSummaryCard key={plugin.pluginId} name={plugin.nodes.length === 1 ? plugin.nodes[0].bundleId : plugin.pluginId} pluginId={plugin.pluginId}
      health={query.error || !plugin.available ? t('plugins.unavailable', 'Unavailable') : t('plugins.nodesObserved', 'Nodes observed')}
      tone="neutral" release={Array.from(new Set(plugin.nodes.map(node => node.version))).join(' · ') || t('plugins.noRelease', 'None selected')}
      nodes={plugin.nodes.map(node => node.nodeId)} onOpen={() => setSelected(plugin.pluginId)} />)}
    </div>
    <RightDrawer open={accessOpen} onClose={() => setAccessOpen(false)} title={t('operator.browserAccess', 'Browser access')} ariaLabel={t('operator.browserAccess', 'Browser access')} width={width} minWidth={360} maxWidth={900} onWidthChange={setWidth} closeLabel={t('common.close', 'Close')} resizeLabel={t('plugins.resize', 'Resize plugin details')}><OperatorAccessPanel /></RightDrawer>
    <RightDrawer open={selected !== null} onClose={() => setSelected(null)} title={selected ?? ''} ariaLabel={t('plugins.details', 'Plugin details')}
      width={width} minWidth={360} maxWidth={900} onWidthChange={setWidth} closeLabel={t('common.close', 'Close')} resizeLabel={t('plugins.resize', 'Resize plugin details')}>
      <div style={{ padding: 24, overflowY: 'auto' }}>
      {query.data?.find(plugin => plugin.pluginId === selected)?.nodes.map(node => <NodeCard key={node.nodeId} pluginId={selected!} node={node} />)}
      </div>
    </RightDrawer>
  </>;
}

/** Remount sensitive drafts when the browser changes its authorization identity. */
export function PluginsPage() {
  const session = useSyncExternalStore(operatorSession.subscribe, operatorSession.snapshot);
  return <Page><PluginInventory key={`${session.mode}:${session.deviceId ?? ''}`} /></Page>;
}

const PageHeading = styled.div`
  display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap;
  h1 { font-size: 28px; color: ${({ theme }) => theme.colors.text}; letter-spacing: -.02em; }
  p { font-size: 14px; color: ${({ theme }) => theme.colors.textSecondary}; margin-top: 6px; }
`;
const HeaderActions = styled.div`display: flex; align-items: center; gap: 10px; flex-wrap: wrap;`;
const Filters = styled.div`display: flex; flex-wrap: wrap; gap: 6px; margin: 22px 0;`;
const FilterButton = styled.button<{ $active: boolean }>`
  border: 0; border-radius: 999px; padding: 6px 12px; cursor: pointer;
  background: ${({ theme, $active }) => $active ? theme.colors.selected : 'transparent'};
  color: ${({ theme, $active }) => $active ? theme.colors.text : theme.colors.textSecondary};
  font: ${({ $active }) => $active ? 600 : 400} 12.5px ${({ theme }) => theme.fonts.body};
  &:disabled { cursor: default; opacity: .6; }
  &:hover:not(:disabled) { background: ${({ theme }) => theme.colors.surfaceHover}; }
`;
const AccessButton = styled(Button)`border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 999px; font-size: 12px; color: ${({ theme }) => theme.colors.textSecondary};`;
const AccessDot = styled.span<{ $direct: boolean }>`width: 7px; height: 7px; border-radius: 50%; background: ${({ theme, $direct }) => $direct ? theme.colors.healthy : theme.colors.textMuted};`;
