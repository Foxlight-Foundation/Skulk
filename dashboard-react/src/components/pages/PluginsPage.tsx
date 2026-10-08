import { derivePluginHealth, type PluginFilter } from './pluginHealth';
import { useState, useSyncExternalStore, type ReactNode } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';
import {
  installOperationIds, pluginRefusalDetail, useGetInstallOperationsQuery, useGetManagedRuntimesQuery, useGetPluginNodesQuery, useGetNodeConfigurationQuery,
  useConfigurePluginNodeMutation, type ConfigurableNode, type NodeConfiguration, type PluginNodes,
} from '../../store/endpoints/plugins';
import { RightDrawer } from '../common/RightDrawer';
import { PluginSummaryCard } from '../common/PluginSummaryCard';
import { Button } from '../common/Button';
import { StatusPill, type StatusTone } from '../common/Surfaces';
import { ManagedRuntimesPanel } from './ManagedRuntimesPanel';
import { PluginConfigurationFields } from './PluginConfigurationFields';
import { humanizeKey, supportedConfigurationSchema } from './pluginSettingsSchema';
import { CapabilitySetupPanel } from './CapabilitySetupPanel';
import type { SetupTarget } from './CatalogInstallProgress';
import { NodeCredentialsPanel } from './NodeCredentialsPanel';
import { NodePreflightPanel } from './NodePreflightPanel';
import { NodeSetupPanel } from './NodeSetupPanel';
import { NodeSetupActionsPanel } from './NodeSetupActionsPanel';
import { NodeProposalsPanel } from './NodeProposalsPanel';
import { OperatorAccessPanel } from './OperatorAccessPanel';
import { PluginCatalogBrowse, type BrowseRetryHandoff } from './PluginCatalogBrowse';
import type { RetryRequest } from './catalogJourney';
import { PluginServiceSetup } from './PluginServiceSetup';
import { operatorSession } from '../../auth/operatorSession';
import { FiExternalLink } from 'react-icons/fi';
import { useClusterState } from '../../hooks/useClusterState';
import { dashboardUrlOn, pluginsPath, tailnetAddress } from '../../utils/hostDashboard';

type PluginsView = 'installed' | 'browse';

function initialView(): PluginsView {
  try { return new URLSearchParams(window.location.search).get('view') === 'browse' ? 'browse' : 'installed'; } catch { return 'installed'; }
}

// Another node's "Manage on {host}" link opens this page on one plugin.
function initialPlugin(): string | null {
  try { return new URLSearchParams(window.location.search).get('plugin') || null; } catch { return null; }
}

/**
 * The other nodes of the cluster, each linking to its own Plugins page at its
 * Tailscale address. Each node manages its own plugins and admits a browser
 * that reaches it over the tailnet, so this is how an owner on any tailnet
 * machine reaches every node's plugins. A node that reports no Tailscale
 * address is named without a link.
 */
function OtherHosts() {
  const { t } = useSkulkTranslation();
  const { topology, localNodeId, capabilityNodes } = useClusterState();
  const hosts = Object.entries(topology?.nodes ?? {})
    .filter(([nodeId]) => nodeId !== localNodeId)
    .map(([nodeId, node]) => ({
      nodeId, name: node.friendly_name ?? nodeId.slice(-8), address: tailnetAddress(node),
      plugins: new Set((capabilityNodes[nodeId] ?? []).map((summary) => summary.pluginId)).size,
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
  if (hosts.length === 0) return null;
  return <OtherHostsRow aria-label={t('plugins.otherHosts', 'Plugins on other nodes')}>
    <span>{t('plugins.otherHostsLabel', 'Other nodes:')}</span>
    {hosts.map((item) => {
      const label = item.plugins === 0 ? item.name
        : item.plugins === 1 ? t('plugins.otherHostOnePlugin', '{host} · 1 plugin', { host: item.name })
          : t('plugins.otherHostPlugins', '{host} · {count} plugins', { host: item.name, count: item.plugins });
      return item.address
        ? <a key={item.nodeId} href={dashboardUrlOn(item.address, pluginsPath())} target="_blank" rel="noopener noreferrer"
          title={t('plugins.otherHostLink', 'Open the Plugins page on {host}', { host: item.name })}>{label}<FiExternalLink aria-hidden /></a>
        : <span key={item.nodeId} title={t('plugins.otherHostNoTailnet', 'This node reports no Tailscale address. Open its dashboard on the node itself.')}>{label}</span>;
    })}
  </OtherHostsRow>;
}

const Page = styled.section`padding: 32px; @media (max-width: 600px) { padding: 16px; } width: 100%; max-width: 1044px; margin: 0 auto; box-sizing: border-box; container-type: inline-size;`;

/** How one node's reported status reads in its settings panel. */
function nodeStatus(status: string, t: SkulkTranslate): { label: string; tone: StatusTone } {
  return ({
    ready: { label: t('plugins.node.running', 'Running'), tone: 'healthy' },
    disabled: { label: t('plugins.node.off', 'Off'), tone: 'neutral' },
    configuration_invalid: { label: t('plugins.node.needsSettings', 'Needs settings'), tone: 'live' },
    failed: { label: t('plugins.node.stopped', 'Stopped'), tone: 'danger' },
    installed: { label: t('plugins.node.starting', 'Starting'), tone: 'live' },
    starting: { label: t('plugins.node.starting', 'Starting'), tone: 'live' },
    degraded: { label: t('plugins.node.restarting', 'Restarting'), tone: 'live' },
  } as Record<string, { label: string; tone: StatusTone }>)[status] ?? { label: humanizeKey(status), tone: 'neutral' };
}

/** Props for one node's settings form. */
interface NodeEditorProps {
  pluginId: string;
  configuration: NodeConfiguration;
  reload: () => Promise<NodeConfiguration>;
  /** Called when the host refuses to turn the node on, so its setup checks can say why. */
  onTurnOnRefused?: () => void;
}

/**
 * A node's settings form and its on/off switch.
 *
 * A draft retains its observed revision until the owner explicitly saves or
 * reloads. Turning the node on or off sends no values, only the revision and
 * schema digest it was read at, so it never depends on whether this form can
 * render the plugin's settings: it waits only for unsaved edits, a request in
 * flight, or a change made elsewhere.
 */
function NodeEditor({ pluginId, configuration, reload, onTurnOnRefused }: NodeEditorProps) {
  const { t } = useSkulkTranslation();
  const [baseline, setBaseline] = useState(configuration);
  const [values, setValues] = useState(configuration.values);
  const [notice, setNotice] = useState<{ text: string; problem: boolean } | null>(null);
  const [reloading, setReloading] = useState(false);
  const [mutate, { isLoading }] = useConfigurePluginNodeMutation();
  const busy = isLoading || reloading;
  const changedElsewhere = configuration.revision !== baseline.revision || configuration.schemaDigest !== baseline.schemaDigest;
  const dirty = JSON.stringify(values) !== JSON.stringify(baseline.values);
  const supported = supportedConfigurationSchema(baseline.configurationSchema);
  const switchBlocked = busy || changedElsewhere || dirty;
  const reloadSettings = async () => {
    setReloading(true);
    try {
      const fresh = await reload();
      setBaseline(fresh);
      setValues(fresh.values);
      setNotice(null);
    } catch {
      setNotice({ text: t('plugins.reloadFailed', 'Settings could not be reloaded. Your draft is preserved.'), problem: true });
    } finally {
      setReloading(false);
    }
  };
  const act = async (operation: 'validate' | 'edit' | 'enable' | 'disable') => {
    setNotice(null);
    try {
      const result = await mutate({ pluginId, nodeId: baseline.nodeId, operation,
        expectedRevision: baseline.revision, expectedSchemaDigest: baseline.schemaDigest,
        ...(operation === 'validate' || operation === 'edit' ? { values } : {}),
      }).unwrap();
      if (operation !== 'validate') setBaseline(result.configuration);
      if (operation === 'edit') setValues(result.configuration.values);
      setNotice({ text: {
        validate: t('plugins.valid', 'Settings are valid.'),
        edit: t('plugins.saved', 'Settings saved.'),
        enable: t('plugins.turnedOn', 'Turned on. It starts in a moment.'),
        disable: t('plugins.turnedOff', 'Turned off.'),
      }[operation], problem: false });
    } catch (error) {
      const detail = pluginRefusalDetail(error);
      if (operation === 'enable') {
        // The plugin runs its setup checks before it starts; a refusal means one of them failed.
        setNotice({ text: `${t('plugins.turnOnRefused', 'It was not turned on.')} ${detail ? t('plugins.hostSaid', 'The host said: {reason}', { reason: detail }) : t('plugins.turnOnRefusedGeneric', 'Check your access, then reload if another operator changed this node.')} ${t('plugins.turnOnSeeChecks', 'Setup checks show what is missing.')}`, problem: true });
        onTurnOnRefused?.();
      } else {
        setNotice({ text: detail ? `${t('plugins.refusedShort', 'The change was refused.')} ${t('plugins.hostSaid', 'The host said: {reason}', { reason: detail })}`
          : t('plugins.refused', 'The change was refused. Check your access and settings; reload if another operator changed this node.'), problem: true });
      }
    }
  };
  // One primary at a time: saving a draft, else turning a node on that is off.
  const switchPrimary = !dirty && !baseline.enabled;
  return <Editor onSubmit={(event) => { event.preventDefault(); void act('edit'); }}>
    {changedElsewhere ? <Notice role="status" $problem>{t('plugins.changedElsewhere', 'This node changed elsewhere. Reload first; your draft is kept.')}</Notice> : null}
    {supported ? <PluginConfigurationFields schema={baseline.configurationSchema} values={values} onChange={setValues} disabled={busy} />
      : <Muted>{t('plugins.unsupportedForm', 'These settings cannot be edited here yet; the plugin’s own configuration tool changes them. You can still turn it on or off here.')}</Muted>}
    {/* The primary comes first: saving a draft, else turning on a node that is off. */}
    <ActionRow>
      {dirty ? <>
        <Button type="submit" variant="solid" disabled={busy || changedElsewhere}>{t('plugins.save', 'Save settings')}</Button>
        <Button type="button" variant="outline" disabled={busy} onClick={() => { setValues(baseline.values); setNotice(null); }}>{t('plugins.discard', 'Discard changes')}</Button>
      </> : null}
      <Button type="button" variant={switchPrimary ? 'solid' : 'outline'} disabled={switchBlocked} onClick={() => void act(baseline.enabled ? 'disable' : 'enable')}>
        {baseline.enabled ? t('plugins.turnOff', 'Turn off') : t('plugins.turnOn', 'Turn on')}
      </Button>
      {supported ? <Button type="button" variant="outline" disabled={busy || changedElsewhere} onClick={() => void act('validate')}>{t('plugins.validate', 'Validate')}</Button> : null}
      <Button type="button" variant="outline" disabled={busy} onClick={() => void reloadSettings()}>{t('plugins.reload', 'Reload settings')}</Button>
    </ActionRow>
    {dirty && !changedElsewhere ? <Muted>{baseline.enabled ? t('plugins.saveBeforeTurningOff', 'Save or discard your changes before turning it off.') : t('plugins.saveBeforeTurningOn', 'Save or discard your changes before turning it on.')}</Muted> : null}
    {!dirty && !baseline.enabled && !changedElsewhere ? <Muted>{t('plugins.turnOnHelp', 'Turning it on runs its setup checks first.')}</Muted> : null}
    {notice ? <Notice role="status" $problem={notice.problem}>{notice.text}</Notice> : null}
  </Editor>;
}

/** Props for one installed node's settings panel. */
interface NodeCardProps {
  pluginId: string;
  node: ConfigurableNode;
  /** The capability's display title; the bundle id when none is known. */
  title: string;
  /** Show the settings open from the start: the plugin's only node, or the node the owner asked for. */
  defaultExpanded?: boolean;
  /** Whether the settings can be folded away, as one of several nodes. */
  collapsible?: boolean;
}

function NodeCard({ pluginId, node, title, defaultExpanded = false, collapsible = true }: NodeCardProps) {
  const { t } = useSkulkTranslation();
  const [expanded, setExpanded] = useState(defaultExpanded || !collapsible);
  const [credentialsOpen, setCredentialsOpen] = useState(false);
  const [checkRequest, setCheckRequest] = useState(0);
  const query = useGetNodeConfigurationQuery({ pluginId, nodeId: node.nodeId }, { skip: !expanded || !node.configurable });
  const status = nodeStatus(node.status, t);
  return <Card aria-label={title}>
    <CardHeader>
      <div>
        <h3>{title}</h3>
        <Meta>{t('plugins.nodeVersion', 'Version {version}', { version: node.version })}{title !== node.bundleId ? <> · <code>{node.bundleId}</code></> : null}</Meta>
      </div>
      <StatusPill tone={status.tone}>{status.label}</StatusPill>
    </CardHeader>
    <Section>
      <SectionHead>
        <h4>{t('plugins.settingsTitle', 'Settings')}</h4>
        {node.configurable && collapsible ? <Button type="button" variant="outline" size="sm" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? t('plugins.hideSettings', 'Hide settings') : t('plugins.showSettings', 'Show settings')}</Button> : null}
      </SectionHead>
      {!node.configurable ? <Muted>{t('plugins.noSettings', 'This node has no configurable settings.')}</Muted> : null}
      {expanded && query.isLoading ? <Muted>{t('plugins.loadingSettings', 'Loading settings…')}</Muted> : null}
      {expanded && query.error ? <Notice role="alert" $problem>{t('plugins.settingsUnavailable', 'Settings are unavailable. Check node health and your plugin permissions.')}</Notice> : null}
      {expanded && query.data ? <NodeEditor pluginId={pluginId} configuration={query.data} reload={() => query.refetch().unwrap()}
        onTurnOnRefused={node.preflightAvailable ? () => setCheckRequest((count) => count + 1) : undefined} /> : null}
    </Section>
    {node.preflightAvailable ? <Section>
      <h4>{t('plugins.setup.checksTitle', 'Setup checks')}</h4>
      <Muted>{t('plugins.checksHelp', 'Check what it needs to run. Checking changes nothing and does not turn it on.')}</Muted>
      <NodePreflightPanel pluginId={pluginId} nodeId={node.nodeId} runRequest={checkRequest} />
    </Section> : null}
    {node.credentialsConfigurable ? <Section>
      <SectionHead>
        <h4>{t('plugins.credentialsTitle', 'Credentials')}</h4>
        <Button type="button" variant="outline" size="sm" aria-expanded={credentialsOpen} onClick={() => setCredentialsOpen(!credentialsOpen)}>{credentialsOpen ? t('plugins.closeCredentials', 'Close credentials') : t('plugins.manageCredentials', 'Manage credentials')}</Button>
      </SectionHead>
      {credentialsOpen ? <NodeCredentialsPanel pluginId={pluginId} nodeId={node.nodeId} /> : null}
    </Section> : null}
    {node.setupActionsAvailable ? <Section><NodeSetupActionsPanel pluginId={pluginId} nodeId={node.nodeId} /></Section> : null}
    {node.setupAvailable ? <Section><NodeSetupPanel pluginId={pluginId} nodeId={node.nodeId} /></Section> : null}
    {node.proposalsAvailable ? <Section><NodeProposalsPanel pluginId={pluginId} nodeId={node.nodeId} actionsAvailable={node.proposalActionsAvailable === true} /></Section> : null}
  </Card>;
}

/** Render configuration declared by installed plugins, without provider-specific UI. */
function PluginInventory() {
  const runtimes = useGetManagedRuntimesQuery();
  const session = useSyncExternalStore(operatorSession.subscribe, operatorSession.snapshot);
  const [accessOpen, setAccessOpen] = useState(false);
  const [filter, setFilter] = useState<PluginFilter>('all');
  const [view, setView] = useState<PluginsView>(initialView);
  const [selected, setSelected] = useState<string | null>(initialPlugin);
  // The node whose settings the owner asked for, opened first in the drawer.
  const [focusNode, setFocusNode] = useState<string | null>(null);
  // An installed plugin's setup page, shown under Installed until it is done.
  const [setup, setSetup] = useState<SetupTarget | null>(null);
  const [width, setWidth] = useState(640);
  const { t } = useSkulkTranslation();
  const { capabilityNodes, localNodeId } = useClusterState();
  const query = useGetPluginNodesQuery(undefined, { pollingInterval: 5000, skipPollingIfUnfocused: true });
  // Shared with the Installed list: a stopped install counts as needing attention.
  const operationIds = installOperationIds(runtimes.data?.installations);
  const installs = useGetInstallOperationsQuery(operationIds, { skip: operationIds.length === 0 });
  const [retryHandoff, setRetryHandoff] = useState<BrowseRetryHandoff | null>(null);
  const counts: Record<PluginFilter, number> = { all: 0, healthy: 0, attention: 0, uninstalled: 0 };
  for (const runtime of runtimes.data?.installations ?? []) {
    counts.all += 1;
    const category = derivePluginHealth(runtime, query.data?.find(plugin => plugin.pluginId === runtime.plugin_id), !!runtimes.error || !!query.error, runtime.operation_state, installs.data?.[runtime.plugin_id] ?? null);
    if (category === 'healthy' || category === 'attention' || category === 'uninstalled') counts[category] += 1;
  }
  counts.all += query.data?.filter(plugin => !runtimes.data?.installations.some(runtime => runtime.plugin_id === plugin.pluginId)).length ?? 0;
  const inventoryKnown = !!runtimes.data && !runtimes.error && !!query.data && !query.error;
  const chooseView = (next: PluginsView) => {
    setView(next);
    // The view is part of the address so Browse can be linked to directly.
    try { history.replaceState(null, '', next === 'browse' ? `${window.location.pathname}?view=browse` : window.location.pathname); } catch { /* the view still changes */ }
  };
  const heading = (registrationAction?: ReactNode) => <PageHeading><div>    <h1>{t('plugins.title', 'Plugins')}</h1>
    <p>{t('plugins.introReference', 'Capability runtimes installed on this host.')} {' '}
      {inventoryKnown ? t('plugins.inventorySummary', '{count} installed · {healthy} healthy.', { count: counts.all - counts.uninstalled, healthy: counts.healthy })
        : runtimes.isLoading || query.isLoading ? t('plugins.loadingInventory', 'Loading inventory…') : t('plugins.inventoryUnknown', 'Inventory unavailable.')}
    </p>
    <OtherHosts />
</div><HeaderActions>
        <AccessButton variant="ghost" size="sm" onClick={() => setAccessOpen(true)}><AccessDot $direct={session.mode === 'direct'} aria-hidden />{session.mode === 'direct' ? t('operator.direct', 'Direct host access') : t('operator.browserAccess', 'Browser access')}</AccessButton>
        {registrationAction}
      </HeaderActions></PageHeading>;
  // A tab opens its own list, leaving any setup page.
  const openTab = (next: PluginsView) => { setSetup(null); chooseView(next); };
  const tabs = <Tabs role="tablist" aria-label={t('plugins.views', 'Plugin views')}>
    <Tab role="tab" type="button" aria-selected={view === 'installed'} $active={view === 'installed'} onClick={() => openTab('installed')}>{t('plugins.tabInstalled', 'Installed')}</Tab>
    <Tab role="tab" type="button" aria-selected={view === 'browse'} $active={view === 'browse'} onClick={() => openTab('browse')}>{t('plugins.tabBrowse', 'Browse')}</Tab>
  </Tabs>;
  const accessDrawer = <RightDrawer open={accessOpen} onClose={() => setAccessOpen(false)} title={t('operator.browserAccess', 'Browser access')} ariaLabel={t('operator.browserAccess', 'Browser access')} width={width} minWidth={360} maxWidth={900} onWidthChange={setWidth} closeLabel={t('common.close', 'Close')} resizeLabel={t('plugins.resize', 'Resize plugin details')}><OperatorAccessPanel /></RightDrawer>;
  // Name the open plugin by its signed release, as the list does; its local id is the fallback.
  const selectedRelease = runtimes.data?.installations.find((runtime) => runtime.plugin_id === selected)?.release;
  const selectedTitle = selectedRelease?.title?.trim() || selectedRelease?.bundle_id || selected || '';
  const closeDetails = () => {
    setSelected(null);
    setFocusNode(null);
    // A plugin named in the address was a one-time deep link; drop it so a reload does not reopen it.
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.has('plugin')) { params.delete('plugin'); const rest = params.toString(); history.replaceState(null, '', `${window.location.pathname}${rest ? `?${rest}` : ''}`); }
    } catch { /* the drawer still closes */ }
  };
  // Setup opens a plugin's settings in the Installed drawer, on the node that needs them.
  const manage = (pluginId: string, nodeId?: string) => { chooseView('installed'); setSelected(pluginId); setFocusNode(nodeId ?? null); };
  // Once installed, a plugin is set up where it now lives: under Installed.
  const openSetup = (target: SetupTarget) => { setSetup(target); chooseView('installed'); };
  // A node is named by the title its host reports, else its plugin's signed
  // title when it is the plugin's only node, else its bundle id.
  const nodeTitle = (plugin: PluginNodes, node: ConfigurableNode): string => {
    const reported = (localNodeId ? capabilityNodes[localNodeId] ?? [] : []).find((summary) => summary.pluginId === plugin.pluginId && summary.nodeId === node.nodeId)?.title?.trim();
    const release = runtimes.data?.installations.find((runtime) => runtime.plugin_id === plugin.pluginId)?.release?.title?.trim();
    return reported || (plugin.nodes.length === 1 ? release : null) || node.bundleId;
  };
  const nodeCards = (pluginId: string) => {
    const plugin = query.data?.find((item) => item.pluginId === pluginId);
    if (!plugin) return null;
    return plugin.nodes.map((node) => <NodeCard key={node.nodeId} pluginId={pluginId} node={node} title={nodeTitle(plugin, node)}
      collapsible={plugin.nodes.length > 1} defaultExpanded={pluginId === selected && node.nodeId === focusNode} />);
  };
  const detailsDrawer = <RightDrawer open={selected !== null} onClose={closeDetails} title={selectedTitle} ariaLabel={t('plugins.details', 'Plugin details')}
    width={width} minWidth={360} maxWidth={900} onWidthChange={setWidth} closeLabel={t('common.close', 'Close')} resizeLabel={t('plugins.resize', 'Resize plugin details')}>
    <DrawerBody>{selected ? nodeCards(selected) : null}</DrawerBody>
  </RightDrawer>;
  // A stopped install is retried where installs are followed, so a retry
  // from Installed opens Browse on that install's progress.
  const retryInstall = (request: RetryRequest) => { setRetryHandoff({ id: Date.now(), request }); chooseView('browse'); };
  const direct = session.mode === 'direct';
  if (view === 'browse') return <><PluginServiceSetup direct={direct} header={<>{heading()}{tabs}</>}>{heading()}{tabs}<BrowseArea><PluginCatalogBrowse onManage={manage} onSetUp={openSetup} retry={retryHandoff} onRetryTaken={() => setRetryHandoff(null)} /></BrowseArea></PluginServiceSetup>{accessDrawer}</>;
  if (setup) return <><PluginServiceSetup direct={direct} header={<>{heading()}{tabs}</>}>{heading()}{tabs}<BrowseArea>
    <CapabilitySetupPanel key={setup.pluginId} target={setup} onManage={manage} onDone={() => setSetup(null)} />
  </BrowseArea>{detailsDrawer}</PluginServiceSetup>{accessDrawer}</>;
  return <PluginServiceSetup direct={direct} header={<>{heading()}{tabs}{accessDrawer}</>}>
    <ManagedRuntimesPanel renderHeader={registrationAction => <>
      {heading(registrationAction)}
      {tabs}
      <Filters aria-label={t('plugins.filters', 'Filter plugins')}>
        {([{ value: 'all', label: t('plugins.all', 'All') }, { value: 'healthy', label: t('plugins.healthy', 'Healthy') }, { value: 'attention', label: t('plugins.needsAttention', 'Needs attention') }, { value: 'uninstalled', label: t('plugins.uninstalled', 'Uninstalled') }] as const).map(option => <FilterButton key={option.value} type="button" $active={filter === option.value} aria-pressed={filter === option.value} disabled={!inventoryKnown} onClick={() => setFilter(option.value)}>{option.label}{inventoryKnown ? ` · ${counts[option.value]}` : ''}</FilterButton>)}
      </Filters>
    </>} filter={filter} onRetryInstall={retryInstall} nodeEvidence={pluginId => query.error ? undefined : query.data?.find(plugin => plugin.pluginId === pluginId)} nodeNames={pluginId => query.data?.find(plugin => plugin.pluginId === pluginId)?.nodes.map(node => node.nodeId) ?? []}
      renderDetails={nodeCards} />
    <Button type="button" variant="ghost" size="sm" disabled={query.isFetching || runtimes.isFetching} onClick={() => { void query.refetch(); void runtimes.refetch(); }}>{t('plugins.refresh', 'Refresh')}</Button>
    {query.isLoading ? <p>{t('plugins.loading', 'Loading plugins…')}</p> : null}
    {query.error ? <p role="alert">{t('plugins.accessRequired', 'Plugin management is unavailable. Open the host dashboard through localhost or Tailscale, or use a paired operator with plugin access.')}</p> : null}

    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 16, marginTop: 20 }}>
    {query.data?.filter(plugin => filter === 'all' && !runtimes.data?.installations.some(runtime => runtime.plugin_id === plugin.pluginId)).map(plugin => <PluginSummaryCard key={plugin.pluginId} name={plugin.nodes.length === 1 ? plugin.nodes[0].bundleId : plugin.pluginId} pluginId={plugin.pluginId}
      health={query.error || !plugin.available ? t('plugins.unavailable', 'Unavailable') : t('plugins.nodesObserved', 'Nodes observed')}
      tone="neutral" release={Array.from(new Set(plugin.nodes.map(node => node.version))).join(' · ') || t('plugins.noRelease', 'None selected')}
      nodes={plugin.nodes.map(node => node.nodeId)} onOpen={() => setSelected(plugin.pluginId)} />)}
    </div>
    {accessDrawer}
    {detailsDrawer}
  </PluginServiceSetup>;
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
const BrowseArea = styled.div`margin-top: 24px;`;
const OtherHostsRow = styled.nav`
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; margin-top: 8px; font-size: 13px;
  color: ${({ theme }) => theme.colors.textSecondary};
  a { display: inline-flex; align-items: center; gap: 4px; color: ${({ theme }) => theme.colors.accentText}; text-decoration: none; }
  a:hover { text-decoration: underline; }
  svg { width: 12px; height: 12px; }
`;
const Tabs = styled.div`display: flex; gap: 4px; margin: 22px 0 0; border-bottom: 1px solid ${({ theme }) => theme.colors.border};`;
const Tab = styled.button<{ $active: boolean }>`
  border: 0; background: transparent; cursor: pointer; padding: 10px 14px; margin-bottom: -1px;
  border-bottom: 2px solid ${({ theme, $active }) => $active ? theme.colors.accentText : 'transparent'};
  color: ${({ theme, $active }) => $active ? theme.colors.text : theme.colors.textSecondary};
  font: ${({ $active }) => $active ? 600 : 500} 14px ${({ theme }) => theme.fonts.body};
  &:hover { color: ${({ theme }) => theme.colors.text}; }
`;
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

const DrawerBody = styled.div`padding: 20px 24px 32px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; @media (max-width: 600px) { padding: 16px; }`;
const Card = styled.article`
  display: flex; flex-direction: column; min-width: 0; border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg}; background: ${({ theme }) => theme.colors.surface}; overflow-wrap: anywhere;
`;
const CardHeader = styled.header`
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 18px 20px 16px;
  h3 { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
`;
const Meta = styled.p`
  margin: 4px 0 0; font-size: 12.5px; color: ${({ theme }) => theme.colors.metadataText};
  code { font: 12px ${({ theme }) => theme.fonts.mono}; }
`;
const Section = styled.section`
  display: flex; flex-direction: column; gap: 12px; padding: 18px 20px; border-top: 1px solid ${({ theme }) => theme.colors.border}; min-width: 0;
  h4 { margin: 0; font: 600 11px ${({ theme }) => theme.fonts.mono}; letter-spacing: .14em; text-transform: uppercase; color: ${({ theme }) => theme.colors.subtleText}; }
`;
const SectionHead = styled.div`display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;`;
const Editor = styled.form`display: flex; flex-direction: column; gap: 16px; min-width: 0;`;
const ActionRow = styled.div`display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding-top: 4px;`;
const Muted = styled.p`margin: 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary};`;
const Notice = styled.p<{ $problem?: boolean }>`
  margin: 0; padding: 10px 12px; border-radius: ${({ theme }) => theme.radii.md}; font-size: 13px; line-height: 1.5;
  border: 1px solid ${({ theme, $problem }) => $problem ? theme.colors.borderDanger : theme.colors.border};
  background: ${({ theme, $problem }) => $problem ? theme.colors.errorBg : theme.colors.surfaceSunken};
  color: ${({ theme }) => theme.colors.text};
`;
