import { useMemo } from 'react';
import styled from 'styled-components';
import { FiCheck, FiCircle, FiExternalLink } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useClusterState } from '../../hooks/useClusterState';
import { useGetPluginNodesQuery } from '../../store/endpoints/plugins';
import { capabilityNodeKey, capabilityNodeTitle, type CapabilityNodeStatus, type CapabilityNodeSummary, type CapabilityNodeSurface } from '../../types/capabilityNodes';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { StatusPill, Surface } from '../common/Surfaces';
import { resolveSurfaceUrl } from '../topology/capabilityActions';
import { NodePreflightPanel } from './NodePreflightPanel';
import type { SetupTarget } from './CatalogInstallProgress';

/** Props for the setup step of one installed plugin. */
export interface CapabilitySetupPanelProps {
  target: SetupTarget;
  /** Open the plugin's settings, credentials and setup actions; the button is hidden without it. */
  onManage?: (pluginId: string) => void;
  onBack: () => void;
}

/** What a node's status asks of the operator. */
type ItemState = 'done' | 'waiting' | 'needed';

// The plugin service decides readiness: it only reports `ready` after the
// node's settings, credentials and setup checks admitted a start and the
// node answered its health check. Transitional states settle by themselves;
// the rest wait for the operator.
const STATUS_STATE: Record<CapabilityNodeStatus, ItemState> = {
  ready: 'done',
  installed: 'waiting',
  starting: 'waiting',
  degraded: 'waiting',
  disabled: 'needed',
  configuration_invalid: 'needed',
  failed: 'needed',
};

/**
 * What an installed plugin still needs, from what its own nodes report.
 *
 * Every capability node of the plugin gets one line from the host's live
 * summary. A node that needs the operator links to the plugin's settings and,
 * when the plugin declares them, offers its setup checks here, worded by the
 * plugin itself. The plugin's screens open once they are ready.
 */
export function CapabilitySetupPanel({ target, onManage, onBack }: CapabilitySetupPanelProps) {
  const { t } = useSkulkTranslation();
  const { capabilityNodes, localNodeId } = useClusterState();
  const plugins = useGetPluginNodesQuery();
  const hosted = useMemo(() => Object.entries(capabilityNodes).flatMap(([hostNodeId, nodes]) => nodes
    .filter((node) => node.pluginId === target.pluginId).map((summary) => ({ hostNodeId, summary }))), [capabilityNodes, target.pluginId]);
  const summaries = useMemo(() => hosted.map((item) => item.summary), [hosted]);
  const withChecks = new Set(plugins.data?.find((plugin) => plugin.pluginId === target.pluginId)?.nodes.filter((node) => node.preflightAvailable).map((node) => node.nodeId) ?? []);
  // The summary carries only surfaces the plugin reported ready to open. As in
  // the topology, a loopback one opens only from a browser on its own host.
  const surfaces = useMemo(() => {
    const unique = new Map<string, { surface: CapabilityNodeSurface; reachable: boolean }>();
    for (const { hostNodeId, summary } of hosted) {
      for (const surface of summary.surfaces) {
        if (surface.kind !== 'link' || !surface.url || unique.has(surface.url)) continue;
        const resolved = resolveSurfaceUrl(surface.url, { isLocalHost: hostNodeId === localNodeId, dashboardHostname: window.location.hostname });
        unique.set(surface.url, { surface: { ...surface, url: resolved.url }, reachable: resolved.reachable });
      }
    }
    return [...unique.values()];
  }, [hosted, localNodeId]);
  const unreachable = surfaces.filter((item) => !item.reachable);

  // A status this dashboard does not know yet is shown as needing a look.
  const itemState = (summary: CapabilityNodeSummary): ItemState => summary.ownerAvailable ? STATUS_STATE[summary.status] ?? 'needed' : 'needed';
  const detail = (summary: CapabilityNodeSummary): string => {
    if (!summary.ownerAvailable) return t('plugins.setup.ownerUnavailable', 'The plugin service on this host is not answering. It reports again once it is back.');
    return {
      ready: t('plugins.setup.statusReady', 'Ready.'),
      installed: t('plugins.setup.statusStarting', 'Starting…'),
      starting: t('plugins.setup.statusStarting', 'Starting…'),
      degraded: t('plugins.setup.statusDegraded', 'It hit a problem and is restarting.'),
      disabled: t('plugins.setup.statusDisabled', 'Turned off. Turn it on in its settings.'),
      configuration_invalid: t('plugins.setup.statusNeedsSetup', 'Waiting for setup: its settings, credentials or setup checks are not complete yet.'),
      failed: t('plugins.setup.statusFailed', 'Stopped. It could not start or keep running; its settings say why.'),
    }[summary.status] ?? summary.status;
  };
  const states = summaries.map(itemState);
  // A plugin's only node usually shares its name; saying it twice reads as noise.
  const rowTitle = (summary: CapabilityNodeSummary): string => summaries.length === 1 && capabilityNodeTitle(summary) === target.title
    ? t('plugins.setup.status', 'Status') : capabilityNodeTitle(summary);
  const needed = states.filter((state) => state === 'needed').length;
  const ready = summaries.length > 0 && states.every((state) => state === 'done');
  const lead = ready ? t('plugins.setup.readyLead', 'It reports ready.')
    : summaries.length === 0 ? t('plugins.setup.waitingLead', 'Waiting for it to report…')
      : needed > 0 ? t('plugins.setup.neededLead', 'It needs you before it can run.')
        : t('plugins.setup.startingLead', 'It is starting.');
  return <Panel aria-labelledby="capability-setup-title">
    <h2 id="capability-setup-title">{ready ? t('plugins.setup.readyTitle', '{title} is ready', { title: target.title }) : t('plugins.setup.title', 'Set up {title}', { title: target.title })}</h2>
    <Lead>{lead}</Lead>
    <Items>
      {summaries.length === 0 ? <Item>
        <Mark aria-hidden="true" $state="waiting"><Spinner size={14} /></Mark>
        <div><h3>{target.title}</h3><p>{t('plugins.setup.notReported', 'The host has not reported it yet. This takes a few seconds after it starts.')}</p></div>
        <Side />
      </Item> : null}
      {summaries.map((summary, index) => {
        const state = states[index];
        return <Item key={capabilityNodeKey(summary)}>
          <Mark aria-hidden="true" $state={state}>{state === 'done' ? <FiCheck /> : state === 'waiting' ? <Spinner size={14} /> : <FiCircle />}</Mark>
          <div>
            <h3>{rowTitle(summary)}</h3>
            <p>{detail(summary)}</p>
            {state === 'needed' && withChecks.has(summary.nodeId) ? <NodePreflightPanel pluginId={summary.pluginId} nodeId={summary.nodeId} /> : null}
          </div>
          <Side>
            <StatusPill tone={state === 'done' ? 'healthy' : state === 'needed' ? 'live' : 'neutral'}>
              {{ done: t('plugins.setup.done', 'Done'), waiting: t('plugins.setup.waiting', 'Waiting'), needed: t('plugins.setup.needed', 'Needed') }[state]}
            </StatusPill>
            {state === 'needed' && onManage ? <Button variant="outline" size="sm" onClick={() => onManage(target.pluginId)}>{t('plugins.setup.openSettings', 'Open settings')}</Button> : null}
          </Side>
        </Item>;
      })}
    </Items>
    <Actions>
      {surfaces.map(({ surface, reachable }, index) => reachable
        ? <OpenLink key={surface.url} href={surface.url} target="_blank" rel="noopener noreferrer" $primary={index === 0}>
          {t('plugins.setup.open', 'Open {surface}', { surface: surface.title })}<FiExternalLink aria-hidden="true" />
        </OpenLink>
        : <Button key={surface.url} variant="outline" disabled>{t('plugins.setup.open', 'Open {surface}', { surface: surface.title })}</Button>)}
      {onManage ? <Button variant="ghost" onClick={() => onManage(target.pluginId)}>{t('plugins.setup.managePlugin', 'Manage plugin')}</Button> : null}
      <Button variant="ghost" onClick={onBack}>{t('plugins.catalog.backToBrowse', 'Back to Browse')}</Button>
    </Actions>
    {unreachable.length > 0 ? <Hint>{t('plugins.setup.onHostOnly', '{surface} opens only from a browser running on the host that runs it.', { surface: unreachable.map((item) => item.surface.title).join(', ') })}</Hint> : null}
    {surfaces.length === 0 && summaries.length > 0 ? <Hint>{ready ? t('plugins.setup.noScreens', 'It opens no screens of its own; its capabilities are ready for clients and tools.')
      : t('plugins.setup.screensLater', 'Its screens appear here once it is ready.')}</Hint> : null}
  </Panel>;
}

const Panel = styled(Surface)`
  max-width: 720px; padding: 28px; display: flex; flex-direction: column; gap: 14px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
`;
const Lead = styled.p`margin: -6px 0 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14px;`;
const Items = styled.ul`list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column;`;
const Item = styled.li`
  display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; gap: 14px; align-items: center; padding: 14px 4px; border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  h3 { margin: 0; font-size: 15px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
  p { margin: 3px 0 0; font-size: 13.5px; line-height: 1.45; color: ${({ theme }) => theme.colors.textSecondary}; overflow-wrap: anywhere; }
  @media (max-width: 600px) { grid-template-columns: 28px minmax(0, 1fr); > div:last-child { grid-column: 2; } }
`;
const Mark = styled.span<{ $state: ItemState }>`
  display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 50%;
  color: ${({ theme, $state }) => $state === 'done' ? theme.colors.healthy : $state === 'needed' ? theme.colors.liveText : theme.colors.textMuted};
`;
const Side = styled.div`display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end;`;
const Actions = styled.div`display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 6px;`;
// The first screen is styled as the primary button: the action this step leads to.
const OpenLink = styled.a<{ $primary: boolean }>`
  display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 10px; font-weight: 600; font-size: 14px; text-decoration: none;
  color: ${({ theme }) => theme.colors.accentText}; border: 1px solid ${({ theme, $primary }) => $primary ? theme.colors.goldDim : theme.colors.border}; background: transparent;
  &:hover { background: ${({ theme }) => theme.colors.goldBg}; }
`;
const Hint = styled.p`margin: 0; font-size: 12.5px; color: ${({ theme }) => theme.colors.metadataText};`;
