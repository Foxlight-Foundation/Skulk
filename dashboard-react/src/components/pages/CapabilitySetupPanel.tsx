import { useMemo, useState } from 'react';
import styled from 'styled-components';
import { FiAlertCircle, FiCheckCircle, FiExternalLink, FiPower, FiXCircle } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useClusterState } from '../../hooks/useClusterState';
import {
  pluginRefusalDetail, pluginRequestRefused, useConfigurePluginNodeMutation, useGetPluginNodesQuery, useLazyGetNodeConfigurationQuery,
} from '../../store/endpoints/plugins';
import { capabilityNodeKey, capabilityNodeTitle, type CapabilityNodeStatus, type CapabilityNodeSummary, type CapabilityNodeSurface } from '../../types/capabilityNodes';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { StatusPill, Surface, type StatusTone } from '../common/Surfaces';
import { NodePreflightPanel } from './NodePreflightPanel';
import type { SetupTarget } from './CatalogInstallProgress';
import { turnOnNodes, type TurnOnRefusal } from './turnOnCapability';
import { hostedCapabilityNodes, openableSurfaces } from './installedCapability';

/** Props for the setup page of one installed plugin. */
export interface CapabilitySetupPanelProps {
  target: SetupTarget;
  /**
   * Open the plugin's settings, credentials and setup actions, on one node's
   * settings when a node is named. The settings actions are hidden without it.
   */
  onManage?: (pluginId: string, nodeId?: string) => void;
  /** Leave the setup page for the list of installed plugins. */
  onDone: () => void;
}

/** What one node's status asks of the operator. */
type ItemState = 'done' | 'waiting' | 'off' | 'settings' | 'stopped';

// The plugin service decides readiness: it only reports `ready` after the
// node's settings, credentials and setup checks admitted a start and the
// node answered its health check. Transitional states settle by themselves;
// a node that is off is turned on from here; the rest wait for its settings.
const STATUS_STATE: Record<CapabilityNodeStatus, ItemState> = {
  ready: 'done',
  installed: 'waiting',
  starting: 'waiting',
  degraded: 'waiting',
  disabled: 'off',
  configuration_invalid: 'settings',
  failed: 'stopped',
};

const STATE_TONE: Record<ItemState, StatusTone> = { done: 'healthy', waiting: 'live', off: 'neutral', settings: 'live', stopped: 'danger' };

/**
 * The page an installed plugin opens on until it runs: what each of its
 * nodes reports, and the one next step.
 *
 * Every capability node of the plugin gets one line from the host's live
 * summary. A plugin that is only turned off is turned on here directly; one
 * that needs settings or credentials opens exactly that node's settings.
 * Once it reports ready, its screens open in a new tab from here (and from
 * its node in the Cluster view); nothing of the plugin is embedded here.
 */
export function CapabilitySetupPanel({ target, onManage, onDone }: CapabilitySetupPanelProps) {
  const { t } = useSkulkTranslation();
  const { capabilityNodes, localNodeId } = useClusterState();
  const plugins = useGetPluginNodesQuery();
  const [readConfiguration] = useLazyGetNodeConfigurationQuery();
  const [configure] = useConfigurePluginNodeMutation();
  const [turning, setTurning] = useState(false);
  // Nodes turned on (here, or by the install that led here) that the host has
  // not yet reported leaving `disabled`.
  const [turnedOn, setTurnedOn] = useState<ReadonlySet<string>>(() => new Set((target.turnedOn ?? []).map((nodeId) => capabilityNodeKey({ pluginId: target.pluginId, nodeId }))));
  // A refused turn-on, here or by the install, with the host's reason.
  const [refusal, setRefusal] = useState<TurnOnRefusal | null>(target.turnOnRefusal ?? null);
  // A decided refusal runs the node's setup checks, which say what is missing.
  const [checkRequest, setCheckRequest] = useState(target.turnOnRefusal && target.turnOnRefusal.status !== null ? 1 : 0);
  const hosted = useMemo(() => hostedCapabilityNodes(capabilityNodes, target.pluginId), [capabilityNodes, target.pluginId]);
  const summaries = useMemo(() => hosted.map((item) => item.summary), [hosted]);
  const withChecks = new Set(plugins.data?.find((plugin) => plugin.pluginId === target.pluginId)?.nodes.filter((node) => node.preflightAvailable).map((node) => node.nodeId) ?? []);
  // As in the topology, a loopback screen opens only from a browser on its own host.
  const surfaces = useMemo(() => openableSurfaces(hosted, localNodeId, window.location.hostname), [hosted, localNodeId]);
  const reachable = surfaces.filter((item) => item.reachable);
  const unreachable = surfaces.filter((item) => !item.reachable);

  // Each new report from the host settles what this page remembers: a node
  // turned on stops counting as such once it leaves `disabled` (so one the
  // owner turns off again later reads as off), and a refusal is over once its
  // node is no longer off. Adjusted while rendering, as React recommends for
  // state that follows a changing input.
  const [reported, setReported] = useState(summaries);
  if (reported !== summaries) {
    setReported(summaries);
    const left = summaries.filter((summary) => summary.status !== 'disabled' && turnedOn.has(capabilityNodeKey(summary)));
    if (left.length > 0) setTurnedOn(new Set([...turnedOn].filter((key) => !left.some((summary) => capabilityNodeKey(summary) === key))));
    if (refusal && summaries.some((summary) => summary.nodeId === refusal.nodeId && summary.status !== 'disabled')) setRefusal(null);
  }

  // A status this dashboard does not know yet, or a silent plugin service, is shown as stopped.
  const itemState = (summary: CapabilityNodeSummary): ItemState => {
    if (!summary.ownerAvailable) return 'stopped';
    if (summary.status === 'disabled' && turnedOn.has(capabilityNodeKey(summary))) return 'waiting';
    return STATUS_STATE[summary.status] ?? 'stopped';
  };
  const detail = (summary: CapabilityNodeSummary): string => {
    if (!summary.ownerAvailable) return t('plugins.setup.ownerUnavailable', 'The plugin service on this host is not answering. It reports again once it is back.');
    if (summary.status === 'disabled' && turnedOn.has(capabilityNodeKey(summary))) return t('plugins.setup.statusTurningOn', 'Turning on…');
    return {
      ready: t('plugins.setup.statusRunning', 'Running.'),
      installed: t('plugins.setup.statusStarting', 'Starting…'),
      starting: t('plugins.setup.statusStarting', 'Starting…'),
      degraded: t('plugins.setup.statusDegraded', 'It hit a problem and is restarting.'),
      disabled: t('plugins.setup.statusOff', 'Turned off.'),
      configuration_invalid: t('plugins.setup.statusNeedsSetup', 'Waiting for setup: its settings, credentials or setup checks are not complete yet.'),
      failed: t('plugins.setup.statusFailed', 'Stopped. It could not start or keep running; its settings say why.'),
    }[summary.status] ?? summary.status;
  };
  const stateLabel = (state: ItemState): string => ({
    done: t('plugins.setup.pillRunning', 'Running'),
    waiting: t('plugins.setup.pillStarting', 'Starting'),
    off: t('plugins.setup.pillOff', 'Off'),
    settings: t('plugins.setup.pillNeedsSettings', 'Needs settings'),
    stopped: t('plugins.setup.pillStopped', 'Stopped'),
  })[state];
  const states = summaries.map(itemState);
  // A plugin's only node usually shares its name; saying it twice reads as noise.
  const rowTitle = (summary: CapabilityNodeSummary): string => summaries.length === 1 && capabilityNodeTitle(summary) === target.title
    ? t('plugins.setup.status', 'Status') : capabilityNodeTitle(summary);
  const ready = summaries.length > 0 && states.every((state) => state === 'done');
  const off = summaries.filter((_summary, index) => states[index] === 'off');
  const needsSettings = summaries.find((_summary, index) => states[index] === 'settings' || states[index] === 'stopped') ?? null;
  // The next step decides the one primary action: the settings a node needs,
  // else turning on what is off, else opening it. A refused turn-on stays on
  // the Turn on step, with the host's reason and the settings beside it; an
  // unconfirmed one may have landed, and turning on again re-reads it first.
  const decided = refusal !== null && refusal.status !== null;
  const next: 'settings' | 'turn-on' | 'open' | null = needsSettings && onManage ? 'settings'
    : off.length > 0 ? 'turn-on'
      : ready && reachable.length > 0 ? 'open' : null;

  const turnOn = async () => {
    if (turning || off.length === 0) return;
    setTurning(true);
    setRefusal(null);
    const outcome = await turnOnNodes({ readConfiguration: (address) => readConfiguration(address, false), configure }, target.pluginId,
      off.map((summary) => summary.nodeId), pluginRefusalDetail, pluginRequestRefused);
    const keys = outcome.enabled.map((nodeId) => capabilityNodeKey({ pluginId: target.pluginId, nodeId }));
    if (keys.length > 0) setTurnedOn((current) => new Set([...current, ...keys]));
    if (outcome.refusal) {
      setRefusal(outcome.refusal);
      // Its own checks say which setting, credential or prerequisite is missing.
      if (outcome.refusal.status !== null && withChecks.has(outcome.refusal.nodeId)) setCheckRequest((count) => count + 1);
    }
    setTurning(false);
  };

  const lead = ready ? (surfaces.length > 0 ? t('plugins.setup.readyLead', 'It is running.') : t('plugins.setup.noScreens', 'It opens no screens of its own; its capabilities are ready for clients and tools.'))
    : summaries.length === 0 ? t('plugins.setup.waitingLead', 'Waiting for it to report…')
      : next === 'settings' ? t('plugins.setup.settingsLead', 'It needs its settings before it can run.')
        : next === 'turn-on' ? t('plugins.setup.offLead', 'It is turned off. Turn it on to start it.')
          : t('plugins.setup.startingLead', 'It is starting. This page updates by itself.');
  const openLabel = (surface: CapabilityNodeSurface) => t('plugins.setup.open', 'Open {surface}', { surface: surface.title });
  // Checks are shown when something needs more than turning on.
  const checked = summaries.filter((summary, index) => withChecks.has(summary.nodeId)
    && (states[index] === 'settings' || states[index] === 'stopped' || (decided && refusal?.nodeId === summary.nodeId)));
  return <Panel aria-labelledby="capability-setup-title">
    <Header>
      <h2 id="capability-setup-title">{ready ? t('plugins.setup.readyTitle', '{title} is ready', { title: target.title }) : t('plugins.setup.installedTitle', '{title} is installed', { title: target.title })}</h2>
      <Lead>{lead}</Lead>
    </Header>
    <Items>
      {summaries.length === 0 ? <Item>
        <Mark aria-hidden="true" $state="waiting"><Spinner size={16} /></Mark>
        <div><h3>{target.title}</h3><p>{t('plugins.setup.notReported', 'The host has not reported it yet. This takes a few seconds after it starts.')}</p></div>
      </Item> : null}
      {summaries.map((summary, index) => {
        const state = states[index];
        return <Item key={capabilityNodeKey(summary)}>
          <Mark aria-hidden="true" $state={state}>{{
            done: <FiCheckCircle />, waiting: <Spinner size={16} />, off: <FiPower />, settings: <FiAlertCircle />, stopped: <FiXCircle />,
          }[state]}</Mark>
          <div><h3>{rowTitle(summary)}</h3><p>{detail(summary)}</p></div>
          <StatusPill tone={STATE_TONE[state]}>{stateLabel(state)}</StatusPill>
        </Item>;
      })}
    </Items>
    {refusal && decided ? <Problem role="alert">
      <strong>{t('plugins.setup.turnOnRefused', 'It was not turned on.')}</strong>{' '}
      {refusal.detail ? t('plugins.setup.hostSaid', 'The host said: {reason}', { reason: refusal.detail })
        : t('plugins.setup.turnOnRefusedStatus', 'The host refused the change (HTTP {status}).', { status: refusal.status ?? '' })}
      {' '}{withChecks.has(refusal.nodeId) ? t('plugins.setup.refusedChecks', 'Its setup checks below show what is missing.') : t('plugins.setup.refusedSettings', 'Its settings show what is missing.')}
    </Problem> : null}
    {refusal && !decided ? <Problem role="alert">
      {t('plugins.setup.turnOnUnconfirmed', 'The host did not confirm turning it on. This page shows its status as soon as it reports again; Turn on checks first and never repeats a change that landed.')}
    </Problem> : null}
    <Actions>
      {next === 'settings' && needsSettings && onManage ? <Button variant="solid" onClick={() => onManage(target.pluginId, needsSettings.nodeId)}>{t('plugins.setup.openSettings', 'Open settings')}</Button> : null}
      {next === 'turn-on' ? <Button variant="solid" loading={turning} onClick={() => void turnOn()}>{turning ? t('plugins.setup.turningOn', 'Turning on…') : t('plugins.setup.turnOn', 'Turn on')}</Button> : null}
      {next === 'open' ? reachable.map(({ surface }, index) => <OpenLink key={surface.url} href={surface.url} target="_blank" rel="noopener noreferrer" $primary={index === 0}>
        {openLabel(surface)}<FiExternalLink aria-hidden="true" />
      </OpenLink>) : null}
      {next === 'settings' && off.length > 0 ? <Button variant="outline" loading={turning} onClick={() => void turnOn()}>{t('plugins.setup.turnOn', 'Turn on')}</Button> : null}
      {ready ? unreachable.map(({ surface }) => <Button key={surface.url} variant="outline" disabled>{openLabel(surface)}</Button>) : null}
      {onManage && next !== 'settings' ? (refusal && decided
        ? <Button variant="outline" onClick={() => onManage(target.pluginId, refusal.nodeId)}>{t('plugins.setup.openSettings', 'Open settings')}</Button>
        : <Button variant="outline" onClick={() => onManage(target.pluginId)}>{t('plugins.setup.settings', 'Settings')}</Button>) : null}
      <Button variant="outline" onClick={onDone}>{t('plugins.setup.done', 'Done')}</Button>
    </Actions>
    {next === 'turn-on' ? <Hint>{t('plugins.setup.turnOnHelp', 'Turning it on runs its setup checks again.')}{target.spendsMoney === false ? ` ${t('plugins.setup.cannotSpend', 'It cannot spend money.')}` : ''}</Hint> : null}
    {next === 'open' ? <Hint>{t('plugins.setup.opensInNewTab', 'Opens in a new tab. You can also open it from its node in the Cluster view.')}</Hint> : null}
    {!ready && summaries.length > 0 && surfaces.length === 0 ? <Hint>{t('plugins.setup.openLater', 'Once it is running, Open {title} appears here and opens it in a new tab.', { title: target.title })}</Hint> : null}
    {unreachable.length > 0 ? <Hint>{t('plugins.setup.onHostOnly', '{surface} opens only from a browser running on the host that runs it.', { surface: unreachable.map((item) => item.surface.title).join(', ') })}</Hint> : null}
    {checked.length > 0 ? <Checks aria-label={t('plugins.preflight', 'Setup checks')}>
      <h3>{t('plugins.setup.checksTitle', 'Setup checks')}</h3>
      <p>{t('plugins.setup.checksHelp', 'They check what it needs to run. Checking changes nothing.')}</p>
      {checked.map((summary) => <div key={capabilityNodeKey(summary)}>
        {checked.length > 1 ? <h4>{capabilityNodeTitle(summary)}</h4> : null}
        <NodePreflightPanel pluginId={summary.pluginId} nodeId={summary.nodeId} runRequest={refusal?.nodeId === summary.nodeId ? checkRequest : 0} />
      </div>)}
    </Checks> : null}
  </Panel>;
}

const Panel = styled(Surface)`
  max-width: 720px; padding: 28px; display: flex; flex-direction: column; gap: 18px;
  @media (max-width: 600px) { padding: 20px 16px; }
`;
const Header = styled.div`
  display: flex; flex-direction: column; gap: 6px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; overflow-wrap: anywhere; }
`;
const Lead = styled.p`margin: 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14px; line-height: 1.5;`;
const Items = styled.ul`
  list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column;
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: ${({ theme }) => theme.radii.lg}; overflow: hidden;
`;
const Item = styled.li`
  display: grid; grid-template-columns: 24px minmax(0, 1fr) auto; gap: 14px; align-items: center; padding: 14px 16px;
  & + & { border-top: 1px solid ${({ theme }) => theme.colors.border}; }
  h3 { margin: 0; font-size: 15px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; overflow-wrap: anywhere; }
  p { margin: 3px 0 0; font-size: 13.5px; line-height: 1.45; color: ${({ theme }) => theme.colors.textSecondary}; overflow-wrap: anywhere; }
  @media (max-width: 600px) { grid-template-columns: 24px minmax(0, 1fr); > :last-child { grid-column: 2; } }
`;
const Mark = styled.span<{ $state: ItemState }>`
  display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; font-size: 20px;
  color: ${({ theme, $state }) => ({ done: theme.colors.healthy, waiting: theme.colors.textMuted, off: theme.colors.textMuted, settings: theme.colors.liveText, stopped: theme.colors.error })[$state]};
`;
const Problem = styled.p`
  margin: 0; padding: 12px 14px; border-radius: ${({ theme }) => theme.radii.lg}; font-size: 13.5px; line-height: 1.5;
  border: 1px solid ${({ theme }) => theme.colors.borderDanger}; background: ${({ theme }) => theme.colors.errorBg}; color: ${({ theme }) => theme.colors.text};
  overflow-wrap: anywhere;
`;
const Actions = styled.div`display: flex; flex-wrap: wrap; align-items: center; gap: 10px;`;
// The first screen is styled as the primary action: what this step leads to.
const OpenLink = styled.a<{ $primary: boolean }>`
  box-sizing: border-box; display: inline-flex; align-items: center; gap: 8px; min-height: 36px; padding: 0 16px;
  border-radius: ${({ theme }) => theme.radii.md}; font: 600 ${({ theme }) => theme.fontSizes.sm} ${({ theme }) => theme.fonts.body}; text-decoration: none;
  color: ${({ theme, $primary }) => $primary ? theme.colors.textOnAccent : theme.colors.textSecondary};
  background: ${({ theme, $primary }) => $primary ? theme.colors.actionFill : 'transparent'};
  border: 1px solid ${({ theme, $primary }) => $primary ? 'transparent' : theme.colors.border};
  &:hover { background: ${({ theme, $primary }) => $primary ? theme.colors.actionHoverFill : theme.colors.goldBg}; }
  &:focus-visible { outline: none; box-shadow: ${({ theme }) => theme.colors.focusRing}; }
`;
const Hint = styled.p`margin: -8px 0 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.metadataText};`;
const Checks = styled.section`
  display: flex; flex-direction: column; gap: 10px; padding-top: 18px; border-top: 1px solid ${({ theme }) => theme.colors.border};
  h3 { margin: 0; font-size: 15px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
  h4 { margin: 4px 0 8px; font-size: 14px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
  > p { margin: -4px 0 4px; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
