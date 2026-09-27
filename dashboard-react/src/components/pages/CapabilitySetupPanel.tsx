import { useCallback, useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { FiCheck, FiCircle, FiExternalLink } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useClusterState } from '../../hooks/useClusterState';
import { useAppDispatch } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import { modelSupportsTextChat, type ModelInfo } from '../../types/models';
import { runDescriptorAction } from '../topology/capabilityActions';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { StatusPill, Surface } from '../common/Surfaces';
import { engineLabel, parseVideoReadiness, servesVideoReadiness, type VideoReadiness } from './catalogJourney';
import type { SetupTarget } from './CatalogInstallProgress';

/** Props for the setup checklist of one installed capability. */
export interface CapabilitySetupPanelProps {
  target: SetupTarget;
  onBack: () => void;
}

type ItemState = 'done' | 'needed' | 'optional' | 'checking';
interface SetupItem { id: string; title: string; detail: string; state: ItemState; action?: { label: string; run: () => void } }

/** The ready chat models: every runner of the placement reports ready. */
function readyModelIds(instances: ReturnType<typeof useClusterState>['instances'], runners: ReturnType<typeof useClusterState>['runners']): string[] {
  const ready: string[] = [];
  for (const wrapper of Object.values(instances)) {
    const inner = wrapper.MlxRingInstance ?? wrapper.MlxJacclInstance ?? wrapper.LlamaRpcInstance;
    const assignments = inner?.shardAssignments;
    const runnerIds = Object.values(assignments?.nodeToRunner ?? {});
    if (!assignments?.modelId || runnerIds.length === 0) continue;
    if (runnerIds.every((runnerId) => runners[runnerId] && 'RunnerReady' in runners[runnerId])) ready.push(assignments.modelId);
  }
  return ready;
}

/**
 * What an installed capability still needs before it is useful, from live
 * facts: the host's own view of the plugin, the capability's readiness
 * report when it serves one, and the cluster's placements.
 */
export function CapabilitySetupPanel({ target, onBack }: CapabilitySetupPanelProps) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch();
  const { localNodeId, capabilityNodes, instances, runners } = useClusterState();
  const [readiness, setReadiness] = useState<VideoReadiness | null>(null);
  const [readinessError, setReadinessError] = useState(false);
  const [checking, setChecking] = useState(false);
  const [chatModels, setChatModels] = useState<Set<string> | null>(null);
  const video = servesVideoReadiness(target.descriptors);
  const summary = useMemo(() => Object.values(capabilityNodes).flat().find((node) => node.pluginId === target.pluginId) ?? null, [capabilityNodes, target.pluginId]);
  const surface = summary?.surfaces.find((item) => item.kind === 'link') ?? null;

  const check = useCallback(async () => {
    if (!video || !localNodeId) return;
    setChecking(true);
    try {
      const reply = await runDescriptorAction(localNodeId, 'video.readiness', {});
      const parsed = reply.ok ? parseVideoReadiness(reply.result) : null;
      setReadiness(parsed);
      setReadinessError(parsed === null);
    } catch {
      setReadinessError(true);
    } finally {
      setChecking(false);
    }
  }, [localNodeId, video]);

  useEffect(() => { void check(); }, [check]);
  useEffect(() => {
    let alive = true;
    fetch('/v1/models', { cache: 'no-store' })
      .then(async (response) => (response.ok ? await response.json() as { data?: ModelInfo[] } : { data: [] }))
      .then((body) => { if (alive) setChatModels(new Set((body.data ?? []).filter((model) => modelSupportsTextChat(model)).map((model) => model.id))); })
      .catch(() => { if (alive) setChatModels(new Set()); });
    return () => { alive = false; };
  }, []);

  const goTo = (route: 'model-store' | 'cluster') => {
    dispatch(uiActions.setActiveRoute(route));
    const path = route === 'cluster' ? '/' : `/${route}`;
    if (window.location.pathname !== path) history.pushState(null, '', path);
  };
  const readyChat = chatModels ? readyModelIds(instances, runners).find((modelId) => chatModels.has(modelId)) ?? null : null;
  const running = summary?.status === 'ready';
  const items: SetupItem[] = [{
    id: 'running', title: t('plugins.setup.running', '{title} is running', { title: target.title }),
    detail: summary ? (running ? t('plugins.setup.runningDone', 'Its process is up on this host.') : t('plugins.setup.runningStatus', 'The host reports it as {status}.', { status: summary.status.replace('_', ' ') }))
      : t('plugins.setup.runningWaiting', 'Waiting for the host to report it.'),
    state: running ? 'done' : summary ? 'needed' : 'checking',
  }];
  if (video) {
    if (!readiness) {
      items.push({ id: 'readiness', title: t('plugins.setup.readiness', 'Rendering readiness'), detail: readinessError ? t('plugins.setup.readinessError', 'It could not report its readiness yet.') : t('plugins.setup.readinessChecking', 'Checking the fleet…'), state: readinessError ? 'needed' : 'checking' });
    } else {
      items.push({
        id: 'models', title: t('plugins.setup.videoModels', 'Video models'),
        detail: readiness.video_models.length > 0 ? t('plugins.setup.videoModelsOn', 'On for this cluster: {count} in the catalog.', { count: readiness.video_models.length })
          : t('plugins.setup.videoModelsOff', 'Off for this cluster. Start Skulk with SKULK_ENABLE_VIDEO_MODELS=1 on every node.'),
        state: readiness.video_models.length > 0 ? 'done' : 'needed',
      });
      const lanes = readiness.lanes.map((lane) => [lane.node, ...lane.backends.slice(0, 1).map(engineLabel)].join(' · '));
      items.push({
        id: 'lanes', title: t('plugins.setup.renderNode', 'A render node'),
        detail: lanes.length > 0 ? lanes.join(', ') : t('plugins.setup.renderNodeNone', 'No node runs a video engine. Video renders need a Linux node with an NVIDIA or AMD GPU.'),
        state: lanes.length > 0 ? 'done' : 'needed',
      });
      const model = readiness.default_model;
      items.push({
        id: 'placed', title: model ? model.split('/').pop() ?? model : t('plugins.setup.renderModel', 'A video model'),
        detail: readiness.default_host ? t('plugins.setup.placed', 'Placed on {host} and ready.', { host: readiness.default_host })
          : t('plugins.setup.notPlaced', 'Not placed on a video engine yet.'),
        state: readiness.default_host ? 'done' : 'needed',
        ...(readiness.default_host ? {} : { action: { label: t('plugins.setup.downloadPlace', 'Download and place'), run: () => goTo('model-store') } }),
      });
    }
    items.push({
      id: 'chat', title: t('plugins.setup.refine', 'Prompt refinement'),
      detail: readyChat ? t('plugins.setup.refineReady', '{model} is ready.', { model: readyChat.split('/').pop() ?? readyChat }) : t('plugins.setup.refineNone', 'Place a chat model to let it refine prompts.'),
      state: readyChat ? 'done' : chatModels === null ? 'checking' : 'optional',
      ...(readyChat || chatModels === null ? {} : { action: { label: t('plugins.setup.placeChat', 'Place a chat model'), run: () => goTo('model-store') } }),
    });
  }
  const needed = items.filter((item) => item.state === 'needed').length;
  const pending = items.some((item) => item.state === 'checking');
  const ready = needed === 0 && !pending && (!video || readiness?.ready === true);
  const reported = readiness && !readiness.ready ? readiness.reasons : [];
  return <Panel aria-labelledby="capability-setup-title">
    <h2 id="capability-setup-title">{ready ? t('plugins.setup.readyTitle', '{title} is ready', { title: target.title }) : t('plugins.setup.title', 'Set up {title}', { title: target.title })}</h2>
    <Lead>{ready ? t('plugins.setup.readyLead', 'Everything it needs is in place.')
      : needed > 0 ? t('plugins.setup.stepsLeft', '{count} left before it is ready.', { count: needed }) : t('plugins.setup.checkingLead', 'Checking what it needs…')}</Lead>
    <Items>{items.map((item) => <Item key={item.id}>
      <Mark aria-hidden="true" $state={item.state}>{item.state === 'done' ? <FiCheck /> : item.state === 'checking' ? <Spinner size={14} /> : <FiCircle />}</Mark>
      <div><h3>{item.title}</h3><p>{item.detail}</p></div>
      <Side>
        <StatusPill tone={item.state === 'done' ? 'healthy' : item.state === 'needed' ? 'live' : 'neutral'}>
          {{ done: t('plugins.setup.done', 'Done'), needed: t('plugins.setup.needed', 'Needed'), optional: t('plugins.setup.optional', 'Optional'), checking: t('plugins.setup.checking', 'Checking') }[item.state]}
        </StatusPill>
        {item.action ? <Button variant="outline" size="sm" onClick={item.action.run}>{item.action.label}</Button> : null}
      </Side>
    </Item>)}</Items>
    {reported.length > 0 ? <Reported>{t('plugins.setup.reported', 'It reports: {reasons}.', { reasons: reported.join('; ') })}</Reported> : null}
    <Actions>
      {surface?.url ? <OpenLink href={surface.url} target="_blank" rel="noopener noreferrer" aria-disabled={!ready || !surface.ready} $disabled={!ready || !surface.ready}
        onClick={(event) => { if (!ready || !surface.ready) event.preventDefault(); }}>
        {t('plugins.setup.open', 'Open {surface}', { surface: surface.title })}<FiExternalLink aria-hidden="true" />
      </OpenLink> : null}
      {video ? <Button variant="ghost" disabled={checking} onClick={() => void check()}>{t('plugins.setup.checkAgain', 'Check again')}</Button> : null}
      <Button variant="ghost" onClick={onBack}>{t('plugins.catalog.backToBrowse', 'Back to Browse')}</Button>
    </Actions>
    {surface?.url && !ready ? <Hint>{t('plugins.setup.openHint', 'Available once the needed steps are done.')}</Hint> : null}
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
const Reported = styled.p`margin: 0; font-size: 13px; color: ${({ theme }) => theme.colors.metadataText}; line-height: 1.5;`;
const Actions = styled.div`display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 6px;`;
// Styled as the primary button: the one action this screen leads to.
const OpenLink = styled.a<{ $disabled: boolean }>`
  display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 10px; font-weight: 600; font-size: 14px; text-decoration: none;
  color: ${({ theme }) => theme.colors.accentText}; border: 1px solid ${({ theme }) => theme.colors.goldDim}; background: transparent;
  opacity: ${({ $disabled }) => $disabled ? .45 : 1}; cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  &:hover { background: ${({ theme, $disabled }) => $disabled ? 'transparent' : theme.colors.goldBg}; }
`;
const Hint = styled.p`margin: 0; font-size: 12.5px; color: ${({ theme }) => theme.colors.metadataText};`;
