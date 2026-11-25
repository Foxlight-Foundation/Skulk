import { useState } from 'react';
import styled from 'styled-components';
import { FiCpu } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useGetRawStateQuery } from '../../store/endpoints/cluster';
import { engineChangeErrorDetail, useChangeEngineMutation, useGetEnginesQuery } from '../../store/endpoints/engines';
import type { EngineChangeRequest, EngineEntry, EngineLoadResult, EngineNodePackage } from '../../types/engines';
import { formatBytes } from '../../utils/format';
import { Button } from '../common/Button';
import { SectionLabel, StatusPill, type StatusTone } from '../common/Surfaces';
import { engineBuildRows, engineProblemSentence, laneLabel, shortBuild } from './enginesView';
import type { Translate } from './remotePairingMessages';

/** How often the open panel reads the report again, in milliseconds. */
const REFRESH_MS = 5000;

const Content = styled.div`
  overflow-y: auto; min-height: 0; flex: 1; padding: 20px;
  display: flex; flex-direction: column; gap: 16px;
`;
const Meta = styled.p`
  margin: 0; font-size: 12.5px; line-height: 1.5; overflow-wrap: anywhere;
  color: ${({ theme }) => theme.colors.textSecondary};
`;
const Mono = styled.span`font: 11.5px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.metadataText};`;
const Group = styled.section`display: flex; flex-direction: column; gap: 10px;`;
const Machines = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 12px;
  background: ${({ theme }) => theme.colors.surface}; overflow: hidden;
`;
const Machine = styled.article`
  padding: 12px 14px; display: grid; grid-template-columns: 32px minmax(0, 1fr); gap: 12px;
  & + & { border-top: 1px solid ${({ theme }) => theme.colors.borderLight}; }
  > svg { width: 32px; height: 32px; padding: 6px; border-radius: 8px; background: ${({ theme }) => theme.colors.selected}; }
`;
const Detail = styled.div`
  min-width: 0; display: flex; flex-direction: column; gap: 6px;
  strong { font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
`;
const Heading = styled.div`display: flex; align-items: center; flex-wrap: wrap; gap: 6px;`;
const Builds = styled.ul`list-style: none; margin: 2px 0 0; padding: 0; display: flex; flex-direction: column; gap: 6px;`;
const Build = styled.li`display: flex; align-items: center; flex-wrap: wrap; gap: 8px;`;
// Kept together so a narrow panel wraps both buttons to one line, not one each.
const BuildActions = styled.span`display: inline-flex; align-items: center; gap: 8px;`;
// The section label is set in capitals; an engine's identity is not.
const EngineId = styled(Mono)`text-transform: none; letter-spacing: 0; margin-left: 6px;`;
const Actions = styled.div`display: flex; gap: 8px; flex-wrap: wrap; margin-top: 2px;`;

function loadSentence(result: EngineLoadResult, t: Translate): string {
  const values = { model: result.modelId, when: new Date(result.atUnixMs).toLocaleString() };
  if (result.outcome === 'loaded') return t('engines.lastLoad.loaded', 'Last load: {model} loaded ({when}).', values);
  if (result.outcome === 'refused') {
    return t('engines.lastLoad.refused', 'Last load: the engine refused {model} ({code}, {when}).', { ...values, code: result.code ?? '' });
  }
  return t('engines.lastLoad.failed', 'Last load: {model} did not load ({when}). The machine\'s log has the reason.', values);
}

/** One engine on one machine: what it holds, and the changes a person may make. */
export function EngineMachineRow({ engine, entry, machineName, busy, onChange }: {
  engine: EngineEntry;
  entry: EngineNodePackage;
  /** The machine's name as the rest of the dashboard shows it. */
  machineName: string;
  /** A change is under way, so no second one may start. */
  busy: boolean;
  onChange: (request: EngineChangeRequest) => void;
}) {
  const { t } = useSkulkTranslation();
  // The build whose removal awaits a second click, or 'all' for the whole engine.
  const [confirming, setConfirming] = useState<string | null>(null);
  const held = entry.package;
  const builds = engineBuildRows(held);
  const target = { engineId: engine.engineId, nodeId: entry.nodeId, lane: held.lane };
  const [tone, status]: [StatusTone, string] = held.status === 'active'
    ? ['healthy', t('engines.status.active', 'Installed')]
    : held.status === 'unusable'
      ? ['danger', t('engines.status.unusable', 'Cannot be used')]
      : ['neutral', t('engines.status.offered', 'Not installed yet')];
  const lastLoad = held.loadResults?.[0];
  const change = (request: EngineChangeRequest) => { setConfirming(null); onChange(request); };
  return <Machine>
    <FiCpu size={24} aria-hidden="true" />
    <Detail>
      <Heading>
        <strong>{machineName}</strong>
        <StatusPill tone={tone}>{status}</StatusPill>
        {held.channel === 'candidate' && <StatusPill>{t('engines.testBuild', 'Test build')}</StatusPill>}
      </Heading>
      <Meta>
        {laneLabel(held.lane, t)}
        {held.engineVersion ? ` · ${t('engines.version', 'version {version}', { version: held.engineVersion })}` : ''}
        {held.status !== 'offered' && held.overheadBytes ? ` · ${t('engines.overhead', 'sets aside {size} of memory beside each model', { size: formatBytes(held.overheadBytes) })}` : ''}
      </Meta>
      {held.status === 'offered' && <Meta>{held.downloadBytes
        ? t('engines.offeredSize', 'This machine installs it the first time a model placed here needs it (about {size} to download).', { size: formatBytes(held.downloadBytes) })
        : t('engines.offered', 'This machine installs it the first time a model placed here needs it.')}</Meta>}
      {held.status === 'unusable' && <Meta role="alert">{engineProblemSentence(held.problem, t)}</Meta>}
      {lastLoad && <Meta>{loadSentence(lastLoad, t)}</Meta>}
      {builds.length > 0 && <Builds aria-label={t('engines.builds', 'Builds on {machine}', { machine: machineName })}>
        {builds.map(row => <Build key={row.build}>
          <Mono title={row.build}>{t('engines.buildLabel', 'build {build}', { build: shortBuild(row.build) })}</Mono>
          {row.role === 'inUse' && <StatusPill tone="live">{t('engines.buildInUse', 'Used for new launches')}</StatusPill>}
          {row.role === 'unusable' && <StatusPill tone="danger">{t('engines.status.unusable', 'Cannot be used')}</StatusPill>}
          {row.role === 'earlier' && <StatusPill>{t('engines.buildEarlier', 'Earlier build')}</StatusPill>}
          {(row.role === 'earlier' || row.role === 'other') && confirming !== row.build && <BuildActions>
            <Button size="sm" variant="outline" disabled={busy} aria-label={t('engines.useBuildOf', 'Use build {build} on {machine}', { build: shortBuild(row.build), machine: machineName })}
              onClick={() => change({ ...target, action: 'activate', build: row.build })}>{t('engines.useBuild', 'Use this build')}</Button>
            <Button size="sm" variant="ghost" disabled={busy} aria-label={t('engines.removeBuildOf', 'Remove build {build} from {machine}', { build: shortBuild(row.build), machine: machineName })}
              onClick={() => setConfirming(row.build)}>{t('engines.removeBuild', 'Remove')}</Button>
          </BuildActions>}
          {confirming === row.build && <BuildActions>
            <Button size="sm" variant="danger" loading={busy} onClick={() => change({ ...target, action: 'remove', build: row.build })}>{t('engines.confirmRemoveBuild', 'Delete this build')}</Button>
            <Button size="sm" disabled={busy} onClick={() => setConfirming(null)}>{t('common.cancel', 'Cancel')}</Button>
          </BuildActions>}
        </Build>)}
      </Builds>}
      {builds.some(row => row.role === 'inUse') && builds.length > 1 && <Meta>{t('engines.runningKeep', 'A model that is already running keeps the build it started on until you stop it.')}</Meta>}
      {held.status !== 'offered' && <>
        {confirming === 'all' && <Meta>{t('engines.confirmRemoveAllHint', 'Every build of {engine} is deleted from {machine}. Model files are not touched, and a model running on it must be stopped first. The machine installs the engine again the first time a model placed here needs it; to prevent that, turn off "Use add-on engines" in Settings.', { engine: engine.name, machine: machineName })}</Meta>}
        <Actions>
          {confirming === 'all'
            ? <>
              <Button size="sm" variant="danger" loading={busy} onClick={() => change({ ...target, action: 'remove' })}>{t('engines.confirmRemoveAll', 'Remove the engine')}</Button>
              <Button size="sm" disabled={busy} onClick={() => setConfirming(null)}>{t('common.cancel', 'Cancel')}</Button>
            </>
            : <Button size="sm" variant="outline" disabled={busy} aria-label={t('engines.removeAllOf', 'Remove {engine} from {machine}', { engine: engine.name, machine: machineName })}
              onClick={() => setConfirming('all')}>{t('engines.removeAll', 'Remove from this machine')}</Button>}
        </Actions>
      </>}
    </Detail>
  </Machine>;
}

/**
 * Every add-on engine the cluster's machines hold or may install, with the
 * changes a person may make to one a machine holds.
 *
 * Changes are immediate and independent of the Settings draft, like device
 * revocation: each is sent to the machine and the panel shows its answer.
 */
export function EnginesPanel() {
  const { t } = useSkulkTranslation();
  const engines = useGetEnginesQuery(undefined, { pollingInterval: REFRESH_MS });
  // Shares the dashboard's one cluster-state subscription; only names are read.
  const cluster = useGetRawStateQuery();
  const [changeEngine, changing] = useChangeEngineMutation();
  const [failure, setFailure] = useState<string | null>(null);
  const machineName = (nodeId: string) => cluster.data?.nodeIdentities?.[nodeId]?.friendlyName || nodeId.slice(-8);
  const onChange = async (request: EngineChangeRequest) => {
    setFailure(null);
    try {
      await changeEngine(request).unwrap();
    } catch (error) {
      setFailure(engineChangeErrorDetail(error) ?? t('engines.changeFailed', 'The machine did not confirm the change. Check what it holds below before trying again.'));
    }
  };
  const report = engines.data;
  return <Content>
    <Meta>{t('engines.intro', 'An add-on engine is installed on each machine that needs it. Here you can see which machines have one, switch a machine to another build it already has, or remove one.')}</Meta>
    {engines.isLoading && <Meta role="status">{t('engines.loading', 'Loading engines…')}</Meta>}
    {engines.isError && !report && <>
      <Meta role="alert">{engineChangeErrorDetail(engines.error) ?? t('engines.unavailable', 'The list of engines is unavailable.')}</Meta>
      <Actions><Button size="sm" onClick={() => void engines.refetch()}>{t('common.retry', 'Retry')}</Button></Actions>
    </>}
    {failure && <Meta role="alert">{failure}</Meta>}
    {report && !report.packagesEnabled && <Meta>{t('engines.turnedOff', 'Add-on engines are turned off, so machines report none. Turn on "Use add-on engines" in Settings to use them.')}</Meta>}
    {report && report.packagesEnabled && report.engines.length === 0 && <Meta>{t('engines.empty', 'No machine has an add-on engine, and none is offered to one.')}</Meta>}
    {report?.engines.map(engine => <Group key={engine.engineId} aria-label={engine.name}>
      <SectionLabel>{engine.name}<EngineId>{engine.engineId}</EngineId></SectionLabel>
      <Machines>{engine.nodes.map(entry => <EngineMachineRow key={`${entry.nodeId}:${entry.package.lane}`} engine={engine} entry={entry}
        machineName={machineName(entry.nodeId)} busy={changing.isLoading} onChange={request => void onChange(request)} />)}</Machines>
    </Group>)}
  </Content>;
}
