import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import { formatBytes } from '../../utils/format';
import type {
  DiagnosticCaptureResponse,
  DiagnosticsProcess,
  MlxMemorySnapshot,
  NodeDiagnostics,
  RunnerFlightRecorderEntry,
  RunnerSupervisorDiagnostics,
} from '../../types/diagnostics';

/** Props for the read-only node diagnostics drawer. */
export interface DiagnosticsDrawerProps {
  /** Node ID to inspect. Pass null to close the drawer. */
  nodeId: string | null;
  /** Called when the user closes the drawer. */
  onClose: () => void;
}

const Overlay = styled.div`
  position: fixed;
  inset: 0;
  z-index: 70;
  background: ${({ theme }) => theme.colors.overlay};
  display: flex;
  justify-content: flex-end;
`;

const Drawer = styled.aside`
  width: min(560px, 100vw);
  height: 100%;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  border-left: 1px solid ${({ theme }) => theme.colors.borderStrong};
  box-shadow: -18px 0 48px ${({ theme }) => theme.colors.shadowStrong};
  padding: 22px;
  overflow: auto;
`;

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
`;

const Title = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xl};
  color: ${({ theme }) => theme.colors.text};
`;

const Subtitle = styled.div`
  margin-top: 4px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  word-break: break-all;
`;

const Section = styled.section`
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: 14px;
  margin-bottom: 12px;
`;

const SectionTitle = styled.h3`
  margin: 0 0 10px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ theme }) => theme.colors.gold};
`;

const Row = styled.div`
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 10px;
  padding: 3px 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

const Key = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Value = styled.div<{ $warn?: boolean }>`
  color: ${({ $warn, theme }) => $warn ? theme.colors.warningText : theme.colors.textSecondary};
  word-break: break-word;
`;

const Warning = styled.div`
  color: ${({ theme }) => theme.colors.warningText};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.45;
  margin-bottom: 6px;
`;

const Pill = styled.span<{ $tone?: 'good' | 'warn' | 'neutral' }>`
  display: inline-flex;
  align-items: center;
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 6px;
  border: 1px solid ${({ $tone, theme }) =>
    $tone === 'good' ? theme.colors.healthy :
    $tone === 'warn' ? theme.colors.warning :
    theme.colors.borderStrong};
  color: ${({ $tone, theme }) =>
    $tone === 'good' ? theme.colors.healthy :
    $tone === 'warn' ? theme.colors.warningText :
    theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
`;

const ProcessLine = styled.div`
  display: grid;
  grid-template-columns: 62px 72px 74px minmax(0, 1fr);
  gap: 8px;
  align-items: baseline;
  padding: 4px 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};

  &:first-child {
    border-top: none;
  }
`;

const Monospace = styled.code`
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const RunnerCard = styled.div`
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
  padding-top: 10px;
  margin-top: 10px;

  &:first-child {
    border-top: none;
    padding-top: 0;
    margin-top: 0;
  }
`;

const MilestoneList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const MilestoneItem = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 8px 10px;
  background: ${({ theme }) => theme.colors.surface};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const TaskActionList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const TaskActionItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 8px 10px;
  background: ${({ theme }) => theme.colors.surface};
`;

const TaskActionMeta = styled.div`
  min-width: 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const TaskActionButtons = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
`;

const CapturePanel = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 10px;
  margin: 10px 0;
  background: ${({ theme }) => theme.colors.surface};
`;

const CaptureActions = styled.div`
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 8px;
`;

const JsonPreview = styled.pre`
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 8px 0 0;
  padding: 8px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

function memoryUsage(bytes?: number | null): string {
  if (bytes == null) return 'unknown';
  return formatBytes(bytes);
}

function processMemory(process: DiagnosticsProcess): string {
  return process.rss ? memoryUsage(process.rss.inBytes) : 'unknown';
}

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

function mlxMemorySummary(snapshot?: MlxMemorySnapshot | null): string {
  if (!snapshot) return 'none reported';
  return [
    `active ${memoryUsage(snapshot.active?.inBytes)}`,
    `cache ${memoryUsage(snapshot.cache?.inBytes)}`,
    `peak ${memoryUsage(snapshot.peak?.inBytes)}`,
    `wired ${memoryUsage(snapshot.wiredLimit?.inBytes)}`,
  ].join(' · ');
}

function phaseTone(runner: RunnerSupervisorDiagnostics): 'good' | 'warn' | 'neutral' {
  if (!runner.processAlive) return 'warn';
  if (runner.secondsInPhase >= 120 && runner.inProgressTasks.length > 0) return 'warn';
  if (runner.secondsInPhase >= 30 && runner.inProgressTasks.length > 0) return 'warn';
  if (runner.statusKind === 'RunnerRunning' || runner.statusKind === 'RunnerReady') return 'good';
  return 'neutral';
}

function recorderLine(entry: RunnerFlightRecorderEntry): string {
  const attrs = Object.entries(entry.attrs ?? {})
    .slice(0, 4)
    .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join('|') : value}`)
    .join(' ');
  return [
    entry.at,
    entry.phase,
    entry.event,
    entry.detail,
    entry.taskId ? `task=${shortId(entry.taskId)}` : null,
    attrs || null,
  ].filter(Boolean).join(' · ');
}

export function DiagnosticsDrawer({ nodeId, onClose }: DiagnosticsDrawerProps) {
  const [diagnostics, setDiagnostics] = useState<NodeDiagnostics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [cancelMessage, setCancelMessage] = useState<string | null>(null);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelActionKey, setCancelActionKey] = useState<string | null>(null);
  const [captureBundle, setCaptureBundle] = useState<DiagnosticCaptureResponse | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [captureActionKey, setCaptureActionKey] = useState<string | null>(null);

  useEffect(() => {
    if (!nodeId) {
      setDiagnostics(null);
      setError(null);
      setLoading(false);
      setCancelMessage(null);
      setCancelError(null);
      setCancelActionKey(null);
      setCaptureBundle(null);
      setCaptureError(null);
      setCaptureActionKey(null);
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/v1/diagnostics/cluster/${encodeURIComponent(nodeId)}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Diagnostics request failed: ${response.status}`);
        }
        return response.json() as Promise<NodeDiagnostics>;
      })
      .then((payload) => setDiagnostics(payload))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : 'Failed to load diagnostics');
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [nodeId, reloadToken]);

  if (!nodeId) return null;

  const runtime = diagnostics?.runtime;
  const currentMemory = diagnostics?.resources.currentMemory;
  const system = diagnostics?.resources.system;

  async function requestRunnerCancel(runnerId: string, taskId: string) {
    const actionKey = `${runnerId}:${taskId}`;
    setCancelActionKey(actionKey);
    setCancelError(null);
    setCancelMessage(null);

    try {
      const response = await fetch(
        `/v1/diagnostics/cluster/${encodeURIComponent(nodeId)}/runners/${encodeURIComponent(runnerId)}/cancel`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ taskId }),
        },
      );
      const payload = await response.json().catch(() => null) as { message?: string; detail?: string } | null;
      if (!response.ok) {
        throw new Error(payload?.detail ?? `Cancellation request failed: ${response.status}`);
      }
      setCancelMessage(payload?.message ?? 'Cancellation requested.');
      setReloadToken((value) => value + 1);
    } catch (err: unknown) {
      setCancelError(err instanceof Error ? err.message : 'Cancellation request failed');
    } finally {
      setCancelActionKey(null);
    }
  }

  async function requestCaptureBundle(runnerId: string, taskId?: string | null) {
    const actionKey = `${runnerId}:${taskId ?? 'runner'}`;
    setCaptureActionKey(actionKey);
    setCaptureError(null);
    setCaptureBundle(null);

    try {
      const response = await fetch(
        `/v1/diagnostics/cluster/${encodeURIComponent(nodeId)}/capture`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            runnerId,
            taskId: taskId ?? undefined,
            includeProcessSamples: true,
            sampleDurationSeconds: 3,
          }),
        },
      );
      const payload = await response.json().catch(() => null) as DiagnosticCaptureResponse | { detail?: string } | null;
      if (!response.ok) {
        throw new Error((payload as { detail?: string } | null)?.detail ?? `Capture request failed: ${response.status}`);
      }
      setCaptureBundle(payload as DiagnosticCaptureResponse);
      setReloadToken((value) => value + 1);
    } catch (err: unknown) {
      setCaptureError(err instanceof Error ? err.message : 'Capture request failed');
    } finally {
      setCaptureActionKey(null);
    }
  }

  async function copyCaptureBundle() {
    if (!captureBundle) return;
    await navigator.clipboard.writeText(JSON.stringify(captureBundle, null, 2));
  }

  function downloadCaptureBundle() {
    if (!captureBundle) return;
    const blob = new Blob([JSON.stringify(captureBundle, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `skulk-diagnostics-${captureBundle.nodeId}-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <Overlay onClick={onClose}>
      <Drawer onClick={(event) => event.stopPropagation()}>
        <Header>
          <div>
            <Title>Node Diagnostics</Title>
            <Subtitle>{runtime?.friendlyName ?? runtime?.hostname ?? shortId(nodeId)} · {shortId(nodeId)}</Subtitle>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
        </Header>

        {loading && <Section><Value>Loading diagnostics…</Value></Section>}
        {error && <Section><Warning>{error}</Warning></Section>}

        {diagnostics && runtime && (
          <>
            {diagnostics.warnings.length > 0 && (
              <Section>
                <SectionTitle>Warnings</SectionTitle>
                {diagnostics.warnings.map((warning) => (
                  <Warning key={warning}>{warning}</Warning>
                ))}
              </Section>
            )}

            <Section>
              <SectionTitle>Runtime</SectionTitle>
              <Row><Key>Role</Key><Value>{runtime.isMaster ? <Pill $tone="good">master</Pill> : <Pill>worker/follower</Pill>}</Value></Row>
              <Row><Key>Master</Key><Value>{runtime.masterNodeId ? shortId(runtime.masterNodeId) : 'unknown'}</Value></Row>
              <Row><Key>Commit</Key><Value>{runtime.skulkCommit}</Value></Row>
              <Row><Key>Version</Key><Value>{runtime.skulkVersion}</Value></Row>
              <Row><Key>Namespace</Key><Value>{runtime.libp2pNamespace ?? 'default'}</Value></Row>
              <Row><Key>CWD</Key><Value>{runtime.cwd}</Value></Row>
              <Row><Key>Config</Key><Value $warn={!runtime.configFileExists}>{runtime.configPath} {runtime.configFileExists ? '' : '(missing)'}</Value></Row>
              <Row><Key>Logging</Key><Value>{runtime.structuredLoggingConfigured ? 'centralized enabled' : 'not configured'}</Value></Row>
            </Section>

            <Section>
              <SectionTitle>Resources</SectionTitle>
              <Row><Key>RAM available</Key><Value>{memoryUsage(currentMemory?.ramAvailable?.inBytes)} / {memoryUsage(currentMemory?.ramTotal?.inBytes)}</Value></Row>
              <Row><Key>Swap available</Key><Value>{memoryUsage(currentMemory?.swapAvailable?.inBytes)} / {memoryUsage(currentMemory?.swapTotal?.inBytes)}</Value></Row>
              <Row><Key>GPU</Key><Value>{system?.gpuUsage != null ? `${Math.round(system.gpuUsage)}%` : 'unknown'}</Value></Row>
              <Row><Key>Temp</Key><Value>{system?.temp != null ? `${Math.round(system.temp)}°C` : 'unknown'}</Value></Row>
              <Row><Key>Power</Key><Value>{system?.sysPower != null ? `${Math.round(system.sysPower)}W` : 'unknown'}</Value></Row>
            </Section>

            <Section>
              <SectionTitle>Placements</SectionTitle>
              {diagnostics.placements.length === 0 ? (
                <Value>No active placements.</Value>
              ) : diagnostics.placements.map((placement) => (
                <div key={placement.instanceId}>
                  <Row><Key>Model</Key><Value>{placement.modelId}</Value></Row>
                  <Row><Key>Instance</Key><Value>{shortId(placement.instanceId)}</Value></Row>
                  <Row>
                    <Key>Master placed</Key>
                    <Value $warn={!placement.masterIsPlacementNode}>
                      {placement.masterIsPlacementNode ? <Pill $tone="good">yes</Pill> : <Pill $tone="warn">no</Pill>}
                    </Value>
                  </Row>
                  {placement.runners.map((runner) => (
                    <Row key={runner.runnerId}>
                      <Key>Rank {runner.deviceRank}</Key>
                      <Value>
                        {runner.friendlyName ?? shortId(runner.nodeId)} · {runner.statusKind ?? 'unknown'} · layers {runner.startLayer}:{runner.endLayer}
                      </Value>
                    </Row>
                  ))}
                  {placement.warnings.map((warning) => (
                    <Warning key={warning}>{warning}</Warning>
                  ))}
                </div>
              ))}
            </Section>

              <Section>
                <SectionTitle>Live Runners</SectionTitle>
              <Warning>
                Cancel task sends a cooperative runner-local cancellation request. A runner wedged in native MLX work may ignore it and still require stronger intervention.
              </Warning>
              {cancelMessage && <Warning>{cancelMessage}</Warning>}
              {cancelError && <Warning>{cancelError}</Warning>}
              {captureError && <Warning>{captureError}</Warning>}
              {captureBundle && (
                <CapturePanel>
                  <Row><Key>Captured</Key><Value>{captureBundle.generatedAt}</Value></Row>
                  <Row><Key>Runner</Key><Value>{captureBundle.runner ? shortId(captureBundle.runner.runnerId) : 'none'}</Value></Row>
                  <Row><Key>MLX memory</Key><Value>{mlxMemorySummary(captureBundle.mlxMemory)}</Value></Row>
                  <Row><Key>Samples</Key><Value>{captureBundle.processSamples.map((sample) => `${sample.name}:${sample.ok ? 'ok' : sample.error ?? 'failed'}`).join(', ') || 'none'}</Value></Row>
                  {captureBundle.warnings.map((warning) => <Warning key={warning}>{warning}</Warning>)}
                  <CaptureActions>
                    <Button variant="outline" size="sm" onClick={() => void copyCaptureBundle()}>Copy JSON</Button>
                    <Button variant="outline" size="sm" onClick={downloadCaptureBundle}>Download JSON</Button>
                  </CaptureActions>
                  <JsonPreview>{JSON.stringify(captureBundle, null, 2)}</JsonPreview>
                </CapturePanel>
              )}
              {diagnostics.supervisorRunners.length === 0 ? (
                <Value>No local runner supervisors reported by this node.</Value>
              ) : diagnostics.supervisorRunners.map((runner) => (
                <RunnerCard key={runner.runnerId}>
                  <Row><Key>Runner</Key><Value>{shortId(runner.runnerId)} · pid {runner.pid ?? 'unknown'}</Value></Row>
                  <Row>
                    <Key>Status</Key>
                    <Value>
                      {runner.statusKind} for {Math.round(runner.secondsInStatus)}s{' '}
                      {runner.processAlive ? <Pill $tone="good">alive</Pill> : <Pill $tone="warn">exited</Pill>}
                      {!runner.processAlive && runner.exitCode != null ? ` exit ${runner.exitCode}` : ''}
                    </Value>
                  </Row>
                  <Row><Key>Shard</Key><Value>rank {runner.deviceRank}/{runner.worldSize} · layers {runner.startLayer}:{runner.endLayer}</Value></Row>
                  <Row><Key>Instance</Key><Value>{shortId(runner.instanceId)} · {runner.modelId}</Value></Row>
                  <Row>
                    <Key>Phase</Key>
                    <Value $warn={phaseTone(runner) === 'warn'}>
                      <Pill $tone={phaseTone(runner)}>{runner.phase}</Pill>{' '}
                      {Math.round(runner.secondsInPhase)}s
                      {runner.phaseDetail ? ` · ${runner.phaseDetail}` : ''}
                    </Value>
                  </Row>
                  <Row><Key>Last progress</Key><Value>{runner.lastProgressAt ?? 'none'}</Value></Row>
                  <Row><Key>Active task</Key><Value>{runner.activeTaskId ? shortId(runner.activeTaskId) : 'none'}</Value></Row>
                  <Row><Key>MLX memory</Key><Value>{mlxMemorySummary(runner.lastMlxMemory)}</Value></Row>
                  <Row><Key>Last task sent</Key><Value>{runner.lastTaskSentAt ?? 'none'}</Value></Row>
                  <Row><Key>Last event</Key><Value>{runner.lastEventType ?? 'none'} {runner.lastEventReceivedAt ? `at ${runner.lastEventReceivedAt}` : ''}</Value></Row>
                  <Row>
                    <Key>Pending</Key>
                    <Value>{runner.pendingTaskIds.map((taskId) => shortId(taskId)).join(', ') || 'none'}</Value>
                  </Row>
                  <Row>
                    <Key>In progress</Key>
                    <Value>
                      {runner.inProgressTasks.map((task) => `${task.taskKind}:${shortId(task.taskId)} (${task.taskStatus})`).join(', ') || 'none'}
                    </Value>
                  </Row>
                  {runner.inProgressTasks.length > 0 && (
                    <TaskActionList>
                      {runner.inProgressTasks.map((task) => {
                        const actionKey = `${runner.runnerId}:${task.taskId}`;
                        return (
                          <TaskActionItem key={actionKey}>
                            <TaskActionMeta title={`${task.taskKind}:${task.taskId}`}>
                              {task.taskKind}:{shortId(task.taskId)} · {task.taskStatus}
                              {task.commandId ? ` · cmd ${shortId(task.commandId)}` : ''}
                            </TaskActionMeta>
                            <TaskActionButtons>
                              <Button
                                size="sm"
                                variant="primary"
                                loading={captureActionKey === actionKey}
                                onClick={() => { void requestCaptureBundle(runner.runnerId, task.taskId); }}
                              >
                                Capture bundle
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                loading={cancelActionKey === actionKey}
                                onClick={() => { void requestRunnerCancel(runner.runnerId, task.taskId); }}
                              >
                                Cancel task
                              </Button>
                            </TaskActionButtons>
                          </TaskActionItem>
                        );
                      })}
                    </TaskActionList>
                  )}
                  {runner.inProgressTasks.length === 0 && (
                    <CaptureActions>
                      <Button
                        size="sm"
                        variant="outline"
                        loading={captureActionKey === `${runner.runnerId}:runner`}
                        onClick={() => { void requestCaptureBundle(runner.runnerId); }}
                      >
                        Capture runner bundle
                      </Button>
                    </CaptureActions>
                  )}
                  <Row>
                    <Key>Cancelled</Key>
                    <Value>{runner.cancelledTaskIds.map((taskId) => shortId(taskId)).join(', ') || 'none'}</Value>
                  </Row>
                  <Row><Key>Completed</Key><Value>{runner.completedTaskCount}</Value></Row>
                  <Row><Key>Milestones</Key><Value>{runner.milestones.length === 0 ? 'none' : `${runner.milestones.length} recorded`}</Value></Row>
                  {runner.milestones.length > 0 && (
                    <MilestoneList>
                      {runner.milestones.slice().reverse().map((milestone, index) => (
                        <MilestoneItem key={`${milestone.at}-${milestone.name}-${index}`}>
                          {milestone.at} · {milestone.name}
                          {milestone.detail ? ` · ${milestone.detail}` : ''}
                        </MilestoneItem>
                      ))}
                    </MilestoneList>
                  )}
                  {runner.flightRecorder.length > 0 && (
                    <>
                      <Row><Key>Flight recorder</Key><Value>{runner.flightRecorder.length} entries</Value></Row>
                      <MilestoneList>
                        {runner.flightRecorder.slice(-8).reverse().map((entry, index) => (
                          <MilestoneItem key={`${entry.at}-${entry.event}-${index}`}>
                            {recorderLine(entry)}
                          </MilestoneItem>
                        ))}
                      </MilestoneList>
                    </>
                  )}
                </RunnerCard>
              ))}
            </Section>

            <Section>
              <SectionTitle>Processes</SectionTitle>
              {diagnostics.processes
                .filter((process) => process.role !== 'other')
                .map((process) => (
                  <ProcessLine key={process.pid}>
                    <span>{process.pid}</span>
                    <span>{process.role}</span>
                    <span>{processMemory(process)}</span>
                    <Monospace title={process.command}>{process.command || process.status || 'unknown'}</Monospace>
                  </ProcessLine>
                ))}
            </Section>
          </>
        )}
      </Drawer>
    </Overlay>
  );
}
