import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import { formatBytes } from '../../utils/format';
import type { NodeDiagnostics, DiagnosticsProcess } from '../../types/diagnostics';

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

export function DiagnosticsDrawer({ nodeId, onClose }: DiagnosticsDrawerProps) {
  const [diagnostics, setDiagnostics] = useState<NodeDiagnostics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!nodeId) {
      setDiagnostics(null);
      setError(null);
      setLoading(false);
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
  }, [nodeId]);

  if (!nodeId) return null;

  const runtime = diagnostics?.runtime;
  const currentMemory = diagnostics?.resources.currentMemory;
  const system = diagnostics?.resources.system;

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
              {diagnostics.supervisorRunners.length === 0 ? (
                <Value>No local runner supervisors reported by this node.</Value>
              ) : diagnostics.supervisorRunners.map((runner) => (
                <div key={runner.runnerId}>
                  <Row><Key>Runner</Key><Value>{shortId(runner.runnerId)} · pid {runner.pid ?? 'unknown'}</Value></Row>
                  <Row><Key>Status</Key><Value>{runner.statusKind} for {Math.round(runner.secondsInStatus)}s</Value></Row>
                  <Row><Key>Shard</Key><Value>rank {runner.deviceRank}/{runner.worldSize} · layers {runner.startLayer}:{runner.endLayer}</Value></Row>
                  <Row><Key>Last event</Key><Value>{runner.lastEventType ?? 'none'} {runner.lastEventReceivedAt ? `at ${runner.lastEventReceivedAt}` : ''}</Value></Row>
                  <Row><Key>In progress</Key><Value>{runner.inProgressTasks.map((task) => `${task.taskKind}:${shortId(task.taskId)}`).join(', ') || 'none'}</Value></Row>
                </div>
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
