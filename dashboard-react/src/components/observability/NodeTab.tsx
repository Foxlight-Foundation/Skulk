import { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import { CenteredSpinner, Spinner } from '../common/Spinner';
import { formatBytes } from '../../utils/format';
import { useClusterState } from '../../hooks/useClusterState';
import { useAppDispatch } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import {
  useGetClusterTimelineQuery,
  useGetNodeDiagnosticsQuery,
  useCancelRunnerTaskMutation,
  useCaptureRunnerBundleMutation,
} from '../../store/endpoints/observability';
import type {
  DiagnosticCaptureResponse,
  DiagnosticsProcess,
  MlxMemorySnapshot,
  RunnerFlightRecorderEntry,
  RunnerSupervisorDiagnostics,
} from '../../types/diagnostics';

/**
 * "Node" tab body for the observability panel — read-only diagnostics for one cluster
 * node. Replaces the standalone DiagnosticsDrawer that used to render with its own
 * overlay; the panel now provides the framing (resizable width, header, close button)
 * and this component is just the data rendering.
 *
 * Hosts a node-selector dropdown at the top so the operator can switch nodes without
 * leaving the panel. The previously-required topology-bug-icon entry path still works
 * (sets the same store field), but is no longer the only way in.
 *
 * If no node is selected the tab renders an empty-state hint below the selector. The
 * diagnostics fetch only fires when a node is selected.
 */
export interface NodeTabProps {
  /** Node ID to inspect. Null when the operator hasn't picked a node yet. */
  nodeId: string | null;
}

/**
 * Provides this tab's scroll surface. ObservabilityPanel.Body has
 * `overflow: hidden` so each tab owns its own scroll behavior; the long
 * runner / process / placement sections scroll inside this Wrap rather
 * than the panel root.
 */
const Wrap = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
`;

const SelectorRow = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 12px;
`;

const SelectorLabel = styled.label`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  flex-shrink: 0;
`;

const NodeSelect = styled.select`
  flex: 1;
  min-width: 0;
  background: ${({ theme }) => theme.colors.bg};
  color: ${({ theme }) => theme.colors.text};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 4px 8px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  outline: none;
  cursor: pointer;

  &:focus {
    border-color: ${({ theme }) => theme.colors.goldDim};
  }

  option {
    background: ${({ theme }) => theme.colors.surface};
    color: ${({ theme }) => theme.colors.text};
  }
`;

const Subtitle = styled.div`
  margin: 0 0 14px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  word-break: break-all;
`;

const EmptyHint = styled.div`
  padding: 24px 12px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.55;
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

export function NodeTab({ nodeId }: NodeTabProps) {
  const cluster = useClusterState();
  const dispatch = useAppDispatch();
  const setSelectedNodeId = (nodeId: string | null) =>
    dispatch(uiActions.setObservabilitySelectedNodeId(nodeId));

  // The cluster timeline carries `masterNodeId`. RTK Query dedups, so when
  // the Live tab is also mounted there's no extra request; when only Node is
  // open this fires once to identify the master.
  const timelineQuery = useGetClusterTimelineQuery();
  const masterNodeId = timelineQuery.data?.masterNodeId ?? null;

  // Build a stable, sorted list of selectable nodes from the cluster topology.
  // Friendly names take precedence; nodes without one are labelled by short id and
  // float to the bottom so the most-recognizable entries surface first.
  const nodeOptions = useMemo(() => {
    const nodes = cluster.topology?.nodes ?? {};
    return Object.entries(nodes)
      .map(([id, info]) => ({
        id,
        label: info.friendly_name?.trim() || shortId(id),
        hasFriendly: Boolean(info.friendly_name?.trim()),
      }))
      .sort((a, b) => {
        if (a.hasFriendly !== b.hasFriendly) return a.hasFriendly ? -1 : 1;
        return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
      });
  }, [cluster.topology]);

  // The persisted `nodeId` may point at a node that no longer exists in this
  // cluster — sessionStorage carries the prior session's selection across
  // restarts, and operators sometimes hop between clusters. Compute an
  // *effective* nodeId that's null whenever the persisted value doesn't
  // match a known topology entry. The query downstream uses the effective
  // id, so we never fire `/v1/diagnostics/cluster/<dead-node-id>` and watch
  // it 404 into the error block.
  const isStaleSelection =
    nodeId != null &&
    nodeOptions.length > 0 &&
    !nodeOptions.some((option) => option.id === nodeId);
  const effectiveNodeId = isStaleSelection ? null : nodeId;

  // Auto-select rules:
  //  1. If the persisted selection is stale, clear it. Lets the next branch
  //     run on the next render with a clean slate.
  //  2. Otherwise, default to the master when nothing is picked. Avoids the
  //     "Pick a node above" empty state on every visit — the master is the
  //     single most useful starting point.
  useEffect(() => {
    if (isStaleSelection) {
      setSelectedNodeId(null);
      return;
    }
    if (effectiveNodeId) return;
    if (!masterNodeId) return;
    if (!nodeOptions.some((option) => option.id === masterNodeId)) return;
    setSelectedNodeId(masterNodeId);
    // setSelectedNodeId is stable (dispatch returns referentially-stable
    // action creators) so omitting it from deps is safe.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveNodeId, isStaleSelection, masterNodeId, nodeOptions]);

  // Diagnostics fetch — RTK Query keys by nodeId, dedups across components,
  // and refetches automatically when a cancel/capture mutation invalidates
  // the NodeDiagnostics cache tag for this node.
  //
  // Use `currentData` (not `data`) so flipping the node selector clears the
  // view immediately — `data` would leave the previous node's diagnostics on
  // screen until the next fetch lands, which mis-attributes that data to the
  // newly-picked node and is exactly what the user reported.
  const diagnosticsQuery = useGetNodeDiagnosticsQuery(effectiveNodeId ?? '', {
    skip: !effectiveNodeId,
  });
  const diagnostics = diagnosticsQuery.currentData ?? null;
  const loading = diagnosticsQuery.isFetching;
  const error = diagnosticsQuery.isError
    ? (diagnosticsQuery.error as { error?: string })?.error ?? 'Failed to load diagnostics'
    : null;

  const [cancelRunnerTask, cancelMutation] = useCancelRunnerTaskMutation();
  const [captureRunnerBundle, captureMutation] = useCaptureRunnerBundleMutation();
  const [cancelMessage, setCancelMessage] = useState<string | null>(null);
  const [cancelActionKey, setCancelActionKey] = useState<string | null>(null);
  const [captureBundle, setCaptureBundle] = useState<DiagnosticCaptureResponse | null>(null);
  const [captureActionKey, setCaptureActionKey] = useState<string | null>(null);

  const cancelError = cancelMutation.isError
    ? (cancelMutation.error as { data?: { detail?: string }; error?: string })?.data?.detail
      ?? (cancelMutation.error as { error?: string })?.error
      ?? 'Cancellation request failed'
    : null;
  const captureError = captureMutation.isError
    ? (captureMutation.error as { data?: { detail?: string }; error?: string })?.data?.detail
      ?? (captureMutation.error as { error?: string })?.error
      ?? 'Capture request failed'
    : null;

  const selector = (
    <SelectorRow>
      <SelectorLabel htmlFor="observability-node-select">Node</SelectorLabel>
      <NodeSelect
        id="observability-node-select"
        value={effectiveNodeId ?? ''}
        onChange={(event) => setSelectedNodeId(event.target.value || null)}
      >
        <option value="">Select node…</option>
        {nodeOptions.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
          </option>
        ))}
      </NodeSelect>
    </SelectorRow>
  );

  if (!effectiveNodeId) {
    return (
      <Wrap>
        {selector}
        {nodeOptions.length === 0 ? (
          <EmptyHint>
            No nodes reported by the cluster yet. Once a node connects, pick it here to inspect its diagnostics.
          </EmptyHint>
        ) : (
          // The auto-select effect above will resolve to master once the
          // timeline query lands. Render a spinner in the meantime rather
          // than a stale "Pick a node above" hint that would flash for the
          // first few hundred ms after entering the tab.
          <CenteredSpinner>
            <Spinner />
          </CenteredSpinner>
        )}
      </Wrap>
    );
  }

  const runtime = diagnostics?.runtime;
  const currentMemory = diagnostics?.resources.currentMemory;
  const system = diagnostics?.resources.system;

  async function requestRunnerCancel(runnerId: string, taskId: string) {
    if (!effectiveNodeId) return;
    const actionKey = `${runnerId}:${taskId}`;
    setCancelActionKey(actionKey);
    setCancelMessage(null);
    try {
      const payload = await cancelRunnerTask({
        nodeId: effectiveNodeId,
        runnerId,
        taskId,
      }).unwrap();
      setCancelMessage(payload.message ?? 'Cancellation requested.');
      // The mutation invalidates the NodeDiagnostics tag, so the query
      // refetches automatically — no manual reload token needed.
    } catch {
      // Error surfaces via cancelMutation.isError.
    } finally {
      setCancelActionKey(null);
    }
  }

  async function requestCaptureBundle(runnerId: string, taskId?: string | null) {
    if (!effectiveNodeId) return;
    const actionKey = `${runnerId}:${taskId ?? 'runner'}`;
    setCaptureActionKey(actionKey);
    setCaptureBundle(null);
    try {
      const payload = await captureRunnerBundle({
        nodeId: effectiveNodeId,
        runnerId,
        taskId,
      }).unwrap();
      setCaptureBundle(payload);
    } catch {
      // Error surfaces via captureMutation.isError.
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
    <Wrap>
      {selector}

      {/*
        While the new node's diagnostics are in flight (or the request errored
        without prior data), don't render the previous node's subtitle or
        sections — that mis-attributes data to the wrong node. A single
        centered spinner fills the remaining space; errors still surface as
        a warning block so an operator can see why nothing loaded.
      */}
      {!diagnostics && !error && (
        <CenteredSpinner>
          <Spinner />
        </CenteredSpinner>
      )}
      {!diagnostics && error && <Section><Warning>{error}</Warning></Section>}

      {diagnostics && runtime && (
        <>
          <Subtitle>{runtime.friendlyName ?? runtime.hostname ?? shortId(effectiveNodeId)} · {shortId(effectiveNodeId)}</Subtitle>
          {error && <Section><Warning>{error}</Warning></Section>}
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
        </>
        )}
    </Wrap>
  );
}
