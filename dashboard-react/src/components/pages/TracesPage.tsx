import { useCallback, useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import type { TraceListItem, TraceListResponse, TracingStateResponse } from '../../types/traces';

export interface TracesPageProps {
  scope: 'cluster' | 'local';
  onScopeChange: (scope: 'cluster' | 'local') => void;
  onOpenTrace: (taskId: string) => void;
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 ? 0 : 1)}${units[index]}`;
}

function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

function traceBasePath(scope: 'cluster' | 'local'): string {
  return scope === 'cluster' ? '/v1/traces/cluster' : '/v1/traces';
}

async function downloadTrace(scope: 'cluster' | 'local', taskId: string): Promise<void> {
  const response = await fetch(`${traceBasePath(scope)}/${encodeURIComponent(taskId)}/raw`);
  if (!response.ok) {
    throw new Error(`Failed to download trace (${response.status})`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `trace_${taskId}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function openInPerfetto(scope: 'cluster' | 'local', taskId: string): Promise<void> {
  const response = await fetch(`${traceBasePath(scope)}/${encodeURIComponent(taskId)}/raw`);
  if (!response.ok) {
    throw new Error(`Failed to open trace (${response.status})`);
  }
  const traceData = await response.arrayBuffer();
  const perfettoWindow = window.open('https://ui.perfetto.dev');
  if (!perfettoWindow) {
    throw new Error('Failed to open Perfetto. Please allow popups.');
  }

  const onMessage = (event: MessageEvent) => {
    if (event.data === 'PONG') {
      window.removeEventListener('message', onMessage);
      perfettoWindow.postMessage(
        {
          perfetto: {
            buffer: traceData,
            title: `Trace ${taskId}`,
          },
        },
        'https://ui.perfetto.dev',
      );
    }
  };

  window.addEventListener('message', onMessage);
  const pingInterval = window.setInterval(() => {
    perfettoWindow.postMessage('PING', 'https://ui.perfetto.dev');
  }, 50);

  window.setTimeout(() => {
    window.clearInterval(pingInterval);
    window.removeEventListener('message', onMessage);
  }, 10000);
}

export function TracesPage({ scope, onScopeChange, onOpenTrace }: TracesPageProps) {
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tracingEnabled, setTracingEnabled] = useState(false);
  const [toggleLoading, setToggleLoading] = useState(false);
  const [taskKindFilter, setTaskKindFilter] = useState<string>('all');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [sourceNodeFilter, setSourceNodeFilter] = useState<string>('all');
  const [categoryOrTagFilter, setCategoryOrTagFilter] = useState('');
  const [toolOnly, setToolOnly] = useState(false);

  const refreshTracingState = useCallback(async () => {
    const response = await fetch('/v1/tracing');
    if (!response.ok) {
      throw new Error(`Failed to load tracing state (${response.status})`);
    }
    const data = (await response.json()) as TracingStateResponse;
    setTracingEnabled(data.enabled);
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [traceResponse] = await Promise.all([
        fetch(traceBasePath(scope)),
        refreshTracingState(),
      ]);
      if (!traceResponse.ok) {
        throw new Error(`Failed to load traces (${traceResponse.status})`);
      }
      const data = (await traceResponse.json()) as TraceListResponse;
      setTraces(data.traces ?? []);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to load traces');
    } finally {
      setLoading(false);
    }
  }, [refreshTracingState, scope]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    setSelectedIds(new Set());
  }, [scope]);

  const toggleTracing = useCallback(async (enabled: boolean) => {
    setToggleLoading(true);
    setError(null);
    try {
      const response = await fetch('/v1/tracing', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) {
        throw new Error(`Failed to update tracing (${response.status})`);
      }
      const data = (await response.json()) as TracingStateResponse;
      setTracingEnabled(data.enabled);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to update tracing');
    } finally {
      setToggleLoading(false);
    }
  }, []);

  const modelOptions = useMemo(
    () => [...new Set(traces.map((trace) => trace.modelId).filter(Boolean))].sort(),
    [traces],
  );
  const sourceNodeOptions = useMemo(
    () =>
      [...new Map(
        traces
          .flatMap((trace) => trace.sourceNodes)
          .map((sourceNode) => [sourceNode.nodeId, sourceNode] as const),
      ).values()].sort((left, right) => {
        const leftLabel = left.friendlyName || left.nodeId;
        const rightLabel = right.friendlyName || right.nodeId;
        return leftLabel.localeCompare(rightLabel);
      }),
    [traces],
  );

  const filteredTraces = useMemo(() => {
    const normalizedCategoryOrTag = categoryOrTagFilter.trim().toLowerCase();

    return traces.filter((trace) => {
      if (taskKindFilter !== 'all' && trace.taskKind !== taskKindFilter) return false;
      if (modelFilter !== 'all' && trace.modelId !== modelFilter) return false;
      if (sourceNodeFilter !== 'all' && !trace.sourceNodes.some((node) => node.nodeId === sourceNodeFilter)) {
        return false;
      }
      if (toolOnly && !trace.hasToolActivity) return false;
      if (!normalizedCategoryOrTag) return true;

      const haystack = [...trace.categories, ...trace.tags].map((value) => value.toLowerCase());
      return haystack.some((value) => value.includes(normalizedCategoryOrTag));
    });
  }, [categoryOrTagFilter, modelFilter, sourceNodeFilter, taskKindFilter, toolOnly, traces]);

  const allSelected = scope === 'local' && filteredTraces.length > 0 && selectedIds.size === filteredTraces.length;

  const selectedCountLabel = useMemo(() => {
    const count = selectedIds.size;
    return count === 1 ? '1 trace selected' : `${count} traces selected`;
  }, [selectedIds]);

  const toggleSelect = useCallback((taskId: string) => {
    if (scope !== 'local') return;
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else {
        next.add(taskId);
      }
      return next;
    });
  }, [scope]);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(allSelected ? new Set() : new Set(filteredTraces.map((trace) => trace.taskId)));
  }, [allSelected, filteredTraces]);

  const handleDelete = useCallback(async () => {
    if (scope !== 'local' || selectedIds.size === 0) return;
    const count = selectedIds.size;
    const confirmed = window.confirm(
      `Delete ${count} trace${count === 1 ? '' : 's'}? This only removes local trace files on this node.`,
    );
    if (!confirmed) return;

    setDeleting(true);
    setError(null);
    try {
      const response = await fetch('/v1/traces/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskIds: [...selectedIds] }),
      });
      if (!response.ok) {
        throw new Error(`Failed to delete traces (${response.status})`);
      }
      setSelectedIds(new Set());
      await refresh();
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to delete traces');
    } finally {
      setDeleting(false);
    }
  }, [refresh, scope, selectedIds]);

  return (
    <Page>
      <Header>
        <div>
          <Title>Traces</Title>
          <Subtitle>Cluster-aware runtime traces for image, text, and embedding requests.</Subtitle>
        </div>
        <HeaderActions>
          <Button variant="outline" size="sm" onClick={() => void refresh()} loading={loading}>
            Refresh
          </Button>
        </HeaderActions>
      </Header>

      <ControlCard>
        <ControlRow>
          <div>
            <ControlLabel>Cluster Tracing</ControlLabel>
            <ControlValue $enabled={tracingEnabled}>{tracingEnabled ? 'Enabled' : 'Disabled'}</ControlValue>
            <ControlHint>Applies to new requests on all nodes.</ControlHint>
          </div>
          <ControlActions>
            <Button
              variant={tracingEnabled ? 'primary' : 'outline'}
              size="sm"
              onClick={() => void toggleTracing(!tracingEnabled)}
              loading={toggleLoading}
            >
              {tracingEnabled ? 'Turn Off' : 'Turn On'}
            </Button>
          </ControlActions>
        </ControlRow>
        <ScopeRow>
          <ScopeLabel>Browse</ScopeLabel>
          <ScopeButtons>
            <Button variant={scope === 'cluster' ? 'primary' : 'outline'} size="sm" onClick={() => onScopeChange('cluster')}>
              Cluster
            </Button>
            <Button variant={scope === 'local' ? 'primary' : 'outline'} size="sm" onClick={() => onScopeChange('local')}>
              Local
            </Button>
          </ScopeButtons>
          <ScopeHint>
            {scope === 'cluster'
              ? 'Read-only cluster view across reachable nodes.'
              : 'Local trace files stored on this node. Deletion stays local-only.'}
          </ScopeHint>
        </ScopeRow>
      </ControlCard>

      <FiltersCard>
        <FilterGrid>
          <FilterField>
            <FilterLabel>Task kind</FilterLabel>
            <FilterSelect value={taskKindFilter} onChange={(event) => setTaskKindFilter(event.target.value)}>
              <option value="all">All</option>
              <option value="image">Image</option>
              <option value="text">Text</option>
              <option value="embedding">Embedding</option>
            </FilterSelect>
          </FilterField>

          <FilterField>
            <FilterLabel>Model</FilterLabel>
            <FilterSelect value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}>
              <option value="all">All</option>
              {modelOptions.map((modelId) => (
                <option key={modelId} value={modelId}>
                  {modelId}
                </option>
              ))}
            </FilterSelect>
          </FilterField>

          <FilterField>
            <FilterLabel>Source node</FilterLabel>
            <FilterSelect value={sourceNodeFilter} onChange={(event) => setSourceNodeFilter(event.target.value)}>
              <option value="all">All</option>
              {sourceNodeOptions.map((sourceNode) => (
                <option key={sourceNode.nodeId} value={sourceNode.nodeId}>
                  {sourceNode.friendlyName || sourceNode.nodeId}
                </option>
              ))}
            </FilterSelect>
          </FilterField>

          <FilterField>
            <FilterLabel>Category / tag</FilterLabel>
            <FilterInput
              value={categoryOrTagFilter}
              onChange={(event) => setCategoryOrTagFilter(event.target.value)}
              placeholder="prefill, decode, tool_call…"
            />
          </FilterField>
        </FilterGrid>

        <FilterActions>
          <Button variant={toolOnly ? 'primary' : 'outline'} size="sm" onClick={() => setToolOnly((current) => !current)}>
            Tool activity only
          </Button>
          {scope === 'local' && selectedIds.size > 0 && (
            <>
              <SelectionLabel>{selectedCountLabel}</SelectionLabel>
              <Button variant="danger" size="sm" onClick={() => void handleDelete()} loading={deleting}>
                Delete Selected
              </Button>
            </>
          )}
        </FilterActions>
      </FiltersCard>

      {loading ? (
        <StateBox>Loading traces…</StateBox>
      ) : error ? (
        <ErrorBox>{error}</ErrorBox>
      ) : filteredTraces.length === 0 ? (
        <StateBox>
          <div>No traces found.</div>
          <StateHint>
            {tracingEnabled
              ? 'Tracing is enabled. Run a new request and it will appear here.'
              : 'Tracing is off. Enable it above to start collecting traces for new requests on all nodes.'}
          </StateHint>
        </StateBox>
      ) : (
        <>
          {scope === 'local' && (
            <Toolbar>
              <Button variant="ghost" size="sm" onClick={toggleSelectAll}>
                {allSelected ? 'Deselect All' : 'Select All'}
              </Button>
            </Toolbar>
          )}
          <TraceList>
            {filteredTraces.map((trace) => {
              const selected = selectedIds.has(trace.taskId);
              return (
                <TraceRow
                  key={trace.taskId}
                  $interactive={scope === 'local'}
                  $selected={selected}
                  onClick={() => toggleSelect(trace.taskId)}
                >
                  <TraceMain>
                    <TraceNameButton
                      onClick={(event) => {
                        event.stopPropagation();
                        onOpenTrace(trace.taskId);
                      }}
                    >
                      {trace.taskId}
                    </TraceNameButton>
                    <TraceMeta>
                      {formatDate(trace.createdAt)} · {formatBytes(trace.fileSize)}
                    </TraceMeta>
                    <BadgeRow>
                      {trace.taskKind && <MetaBadge>{trace.taskKind}</MetaBadge>}
                      {trace.modelId && <MetaBadge>{trace.modelId}</MetaBadge>}
                      {trace.hasToolActivity && <MetaBadge>tool_call</MetaBadge>}
                      {trace.sourceNodes.map((sourceNode) => (
                        <MetaBadge key={`${trace.taskId}-${sourceNode.nodeId}`}>
                          {sourceNode.friendlyName || sourceNode.nodeId}
                        </MetaBadge>
                      ))}
                    </BadgeRow>
                    {trace.tags.length > 0 && <TagLine>{trace.tags.join(' · ')}</TagLine>}
                  </TraceMain>
                  <TraceActions onClick={(event) => event.stopPropagation()}>
                    <Button variant="outline" size="sm" onClick={() => onOpenTrace(trace.taskId)}>
                      View Stats
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => void downloadTrace(scope, trace.taskId)}>
                      Download
                    </Button>
                    <Button variant="primary" size="sm" onClick={() => void openInPerfetto(scope, trace.taskId)}>
                      Perfetto
                    </Button>
                  </TraceActions>
                </TraceRow>
              );
            })}
          </TraceList>
        </>
      )}
    </Page>
  );
}

const Page = styled.div`
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
`;

const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const Title = styled.h1`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xl};
  color: ${({ theme }) => theme.colors.gold};
`;

const Subtitle = styled.p`
  margin: 6px 0 0;
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ControlCard = styled.section`
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 18px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  background: ${({ theme }) => theme.colors.surfaceAlt};
`;

const ControlRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
`;

const ControlLabel = styled.div`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const ControlValue = styled.div<{ $enabled: boolean }>`
  margin-top: 6px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-weight: 600;
  color: ${({ $enabled, theme }) => ($enabled ? theme.colors.gold : theme.colors.text)};
`;

const ControlHint = styled.p`
  margin: 6px 0 0;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ControlActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ScopeRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
`;

const ScopeLabel = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  text-transform: uppercase;
  letter-spacing: 0.08em;
`;

const ScopeButtons = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ScopeHint = styled.span`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const FiltersCard = styled.section`
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surface};
`;

const FilterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
`;

const FilterField = styled.label`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const FilterLabel = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const FilterSelect = styled.select`
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceElevated};
  color: ${({ theme }) => theme.colors.text};
  padding: 10px 12px;
`;

const FilterInput = styled.input`
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceElevated};
  color: ${({ theme }) => theme.colors.text};
  padding: 10px 12px;
`;

const FilterActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const StateBox = styled.div`
  padding: 24px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px dashed ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const StateHint = styled.div`
  margin-top: 8px;
`;

const ErrorBox = styled(StateBox)`
  border-style: solid;
  border-color: ${({ theme }) => theme.colors.error};
  color: ${({ theme }) => theme.colors.errorText};
`;

const Toolbar = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const SelectionLabel = styled.span`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const TraceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const TraceRow = styled.div<{ $selected: boolean; $interactive: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ $selected, theme }) => ($selected ? theme.colors.goldDim : theme.colors.border)};
  background: ${({ $selected, theme }) => ($selected ? theme.colors.goldBg : theme.colors.surface)};
  cursor: ${({ $interactive }) => ($interactive ? 'pointer' : 'default')};
  flex-wrap: wrap;
`;

const TraceMain = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
`;

const TraceNameButton = styled.button`
  all: unset;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.gold};
  font-weight: 600;
  overflow-wrap: anywhere;
`;

const TraceMeta = styled.div`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const BadgeRow = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
`;

const MetaBadge = styled.span`
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceElevated};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

const TagLine = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  overflow-wrap: anywhere;
`;

const TraceActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;
