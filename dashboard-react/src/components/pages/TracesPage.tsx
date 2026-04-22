import { useCallback, useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import type { TraceListItem } from '../../types/traces';

export interface TracesPageProps {
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

async function downloadTrace(taskId: string): Promise<void> {
  const response = await fetch(`/v1/traces/${encodeURIComponent(taskId)}/raw`);
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

async function openInPerfetto(taskId: string): Promise<void> {
  const response = await fetch(`/v1/traces/${encodeURIComponent(taskId)}/raw`);
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

export function TracesPage({ onOpenTrace }: TracesPageProps) {
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/v1/traces');
      if (!response.ok) {
        throw new Error(`Failed to load traces (${response.status})`);
      }
      const data = (await response.json()) as { traces?: TraceListItem[] };
      setTraces(data.traces ?? []);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to load traces');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const allSelected = traces.length > 0 && selectedIds.size === traces.length;

  const selectedCountLabel = useMemo(() => {
    const count = selectedIds.size;
    return count === 1 ? '1 trace selected' : `${count} traces selected`;
  }, [selectedIds]);

  const toggleSelect = useCallback((taskId: string) => {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else {
        next.add(taskId);
      }
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(allSelected ? new Set() : new Set(traces.map((trace) => trace.taskId)));
  }, [allSelected, traces]);

  const handleDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    const count = selectedIds.size;
    const confirmed = window.confirm(
      `Delete ${count} trace${count === 1 ? '' : 's'}? This cannot be undone.`,
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
  }, [refresh, selectedIds]);

  return (
    <Page>
      <Header>
        <div>
          <Title>Traces</Title>
          <Subtitle>Distributed tracing artifacts for request and runtime analysis.</Subtitle>
        </div>
        <HeaderActions>
          {selectedIds.size > 0 && (
            <>
              <SelectionLabel>{selectedCountLabel}</SelectionLabel>
              <Button variant="danger" size="sm" onClick={() => void handleDelete()} loading={deleting}>
                Delete Selected
              </Button>
            </>
          )}
          <Button variant="outline" size="sm" onClick={() => void refresh()} loading={loading}>
            Refresh
          </Button>
        </HeaderActions>
      </Header>

      {loading ? (
        <StateBox>Loading traces…</StateBox>
      ) : error ? (
        <ErrorBox>{error}</ErrorBox>
      ) : traces.length === 0 ? (
        <StateBox>
          <div>No traces found.</div>
          <StateHint>Run Skulk with `SKULK_TRACING_ENABLED=1` to collect traces.</StateHint>
        </StateBox>
      ) : (
        <>
          <Toolbar>
            <Button variant="ghost" size="sm" onClick={toggleSelectAll}>
              {allSelected ? 'Deselect All' : 'Select All'}
            </Button>
          </Toolbar>
          <TraceList>
            {traces.map((trace) => {
              const selected = selectedIds.has(trace.taskId);
              return (
                <TraceRow key={trace.taskId} $selected={selected} onClick={() => toggleSelect(trace.taskId)}>
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
                  </TraceMain>
                  <TraceActions onClick={(event) => event.stopPropagation()}>
                    <Button variant="outline" size="sm" onClick={() => onOpenTrace(trace.taskId)}>
                      View Stats
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => void downloadTrace(trace.taskId)}>
                      Download
                    </Button>
                    <Button variant="primary" size="sm" onClick={() => void openInPerfetto(trace.taskId)}>
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
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SelectionLabel = styled.span`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Toolbar = styled.div`
  display: flex;
  align-items: center;
  justify-content: flex-start;
`;

const TraceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const TraceRow = styled.div<{ $selected: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ $selected, theme }) => ($selected ? theme.colors.goldDim : theme.colors.border)};
  background: ${({ $selected, theme }) => ($selected ? theme.colors.goldBg : theme.colors.surfaceSunken)};
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;

  &:hover {
    border-color: ${({ theme }) => theme.colors.goldDim};
  }
`;

const TraceMain = styled.div`
  min-width: 0;
  flex: 1;
`;

const TraceNameButton = styled.button`
  all: unset;
  cursor: pointer;
  display: block;
  min-width: 0;
  color: ${({ theme }) => theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;

  &:hover {
    color: ${({ theme }) => theme.colors.gold};
  }
`;

const TraceMeta = styled.div`
  margin-top: 6px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

const TraceActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
`;

const StateBox = styled.div`
  padding: 32px 24px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  text-align: center;
`;

const StateHint = styled.div`
  margin-top: 8px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

const ErrorBox = styled(StateBox)`
  color: ${({ theme }) => theme.colors.errorText};
  border-color: ${({ theme }) => theme.colors.errorBg};
  background: ${({ theme }) => theme.colors.errorBg};
`;
