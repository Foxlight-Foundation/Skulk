/**
 * TracesPage  —  "/traces" and "/traces/:taskId"
 *
 * Inference trace viewer with Perfetto integration.
 * Mirrors the Svelte traces/+page.svelte implementation.
 */
import React, { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import styled from 'styled-components';
import { listTraces, deleteTracesBatch, getTraceRawUrl } from '../api/client';
import type { TraceListItem } from '../api/types';

// ─── Styled components ────────────────────────────────────────────────────────

const Page = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  overflow-y: auto;
  max-width: 1200px;
`;

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 24px;
`;

const TitleBlock = styled.div``;

const Title = styled.h1`
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
`;

const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ActionButton = styled.button<{ $danger?: boolean; $primary?: boolean }>`
  background: ${({ theme, $primary }) => $primary ? theme.colors.yellow : 'transparent'};
  color: ${({ theme, $danger, $primary }) =>
    $primary ? theme.colors.black : $danger ? theme.colors.destructive : theme.colors.lightGray};
  border: 1px solid ${({ theme, $danger, $primary }) =>
    $primary ? theme.colors.yellow : $danger ? theme.colors.destructive : `${theme.colors.mediumGray}`};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 5px 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) {
    border-color: ${({ theme, $danger }) =>
      $danger ? theme.colors.destructive : theme.colors.yellow};
    color: ${({ theme, $danger, $primary }) =>
      $primary ? theme.colors.black : $danger ? theme.colors.destructive : theme.colors.yellow};
    opacity: ${({ $danger }) => $danger ? 0.85 : 1};
  }

  &:disabled { opacity: 0.4; cursor: default; }
`;

const SelectAllBtn = styled.button`
  background: none;
  border: none;
  padding: 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const TraceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SelectRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 4px;
  margin-bottom: 4px;
`;

const TraceCard = styled.div<{ $selected?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  border-radius: ${({ theme }) => theme.radius.sm};
  border-left: 2px solid ${({ theme, $selected }) =>
    $selected ? theme.colors.yellow : 'transparent'};
  border-top: 1px solid ${({ theme }) => theme.colors.mediumGray};
  border-right: 1px solid ${({ theme }) => theme.colors.mediumGray};
  border-bottom: 1px solid ${({ theme }) => theme.colors.mediumGray};
  background: ${({ theme, $selected }) =>
    $selected ? `${theme.colors.yellow}10` : `${theme.colors.black}80`};
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover {
    background: ${({ theme, $selected }) =>
      $selected ? `${theme.colors.yellow}10` : `${theme.colors.mediumGray}30`};
  }
`;

const TraceInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const TaskIdText = styled.div<{ $selected?: boolean }>`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme, $selected }) => $selected ? theme.colors.yellow : theme.colors.foreground};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const TraceMeta = styled.div`
  margin-top: 3px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const TraceActions = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
`;

const EmptyState = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.mediumGray};
  border-radius: ${({ theme }) => theme.radius.sm};
  background: ${({ theme }) => `${theme.colors.black}80`};
  padding: 48px 24px;
  text-align: center;
`;

const EmptyTitle = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.lightGray};
  margin-bottom: 8px;
`;

const EmptyHint = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => `${theme.colors.lightGray}80`};
`;

const ErrorBox = styled.div`
  border: 1px solid ${({ theme }) => `${theme.colors.destructive}50`};
  background: ${({ theme }) => `${theme.colors.destructive}10`};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 16px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.destructive};
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
}

function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

async function downloadTrace(taskId: string) {
  const response = await fetch(getTraceRawUrl(taskId));
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `trace_${taskId}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function openInPerfetto(taskId: string) {
  const response = await fetch(getTraceRawUrl(taskId));
  const traceData = await response.arrayBuffer();

  const perfettoWindow = window.open('https://ui.perfetto.dev');
  if (!perfettoWindow) {
    alert('Failed to open Perfetto. Please allow popups.');
    return;
  }

  const onMessage = (e: MessageEvent) => {
    if (e.data === 'PONG') {
      window.removeEventListener('message', onMessage);
      clearInterval(pingInterval);
      perfettoWindow.postMessage(
        { perfetto: { buffer: traceData, title: `Trace ${taskId}` } },
        'https://ui.perfetto.dev',
      );
    }
  };
  window.addEventListener('message', onMessage);

  const pingInterval = setInterval(() => {
    perfettoWindow.postMessage('PING', 'https://ui.perfetto.dev');
  }, 50);

  setTimeout(() => {
    clearInterval(pingInterval);
    window.removeEventListener('message', onMessage);
  }, 10000);
}

// ─── Component ────────────────────────────────────────────────────────────────

const TracesPage: React.FC = () => {
  const { taskId } = useParams<{ taskId?: string }>();

  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);

  const allSelected = traces.length > 0 && selectedIds.size === traces.length;

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listTraces();
      const items = Array.isArray(response)
        ? (response as TraceListItem[])
        : (response as { traces?: TraceListItem[] }).traces ?? [];
      setTraces(items);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load traces');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(traces.map((t) => t.taskId)));
    }
  };

  const handleDelete = async () => {
    if (selectedIds.size === 0) return;
    const count = selectedIds.size;
    if (!confirm(`Delete ${count} trace${count === 1 ? '' : 's'}? This cannot be undone.`)) return;

    setDeleting(true);
    try {
      await deleteTracesBatch([...selectedIds]);
      setSelectedIds(new Set());
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete traces');
    } finally {
      setDeleting(false);
    }
  };

  // If viewing a specific trace's detail, show a simple detail view
  if (taskId) {
    return (
      <Page>
        <Header>
          <TitleBlock>
            <Title>Trace — {taskId}</Title>
          </TitleBlock>
          <HeaderActions>
            <ActionButton type="button" onClick={() => void downloadTrace(taskId)}>
              Download
            </ActionButton>
            <ActionButton
              type="button"
              $primary
              onClick={() => void openInPerfetto(taskId)}
            >
              View in Perfetto
            </ActionButton>
          </HeaderActions>
        </Header>
      </Page>
    );
  }

  return (
    <Page>
      <Header>
        <TitleBlock>
          <Title>Inference Traces</Title>
        </TitleBlock>
        <HeaderActions>
          {selectedIds.size > 0 && (
            <ActionButton
              type="button"
              $danger
              onClick={() => void handleDelete()}
              disabled={deleting}
            >
              {deleting ? 'Deleting…' : `Delete (${selectedIds.size})`}
            </ActionButton>
          )}
          <ActionButton type="button" onClick={() => void refresh()} disabled={loading}>
            Refresh
          </ActionButton>
        </HeaderActions>
      </Header>

      {loading ? (
        <EmptyState>
          <EmptyTitle>Loading traces…</EmptyTitle>
        </EmptyState>
      ) : error ? (
        <ErrorBox>{error}</ErrorBox>
      ) : traces.length === 0 ? (
        <EmptyState>
          <EmptyTitle>No traces recorded</EmptyTitle>
          <EmptyHint>Run exo with EXO_TRACING_ENABLED=1 to collect traces.</EmptyHint>
        </EmptyState>
      ) : (
        <TraceList>
          <SelectRow>
            <SelectAllBtn type="button" onClick={toggleSelectAll}>
              {allSelected ? 'Deselect all' : 'Select all'}
            </SelectAllBtn>
          </SelectRow>
          {traces.map((trace) => {
            const isSelected = selectedIds.has(trace.taskId);
            return (
              <TraceCard
                key={trace.taskId}
                $selected={isSelected}
                onClick={() => toggleSelect(trace.taskId)}
              >
                <TraceInfo>
                  <TaskIdText $selected={isSelected}>{trace.taskId}</TaskIdText>
                  <TraceMeta>
                    {formatDate(trace.createdAt)} &bull; {formatBytes(trace.fileSize)}
                  </TraceMeta>
                </TraceInfo>
                <TraceActions onClick={(e) => e.stopPropagation()}>
                  <ActionButton
                    type="button"
                    onClick={() => void downloadTrace(trace.taskId)}
                  >
                    Download
                  </ActionButton>
                  <ActionButton
                    type="button"
                    $primary
                    onClick={() => void openInPerfetto(trace.taskId)}
                  >
                    View Trace
                  </ActionButton>
                </TraceActions>
              </TraceCard>
            );
          })}
        </TraceList>
      )}
    </Page>
  );
};

export default TracesPage;
