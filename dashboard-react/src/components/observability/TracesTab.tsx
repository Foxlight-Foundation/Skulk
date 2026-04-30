import { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import { useAppDispatch } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import {
  useGetTracesListQuery,
  useGetTraceQuery,
  type TraceScope,
} from '../../store/endpoints/observability';
import {
  traceEventToWaterfall,
  type TraceEventLike,
} from '../../types/observabilityEvents';
import type { TraceEventResponse } from '../../types/traces';
import { TraceWaterfall } from './TraceWaterfall';

/**
 * "Traces" tab body for the observability panel.
 *
 * Shows the saved-trace list and, when a trace is selected, renders the trace
 * inline as a Skulk-native waterfall (no popup, no third-party hosted UI). The
 * scope toggle is local-vs-cluster; cluster mode hits `/v1/traces/cluster` to
 * fan out across reachable peers, local mode is just this node's saved traces.
 *
 * Filters, multi-select, and bulk-delete intentionally stay in `TracesPage` —
 * that is the heavyweight admin surface. This tab is the in-panel viewer; the
 * footer link below the list jumps to the legacy page when those features are
 * needed.
 */

/**
 * Provides this tab's scroll surface. ObservabilityPanel.Body has
 * `overflow: hidden` so each tab owns its own scroll behavior; we route
 * the user's scroll wheel through Wrap rather than the panel root.
 */
const Wrap = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
`;

const ScopeToggle = styled.div`
  display: inline-flex;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  overflow: hidden;
`;

const ScopeButton = styled.button<{ $active: boolean }>`
  all: unset;
  padding: 4px 12px;
  cursor: pointer;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  background: ${({ $active, theme }) => ($active ? theme.colors.goldBg : 'transparent')};
  color: ${({ $active, theme }) => ($active ? theme.colors.goldStrong : theme.colors.textSecondary)};

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
  }

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.goldDim};
    outline-offset: -2px;
  }
`;

const ListWrap = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  max-height: 240px;
  overflow-y: auto;
`;

const ListRow = styled.button<{ $selected: boolean }>`
  all: unset;
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr) auto;
  gap: 10px;
  width: 100%;
  box-sizing: border-box;
  padding: 8px 10px;
  cursor: pointer;
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  background: ${({ $selected, theme }) => ($selected ? theme.colors.goldBg : 'transparent')};

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background: ${({ $selected, theme }) =>
      $selected ? theme.colors.goldBg : theme.colors.surfaceHover};
    color: ${({ theme }) => theme.colors.text};
  }

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.goldDim};
    outline-offset: -2px;
  }
`;

const RowMain = styled.span`
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const RowSecondary = styled.span`
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const RowTime = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
`;

const Notice = styled.div`
  padding: 12px 10px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const ErrorNotice = styled(Notice)`
  color: ${({ theme }) => theme.colors.errorText};
`;

const DetailWrap = styled.section`
  display: flex;
  flex-direction: column;
  gap: 8px;
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  padding: 10px 12px;
`;

const DetailHeader = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
`;

const DetailMeta = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
`;

const SelectedRow = styled.div`
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 8px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  padding: 2px 0;
`;

const SelectedKey = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SelectedValue = styled.span`
  word-break: break-word;
`;

const FooterLink = styled.button`
  all: unset;
  cursor: pointer;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  text-decoration: underline dotted;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.goldDim};
    outline-offset: 2px;
  }
`;

type Scope = TraceScope;

function basePath(scope: Scope): string {
  return scope === 'cluster' ? '/v1/traces/cluster' : '/v1/traces';
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** idx;
  return `${value.toFixed(value >= 10 ? 0 : 1)}${units[idx]}`;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString();
}

function formatDuration(microseconds: number): string {
  if (microseconds < 1_000) return `${microseconds.toFixed(0)}us`;
  if (microseconds < 1_000_000) return `${(microseconds / 1_000).toFixed(2)}ms`;
  return `${(microseconds / 1_000_000).toFixed(2)}s`;
}

function totalWallTimeUs(events: readonly TraceEventResponse[]): number {
  if (events.length === 0) return 0;
  let min = Infinity;
  let max = -Infinity;
  for (const event of events) {
    if (event.startUs < min) min = event.startUs;
    const end = event.startUs + event.durationUs;
    if (end > max) max = end;
  }
  return max - min;
}

async function downloadRawTrace(scope: Scope, taskId: string): Promise<void> {
  const response = await fetch(`${basePath(scope)}/${encodeURIComponent(taskId)}/raw`);
  if (!response.ok) throw new Error(`Failed to download trace (${response.status})`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `trace_${taskId}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function TracesTab() {
  const dispatch = useAppDispatch();

  const [scope, setScope] = useState<Scope>('local');
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TraceEventLike | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Cache key includes scope, so flipping the toggle yields a fresh fetch
  // (RTK Query naturally treats the two scopes as separate entries).
  const listQuery = useGetTracesListQuery(scope);
  const traceQuery = useGetTraceQuery(
    selectedTaskId ? { scope, taskId: selectedTaskId } : { scope, taskId: '' },
    { skip: !selectedTaskId },
  );

  const traces = listQuery.data?.traces ?? null;
  const listError = listQuery.isError
    ? (listQuery.error as { error?: string })?.error ?? 'Failed to load traces'
    : null;
  const listLoading = listQuery.isLoading;

  const traceData = traceQuery.data ?? null;
  const traceError = traceQuery.isError
    ? (traceQuery.error as { error?: string })?.error ?? 'Failed to load trace'
    : null;
  const traceLoading = traceQuery.isFetching;

  // Selection doesn't necessarily survive a scope flip (a local trace may not
  // exist under the cluster proxy or vice versa) — clear it explicitly when
  // scope changes.
  useEffect(() => {
    setSelectedTaskId(null);
    setSelectedEvent(null);
  }, [scope]);

  useEffect(() => {
    setSelectedEvent(null);
  }, [selectedTaskId]);

  const waterfallEvents = useMemo<TraceEventLike[]>(() => {
    if (!traceData) return [];
    return traceEventToWaterfall(traceData.traces);
  }, [traceData]);

  const totalUs = useMemo(
    () => (traceData ? totalWallTimeUs(traceData.traces) : 0),
    [traceData],
  );

  const selectedTraceListItem = useMemo(
    () => (selectedTaskId && traces ? traces.find((trace) => trace.taskId === selectedTaskId) ?? null : null),
    [selectedTaskId, traces],
  );

  async function handleDownload() {
    if (!selectedTaskId) return;
    setDownloadError(null);
    try {
      await downloadRawTrace(scope, selectedTaskId);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : 'Download failed');
    }
  }

  const openLegacyTraces = () => {
    dispatch(uiActions.setActiveRoute('traces'));
    dispatch(uiActions.closeObservability());
  };

  return (
    <Wrap>
      <ScopeToggle role="tablist" aria-label="Trace scope">
        <ScopeButton
          role="tab"
          aria-selected={scope === 'local'}
          $active={scope === 'local'}
          onClick={() => setScope('local')}
        >
          This node
        </ScopeButton>
        <ScopeButton
          role="tab"
          aria-selected={scope === 'cluster'}
          $active={scope === 'cluster'}
          onClick={() => setScope('cluster')}
        >
          Cluster
        </ScopeButton>
      </ScopeToggle>

      <ListWrap>
        {listLoading && <Notice>Loading traces…</Notice>}
        {listError && <ErrorNotice>{listError}</ErrorNotice>}
        {!listLoading && !listError && traces && traces.length === 0 && (
          <Notice>
            No saved traces yet. Enable tracing for the cluster and re-run a request to record one.
          </Notice>
        )}
        {!listLoading && traces && traces.map((trace) => (
          <ListRow
            key={trace.taskId}
            $selected={selectedTaskId === trace.taskId}
            onClick={() => setSelectedTaskId(trace.taskId)}
            type="button"
          >
            <RowMain title={trace.taskId}>
              {trace.modelId ?? trace.taskId}
            </RowMain>
            <RowSecondary>
              {trace.taskKind ?? 'unknown'} · {formatBytes(trace.fileSize)}
            </RowSecondary>
            <RowTime>{formatTime(trace.createdAt)}</RowTime>
          </ListRow>
        ))}
      </ListWrap>

      {selectedTaskId && (
        <DetailWrap>
          <DetailHeader>
            <DetailMeta>
              {selectedTraceListItem?.modelId && <span>{selectedTraceListItem.modelId}</span>}
              {selectedTraceListItem?.taskKind && <span>{selectedTraceListItem.taskKind}</span>}
              {totalUs > 0 && <span>wall {formatDuration(totalUs)}</span>}
              {traceData?.sourceNodes && traceData.sourceNodes.length > 0 && (
                <span>{traceData.sourceNodes.length} source nodes</span>
              )}
            </DetailMeta>
            <Button
              variant="outline"
              size="sm"
              onClick={() => { void handleDownload(); }}
            >
              Download JSON
            </Button>
          </DetailHeader>
          {downloadError && <ErrorNotice>{downloadError}</ErrorNotice>}
          {traceLoading && <Notice>Loading trace…</Notice>}
          {traceError && <ErrorNotice>{traceError}</ErrorNotice>}
          {traceData && !traceLoading && !traceError && (
            <>
              <TraceWaterfall
                events={waterfallEvents}
                selectedId={selectedEvent?.id ?? null}
                onSelect={setSelectedEvent}
              />
              {selectedEvent && (
                <div>
                  <SelectedRow>
                    <SelectedKey>Event</SelectedKey>
                    <SelectedValue>{selectedEvent.name}</SelectedValue>
                  </SelectedRow>
                  <SelectedRow>
                    <SelectedKey>Lane</SelectedKey>
                    <SelectedValue>{selectedEvent.laneLabel}</SelectedValue>
                  </SelectedRow>
                  <SelectedRow>
                    <SelectedKey>Category</SelectedKey>
                    <SelectedValue>{selectedEvent.category}</SelectedValue>
                  </SelectedRow>
                  <SelectedRow>
                    <SelectedKey>Duration</SelectedKey>
                    <SelectedValue>{formatDuration(selectedEvent.durationUs)}</SelectedValue>
                  </SelectedRow>
                  {selectedEvent.tags && selectedEvent.tags.length > 0 && (
                    <SelectedRow>
                      <SelectedKey>Tags</SelectedKey>
                      <SelectedValue>{selectedEvent.tags.join(', ')}</SelectedValue>
                    </SelectedRow>
                  )}
                  {selectedEvent.attrs &&
                    Object.entries(selectedEvent.attrs).map(([key, value]) => (
                      <SelectedRow key={key}>
                        <SelectedKey>{key}</SelectedKey>
                        <SelectedValue>
                          {Array.isArray(value) ? value.join(', ') : String(value)}
                        </SelectedValue>
                      </SelectedRow>
                    ))}
                </div>
              )}
            </>
          )}
        </DetailWrap>
      )}

      <FooterLink type="button" onClick={openLegacyTraces}>
        Open legacy traces page (filters, multi-select, bulk delete)
      </FooterLink>
    </Wrap>
  );
}
