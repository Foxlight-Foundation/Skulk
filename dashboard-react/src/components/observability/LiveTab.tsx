import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import type {
  ClusterTimeline,
  ClusterTimelineEntry,
  ClusterTimelineRunner,
} from '../../types/diagnostics';
import type { TracingStateResponse } from '../../types/traces';

/**
 * "Live" tab body for the observability panel. Polls the cluster-timeline
 * fan-out (`/v1/diagnostics/cluster/timeline`) and the cluster-wide tracing
 * toggle (`/v1/tracing`), then renders three blocks:
 *
 *  - **Header strip** with master id, tracing toggle, hang-rate count, and a
 *    "refresh now" button. The hang-rate is a coarse health indicator — count
 *    of runners stuck in their current phase past a threshold *while* an
 *    active task is bound to that runner (idle-phase stalls aren't hangs).
 *  - **Runner synopsis grid** — one card per runner showing rank, phase, time
 *    stuck, and the active task. Cards highlight when the runner is in the
 *    hang-rate count.
 *  - **Cross-rank flight-recorder feed** — the merged timeline tail, newest
 *    first. This is the rank-disagreement view the cluster-timeline endpoint
 *    was built for in the first place.
 *
 * Polling cadence is 4s while the tab is mounted; the cadence is short enough
 * to feel live but long enough that operators inspecting one record have time
 * to read it before it scrolls away.
 *
 * The `/events` event-stream tail noted in the original Phase 3 scope is left
 * as a follow-up — heavier surface (live SSE, debug-only data), and the
 * timeline view already covers the more common debugging case.
 */

/** Seconds-stuck threshold above which a phase-bound runner counts toward hangs. */
const HANG_PHASE_SECONDS = 30;
const REFRESH_MS = 4000;
const TIMELINE_TAIL_LIMIT = 80;

const Wrap = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const HeaderStrip = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
`;

const HeaderField = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
`;

const HeaderLabel = styled.span`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 0.04em;
`;

const HeaderValue = styled.span<{ $tone?: 'good' | 'warn' | 'neutral' }>`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ $tone, theme }) =>
    $tone === 'good' ? theme.colors.healthy :
    $tone === 'warn' ? theme.colors.warningText :
    theme.colors.text};
  word-break: break-all;
`;

const ToggleButton = styled.button<{ $on: boolean }>`
  all: unset;
  cursor: pointer;
  width: 36px;
  height: 20px;
  border-radius: 10px;
  position: relative;
  flex-shrink: 0;
  transition: background 0.2s;
  background: ${({ $on, theme }) => ($on ? theme.colors.gold : theme.colors.surfaceSunken)};
  border: 1px solid ${({ theme }) => theme.colors.border};

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.goldDim};
  }

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $on }) => ($on ? '18px' : '2px')};
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.surface};
    box-shadow: 0 1px 2px ${({ theme }) => theme.colors.shadow};
    transition: left 0.2s;
  }
`;

const Section = styled.section`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SectionTitle = styled.h3`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ theme }) => theme.colors.gold};
`;

const RunnerGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px;
`;

const RunnerCard = styled.div<{ $hung: boolean }>`
  border: 1px solid ${({ $hung, theme }) => ($hung ? theme.colors.warning : theme.colors.borderLight)};
  background: ${({ $hung, theme }) => ($hung ? theme.colors.warningBg : theme.colors.surface)};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 8px 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const RunnerLine = styled.div`
  display: flex;
  align-items: baseline;
  gap: 6px;
  word-break: break-word;
`;

const RunnerLabel = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  flex-shrink: 0;
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
  font-size: 10px;
`;

const TimelineList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 360px;
  overflow-y: auto;
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 6px 8px;
  background: ${({ theme }) => theme.colors.surfaceSunken};
`;

const TimelineItem = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.45;
  padding: 3px 0;
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
  word-break: break-word;

  &:last-child {
    border-bottom: none;
  }
`;

const TimelineMeta = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  margin-right: 6px;
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

const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

function isHung(runner: ClusterTimelineRunner): boolean {
  return Boolean(
    runner.processAlive &&
    runner.activeTaskId &&
    runner.secondsInPhase >= HANG_PHASE_SECONDS,
  );
}

function formatLocalTime(at: string): string {
  // The timeline always reports UTC ISO timestamps. Render in the user's
  // locale; absolute time + the cardinality of entries beats reverse-engineering
  // a "12s ago" string from a timestamp the server might have emitted before
  // the local clock if the cluster nodes are slightly out of sync.
  const parsed = new Date(at);
  if (Number.isNaN(parsed.getTime())) return at;
  return parsed.toLocaleTimeString([], { hour12: false }) +
    `.${String(parsed.getMilliseconds()).padStart(3, '0')}`;
}

function summarizeAttrs(attrs: Record<string, unknown>): string {
  const entries = Object.entries(attrs).slice(0, 4);
  if (entries.length === 0) return '';
  return entries
    .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join('|') : String(value)}`)
    .join(' ');
}

export function LiveTab() {
  const [timeline, setTimeline] = useState<ClusterTimeline | null>(null);
  const [tracingEnabled, setTracingEnabled] = useState<boolean | null>(null);
  const [tracingToggling, setTracingToggling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tracingError, setTracingError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Track whether the component is still mounted so cleanup-time race conditions
  // don't surface as console errors. The polling timer uses this to short-circuit
  // before commits.
  const mountedRef = useRef(true);

  const fetchAll = useCallback(async () => {
    try {
      setRefreshing(true);
      const [timelineResponse, tracingResponse] = await Promise.all([
        fetch('/v1/diagnostics/cluster/timeline'),
        fetch('/v1/tracing'),
      ]);
      if (!mountedRef.current) return;
      if (!timelineResponse.ok) {
        throw new Error(`Timeline request failed: ${timelineResponse.status}`);
      }
      const timelineData = (await timelineResponse.json()) as ClusterTimeline;
      if (!mountedRef.current) return;
      setTimeline(timelineData);
      setError(null);

      if (tracingResponse.ok) {
        const tracingData = (await tracingResponse.json()) as TracingStateResponse;
        if (!mountedRef.current) return;
        setTracingEnabled(tracingData.enabled);
      }
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : 'Failed to load cluster timeline');
    } finally {
      if (mountedRef.current) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void fetchAll();
    const handle = window.setInterval(() => { void fetchAll(); }, REFRESH_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(handle);
    };
  }, [fetchAll]);

  async function toggleTracing() {
    if (tracingEnabled == null) return;
    const next = !tracingEnabled;
    setTracingToggling(true);
    setTracingError(null);
    try {
      const response = await fetch('/v1/tracing', {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ enabled: next }),
      });
      if (!response.ok) throw new Error(`Tracing toggle failed: ${response.status}`);
      const data = (await response.json()) as TracingStateResponse;
      setTracingEnabled(data.enabled);
    } catch (err) {
      setTracingError(err instanceof Error ? err.message : 'Tracing toggle failed');
    } finally {
      setTracingToggling(false);
    }
  }

  const hangCount = useMemo(() => {
    if (!timeline) return 0;
    return timeline.runners.filter(isHung).length;
  }, [timeline]);

  const recentTimeline = useMemo(() => {
    if (!timeline) return [] as ClusterTimelineEntry[];
    // Newest first. The API returns ascending; reverse and cap at the tail
    // limit so the panel doesn't grind under thousands of entries.
    const reversed = [...timeline.timeline].reverse();
    return reversed.slice(0, TIMELINE_TAIL_LIMIT);
  }, [timeline]);

  const reachableLabel = timeline
    ? timeline.unreachableNodes.length === 0
      ? 'all reachable'
      : `${timeline.unreachableNodes.length} unreachable`
    : '—';

  return (
    <Wrap>
      <HeaderStrip>
        <HeaderField>
          <HeaderLabel>Master</HeaderLabel>
          <HeaderValue>
            {timeline?.masterNodeId ? shortId(timeline.masterNodeId) : '—'}
          </HeaderValue>
        </HeaderField>
        <HeaderField>
          <HeaderLabel>Connectivity</HeaderLabel>
          <HeaderValue $tone={timeline && timeline.unreachableNodes.length > 0 ? 'warn' : 'good'}>
            {reachableLabel}
          </HeaderValue>
        </HeaderField>
        <HeaderField>
          <HeaderLabel>Hangs</HeaderLabel>
          <HeaderValue $tone={hangCount > 0 ? 'warn' : 'good'}>
            {hangCount} runner{hangCount === 1 ? '' : 's'} stuck
          </HeaderValue>
        </HeaderField>
        <HeaderField>
          <HeaderLabel>Tracing</HeaderLabel>
          {tracingEnabled == null ? (
            <HeaderValue>—</HeaderValue>
          ) : (
            <ToggleButton
              $on={tracingEnabled}
              role="switch"
              aria-checked={tracingEnabled}
              aria-label={tracingEnabled ? 'Disable cluster tracing' : 'Enable cluster tracing'}
              disabled={tracingToggling}
              onClick={() => { void toggleTracing(); }}
            />
          )}
        </HeaderField>
        <HeaderActions>
          <Button
            variant="outline"
            size="sm"
            loading={refreshing}
            onClick={() => { void fetchAll(); }}
          >
            Refresh
          </Button>
        </HeaderActions>
      </HeaderStrip>

      {error && <ErrorNotice>{error}</ErrorNotice>}
      {tracingError && <ErrorNotice>{tracingError}</ErrorNotice>}

      <Section>
        <SectionTitle>Runners</SectionTitle>
        {timeline && timeline.runners.length === 0 && (
          <Notice>No runners reported across the cluster.</Notice>
        )}
        {timeline && timeline.runners.length > 0 && (
          <RunnerGrid>
            {timeline.runners.map((runner) => {
              const hung = isHung(runner);
              return (
                <RunnerCard key={runner.runnerId} $hung={hung}>
                  <RunnerLine>
                    <Pill $tone={runner.processAlive ? 'good' : 'warn'}>
                      rank {runner.deviceRank}/{runner.worldSize}
                    </Pill>
                    <RunnerLabel>{shortId(runner.nodeId)}</RunnerLabel>
                  </RunnerLine>
                  <RunnerLine>
                    <RunnerLabel>model</RunnerLabel>
                    {runner.modelId}
                  </RunnerLine>
                  <RunnerLine>
                    <RunnerLabel>phase</RunnerLabel>
                    <Pill $tone={hung ? 'warn' : 'neutral'}>{runner.phase}</Pill>
                    <span>{Math.round(runner.secondsInPhase)}s</span>
                  </RunnerLine>
                  {runner.phaseDetail && (
                    <RunnerLine>
                      <RunnerLabel>detail</RunnerLabel>
                      <span>{runner.phaseDetail}</span>
                    </RunnerLine>
                  )}
                  {runner.activeTaskId && (
                    <RunnerLine>
                      <RunnerLabel>task</RunnerLabel>
                      {shortId(runner.activeTaskId)}
                    </RunnerLine>
                  )}
                </RunnerCard>
              );
            })}
          </RunnerGrid>
        )}
      </Section>

      <Section>
        <SectionTitle>Cross-rank timeline (newest first)</SectionTitle>
        {timeline && recentTimeline.length === 0 && (
          <Notice>No flight-recorder entries reported yet.</Notice>
        )}
        {recentTimeline.length > 0 && (
          <TimelineList>
            {recentTimeline.map((entry, idx) => {
              const attrSummary = summarizeAttrs(entry.attrs);
              return (
                <TimelineItem key={`${entry.at}-${entry.runnerId}-${idx}`}>
                  <TimelineMeta>{formatLocalTime(entry.at)}</TimelineMeta>
                  <TimelineMeta>r{entry.deviceRank}</TimelineMeta>
                  <TimelineMeta>{entry.phase}</TimelineMeta>
                  {entry.event}
                  {entry.detail && <span> · {entry.detail}</span>}
                  {attrSummary && <span> · {attrSummary}</span>}
                </TimelineItem>
              );
            })}
          </TimelineList>
        )}
      </Section>
    </Wrap>
  );
}
