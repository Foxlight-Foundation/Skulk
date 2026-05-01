import { useMemo, useState } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { FiRefreshCw } from 'react-icons/fi';
import { Button } from '../common/Button';
import { CenteredSpinner, Spinner } from '../common/Spinner';
import { useAppSelector } from '../../store/hooks';
import {
  useGetClusterTimelineQuery,
  useGetTracingStateQuery,
  useSetTracingStateMutation,
} from '../../store/endpoints/observability';
import type {
  ClusterTimelineEntry,
  ClusterTimelineRunner,
} from '../../types/diagnostics';

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
  flex: 1;
  min-height: 0;
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

/**
 * Same as Section but claims the remaining vertical space inside Wrap. Used
 * for the cross-rank timeline so the entry list grows with the panel rather
 * than capping at a hard pixel height and leaving empty space below.
 */
const FillSection = styled.section`
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
  min-height: 0;
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
  flex: 1;
  min-height: 0;
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

/**
 * Refresh icon that spins while the user-initiated refresh is in flight.
 * The button itself stays fixed-width — only the inner icon rotates — so
 * polling-driven state changes can never reflow the header strip.
 */
const refreshSpin = keyframes`
  to { transform: rotate(360deg); }
`;

const RefreshIcon = styled(FiRefreshCw)<{ $spinning: boolean }>`
  ${({ $spinning }) =>
    $spinning &&
    css`
      animation: ${refreshSpin} 0.7s linear infinite;
    `}
`;

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

/**
 * Phases that explicitly aren't "doing work right now" — time spent in any of
 * these is normal regardless of duration. A runner can sit idle for hours
 * between requests and we shouldn't paint that as a hang; same for terminal
 * states like `completion`, `error`, and `shutdown_cleanup` where the runner
 * is past the work it could be stuck on.
 */
const NON_WORKING_PHASES: ReadonlySet<string> = new Set([
  'idle',
  'completion',
  'error',
  'shutdown_cleanup',
  'cancel_observed',
  'created',
]);

function isHung(runner: ClusterTimelineRunner): boolean {
  if (!runner.processAlive) return false;
  if (NON_WORKING_PHASES.has(runner.phase)) return false;
  // Active task association alone isn't enough — we also require a working
  // phase, because `activeTaskId` can linger briefly after a task completes
  // before the next phase transition clears it.
  return Boolean(runner.activeTaskId && runner.secondsInPhase >= HANG_PHASE_SECONDS);
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
  // Pause polling whenever the panel is closed. RTK Query treats `skip: true`
  // as "this consumer doesn't need data right now"; the cache entry is kept
  // and polling resumes on the next render with `skip: false`.
  const panelOpen = useAppSelector((s) => s.ui.observabilityPanelOpen);

  const timelineQuery = useGetClusterTimelineQuery(undefined, {
    pollingInterval: REFRESH_MS,
    skip: !panelOpen,
  });
  const tracingQuery = useGetTracingStateQuery(undefined, { skip: !panelOpen });
  const [setTracingState, tracingMutation] = useSetTracingStateMutation();

  // Stale-while-revalidate: render whatever was last fetched while a refresh
  // is in flight, never blowing away the rendered view. The first-load
  // skeleton is therefore the only state that hides the data sections.
  const timeline = timelineQuery.data ?? null;
  const tracingEnabled = tracingQuery.data?.enabled ?? null;

  // Surface refresh status only for explicit user-initiated refetches, not
  // for the background poll. RTK Query's `isFetching` flips true on every
  // 4-second poll; if we wired the Refresh button's spinner directly to it
  // the button would change width on every tick, reflowing the header strip.
  const [userRefreshing, setUserRefreshing] = useState(false);
  const handleManualRefresh = async () => {
    setUserRefreshing(true);
    try {
      await timelineQuery.refetch();
    } finally {
      setUserRefreshing(false);
    }
  };

  const error = timelineQuery.isError
    ? (timelineQuery.error as { error?: string })?.error ?? 'Failed to load cluster timeline'
    : null;
  const tracingError = tracingMutation.isError
    ? (tracingMutation.error as { error?: string })?.error ?? 'Tracing toggle failed'
    : null;
  const tracingToggling = tracingMutation.isLoading;

  async function toggleTracing() {
    if (tracingEnabled == null) return;
    try {
      await setTracingState(!tracingEnabled).unwrap();
    } catch {
      // Error surfaces via tracingMutation.isError, picked up above.
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
        <Button
          variant="outline"
          size="sm"
          icon
          onClick={() => { void handleManualRefresh(); }}
          aria-label="Refresh cluster timeline"
          title="Refresh"
        >
          <RefreshIcon $spinning={userRefreshing} size={14} />
        </Button>
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
      </HeaderStrip>

      {error && <ErrorNotice>{error}</ErrorNotice>}
      {tracingError && <ErrorNotice>{tracingError}</ErrorNotice>}

      {!timeline && !error && (
        <CenteredSpinner>
          <Spinner />
        </CenteredSpinner>
      )}

      {timeline && (
        <Section>
        <SectionTitle>Runners</SectionTitle>
        {timeline.runners.length === 0 && (
          <Notice>No runners reported across the cluster.</Notice>
        )}
        {timeline.runners.length > 0 && (
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
      )}

      {timeline && (
        <FillSection>
        <SectionTitle>Cross-rank timeline (newest first)</SectionTitle>
        {recentTimeline.length === 0 && (
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
        </FillSection>
      )}
    </Wrap>
  );
}
