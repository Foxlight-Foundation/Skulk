import { useCallback, useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import type { TraceCategoryStats, TraceStatsResponse } from '../../types/traces';

export interface TraceDetailPageProps {
  taskId: string;
  scope: 'cluster' | 'local';
  onBack: () => void;
}

interface PhaseData {
  name: string;
  subcategories: { name: string; stats: TraceCategoryStats }[];
  totalUs: number;
  stepCount: number;
}

function traceBasePath(scope: 'cluster' | 'local'): string {
  return scope === 'cluster' ? '/v1/traces/cluster' : '/v1/traces';
}

function formatDuration(microseconds: number): string {
  if (microseconds < 1_000) return `${microseconds.toFixed(0)}us`;
  if (microseconds < 1_000_000) return `${(microseconds / 1_000).toFixed(2)}ms`;
  return `${(microseconds / 1_000_000).toFixed(2)}s`;
}

function formatPercentage(part: number, total: number): string {
  if (total === 0) return '0.0%';
  return `${((part / total) * 100).toFixed(1)}%`;
}

function parsePhases(byCategory: Record<string, TraceCategoryStats>): PhaseData[] {
  const phases = new Map<string, { subcategories: Map<string, TraceCategoryStats>; outerStats: TraceCategoryStats | null }>();

  for (const [category, categoryStats] of Object.entries(byCategory)) {
    if (category.includes('/')) {
      const [phaseName, subcategoryName] = category.split('/', 2);
      const existing = phases.get(phaseName) ?? { subcategories: new Map(), outerStats: null };
      existing.subcategories.set(subcategoryName, categoryStats);
      phases.set(phaseName, existing);
      continue;
    }

    const existing = phases.get(category) ?? { subcategories: new Map(), outerStats: null };
    existing.outerStats = categoryStats;
    phases.set(category, existing);
  }

  return [...phases.entries()]
    .filter(([, data]) => data.outerStats !== null)
    .map(([name, data]) => ({
      name,
      subcategories: [...data.subcategories.entries()]
        .map(([subcategoryName, stats]) => ({ name: subcategoryName, stats }))
        .sort((left, right) => right.stats.totalUs - left.stats.totalUs),
      totalUs: data.outerStats!.totalUs,
      stepCount: data.outerStats!.count,
    }))
    .sort((left, right) => right.totalUs - left.totalUs);
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

export function TraceDetailPage({ taskId, scope, onBack }: TraceDetailPageProps) {
  const [stats, setStats] = useState<TraceStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${traceBasePath(scope)}/${encodeURIComponent(taskId)}/stats`);
      if (!response.ok) {
        throw new Error(`Failed to load trace (${response.status})`);
      }
      setStats((await response.json()) as TraceStatsResponse);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to load trace');
    } finally {
      setLoading(false);
    }
  }, [scope, taskId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const phases = useMemo(() => (stats ? parsePhases(stats.byCategory) : []), [stats]);
  const sortedRanks = useMemo(
    () => (stats ? Object.keys(stats.byRank).map(Number).sort((left, right) => left - right) : []),
    [stats],
  );
  const nodeCount = sortedRanks.length || 1;

  return (
    <Page>
      <Header>
        <div>
          <Button variant="ghost" size="sm" onClick={onBack}>← All Traces</Button>
          <Title>Trace</Title>
          <TaskIdLabel>{taskId}</TaskIdLabel>
          <ScopeLabel>{scope === 'cluster' ? 'Cluster view' : 'Local view'}</ScopeLabel>
        </div>
        <HeaderActions>
          <Button variant="outline" size="sm" onClick={() => void refresh()} loading={loading}>
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={() => void downloadTrace(scope, taskId)} disabled={loading || !!error}>
            Download
          </Button>
          <Button variant="primary" size="sm" onClick={() => void openInPerfetto(scope, taskId)} disabled={loading || !!error}>
            Perfetto
          </Button>
        </HeaderActions>
      </Header>

      {loading ? (
        <StateBox>Loading trace data…</StateBox>
      ) : error ? (
        <ErrorBox>{error}</ErrorBox>
      ) : stats ? (
        <>
          <SummaryCard>
            <SectionLabel>Summary</SectionLabel>
            <SummaryValue>{formatDuration(stats.totalWallTimeUs)}</SummaryValue>
            <SummaryHint>Total wall time</SummaryHint>
            {stats.sourceNodes.length > 0 && (
              <SourceNodes>
                {stats.sourceNodes.map((sourceNode) => (
                  <SourceNodeBadge key={sourceNode.nodeId}>
                    {sourceNode.friendlyName || sourceNode.nodeId}
                  </SourceNodeBadge>
                ))}
              </SourceNodes>
            )}
          </SummaryCard>

          {phases.length > 0 && (
            <SectionCard>
              <SectionLabel>By Phase</SectionLabel>
              <SectionHint>Average per rank represented in the trace file.</SectionHint>
              <Stack>
                {phases.map((phase) => {
                  const normalizedTotal = phase.totalUs / nodeCount;
                  const normalizedStepCount = phase.stepCount / nodeCount;
                  const averageStepDuration = normalizedStepCount > 0 ? normalizedTotal / normalizedStepCount : 0;
                  return (
                    <PhaseBlock key={phase.name}>
                      <PhaseHeader>
                        <PhaseName>{phase.name}</PhaseName>
                        <PhaseNumbers>
                          <StrongValue>{formatDuration(normalizedTotal)}</StrongValue>
                          <MutedValue>
                            ({normalizedStepCount.toFixed(1)} steps, {formatDuration(averageStepDuration)}/step)
                          </MutedValue>
                        </PhaseNumbers>
                      </PhaseHeader>
                      {phase.subcategories.length > 0 && (
                        <SubcategoryList>
                          {phase.subcategories.map((subcategory) => {
                            const normalizedSubcategory = subcategory.stats.totalUs / nodeCount;
                            return (
                              <SubcategoryRow key={`${phase.name}/${subcategory.name}`}>
                                <span>{subcategory.name}</span>
                                <MutedValue>
                                  {formatDuration(normalizedSubcategory)} ({formatPercentage(normalizedSubcategory, normalizedTotal)})
                                </MutedValue>
                              </SubcategoryRow>
                            );
                          })}
                        </SubcategoryList>
                      )}
                    </PhaseBlock>
                  );
                })}
              </Stack>
            </SectionCard>
          )}

          {sortedRanks.length > 0 && (
            <SectionCard>
              <SectionLabel>By Rank</SectionLabel>
              <RankGrid>
                {sortedRanks.map((rank) => {
                  const rankStats = stats.byRank[String(rank)] ?? stats.byRank[rank];
                  return (
                    <RankCard key={rank}>
                      <RankTitle>Rank {rank}</RankTitle>
                      <Stack>
                        {Object.entries(rankStats.byCategory)
                          .sort((left, right) => right[1].totalUs - left[1].totalUs)
                          .map(([category, categoryStats]) => (
                            <SubcategoryRow key={`${rank}-${category}`}>
                              <span>{category}</span>
                              <MutedValue>{formatDuration(categoryStats.totalUs)}</MutedValue>
                            </SubcategoryRow>
                          ))}
                      </Stack>
                    </RankCard>
                  );
                })}
              </RankGrid>
            </SectionCard>
          )}
        </>
      ) : null}
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
  margin: 12px 0 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xl};
  color: ${({ theme }) => theme.colors.gold};
`;

const TaskIdLabel = styled.p`
  margin: 6px 0 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  word-break: break-all;
`;

const ScopeLabel = styled.p`
  margin: 6px 0 0;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const SummaryCard = styled.div`
  padding: 20px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceSunken};
`;

const SummaryValue = styled.div`
  margin-top: 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xxl};
  color: ${({ theme }) => theme.colors.gold};
`;

const SummaryHint = styled.div`
  margin-top: 6px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SourceNodes = styled.div`
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
`;

const SourceNodeBadge = styled.span`
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceElevated};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

const SectionCard = styled.div`
  padding: 20px;
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surfaceSunken};
`;

const SectionLabel = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ theme }) => theme.colors.text};
`;

const SectionHint = styled.div`
  margin-top: 6px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Stack = styled.div`
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const PhaseBlock = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const PhaseHeader = styled.div`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
`;

const PhaseName = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.text};
`;

const PhaseNumbers = styled.div`
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
`;

const StrongValue = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.gold};
`;

const MutedValue = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SubcategoryList = styled.div`
  margin-left: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SubcategoryRow = styled.div`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const RankGrid = styled.div`
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
`;

const RankCard = styled.div`
  padding: 16px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surface};
`;

const RankTitle = styled.h3`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.gold};
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

const ErrorBox = styled(StateBox)`
  color: ${({ theme }) => theme.colors.errorText};
  border-color: ${({ theme }) => theme.colors.errorBg};
  background: ${({ theme }) => theme.colors.errorBg};
`;
