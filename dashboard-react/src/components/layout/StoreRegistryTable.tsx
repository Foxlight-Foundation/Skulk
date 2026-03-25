import { useMemo } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { formatBytes } from '../../utils/format';
import { Button } from '../common/Button';

/* ================================================================
   Types
   ================================================================ */

export interface StoreRegistryEntry {
  model_id: string;
  total_bytes: number;
  files: Array<{ name: string }>;
  downloaded_at: string;
}

export interface StoreDownloadProgress {
  modelId: string;
  progress: number;
  status: string;
}

export interface StoreRegistryTableProps {
  entries: StoreRegistryEntry[];
  activeDownloads?: StoreDownloadProgress[];
  loading?: boolean;
  activeModelIds?: string[];
  onRefresh: () => void;
  onInfo: (entry: StoreRegistryEntry) => void;
  onDelete: (entry: StoreRegistryEntry, isActive: boolean) => void;
}

/* ---- helpers ---- */

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return 'just now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
}

/* ================================================================
   Styles
   ================================================================ */

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.4; }
`;

const shimmer = keyframes`
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
`;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const HeaderRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const HeaderText = styled.span`
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textSecondary};
`;


const ShimmerRow = styled.div`
  height: 48px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: linear-gradient(90deg, rgba(80,80,80,0.1) 25%, rgba(80,80,80,0.2) 50%, rgba(80,80,80,0.1) 75%);
  background-size: 200% 100%;
  animation: ${shimmer} 1.5s ease-in-out infinite;
`;

const EmptyBox = styled.div`
  border: 1px solid rgba(80, 80, 80, 0.3);
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(0, 0, 0, 0.3);
  padding: 24px;
  text-align: center;
  font-size: 13px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Table = styled.div`
  border: 1px solid rgba(80, 80, 80, 0.3);
  border-radius: ${({ theme }) => theme.radii.md};
  overflow: hidden;
`;

const THead = styled.div`
  display: grid;
  grid-template-columns: 1fr 80px 60px 100px 60px;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(0, 0, 0, 0.4);
  font-size: 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const TRow = styled.div<{ $highlight?: boolean }>`
  display: grid;
  grid-template-columns: 1fr 80px 60px 100px 60px;
  gap: 8px;
  padding: 10px 12px;
  align-items: center;
  border-top: 1px solid rgba(80, 80, 80, 0.2);
  transition: background 0.15s;

  &:hover { background: rgba(80, 80, 80, 0.1); }

  ${({ $highlight }) =>
    $highlight &&
    css`background: rgba(255, 215, 0, 0.05);`}
`;

const ModelCell = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
`;

const ModelId = styled.span`
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ActiveBadge = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  font-size: 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  color: #4ade80;
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.2);
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 6px;
`;

const PulseDot = styled.span`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #4ade80;
  animation: ${pulse} 1.5s ease-in-out infinite;
`;

const Cell = styled.div<{ $align?: string }>`
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textSecondary};
  text-align: ${({ $align }) => $align ?? 'left'};
`;

const ProgressTrack = styled.div`
  width: 96px;
  height: 6px;
  background: rgba(80, 80, 80, 0.3);
  border-radius: 3px;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $pct: number }>`
  width: ${({ $pct }) => $pct}%;
  height: 100%;
  background: #FFD700;
  border-radius: 3px;
  transition: width 0.3s ease-out;
`;

const ProgressText = styled.span`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: #FFD700;
`;

const ActionsCell = styled.div`
  display: flex;
  gap: 4px;
  opacity: 0.4;
  transition: opacity 0.15s;
  ${TRow}:hover & { opacity: 1; }
`;


/* ================================================================
   Component
   ================================================================ */

export function StoreRegistryTable({
  entries,
  activeDownloads = [],
  loading = false,
  activeModelIds = [],
  onRefresh,
  onInfo,
  onDelete,
}: StoreRegistryTableProps) {
  const registeredIds = useMemo(() => new Set(entries.map((e) => e.model_id)), [entries]);
  const pendingDownloads = useMemo(
    () => activeDownloads.filter((d) => !registeredIds.has(d.modelId)),
    [activeDownloads, registeredIds],
  );
  const downloadMap = useMemo(() => {
    const m = new Map<string, StoreDownloadProgress>();
    for (const d of activeDownloads) m.set(d.modelId, d);
    return m;
  }, [activeDownloads]);

  const isActive = (id: string) => activeModelIds.includes(id);
  const downloadingCount = activeDownloads.length;

  return (
    <Container>
      <HeaderRow>
        <HeaderText>
          {entries.length} model{entries.length !== 1 ? 's' : ''} in store
          {downloadingCount > 0 && `, ${downloadingCount} downloading`}
        </HeaderText>
        <Button variant="outline" size="sm" onClick={onRefresh}>Refresh</Button>
      </HeaderRow>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[1, 2, 3, 4].map((i) => <ShimmerRow key={i} />)}
        </div>
      ) : entries.length === 0 && pendingDownloads.length === 0 ? (
        <EmptyBox>No models in store.</EmptyBox>
      ) : (
        <Table>
          <THead>
            <div>Model</div>
            <div style={{ textAlign: 'right' }}>Size</div>
            <div style={{ textAlign: 'right' }}>Files</div>
            <div style={{ textAlign: 'right' }}>Status</div>
            <div />
          </THead>

          {/* Pending downloads (not yet registered) */}
          {pendingDownloads.map((dl) => (
            <TRow key={dl.modelId} $highlight>
              <ModelCell>
                <ModelId title={dl.modelId}>{dl.modelId}</ModelId>
              </ModelCell>
              <Cell $align="right">—</Cell>
              <Cell $align="right">—</Cell>
              <Cell $align="right">
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                  <ProgressTrack>
                    <ProgressFill $pct={dl.progress * 100} />
                  </ProgressTrack>
                  <ProgressText>{(dl.progress * 100).toFixed(0)}%</ProgressText>
                </div>
              </Cell>
              <Cell />
            </TRow>
          ))}

          {/* Registered entries */}
          {entries.map((entry) => {
            const dl = downloadMap.get(entry.model_id);
            const active = isActive(entry.model_id);
            return (
              <TRow key={entry.model_id}>
                <ModelCell>
                  <ModelId title={entry.model_id}>{entry.model_id}</ModelId>
                  {active && (
                    <ActiveBadge><PulseDot /> Active</ActiveBadge>
                  )}
                </ModelCell>
                <Cell $align="right">{formatBytes(entry.total_bytes)}</Cell>
                <Cell $align="right">{entry.files.length}</Cell>
                <Cell $align="right">
                  {dl ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                      <ProgressTrack>
                        <ProgressFill $pct={dl.progress * 100} />
                      </ProgressTrack>
                      <ProgressText>{(dl.progress * 100).toFixed(0)}%</ProgressText>
                    </div>
                  ) : (
                    timeAgo(entry.downloaded_at)
                  )}
                </Cell>
                <ActionsCell>
                  <Button variant="ghost" size="sm" icon onClick={() => onInfo(entry)} title="Model info">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" />
                    </svg>
                  </Button>
                  <Button variant="danger" size="sm" icon onClick={() => onDelete(entry, active)} title="Delete model">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </Button>
                </ActionsCell>
              </TRow>
            );
          })}
        </Table>
      )}
    </Container>
  );
}
