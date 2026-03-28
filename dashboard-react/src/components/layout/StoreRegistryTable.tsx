import { useMemo } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { FiTrash2, FiExternalLink, FiRefreshCw } from 'react-icons/fi';
import { MdPlayArrow, MdClose } from 'react-icons/md';
import { formatBytes } from '../../utils/format';
import { Button } from '../common/Button';
import { InfoTooltip } from '../common/InfoTooltip';
import { ClusterCard, type ClusterCardProps } from '../cluster/ClusterCard';

/* ================================================================
   Types
   ================================================================ */

export interface StoreRegistryEntry {
  model_id: string;
  total_bytes: number;
  files: string[];
  downloaded_at: string;
}

export interface StoreDownloadProgress {
  modelId: string;
  progress: number;
  status: string;
}

export interface ModelCardInfo {
  family?: string;
  quantization?: string;
  baseModel?: string;
  supportsTensor?: boolean;
  capabilities?: string[];
}

export interface StoreRegistryTableProps {
  entries: StoreRegistryEntry[];
  activeDownloads?: StoreDownloadProgress[];
  loading?: boolean;
  activeModelIds?: string[];
  modelCards?: Record<string, ModelCardInfo>;
  actions?: React.ReactNode;
  onRefresh: () => void;
  onDelete: (entry: StoreRegistryEntry, isActive: boolean) => void;
  onLaunch?: (modelId: string) => void;
  onStop?: (modelId: string) => void;
  clusterCards?: Record<string, Omit<ClusterCardProps, 'onLaunch'>>;
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

const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const HeaderText = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
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
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Table = styled.div`
  border: 1px solid rgba(80, 80, 80, 0.3);
  border-radius: ${({ theme }) => theme.radii.md};
  overflow: hidden;
`;

const THead = styled.div`
  display: grid;
  grid-template-columns: 36px 1fr 80px 60px 100px 60px;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(0, 0, 0, 0.4);
  font-size: ${({ theme }) => theme.fontSizes.tableHead};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const TRow = styled.div<{ $highlight?: boolean }>`
  display: grid;
  grid-template-columns: 36px 1fr 80px 60px 100px 60px;
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
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ReadyBadge = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: #4ade80;
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.2);
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 6px;
`;

const ActiveBadge = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.gold};
  background: ${({ theme }) => theme.colors.goldBg};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
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

const ActiveDot = styled.span`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: ${({ theme }) => theme.colors.gold};
  animation: ${pulse} 1.5s ease-in-out infinite;
`;

const Cell = styled.div<{ $align?: string }>`
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-family: ${({ theme }) => theme.fonts.body};
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
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: #FFD700;
`;

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

const RefreshBtn = styled.button<{ $spinning: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.textSecondary};
  transition: color 0.15s, background 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.gold};
    background: ${({ theme }) => theme.colors.goldBg};
  }

  svg {
    ${({ $spinning }) => $spinning && css`animation: ${spin} 0.8s linear infinite;`}
  }
`;

const PlayCell = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

const PlayBtn = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.gold};
  background: rgba(255, 215, 0, 0.15);
  transition: background 0.15s, transform 0.1s;

  &:hover {
    background: rgba(255, 215, 0, 0.3);
    transform: scale(1.1);
  }
`;

const StopBtn = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.error};
  background: rgba(239, 68, 68, 0.15);
  transition: background 0.15s, transform 0.1s;

  &:hover {
    background: rgba(239, 68, 68, 0.3);
    transform: scale(1.1);
  }
`;

const ActionsCell = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  opacity: 0.7;
  transition: opacity 0.15s;
  ${TRow}:hover & { opacity: 1; }
`;


/* ================================================================
   Component
   ================================================================ */

const LinkIcon = () => <FiExternalLink size={14} style={{ flexShrink: 0 }} />;

function ModelInfoContent({ entry, card }: { entry: StoreRegistryEntry; card?: ModelCardInfo }) {
  const hfUrl = entry.model_id.includes('/')
    ? `https://huggingface.co/${entry.model_id}`
    : null;

  return (
    <div style={{ minWidth: 240 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{ color: '#FFD700', fontWeight: 600 }}>
          {entry.model_id}
        </span>
        {hfUrl && (
          <a
            href={hfUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'rgba(255,255,255,0.5)', display: 'flex', transition: 'color 0.15s' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#FFD700'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
            title="Open on HuggingFace"
          >
            <LinkIcon />
          </a>
        )}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Size</span>
        <span>{formatBytes(entry.total_bytes)}</span>
        {card?.baseModel && (
          <>
            <span style={{ color: 'rgba(255,255,255,0.45)' }}>Base model</span>
            <span>{card.baseModel}</span>
          </>
        )}
        {card?.family && (
          <>
            <span style={{ color: 'rgba(255,255,255,0.45)' }}>Family</span>
            <span>{card.family}</span>
          </>
        )}
        {card?.quantization && (
          <>
            <span style={{ color: 'rgba(255,255,255,0.45)' }}>Quantization</span>
            <span>{card.quantization}</span>
          </>
        )}
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Tensor parallel</span>
        <span style={{ color: card?.supportsTensor ? '#4ade80' : 'rgba(255,255,255,0.7)' }}>
          {card?.supportsTensor ? 'Yes' : 'No'}
        </span>
        {card?.capabilities && card.capabilities.length > 0 && (
          <>
            <span style={{ color: 'rgba(255,255,255,0.45)' }}>Capabilities</span>
            <span>{card.capabilities.join(', ')}</span>
          </>
        )}
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Files</span>
        <span>{entry.files.length}</span>
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Downloaded</span>
        <span>{new Date(entry.downloaded_at).toLocaleString()}</span>
      </div>
      {entry.files.length > 0 && (
        <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 6 }}>
          <div style={{ color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            Files
          </div>
          <div style={{ color: 'rgba(255,255,255,0.6)', maxHeight: 120, overflowY: 'auto' }}>
            {entry.files.map((f) => (
              <div key={f}>{f}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function StoreRegistryTable({
  entries,
  activeDownloads = [],
  loading = false,
  activeModelIds = [],
  modelCards = {},
  actions,
  onRefresh,
  onDelete,
  onLaunch,
  onStop,
  clusterCards = {},
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
        <HeaderActions>
          {actions}
          <RefreshBtn onClick={onRefresh} $spinning={loading} title="Refresh">
            <FiRefreshCw size={16} />
          </RefreshBtn>
        </HeaderActions>
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
            <div />
            <div>Model</div>
            <div style={{ textAlign: 'right' }}>Size</div>
            <div style={{ textAlign: 'right' }}>Files</div>
            <div style={{ textAlign: 'right' }}>Status</div>
            <div />
          </THead>

          {/* Pending downloads (not yet registered) */}
          {pendingDownloads.map((dl) => (
            <TRow key={dl.modelId} $highlight>
              <Cell />
              <ModelCell>
                <ModelId title={dl.modelId}>{dl.modelId}</ModelId>
              </ModelCell>
              <Cell />
              <Cell />
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
                <PlayCell>
                  {active && onStop ? (
                    <StopBtn onClick={() => onStop(entry.model_id)} title="Stop model">
                      <MdClose size={20} />
                    </StopBtn>
                  ) : !active && !dl && onLaunch ? (
                    <PlayBtn onClick={() => onLaunch(entry.model_id)} title="Launch model">
                      <MdPlayArrow size={20} />
                    </PlayBtn>
                  ) : null}
                </PlayCell>
                <ModelCell>
                  <ModelId title={entry.model_id}>{entry.model_id}</ModelId>
                  {active && (() => {
                    const card = clusterCards[entry.model_id];
                    const ready = card?.isReady ?? false;
                    const badge = ready
                      ? <ReadyBadge><PulseDot /> Ready</ReadyBadge>
                      : <ActiveBadge><ActiveDot /> Loading</ActiveBadge>;
                    return card ? (
                      <InfoTooltip
                        placement="bottom"
                        delay={100}
                        content={<ClusterCard {...card} />}
                      >
                        {badge}
                      </InfoTooltip>
                    ) : badge;
                  })()}
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
                  <InfoTooltip
                    content={<ModelInfoContent entry={entry} card={modelCards[entry.model_id]} />}
                    placement="left"
                    filled
                    delay={100}
                  />
                  <Button variant="danger" size="sm" icon onClick={() => onDelete(entry, active)} title="Delete model">
                    <FiTrash2 size={18} />
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
