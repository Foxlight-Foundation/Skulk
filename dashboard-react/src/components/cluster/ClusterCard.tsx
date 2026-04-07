import { useMemo } from 'react';
import styled, { keyframes, useTheme } from 'styled-components';
import { FiExternalLink } from 'react-icons/fi';
import { MdPlayArrow } from 'react-icons/md';
import { DeviceIcon } from '../topology/DeviceIcon';
import type { Theme } from '../../theme';

/* ── Types ────────────────────────────────────────────── */

export interface ClusterCardNode {
  nodeId: string;
  name: string;
  memoryUsedPercent: number;
}

export interface ClusterCardDownload {
  nodeId: string;
  nodeName: string;
  percent: number;
}

export interface ClusterCardProps {
  modelId: string;
  modelName?: string;
  sizeBytes?: number;
  sharding: 'Pipeline' | 'Tensor';
  instanceType: 'MlxRing' | 'MlxJaccl';
  nodes: ClusterCardNode[];
  isReady?: boolean;
  downloads?: ClusterCardDownload[];
  onLaunch?: () => void;
  className?: string;
}

/* ── Helpers ──────────────────────────────────────────── */

function formatSize(bytes: number): string {
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(1)}TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)}GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)}MB`;
  return `${bytes}B`;
}

function hfUrl(modelId: string): string | null {
  return modelId.includes('/') ? `https://huggingface.co/${modelId}` : null;
}

/* ── Mini Topology ────────────────────────────────────── */

function MiniTopology({ nodes }: { nodes: ClusterCardNode[] }) {
  const theme = useTheme() as Theme;
  const count = nodes.length;
  const iconW = 48;
  const iconH = 40;
  const w = 240;
  const h = count === 1 ? 80 : 160;
  const cx = w / 2;
  const cy = h / 2;
  const r = count <= 2 ? 50 : Math.min(w, h) * 0.32;

  const positions = useMemo(() => {
    if (count === 1) return [{ x: cx, y: cy - 8 }];
    if (count === 2) return [
      { x: cx - r, y: cy - 8 },
      { x: cx + r, y: cy - 8 },
    ];
    return nodes.map((_, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / count;
      return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) - 4 };
    });
  }, [count, cx, cy, r, nodes.length]);

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Edges */}
      {count > 1 && positions.map((p, i) => {
        const next = positions[(i + 1) % count];
        return (
          <line
            key={`e${i}`}
            x1={p.x} y1={p.y} x2={next.x} y2={next.y}
            stroke={theme.colors.meshLine}
            strokeWidth={1}
            strokeDasharray="4 3"
          />
        );
      })}
      {/* Nodes */}
      {positions.map((p, i) => {
        const node = nodes[i];
        return (
          <g key={node.nodeId}>
            <g transform={`translate(${p.x - iconW / 2}, ${p.y - iconH / 2})`}>
              <DeviceIcon
                model="mac-mini"
                ramPercent={node.memoryUsedPercent}
                width={iconW}
                height={iconH}
                wireColor={theme.colors.goldDim}
                clipId={`cc-${node.nodeId}`}
              />
            </g>
            {/* Memory percent + name */}
            <text
              x={p.x} y={p.y + iconH / 2 + 14}
              textAnchor="middle"
              fill={theme.colors.textSecondary}
              fontSize={11}
              fontFamily="'Outfit', sans-serif"
            >
              {node.memoryUsedPercent}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ── Component ────────────────────────────────────────── */

export function ClusterCard({
  modelId,
  modelName,
  sizeBytes,
  sharding,
  instanceType,
  nodes,
  isReady = false,
  downloads,
  onLaunch,
  className,
}: ClusterCardProps) {
  const link = hfUrl(modelId);
  const displayName = modelName ?? modelId.split('/').pop() ?? modelId;
  const showDownloads = !isReady && downloads && downloads.length > 0;
  const showLaunch = !isReady && onLaunch;

  return (
    <Card className={className}>
      {/* Header */}
      <Header>
        <ModelRow>
          <ModelName>{displayName}</ModelName>
          {link && (
            <LinkBtn href={link} target="_blank" rel="noopener noreferrer" title="Open on HuggingFace">
              <FiExternalLink size={13} />
            </LinkBtn>
          )}
          {sizeBytes != null && <SizeBadge>{formatSize(sizeBytes)}</SizeBadge>}
        </ModelRow>
        <ModelIdText>{modelId}</ModelIdText>
      </Header>

      {/* Sharding / Instance type badges */}
      <BadgeRow>
        <TypeBadge>{sharding}</TypeBadge>
        <TypeBadge>{instanceType === 'MlxRing' ? 'MLX Ring' : 'MLX Jaccl'}</TypeBadge>
        {isReady && <ReadyBadge><PulseDot /> Ready</ReadyBadge>}
      </BadgeRow>

      {/* Download progress */}
      {showDownloads && (
        <DownloadSection>
          <SectionLabel>Download Progress</SectionLabel>
          {downloads.map((dl) => (
            <DownloadRow key={dl.nodeId}>
              <DownloadNode>{dl.nodeName}</DownloadNode>
              <DownloadBar>
                <DownloadFill $pct={dl.percent} />
              </DownloadBar>
              <DownloadPct>{dl.percent}%</DownloadPct>
            </DownloadRow>
          ))}
        </DownloadSection>
      )}

      {/* Mini topology */}
      <TopoWrap>
        <MiniTopology nodes={nodes} />
      </TopoWrap>

      {/* Launch button */}
      {showLaunch && (
        <LaunchBtn onClick={onLaunch}>
          <MdPlayArrow size={18} /> Launch
        </LaunchBtn>
      )}
    </Card>
  );
}

/* ── Styles ───────────────────────────────────────────── */

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
`;

const Card = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 280px;
  transition: border-color 0.2s;

  &:hover {
    border-color: ${({ theme }) => theme.colors.goldDim};
  }
`;

const Header = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
`;

const ModelRow = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
`;

const ModelName = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
`;

const LinkBtn = styled.a`
  color: ${({ theme }) => theme.colors.textMuted};
  display: flex;
  flex-shrink: 0;
  transition: color 0.15s;
  &:hover { color: ${({ theme }) => theme.colors.gold}; }
`;

const SizeBadge = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.gold};
  flex-shrink: 0;
`;

const ModelIdText = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const BadgeRow = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
`;

const TypeBadge = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 3px 10px;
`;

const ReadyBadge = styled.span`
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.healthy};
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.2);
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 3px 10px;
`;

const PulseDot = styled.span`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: ${({ theme }) => theme.colors.healthy};
  animation: ${pulse} 1.5s ease-in-out infinite;
`;

const SectionLabel = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  margin-bottom: 4px;
`;

const DownloadSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const DownloadRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const DownloadNode = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  width: 48px;
  flex-shrink: 0;
`;

const DownloadBar = styled.div`
  flex: 1;
  height: 4px;
  background: ${({ theme }) => theme.colors.borderLight};
  border-radius: 2px;
  overflow: hidden;
`;

const DownloadFill = styled.div<{ $pct: number }>`
  width: ${({ $pct }) => $pct}%;
  height: 100%;
  background: ${({ theme }) => theme.colors.gold};
  border-radius: 2px;
  transition: width 0.3s ease-out;
`;

const DownloadPct = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.gold};
  width: 36px;
  text-align: right;
  flex-shrink: 0;
`;

const TopoWrap = styled.div`
  display: flex;
  justify-content: center;
  padding: 4px 0;
`;

const LaunchBtn = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.textSecondary};
  transition: all 0.15s;

  &:hover {
    border-color: ${({ theme }) => theme.colors.goldDim};
    color: ${({ theme }) => theme.colors.gold};
    background: ${({ theme }) => theme.colors.goldBg};
  }
`;
