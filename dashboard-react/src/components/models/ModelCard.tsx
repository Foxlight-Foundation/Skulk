import styled, { css, keyframes, useTheme } from 'styled-components';
import type { NodeInfo } from '../../types/topology';
import type { DownloadProgress, PlacementPreview } from '../../types/models';
import { formatBytes } from '../../utils/format';
import { Button } from '../common/Button';
import type { Theme } from '../../theme';

/* ================================================================
   Types
   ================================================================ */

export interface NodeDownloadStatus {
  nodeId: string;
  nodeName: string;
  status: 'completed' | 'partial' | 'pending' | 'downloading';
  percentage: number;
  progress: DownloadProgress | null;
}

export interface ModelCardDownloadStatus {
  isDownloading: boolean;
  progress: DownloadProgress | null;
  perNode?: NodeDownloadStatus[];
}

export interface ModelCardProps {
  model: { id: string; name?: string; storage_size_megabytes?: number };
  isLaunching?: boolean;
  downloadStatus?: ModelCardDownloadStatus | null;
  nodes?: Record<string, NodeInfo>;
  sharding?: 'Pipeline' | 'Tensor';
  runtime?: 'MlxRing' | 'MlxJaccl';
  onLaunch?: () => void;
  tags?: string[];
  apiPreview?: PlacementPreview | null;
  modelIdOverride?: string | null;
  hideActions?: boolean;
}

/* ================================================================
   Helpers
   ================================================================ */

function estimateMemoryGB(modelId: string, modelName?: string): number {
  const combined = `${modelId} ${modelName || ''}`.toLowerCase();
  const is4bit = combined.includes('4bit') || combined.includes('4-bit') || combined.includes(':4bit');
  const is8bit = combined.includes('8bit') || combined.includes('8-bit') || combined.includes(':8bit');
  const quantMultiplier = is4bit ? 0.5 : is8bit ? 1 : 2;
  const id = modelId.toLowerCase();

  if (id.includes('deepseek-v3')) return Math.round(685 * quantMultiplier);
  if (id.includes('deepseek-v2')) return Math.round(236 * quantMultiplier);
  if (id.includes('llama-4')) return Math.round(400 * quantMultiplier);

  const paramMatch = id.match(/(\d+(?:\.\d+)?)\s*b(?![a-z])/i);
  if (paramMatch) return Math.max(4, Math.round(parseFloat(paramMatch[1]) * quantMultiplier));

  return 16;
}

function formatSpeed(bps: number): string {
  if (!bps || bps <= 0) return '0 B/s';
  return formatBytes(bps) + '/s';
}

function clamp(v: number, min = 0, max = 100): number {
  return Math.min(max, Math.max(min, v));
}

type DeviceType = 'macbook' | 'studio' | 'mini' | 'unknown';

function getDeviceType(name: string): DeviceType {
  const lower = name.toLowerCase();
  if (lower.includes('macbook')) return 'macbook';
  if (lower.includes('studio')) return 'studio';
  if (lower.includes('mini')) return 'mini';
  return 'unknown';
}

interface PlacementNode {
  id: string;
  deviceType: DeviceType;
  totalGB: number;
  currentPercent: number;
  newPercent: number;
  modelUsageGB: number;
  isUsed: boolean;
  x: number;
  y: number;
  iconSize: number;
}

function computePlacement(
  nodes: Record<string, NodeInfo>,
  apiPreview: PlacementPreview | null,
): { nodes: PlacementNode[]; canFit: boolean; topoWidth: number; topoHeight: number; error: string | null } {
  const ids = Object.keys(nodes);
  const n = ids.length;
  if (n === 0) return { nodes: [], canFit: false, topoWidth: 260, topoHeight: 90, error: null };

  const iconSize = n === 1 ? 50 : 36;
  const topoWidth = 260;
  const topoHeight = n === 1 ? 90 : n === 2 ? 140 : n * 50 + 20;
  const cx = topoWidth / 2;
  const cy = topoHeight / 2;
  const radius = n === 1 ? 0 : n === 2 ? 45 : Math.min(topoWidth, topoHeight) * 0.32;

  const hasPreview = apiPreview !== null && apiPreview.error === null && apiPreview.memory_delta_by_node !== null;
  const memDelta = apiPreview?.memory_delta_by_node ?? {};
  const GB = 1024 * 1024 * 1024;

  const placementNodes: PlacementNode[] = ids.map((id, i) => {
    const info = nodes[id];
    const totalBytes = info.macmon_info?.memory?.ram_total ?? info.system_info?.memory ?? 0;
    const usedBytes = info.macmon_info?.memory?.ram_usage ?? 0;
    const totalGB = totalBytes / GB;
    const usedGB = Math.max(totalBytes - (totalBytes - usedBytes), 0) / GB;
    const deltaBytes = memDelta[id] ?? 0;
    const modelUsageGB = deltaBytes / GB;
    const isUsed = deltaBytes > 0;
    const safeTotal = Math.max(totalGB, 0.001);
    const currentPercent = clamp((usedGB / safeTotal) * 100);
    const newPercent = clamp(((usedGB + modelUsageGB) / safeTotal) * 100);
    const angle = n === 1 ? 0 : (i / n) * Math.PI * 2 - Math.PI / 2;

    return {
      id,
      deviceType: getDeviceType(info.system_info?.model_id ?? ''),
      totalGB,
      currentPercent,
      newPercent,
      modelUsageGB,
      isUsed,
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      iconSize,
    };
  });

  return { nodes: placementNodes, canFit: hasPreview, topoWidth, topoHeight, error: apiPreview?.error ?? null };
}

/* ================================================================
   Styled components
   ================================================================ */

const pulseSlow = keyframes`
  0%, 100% { opacity: 0.8; }
  50%      { opacity: 1; }
`;

const Card = styled.div<{ $canFit: boolean }>`
  position: relative;
  background: transparent;
  padding: 12px;
`;

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
`;

const ModelName = styled.div<{ $canFit: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ $canFit, theme}) => ($canFit ? theme.colors.gold : '#f87171')};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ModelId = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(179, 179, 179, 0.8);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 2px;
`;

const MemorySize = styled.div<{ $canFit: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ $canFit, theme}) => ($canFit ? theme.colors.gold : '#f87171')};
  flex-shrink: 0;
`;

const HfLink = styled.a`
  color: ${({ theme }) => theme.colors.textSecondary};
  flex-shrink: 0;
  transition: color 0.15s;
  &:hover { color: ${({ theme }) => theme.colors.gold}; }
`;

const BadgeRow = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
`;

const Badge = styled.span<{ $color?: string }>`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.radii.sm};
  background: ${({ theme }) => theme.colors.surfaceHover};
  border: 1px solid ${({ theme }) => theme.colors.border};
  color: ${({ $color, theme }) => $color ?? theme.colors.textSecondary};
`;

const TagBadge = styled.span<{ $variant: 'green' | 'purple' }>`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.radii.sm};
  ${({ $variant }) =>
    $variant === 'green'
      ? css`background: rgba(34,197,94,0.2); color: ${({ theme }) => theme.colors.healthy}; border: 1px solid rgba(34,197,94,0.3);`
      : css`background: rgba(168,85,247,0.2); color: #c084fc; border: 1px solid rgba(168,85,247,0.3);`}
`;

const SectionTitle = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.border};
  margin-bottom: 4px;
`;

const DownloadRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
`;

const ProgressTrack = styled.div`
  flex: 1;
  height: 4px;
  background: rgba(80, 80, 80, 0.3);
  border-radius: 2px;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $status: string }>`
  height: 100%;
  transition: width 0.3s;
  ${({ $status }) => {
    if ($status === 'downloading') return css`background: rgba(59,130,246,0.7);`;
    if ($status === 'completed') return css`background: ${({ theme }) => theme.colors.goldDim};`;
    return css`background: ${({ theme }) => theme.colors.border};`;
  }}
`;

const PreviewBox = styled.div`
  margin-bottom: 12px;
  background: ${({ theme }) => theme.colors.shadow};
  border: 1px solid rgba(80, 80, 80, 0.2);
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 8px;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,215,0,0.02) 2px, rgba(255,215,0,0.02) 4px);
    pointer-events: none;
  }
`;

const LaunchBtn = styled(Button)<{ $canFit: boolean; $launching: boolean }>`
  width: 100%;
  font-size: ${({ theme }) => theme.fontSizes.tableBody};

  ${({ $launching }) =>
    $launching &&
    css`
      color: ${({ theme }) => theme.colors.gold};
      border-color: ${({ theme }) => theme.colors.goldDim};
      cursor: wait;
    `}

  ${({ $canFit, $launching }) =>
    !$canFit &&
    !$launching &&
    css`
      background: ${({ theme }) => theme.colors.errorBg};
      color: rgba(248, 113, 113, 1);
      border-color: ${({ theme }) => theme.colors.errorBg};
      cursor: not-allowed;
      opacity: 1;
    `}
`;

const LaunchSpinner = styled.span`
  display: inline-block;
  width: 8px;
  height: 8px;
  border: 1px solid ${({ theme }) => theme.colors.gold};
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
  @keyframes spin { to { transform: rotate(360deg); } }
`;

const ModelFill = styled.rect`
  animation: ${pulseSlow} 1.5s ease-in-out infinite;
`;

/* ================================================================
   Component
   ================================================================ */

export function ModelCard({
  model,
  isLaunching = false,
  downloadStatus = null,
  nodes = {},
  sharding = 'Pipeline',
  runtime = 'MlxRing',
  onLaunch,
  tags = [],
  apiPreview = null,
  modelIdOverride = null,
  hideActions = false,
}: ModelCardProps) {
  const estimatedMemory = model.storage_size_megabytes
    ? Math.round(model.storage_size_megabytes / 1024)
    : estimateMemoryGB(model.id, model.name);

  const theme = useTheme() as Theme;
  // Drop the SVG glow filter in light mode — looks like a hard drop-shadow against
  // the soft blue background.
  const useGlow = theme.colors.bg === '#000000';
  const placement = computePlacement(nodes, apiPreview);
  const canFit = apiPreview ? apiPreview.error === null : placement.canFit;
  const perNode = downloadStatus?.perNode ?? [];
  const hfId = modelIdOverride ?? model.id;
  const filterId = model.id.replace(/[^a-zA-Z0-9]/g, '');

  const runtimeLabel = runtime === 'MlxRing' ? 'MLX Ring' : runtime === 'MlxJaccl' ? 'MLX RDMA' : runtime;

  return (
    <Card $canFit={canFit}>
      {/* Header: name + memory */}
      <Header>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ModelName $canFit={canFit} title={model.name || model.id}>
              {model.name || model.id}
            </ModelName>
            {hfId && (
              <HfLink
                href={`https://huggingface.co/${hfId}`}
                target="_blank"
                rel="noreferrer noopener"
                aria-label="View on HuggingFace"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 3h7v7" /><path d="M10 14l11-11" /><path d="M21 14v6a1 1 0 0 1-1 1h-16a1 1 0 0 1-1-1v-16a1 1 0 0 1 1-1h6" />
                </svg>
              </HfLink>
            )}
            {tags.map((tag) => (
              <TagBadge key={tag} $variant={tag === 'FASTEST' ? 'green' : 'purple'}>
                {tag}
              </TagBadge>
            ))}
          </div>
          {model.name && model.name !== model.id && (
            <ModelId title={model.id}>{model.id}</ModelId>
          )}
        </div>
        <MemorySize $canFit={canFit}>{estimatedMemory}GB</MemorySize>
      </Header>

      {/* Sharding + runtime badges */}
      <BadgeRow>
        <Badge title={sharding === 'Pipeline'
          ? 'Pipeline: splits model into sequential stages across devices.'
          : 'Tensor: splits each layer across devices. Best with high-bandwidth connections.'}>
          {sharding}
        </Badge>
        <Badge title={runtime === 'MlxRing'
          ? 'Ring: standard networking. Works over any connection.'
          : 'RDMA: direct memory access over Thunderbolt.'}>
          {runtimeLabel}
        </Badge>
      </BadgeRow>

      {/* Per-node download progress */}
      {perNode.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <SectionTitle>Download progress</SectionTitle>
          {perNode.map((nd) => (
            <DownloadRow key={nd.nodeId}>
              <span style={{ color: theme.colors.textMuted, width: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={nd.nodeId}>
                {nd.nodeName}
              </span>
              <ProgressTrack>
                <ProgressFill $status={nd.status} style={{ width: `${nd.percentage}%` }} />
              </ProgressTrack>
              <span style={{
                textAlign: 'right',
                color: nd.status === 'completed' ? theme.colors.goldDim : nd.status === 'downloading' ? theme.colors.info : theme.colors.textMuted,
              }}>
                {nd.status === 'downloading' && nd.progress
                  ? `${Math.round(nd.percentage)}% ${formatSpeed(nd.progress.speed)}`
                  : `${nd.percentage > 0 ? Math.round(nd.percentage) : 0}%`}
              </span>
            </DownloadRow>
          ))}
        </div>
      )}

      {/* Mini topology placement preview */}
      {placement.nodes.length > 0 && (
        <PreviewBox>
          <svg
            width="100%"
            height={placement.topoHeight}
            viewBox={`0 0 ${placement.topoWidth} ${placement.topoHeight}`}
            style={{ overflow: 'visible' }}
          >
            <defs>
              <filter id={`nodeGlow-${filterId}`} x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id={`memGlow-${filterId}`} x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Connection lines */}
            {placement.nodes.length > 1 && placement.nodes.map((n, i) =>
              placement.nodes.slice(i + 1).map((n2) => (
                <line
                  key={`${n.id}-${n2.id}`}
                  x1={n.x} y1={n.y} x2={n2.x} y2={n2.y}
                  stroke={n.isUsed && n2.isUsed ? theme.colors.gold : theme.colors.border}
                  strokeWidth={1}
                  strokeDasharray={n.isUsed && n2.isUsed ? '4,2' : '2,4'}
                  opacity={n.isUsed && n2.isUsed ? 0.5 : 0.25}
                />
              )),
            )}

            {/* Node icons */}
            {placement.nodes.map((node) => (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                opacity={node.isUsed ? 1 : (theme.colors.bg === '#000000' ? 0.7 : 0.9)}
                filter={node.isUsed && useGlow ? `url(#nodeGlow-${filterId})` : undefined}
              >
                <PlacementDeviceIcon node={node} filterId={filterId} />
                <text
                  y={node.iconSize / 2 + 12}
                  textAnchor="middle"
                  fontSize={8}
                  fontFamily="SF Mono, Monaco, monospace"
                  fill={node.isUsed ? (node.newPercent > 90 ? theme.colors.error : theme.colors.gold) : theme.colors.textMuted}
                >
                  {node.newPercent.toFixed(0)}%
                </text>
              </g>
            ))}
          </svg>
        </PreviewBox>
      )}

      {/* Launch button */}
      {!hideActions && (
        <LaunchBtn
          variant="outline"
          size="lg"
          block
          $canFit={canFit}
          $launching={isLaunching}
          disabled={isLaunching || !canFit}
          onClick={onLaunch}
        >
          {isLaunching ? (
            <><LaunchSpinner /> LAUNCHING...</>
          ) : !canFit ? (
            'INSUFFICIENT MEMORY'
          ) : (
            '▸ LAUNCH'
          )}
        </LaunchBtn>
      )}
    </Card>
  );
}

/* ================================================================
   Mini device icon for placement preview
   ================================================================ */

function PlacementDeviceIcon({ node, filterId }: { node: PlacementNode; filterId: string }) {
  const theme = useTheme() as Theme;
  const useGlow = theme.colors.bg === '#000000';
  const s = node.iconSize;
  const half = s / 2;
  const stroke = node.isUsed ? theme.colors.gold : theme.colors.border;
  // Idle device case (was hardcoded #1a1a1a) and existing-RAM-usage shade (was #4B5563).
  const bodyFill = theme.colors.deviceBody;
  const usedFill = theme.colors.gpuBarBg;
  const memGlow = useGlow ? `url(#memGlow-${filterId})` : undefined;

  switch (node.deviceType) {
    case 'macbook':
      return (
        <g transform={`translate(${-half}, ${-half})`}>
          <rect x={2} y={0} width={s - 4} height={s * 0.65} rx={2} fill="none" stroke={stroke} strokeWidth={1.5} />
          <rect x={4} y={2} width={s - 8} height={screenH} fill={bodyFill} />
          <rect x={4} y={2 + screenH - currentFillH} width={s - 8} height={currentFillH} fill={usedFill} />
          {node.modelUsageGB > 0 && node.isUsed && (
            <ModelFill x={4} y={2 + screenH - currentFillH - modelFillH} width={s - 8} height={modelFillH} fill={theme.colors.gold} filter={memGlow} />
          )}
          <path d={`M 0 ${s * 0.68} L ${s} ${s * 0.68} L ${s - 2} ${s * 0.78} L 2 ${s * 0.78} Z`} fill="none" stroke={stroke} strokeWidth={1.5} />
        </g>
      );
    case 'studio':
      return (
        <g transform={`translate(${-half}, ${-half})`}>
          <rect x={2} y={2} width={s - 4} height={s - 4} rx={4} fill="none" stroke={stroke} strokeWidth={1.5} />
          <rect x={4} y={4} width={s - 8} height={s - 8} fill={bodyFill} />
          <rect x={4} y={4 + (s - 8) * (1 - node.currentPercent / 100)} width={s - 8} height={(s - 8) * (node.currentPercent / 100)} fill={usedFill} />
          {node.modelUsageGB > 0 && node.isUsed && (
            <ModelFill x={4} y={4 + (s - 8) * (1 - node.newPercent / 100)} width={s - 8} height={(s - 8) * ((node.newPercent - node.currentPercent) / 100)} fill={theme.colors.gold} filter={memGlow} />
          )}
        </g>
      );
    case 'mini':
      return (
        <g transform={`translate(${-half}, ${-half})`}>
          <rect x={2} y={s * 0.3} width={s - 4} height={s * 0.4} rx={3} fill="none" stroke={stroke} strokeWidth={1.5} />
          <rect x={4} y={s * 0.32} width={s - 8} height={s * 0.36} fill={bodyFill} />
          <rect x={4} y={s * 0.32 + s * 0.36 * (1 - node.currentPercent / 100)} width={s - 8} height={s * 0.36 * (node.currentPercent / 100)} fill={usedFill} />
          {node.modelUsageGB > 0 && node.isUsed && (
            <ModelFill x={4} y={s * 0.32 + s * 0.36 * (1 - node.newPercent / 100)} width={s - 8} height={s * 0.36 * ((node.newPercent - node.currentPercent) / 100)} fill={theme.colors.gold} filter={memGlow} />
          )}
        </g>
      );
    default:
      return (
        <g transform={`translate(${-half}, ${-half})`}>
          <polygon
            points={`${half},0 ${s},${s * 0.25} ${s},${s * 0.75} ${half},${s} 0,${s * 0.75} 0,${s * 0.25}`}
            fill={node.isUsed ? theme.colors.goldBg : bodyFill}
            stroke={stroke}
            strokeWidth={1.5}
          />
        </g>
      );
  }
}
