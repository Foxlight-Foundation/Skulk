import styled, { css, keyframes, useTheme } from 'styled-components';
import { FiDownload, FiCheck } from 'react-icons/fi';
import type { Theme } from '../../theme';
import type {
  ModelGroup,
  ModelInfo,
  ModelFitStatus,
  DownloadAvailability,
  InstanceStatus,
  PickerMode,
} from '../../types/models';
import { formatBytes } from '../../utils/format';
import { InfoTooltip } from '../common/InfoTooltip';

export interface ModelPickerGroupProps {
  group: ModelGroup;
  isExpanded: boolean;
  isFavorite: boolean;
  isHighlighted?: boolean;
  selectedModelId: string | null;
  canModelFit: (id: string) => boolean;
  getModelFitStatus: (id: string) => ModelFitStatus;
  onToggleExpand: () => void;
  onSelectModel: (modelId: string) => void;
  onToggleFavorite: (groupId: string) => void;
  onShowInfo?: (group: ModelGroup) => void;
  downloadStatusMap?: Map<string, DownloadAvailability>;
  launchedAt?: number;
  instanceStatuses?: Record<string, InstanceStatus>;
  mode?: PickerMode;
}

/* ---------- helpers ---------- */

function sizeText(mb?: number): string {
  if (!mb) return '';
  return formatBytes(mb * 1024 * 1024);
}

function fitColor(status: ModelFitStatus, theme: Theme): string {
  if (status === 'fits_now') return theme.colors.text;
  if (status === 'fits_cluster_capacity') return theme.colors.warning;
  return theme.colors.error;
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

const CAPABILITY_ICONS: Record<string, string> = {
  thinking: '🧠',
  code: '💻',
  vision: '👁',
  image_gen: '🎨',
  image_edit: '✏️',
};

function bestInstanceStatus(
  variants: ModelInfo[],
  statuses?: Record<string, InstanceStatus>,
): InstanceStatus | null {
  if (!statuses) return null;
  const order = ['ready', 'loading', 'downloading'];
  let best: InstanceStatus | null = null;
  let bestRank = Infinity;
  for (const v of variants) {
    const s = statuses[v.id];
    if (!s) continue;
    const rank = order.indexOf(s.statusClass);
    if (rank !== -1 && rank < bestRank) {
      bestRank = rank;
      best = s;
    }
  }
  return best;
}

/* ---------- styles ---------- */

const glowAnim = keyframes`
  0%, 100% { box-shadow: 0 0 4px rgba(34,197,94,0.4); }
  50%      { box-shadow: 0 0 12px rgba(34,197,94,0.7); }
`;

const Row = styled.div<{ $disabled: boolean; $highlighted: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  cursor: pointer;
  transition: background 0.15s;
  user-select: none;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
  }

  ${({ $disabled }) =>
    $disabled &&
    css`
      opacity: 0.5;
      pointer-events: none;
    `}

  ${({ $highlighted }) =>
    $highlighted &&
    css`
      animation: ${glowAnim} 1.5s ease-in-out 3;
    `}
`;

const Chevron = styled.span<{ $open: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  transition: transform 0.15s;
  width: 14px;
  text-align: center;
  ${({ $open }) => $open && css`transform: rotate(90deg);`}
`;

const Name = styled.span`
  flex: 1;
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const Caps = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  margin-left: 4px;
`;

const Badge = styled.span<{ $color?: string }>`
  font-size: ${({ theme }) => theme.fontSizes.label};
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.radii.sm};
  background: ${({ theme }) => theme.colors.surfaceHover};
  color: ${({ $color, theme }) => $color ?? theme.colors.textSecondary};
`;

const StatusDot = styled.span<{ $class: string }>`
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  ${({ $class }) => {
    if ($class === 'ready') return css`background: ${({ theme }) => theme.colors.accent};`;
    if ($class === 'loading') return css`background: ${({ theme }) => theme.colors.warning};`;
    if ($class === 'downloading') return css`background: ${({ theme }) => theme.colors.info};`;
    return css`background: ${({ theme }) => theme.colors.textMuted};`;
  }}
`;

const FavStar = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ $active }) => ($active ? '#fbbf24' : '#555')};
  transition: color 0.15s;
  &:hover {
    color: #fbbf24;
  }
`;


const VariantList = styled.div`
  padding: 4px 0 4px 32px;
  display: flex;
  flex-direction: column;
  gap: 2px;
`;

const VariantRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: ${({ theme }) => theme.radii.sm};
  cursor: pointer;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  transition: background 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
  }
`;

const InfoIconWrapper = styled.span`
  transform: translateY(2px);
`;

/* ---------- component ---------- */

function ModelGroupInfo({ group }: { group: ModelGroup }) {
  const theme = useTheme() as Theme;
  const v = group.smallestVariant;
  const resolved = v.resolved_capabilities;
  return (
    <div style={{ minWidth: 220 }}>
      <div style={{ color: theme.colors.gold, fontWeight: 600, marginBottom: 6 }}>
        {group.name}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        {v.family && (
          <>
            <span style={{ color: theme.colors.textMuted }}>Family</span>
            <span>{v.family}</span>
          </>
        )}
        <span style={{ color: theme.colors.textMuted }}>Variants</span>
        <span>{group.variants.length}</span>
        <span style={{ color: theme.colors.textMuted }}>Smallest</span>
        <span>{v.storage_size_megabytes ? formatBytes(v.storage_size_megabytes * 1024 * 1024) : '—'}</span>
        <span style={{ color: theme.colors.textMuted }}>Tensor parallel</span>
        <span style={{ color: v.supports_tensor ? theme.colors.healthy : theme.colors.textSecondary }}>
          {v.supports_tensor ? 'Yes' : 'No'}
        </span>
        {group.capabilities.length > 0 && (
          <>
            <span style={{ color: theme.colors.textMuted }}>Capabilities</span>
            <span>{group.capabilities.join(', ')}</span>
          </>
        )}
        {resolved && (
          <>
            <span style={{ color: theme.colors.textMuted }}>Thinking toggle</span>
            <span>{resolved.supports_thinking_toggle ? 'Supported' : 'Not supported'}</span>
            <span style={{ color: theme.colors.textMuted }}>Tool calling</span>
            <span>{resolved.supports_tool_calling ? 'Supported' : 'Not supported'}</span>
            <span style={{ color: theme.colors.textMuted }}>Image input</span>
            <span>{resolved.supports_image_input ? 'Supported' : 'Not supported'}</span>
            <span style={{ color: theme.colors.textMuted }}>Audio input</span>
            <span>{resolved.supports_audio_input ? 'Supported' : 'Not supported'}</span>
          </>
        )}
      </div>
      {group.hasMultipleVariants && (
        <div style={{ marginTop: 8, borderTop: `1px solid ${theme.colors.borderLight}`, paddingTop: 6 }}>
          <div style={{ color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            Quantizations
          </div>
          {group.variants.map((variant) => (
            <div key={variant.id} style={{ color: theme.colors.textSecondary }}>
              {variant.quantization || '—'} · {variant.storage_size_megabytes ? formatBytes(variant.storage_size_megabytes * 1024 * 1024) : '—'}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ModelPickerGroup({
  group,
  isExpanded,
  isFavorite,
  isHighlighted = false,
  selectedModelId,
  canModelFit,
  getModelFitStatus,
  onToggleExpand,
  onSelectModel,
  onToggleFavorite,
  onShowInfo: _onShowInfo,
  downloadStatusMap,
  launchedAt,
  instanceStatuses,
  mode: _mode = 'launch',
}: ModelPickerGroupProps) {
  const theme = useTheme() as Theme;
  const DownloadIcon = () => <FiDownload size={14} color={theme.colors.accent} />;
  const CheckMark = () => <FiCheck size={16} color={theme.colors.accent} strokeWidth={2.5} />;
  const { variants, hasMultipleVariants } = group;
  const singleVariant = !hasMultipleVariants ? variants[0] : null;

  const anyFits = variants.some((v) => canModelFit(v.id));
  const anyHasInstance = variants.some((v) => instanceStatuses?.[v.id]);
  const disabled = !anyFits && !anyHasInstance;

  const groupFit = variants.reduce<ModelFitStatus>((best, v) => {
    const s = getModelFitStatus(v.id);
    if (s === 'fits_now') return 'fits_now';
    if (s === 'fits_cluster_capacity' && best !== 'fits_now') return 'fits_cluster_capacity';
    return best;
  }, 'too_large');

  const groupDownload = variants.find((v) => downloadStatusMap?.get(v.id)?.available);
  const instanceStatus = bestInstanceStatus(variants, instanceStatuses);

  const isSelected = singleVariant ? selectedModelId === singleVariant.id : false;

  const caps = group.capabilities.filter((c) => c in CAPABILITY_ICONS);

  const handleRowClick = () => {
    if (singleVariant) {
      onSelectModel(singleVariant.id);
    } else {
      onToggleExpand();
    }
  };

  // Size display
  let sizeDisplay: React.ReactNode;
  if (hasMultipleVariants) {
    const smallest = sizeText(variants[0].storage_size_megabytes);
    const largest = sizeText(variants[variants.length - 1].storage_size_megabytes);
    sizeDisplay = (
      <Badge>
        {variants.length} variants{smallest && largest ? ` (${smallest}–${largest})` : ''}
      </Badge>
    );
  } else if (singleVariant?.storage_size_megabytes) {
    sizeDisplay = (
      <Badge $color={fitColor(groupFit, theme)}>
        {sizeText(singleVariant.storage_size_megabytes)}
      </Badge>
    );
  }

  return (
    <div>
      <Row $disabled={disabled} $highlighted={isHighlighted} onClick={handleRowClick}>
        {/* Chevron */}
        {hasMultipleVariants ? (
          <Chevron $open={isExpanded}>▶</Chevron>
        ) : (
          <span style={{ width: 14 }} />
        )}

        {/* Name + caps */}
        <Name>
          {group.name}
          {caps.length > 0 && <Caps>{caps.map((c) => CAPABILITY_ICONS[c]).join('')}</Caps>}
        </Name>

        {/* Size */}
        {sizeDisplay}

        {/* Time ago */}
        {launchedAt != null && (
          <Badge>{timeAgo(launchedAt)}</Badge>
        )}

        {/* Download / in-store indicator */}
        {groupDownload ? <CheckMark /> : <DownloadIcon />}

        {/* Instance status dot */}
        {instanceStatus && <StatusDot $class={instanceStatus.statusClass} />}

        {/* Selected check */}
        {isSelected && !groupDownload && <CheckMark />}

        {/* Favorite */}
        <FavStar
          $active={isFavorite}
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavorite(group.id);
          }}
          title={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
        >
          {isFavorite ? '★' : '☆'}
        </FavStar>

        {/* Info */}
        <InfoIconWrapper onClick={(e) => e.stopPropagation()}>
          <InfoTooltip
            filled
            size={16}
            placement="right"
            delay={100}
            content={<ModelGroupInfo group={group} />}
          />
        </InfoIconWrapper>
      </Row>

      {/* Expanded variants */}
      {isExpanded && hasMultipleVariants && (
        <VariantList>
          {variants.map((v) => {
            const vFit = getModelFitStatus(v.id);
            const vDownload = downloadStatusMap?.get(v.id);
            const vInstance = instanceStatuses?.[v.id];
            const vSelected = selectedModelId === v.id;

            return (
              <VariantRow key={v.id} onClick={() => onSelectModel(v.id)}>
                {v.quantization && <Badge>{v.quantization}</Badge>}
                <span style={{ color: fitColor(vFit, theme), flex: 1 }}>
                  {sizeText(v.storage_size_megabytes)}
                </span>
                {vDownload?.available ? <CheckMark /> : <DownloadIcon />}
                {vInstance && <StatusDot $class={vInstance.statusClass} />}
              </VariantRow>
            );
          })}
        </VariantList>
      )}
    </div>
  );
}
