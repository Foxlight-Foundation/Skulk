import styled, { css, keyframes, useTheme } from 'styled-components';
import { FiCheck, FiChevronDown, FiDownload, FiStar } from 'react-icons/fi';
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
import { Button } from '../common/Button';
import { InfoTooltip } from '../common/InfoTooltip';
import { buildTagColors, CapabilityTagBadge } from '../common/capabilityTags';
import { FamilyAvatar } from './FamilyAvatar';
import { deriveFormatLabel, QuantBadge } from './quantBadge';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

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

function timeAgo(ts: number, t: SkulkTranslate): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return t('storeRegistry.time.justNow', 'just now');
  if (mins < 60) return t('storeRegistry.time.minutesAgo', '{count}m ago', { count: mins });
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return t('storeRegistry.time.hoursAgo', '{count}h ago', { count: hrs });
  return t('storeRegistry.time.daysAgo', '{count}d ago', { count: Math.floor(hrs / 24) });
}

/** Capabilities surfaced as chips on the row. `text` is implied and omitted. */
const CHIP_CAPABILITIES = new Set([
  'thinking', 'vision', 'code', 'image_gen', 'image_edit', 'embedding', 'tts', 'stt',
]);

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

const GroupContainer = styled.div`
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};

  &:first-child {
    border-top: none;
  }
`;

const Row = styled.div<{ $disabled: boolean; $highlighted: boolean; $expandable: boolean }>`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  cursor: ${({ $expandable }) => ($expandable ? 'pointer' : 'default')};
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

const Identity = styled.div`
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
`;

const Name = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const MetaLine = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  overflow: hidden;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
`;

const MetaText = styled.span`
  overflow: hidden;
  text-overflow: ellipsis;
`;

const InStoreChip = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.healthy};
  background: ${({ theme }) => theme.colors.accentBg};
  border: 1px solid ${({ theme }) => theme.colors.accentBg};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 2px 8px;
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
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.textMuted)};
  transition: color 0.15s, background 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.gold};
    background: ${({ theme }) => theme.colors.goldBg};
  }

  svg {
    ${({ $active }) => $active && css`fill: currentColor;`}
  }
`;

const Chevron = styled.button<{ $open: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.textMuted};
  flex-shrink: 0;
  transition: color 0.15s, background 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
    background: ${({ theme }) => theme.colors.surfaceHover};
  }

  svg {
    transition: transform 0.15s;
    ${({ $open }) => $open && css`transform: rotate(180deg);`}
  }
`;

const VariantPanel = styled.div`
  margin: 0 14px 10px 62px;
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  overflow: hidden;
`;

const VariantRow = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 12px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};

  & + & {
    border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
  }
`;

const ActionArea = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
`;

/* ---------- component ---------- */

function ModelGroupInfo({ group, title }: { group: ModelGroup; title: string }) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const v = group.smallestVariant;
  const resolved = v.resolved_capabilities;
  return (
    <div style={{ minWidth: 220 }}>
      <div style={{ color: theme.colors.gold, fontWeight: 600, marginBottom: 6 }}>
        {title}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        {v.family && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.family', 'Family')}</span>
            <span>{v.family}</span>
          </>
        )}
        <span style={{ color: theme.colors.textMuted }}>{t('modelPickerGroup.variants', 'Variants')}</span>
        <span>{group.variants.length}</span>
        <span style={{ color: theme.colors.textMuted }}>{t('modelPickerGroup.smallest', 'Smallest')}</span>
        <span>{v.storage_size_megabytes ? formatBytes(v.storage_size_megabytes * 1024 * 1024) : '—'}</span>
        <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.tensorParallel', 'Tensor parallel')}</span>
        <span style={{ color: v.supports_tensor ? theme.colors.healthy : theme.colors.textSecondary }}>
          {v.supports_tensor ? t('common.yes', 'Yes') : t('common.no', 'No')}
        </span>
        {group.capabilities.length > 0 && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.capabilities', 'Capabilities')}</span>
            <span>{group.capabilities.join(', ')}</span>
          </>
        )}
        {resolved && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.thinkingToggle', 'Thinking toggle')}</span>
            <span>{resolved.supports_thinking_toggle ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.toolCalling', 'Tool calling')}</span>
            <span>{resolved.supports_tool_calling ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.imageInput', 'Image input')}</span>
            <span>{resolved.supports_image_input ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.audioInput', 'Audio input')}</span>
            <span>{resolved.supports_audio_input ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.speechSynthesis', 'Speech synthesis')}</span>
            <span>{resolved.supports_speech_synthesis ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.transcription', 'Transcription')}</span>
            <span>{resolved.supports_transcription ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.realtimeAudio', 'Realtime audio')}</span>
            <span>{resolved.supports_realtime_audio ? t('common.supported', 'Supported') : t('common.notSupported', 'Not supported')}</span>
          </>
        )}
      </div>
      {group.hasMultipleVariants && (
        <div style={{ marginTop: 8, borderTop: `1px solid ${theme.colors.borderLight}`, paddingTop: 6 }}>
          <div style={{ color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            {t('modelPickerGroup.quantizations', 'Quantizations')}
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
  canModelFit,
  getModelFitStatus,
  onToggleExpand,
  onSelectModel,
  onToggleFavorite,
  downloadStatusMap,
  launchedAt,
  instanceStatuses,
}: ModelPickerGroupProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const TAG_COLORS = buildTagColors(theme);
  const { variants, hasMultipleVariants } = group;
  const singleVariant = !hasMultipleVariants ? variants[0] : null;

  // The group key is the card's human-readable base model ("GPT-OSS 20B")
  // when one exists; the fallback group.name is a raw repo tail. Prefer the
  // readable one as the row title.
  const title = group.smallestVariant.base_model ?? group.name;

  const anyFits = variants.some((v) => canModelFit(v.id));
  const anyHasInstance = variants.some((v) => instanceStatuses?.[v.id]);
  const disabled = !anyFits && !anyHasInstance;

  const groupDownload = variants.find((v) => downloadStatusMap?.get(v.id)?.available);
  const instanceStatus = bestInstanceStatus(variants, instanceStatuses);

  const chips = group.capabilities.filter((c) => CHIP_CAPABILITIES.has(c));
  const contextLength = group.smallestVariant.context_length;

  // Artifact format (GGUF, MLX) is placement-relevant truth. Show it on the
  // group row when every variant shares one format; otherwise it appears per
  // variant in the expanded size panel.
  const variantFormats = variants.map((v) => deriveFormatLabel(v.id));
  const uniformFormat = variantFormats.every((f) => f === variantFormats[0])
    ? variantFormats[0]
    : null;

  // Size summary for the meta line
  const smallest = sizeText(variants[0].storage_size_megabytes);
  const largest = sizeText(variants[variants.length - 1].storage_size_megabytes);
  const sizeSummary = hasMultipleVariants
    ? (smallest && largest ? `${smallest} – ${largest}` : '')
    : smallest;

  const inStoreChip = (
    <InStoreChip>
      <FiCheck size={12} strokeWidth={2.5} />
      {t('modelPickerGroup.inStore', 'In store')}
    </InStoreChip>
  );

  return (
    <GroupContainer>
      <Row
        $disabled={disabled}
        $highlighted={isHighlighted}
        $expandable={hasMultipleVariants}
        onClick={hasMultipleVariants ? onToggleExpand : undefined}
        role={hasMultipleVariants ? 'button' : undefined}
        tabIndex={hasMultipleVariants && !disabled ? 0 : undefined}
        aria-expanded={hasMultipleVariants ? isExpanded : undefined}
        aria-label={hasMultipleVariants
          ? t('modelPickerGroup.expandGroup', 'Expand {groupName}', { groupName: title })
          : undefined}
        onKeyDown={(event) => {
          if (hasMultipleVariants && !disabled && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            onToggleExpand();
          }
        }}
      >
        <FamilyAvatar name={group.family || title} />

        {/* Identity: title + meta line */}
        <Identity>
          <Name title={singleVariant?.id ?? group.name}>{title}</Name>
          <MetaLine>
            {chips.map((c) => {
              const colors = TAG_COLORS[c];
              if (!colors) return null;
              return (
                <CapabilityTagBadge key={c} $color={colors.color} $bg={colors.bg} $border={colors.border}>
                  {c.replace('_', ' ')}
                </CapabilityTagBadge>
              );
            })}
            {uniformFormat && <QuantBadge>{uniformFormat}</QuantBadge>}
            {singleVariant?.quantization && (
              <QuantBadge>{singleVariant.quantization}</QuantBadge>
            )}
            <MetaText>
              {[
                hasMultipleVariants
                  ? t('modelPickerGroup.sizeCount', '{count} sizes', { count: variants.length })
                  : null,
                sizeSummary,
                contextLength
                  ? t('modelPickerGroup.contextMeta', '{count}k context', {
                      count: Math.round(contextLength / 1024),
                    })
                  : null,
                launchedAt != null ? timeAgo(launchedAt, t) : null,
              ].filter(Boolean).join(' · ')}
            </MetaText>
          </MetaLine>
        </Identity>

        {/* Instance status dot */}
        {instanceStatus && (
          <StatusDot
            $class={instanceStatus.statusClass}
            title={instanceStatus.statusClass}
          />
        )}

        {/* Favorite */}
        <FavStar
          $active={isFavorite}
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavorite(group.id);
          }}
          title={isFavorite
            ? t('modelPickerGroup.removeFromFavorites', 'Remove from favorites')
            : t('modelPickerGroup.addToFavorites', 'Add to favorites')}
        >
          <FiStar size={15} />
        </FavStar>

        {/* Info */}
        <span onClick={(e) => e.stopPropagation()} style={{ display: 'flex' }}>
          <InfoTooltip
            filled
            size={16}
            placement="right"
            delay={100}
            content={<ModelGroupInfo group={group} title={title} />}
          />
        </span>

        {/* Primary action */}
        <ActionArea onClick={(e) => e.stopPropagation()}>
          {hasMultipleVariants ? (
            <>
              {groupDownload && inStoreChip}
              <Chevron
                type="button"
                $open={isExpanded}
                onClick={onToggleExpand}
                aria-expanded={isExpanded}
                aria-label={t('modelPickerGroup.expandGroup', 'Expand {groupName}', { groupName: title })}
                tabIndex={-1}
              >
                <FiChevronDown size={16} />
              </Chevron>
            </>
          ) : groupDownload ? (
            inStoreChip
          ) : (
            <Button
              variant="primary"
              size="sm"
              onClick={() => singleVariant && onSelectModel(singleVariant.id)}
              aria-label={t('modelPickerGroup.selectModel', 'Download {modelId}', {
                modelId: singleVariant?.id ?? title,
              })}
            >
              <FiDownload size={13} />
              {t('modelPickerGroup.download', 'Download')}
            </Button>
          )}
        </ActionArea>
      </Row>

      {/* Expanded variants */}
      {isExpanded && hasMultipleVariants && (
        <VariantPanel>
          {variants.map((v) => {
            const vFit = getModelFitStatus(v.id);
            const vDownload = downloadStatusMap?.get(v.id);
            const vInstance = instanceStatuses?.[v.id];

            return (
              <VariantRow key={v.id}>
                {uniformFormat === null && deriveFormatLabel(v.id) && (
                  <QuantBadge>{deriveFormatLabel(v.id)}</QuantBadge>
                )}
                <QuantBadge>{v.quantization ?? '—'}</QuantBadge>
                <span style={{ color: fitColor(vFit, theme), fontWeight: 500, flex: 1 }}>
                  {sizeText(v.storage_size_megabytes)}
                </span>
                {vInstance && <StatusDot $class={vInstance.statusClass} title={vInstance.statusClass} />}
                {vDownload?.available ? (
                  inStoreChip
                ) : (
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => onSelectModel(v.id)}
                    aria-label={t('modelPickerGroup.selectModel', 'Download {modelId}', {
                      modelId: v.id,
                    })}
                  >
                    <FiDownload size={13} />
                    {t('modelPickerGroup.download', 'Download')}
                  </Button>
                )}
              </VariantRow>
            );
          })}
        </VariantPanel>
      )}
    </GroupContainer>
  );
}
