import type { StoreDownloadProgress } from '../layout/StoreRegistryTable';
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
import { HuggingFaceLink } from './HuggingFaceLink';
import { deriveFormatLabel, QuantBadge } from './quantBadge';
import { BurstChip } from './BurstChip';
import type { BurstInfo } from './burst';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

/** Grouped model metadata, observed availability, and existing variant actions. */
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
  /** Observed transfers from the existing store poll, never synthetic progress. */
  activeDownloads?: StoreDownloadProgress[];
  /** Enter the existing placement workflow for a downloaded variant. */
  onLaunch?: (modelId: string) => void;
  /** Cancel a store transfer through the existing download controller. */
  onCancelDownload?: (modelId: string) => void;
  downloadStatusMap?: Map<string, DownloadAvailability>;
  launchedAt?: number;
  instanceStatuses?: Record<string, InstanceStatus>;
  mode?: PickerMode;
  /** Burst verdict per variant id; null means locally placeable. */
  getBurstInfo?: (variantId: string) => BurstInfo | null;
  /** Fleet total memory for burst tooltips. */
  fleetMemoryBytes?: number;
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

const GroupContainer = styled.div<{ $downloading: boolean }>`
  flex-shrink: 0;
  margin: 10px 16px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 12px;
  background: ${({ theme }) => theme.colors.surface}; overflow: hidden;
  ${({ $downloading, theme }) => $downloading && css`background: ${theme.colors.liveBg}; border-color: ${theme.colors.borderLive};`}
  @media (max-width: 480px) { margin: 8px; }
`;

const Row = styled.div<{ $highlighted: boolean; $expandable: boolean }>`
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 12px 14px;
  @media (max-width: 600px) { padding: 12px; gap: 8px; flex-wrap: wrap; }
  cursor: ${({ $expandable }) => ($expandable ? 'pointer' : 'default')};
  transition: background 0.15s;
  user-select: none;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
  }

  ${({ $highlighted }) =>
    $highlighted &&
    css`
      animation: ${glowAnim} 1.5s ease-in-out 3;
    `}
`;

const Identity = styled.div`
  flex: 1;
  @media (max-width: 600px) { flex: 1 1 calc(100% - 64px); }
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
`;

const Name = styled.span`
  button { all: unset; cursor: pointer; font: inherit; color: inherit; }
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  @media (max-width: 600px) { white-space: normal; overflow-wrap: anywhere; }
`;

const MetaLine = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  overflow: hidden;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
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

const RegistryChip = styled.span`
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.healthy};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 6px;
`;

const ProvenanceChip = styled(RegistryChip)`
  color: ${({ theme }) => theme.colors.textSecondary};
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
  color: ${({ $active, theme }) => ($active ? theme.colors.live : theme.colors.textMuted)};
  transition: color 0.15s, background 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.live};
    background: ${({ theme }) => theme.colors.liveBg};
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
  color: ${({ theme }) => theme.colors.textSecondary};
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
  margin: 0 14px 10px 72px;

  @media (max-width: 640px) {
    margin-left: 14px;
  }
  min-width: 0;
`;

const VariantRow = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) 68px minmax(0, 1fr) 160px;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  > div { min-width: 0; overflow-wrap: anywhere; }
  @media (max-width: 900px) {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    > div:first-child, > div:last-child { grid-column: 1 / -1; }
  }
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};

  & + & {
    border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
  }
`;
const VariantHeader = styled(VariantRow)`
  font: 600 10px ${({ theme }) => theme.fonts.mono}; text-transform: uppercase; letter-spacing: .08em;
  @media (max-width: 900px) { display: none; }
`;
const VariantTags = styled.div`display: flex; flex-wrap: wrap; gap: 5px; align-items: center;`;
const VariantActions = styled(VariantTags)`justify-content: flex-end;`;

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
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: theme.colors.gold, fontWeight: 600, marginBottom: 6 }}>
        {title}
        <HuggingFaceLink repoId={v.hugging_face_id ?? v.id} size={12} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        {v.family && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.family', 'Family')}</span>
            <span>{v.family}</span>
          </>
        )}
        <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.catalogSource', 'Catalog source')}</span>
        <span>{v.catalog_source === 'registry' ? t('modelInfo.signedRegistry', 'Signed registry') : (v.catalog_source ?? 'bundled')}</span>
        {v.registry_card_id && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.registryCard', 'Registry card')}</span>
            <code title={v.registry_card_id}>{`${v.registry_card_id.slice(0, 15)}…`}</code>
          </>
        )}
        {v.registry_provenance && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('modelInfo.provenance', 'Provenance')}</span>
            <span>{v.registry_provenance}</span>
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

/** Render a model group and its expandable, responsive variant details. */
export function ModelPickerGroup({
  group,
  isExpanded,
  isFavorite,
  isHighlighted = false,
  getModelFitStatus,
  onToggleExpand,
  onSelectModel,
  onToggleFavorite,
  activeDownloads,
  onLaunch,
  onCancelDownload,
  downloadStatusMap,
  launchedAt,
  instanceStatuses,
  getBurstInfo,
  fleetMemoryBytes,
}: ModelPickerGroupProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const TAG_COLORS = buildTagColors(theme);
  const { variants, hasMultipleVariants } = group;
  const singleVariant = !hasMultipleVariants ? variants[0] : null;

  // The group key is the card's human-readable base model ("GPT-OSS 20B")
  // when one exists; the fallback group.name is a raw repo tail. Prefer the
  // readable one as the row title.
  // Truthy fallback: generated custom cards leave base_model at "" and a
  // nullish check would render blank titles for user-added models.
  const title = group.smallestVariant.base_model || group.name;

  const groupDownload = variants.find((v) => downloadStatusMap?.get(v.id)?.available);
  const instanceStatus = bestInstanceStatus(variants, instanceStatuses);

  const chips = group.capabilities.filter((c) => CHIP_CAPABILITIES.has(c));
  const contextLength = group.smallestVariant.context_length;
  const provenanceValues = new Set(
    variants.map((variant) => variant.registry_provenance),
  );
  const uniformProvenance = provenanceValues.size === 1
    ? variants[0]?.registry_provenance ?? null
    : null;
  const provenanceLabel = (value: NonNullable<ModelInfo['registry_provenance']>) => ({
    foxlight: t('modelInfo.provenanceFoxlight', 'Foxlight'),
    agent: t('modelInfo.provenanceAgent', 'Agent'),
    community: t('modelInfo.provenanceCommunity', 'Community'),
  })[value];

  // Artifact format (GGUF, MLX) is placement-relevant truth. Show it on the
  // group row when every variant shares one format; otherwise it appears per
  // variant in the expanded size panel.
  const variantFormats = variants.map((v) => deriveFormatLabel(v.id));
  const uniformFormat = variantFormats.every((f) => f === variantFormats[0])
    ? variantFormats[0]
    : null;

  // Group burst verdict: burst only when NO variant is locally placeable.
  // The representative info is the least-demanding variant's (the smallest
  // one still needs at least that much).
  const variantBursts = variants.map((v) => getBurstInfo?.(v.id) ?? null);
  const groupBurst = variantBursts.every((b) => b !== null)
    ? variantBursts[0]
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

  const variantAction = (model: ModelInfo) => {
    const transfer = activeDownloads?.find(item => item.modelId === model.id);
    if (transfer && !['complete', 'completed', 'failed', 'cancelled'].includes(transfer.status)) {
      const progress = Math.max(0, Math.min(100, transfer.progress * 100));
      return <div style={{ minWidth: 100, maxWidth: '100%' }}><progress aria-label={t('modelPickerGroup.downloading', 'Downloading {modelId}', { modelId: model.id })} max={100} value={progress} style={{ width: '100%', accentColor: theme.colors.live }} /><span style={{ fontSize: 11 }}>{Math.round(progress)}%</span>{onCancelDownload && <Button size="sm" onClick={() => onCancelDownload(model.id)}>{t('common.cancel', 'Cancel')}</Button>}</div>;
    }
    if (downloadStatusMap?.get(model.id)?.available) return <>{inStoreChip}{onLaunch && <Button variant="solid" size="sm" onClick={() => onLaunch(model.id)}>{t('common.launch', 'Launch')}</Button>}</>;
    // Downloading into the store does not require present placement capacity.
    return <><Button variant="primary" size="sm" onClick={() => onSelectModel(model.id)} aria-label={t('modelPickerGroup.selectModel', 'Download {modelId}', { modelId: model.id })}><FiDownload size={13} />{t('modelPickerGroup.download', 'Download')}</Button>{transfer?.status === 'failed' && <span role="status" style={{ color: theme.colors.error }}>{transfer.error || t('modelPickerGroup.downloadFailed', 'Download failed')}</span>}</>;
  };

  return (
    <GroupContainer $downloading={variants.some(variant => activeDownloads?.some(item => item.modelId === variant.id && !['completed', 'complete', 'failed', 'cancelled'].includes(item.status)))}>
      <Row
        $highlighted={isHighlighted}
        $expandable={hasMultipleVariants}
        onClick={hasMultipleVariants ? onToggleExpand : undefined}
      >
        <FamilyAvatar name={group.family || title} />

        {/* Identity: title + meta line */}
        <Identity>
          <Name title={singleVariant?.id ?? group.name}>{hasMultipleVariants ? <button type="button" onClick={event => { event.stopPropagation(); onToggleExpand(); }} aria-expanded={isExpanded} aria-label={t('modelPickerGroup.expandGroup', 'Expand {groupName}', { groupName: title })}>{title}</button> : title}</Name>
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
            {variants.every((variant) => variant.catalog_source === 'registry') && (
              <RegistryChip>{t('modelInfo.signedRegistry', 'Signed registry')}</RegistryChip>
            )}
            {uniformProvenance && (
              <ProvenanceChip title={t('modelInfo.provenance', 'Provenance')}>
                {provenanceLabel(uniformProvenance)}
              </ProvenanceChip>
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

        {singleVariant && (
          <HuggingFaceLink repoId={singleVariant.hugging_face_id ?? singleVariant.id} />
        )}

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
          {groupBurst && <BurstChip info={groupBurst} fleetMemoryBytes={fleetMemoryBytes} />}
          {hasMultipleVariants ? (
            <>
              {groupDownload && inStoreChip}
              {groupDownload && onLaunch && <Button variant="solid" size="sm" onClick={() => onLaunch(groupDownload.id)}>{t('common.launch', 'Launch')}</Button>}
              <Chevron
                type="button"
                $open={isExpanded}
                onClick={onToggleExpand}
                aria-expanded={isExpanded}
                aria-label={t('modelPickerGroup.expandGroup', 'Expand {groupName}', { groupName: title })}
              >
                <FiChevronDown size={16} />
              </Chevron>
            </>
          ) : singleVariant ? variantAction(singleVariant) : null          }
        </ActionArea>
      </Row>

      {/* Expanded variants */}
      {isExpanded && hasMultipleVariants && (
        <VariantPanel>
          <VariantHeader aria-hidden="true"><div>{t('modelPickerGroup.variant', 'Variant')}</div><div>{t('modelPickerGroup.formatQuant', 'Format · quant')}</div><div>{t('common.size', 'Size')}</div><div>{t('modelPickerGroup.cachedOn', 'Cached on')}</div><div /></VariantHeader>
          {variants.map((v) => {
            const vFit = getModelFitStatus(v.id);
            const vInstance = instanceStatuses?.[v.id];

            return (
              <VariantRow key={v.id}>
                <div style={{ fontFamily: theme.fonts.mono, fontSize: 12.5 }}>{v.id.split('/').pop()} <HuggingFaceLink repoId={v.hugging_face_id ?? v.id} /></div>
                <VariantTags>{deriveFormatLabel(v.id) && (
                  <QuantBadge>{deriveFormatLabel(v.id)}</QuantBadge>
                )}
                <QuantBadge>{v.quantization ?? '—'}</QuantBadge>
                {!uniformProvenance && v.registry_provenance && (
                  <ProvenanceChip title={t('modelInfo.provenance', 'Provenance')}>
                    {provenanceLabel(v.registry_provenance)}
                  </ProvenanceChip>
                )}
                {groupBurst === null && (() => {
                  const b = getBurstInfo?.(v.id) ?? null;
                  return b ? <BurstChip info={b} fleetMemoryBytes={fleetMemoryBytes} /> : null;
                })()}
                </VariantTags>
                <div style={{ color: fitColor(vFit, theme), fontFamily: theme.fonts.mono, fontSize: 12.5 }}>
                  {sizeText(v.storage_size_megabytes)}
                </div>
                <VariantTags style={{ fontSize: 11.5, fontFamily: theme.fonts.mono, color: theme.colors.textMuted }}>{vInstance && <StatusDot $class={vInstance.statusClass} title={vInstance.statusClass} />}{downloadStatusMap?.get(v.id)?.nodeNames.join(' · ') || '—'}</VariantTags>
                <VariantActions>{variantAction(v)}</VariantActions>
              </VariantRow>
            );
          })}
        </VariantPanel>
      )}
    </GroupContainer>
  );
}
