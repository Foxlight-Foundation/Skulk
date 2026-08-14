import { useState } from 'react';
import styled, { useTheme } from 'styled-components';
import { FiCheck, FiChevronDown, FiDownload, FiHeart, FiLock } from 'react-icons/fi';
import type { HuggingFaceModel } from '../../types/models';
import { Button } from '../common/Button';
import { InfoTooltip } from '../common/InfoTooltip';
import { FamilyAvatar } from './FamilyAvatar';
import { familyTokensFromName } from './familyTokens';
import { HfModelDossier } from './HfModelDossier';
import { HuggingFaceLink } from './HuggingFaceLink';
import { deriveFormatLabel, deriveQuantLabel, QuantBadge } from './quantBadge';
import { BurstChip } from './BurstChip';
import type { BurstInfo } from './burst';
import type { Theme } from '../../theme';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { buildTagColors, CapabilityTagBadge } from '../common/capabilityTags';
import { formatBytes } from '../../utils/format';

export interface HuggingFaceResultItemProps {
  model: HuggingFaceModel;
  isAdded: boolean;
  isAdding: boolean;
  isInStore?: boolean;
  onAdd: () => void;
  onSelect: () => void;
  downloadedOnNodes?: string[];
  /** Burst verdict for this result; null means locally placeable. */
  burst?: BurstInfo | null;
  /** Fleet total memory for the burst tooltip. */
  fleetMemoryBytes?: number;
  /** Download one specific quant (repo-relative first shard) of this repo. */
  onSelectQuant?: (ggufFile: string) => void;
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n < 1e10 ? 1 : 0)}B`;
  return `${Math.round(n / 1e6)}M`;
}

/** Derivation-kind labels for the top-layer classification micro-chip. */
const RELATION_CHIP: Readonly<Record<string, string>> = {
  quantized: 'quant',
  finetune: 'tune',
  merge: 'merge',
  adapter: 'adapter',
};

/** Hugging Face pipeline tags mapped onto the catalog's capability chips. */
const PIPELINE_CAPABILITY: Readonly<Record<string, string>> = {
  'image-text-to-text': 'vision',
  'automatic-speech-recognition': 'stt',
  'text-to-speech': 'tts',
  'text-to-audio': 'tts',
  'feature-extraction': 'embedding',
  'sentence-similarity': 'embedding',
  'text-to-image': 'image_gen',
  'image-to-image': 'image_edit',
};

const Row = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
  transition: background 0.15s;

  &:first-child {
    border-top: none;
  }

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
  }
`;

const Info = styled.div`
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
`;

const ModelName = styled.div`
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
  gap: 10px;
  min-width: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
`;

const MetaItem = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 3px;
  flex-shrink: 0;
`;

const Author = styled.span`
  overflow: hidden;
  text-overflow: ellipsis;
`;

const MatchedFileChip = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  color: ${({ theme }) => theme.colors.textSecondary};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 0 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
`;

const GatedChip = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 3px;
  flex-shrink: 0;
  font-size: 10px;
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textSecondary};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 0 6px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
`;

interface QuantOption {
  gguf_file: string;
  label: string;
  total_bytes: number;
  shard_count: number;
}

const QuantPanel = styled.div`
  margin: 0 14px 10px 62px;
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  overflow: hidden;
`;

const QuantRow = styled.div`
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
    ${({ $open }) => ($open ? 'transform: rotate(180deg);' : '')}
  }
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

export function HuggingFaceResultItem({
  model,
  isAdded,
  isAdding,
  isInStore = false,
  onAdd,
  onSelect,
  burst = null,
  fleetMemoryBytes,
  onSelectQuant,
}: HuggingFaceResultItemProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const shortName = model.id.includes('/')
    ? model.id.slice(model.id.indexOf('/') + 1)
    : model.id;
  // The search API often returns an empty author; the repo org is the author.
  const author = model.author || (model.id.includes('/') ? model.id.split('/')[0] : '');
  const quantLabel = deriveQuantLabel(model.id, model.matched_file, model.tags);
  const formatLabel = deriveFormatLabel(model.id, model.matched_file, model.tags);
  const TAG_COLORS = buildTagColors(theme);
  const capability = model.pipeline_tag ? PIPELINE_CAPABILITY[model.pipeline_tag] : undefined;
  const capabilityColors = capability ? TAG_COLORS[capability] : undefined;
  // A pipeline outside the chat/audio/vision families Skulk serves is worth
  // naming so nobody downloads a video-diffusion repo into an LLM cluster.
  const unfamiliarPipeline = model.pipeline_tag
    && model.pipeline_tag !== 'text-generation'
    && !capability
    ? model.pipeline_tag
    : null;
  const [quantsOpen, setQuantsOpen] = useState(false);
  const [quantOptions, setQuantOptions] = useState<QuantOption[] | null>(null);

  const toggleQuants = async () => {
    const next = !quantsOpen;
    setQuantsOpen(next);
    if (next && quantOptions === null) {
      try {
        const res = await fetch(`/models/gguf-quants?model_id=${encodeURIComponent(model.id)}`);
        if (!res.ok) { setQuantOptions([]); return; }
        const data = (await res.json()) as { options?: QuantOption[] };
        setQuantOptions(data.options ?? []);
      } catch {
        setQuantOptions([]);
      }
    }
  };

  const relationChip = model.base_model_relation
    ? RELATION_CHIP[model.base_model_relation]
    : undefined;
  // Mark ladder: the author when it is itself a brand, then the base
  // model's org (lineage), then family tokens from either repo name.
  const baseOrg = model.base_model_repo?.includes('/')
    ? model.base_model_repo.split('/', 1)[0]
    : undefined;
  const markCandidates = [
    ...(author ? [author] : []),
    ...(baseOrg ? [baseOrg] : []),
    ...familyTokensFromName(model.id),
    ...(model.base_model_repo ? familyTokensFromName(model.base_model_repo) : []),
  ];


  const tooltipContent = <HfModelDossier model={model} author={author} />;

  return (
    <div>
    <Row>
      <FamilyAvatar name={author || shortName} markCandidates={markCandidates} />

      <Info>
        <ModelName title={model.id}>{shortName}</ModelName>
        <MetaLine>
          {capabilityColors && capability && (
            <CapabilityTagBadge
              $color={capabilityColors.color}
              $bg={capabilityColors.bg}
              $border={capabilityColors.border}
            >
              {capability.replace('_', ' ')}
            </CapabilityTagBadge>
          )}
          {unfamiliarPipeline && <QuantBadge>{unfamiliarPipeline}</QuantBadge>}
          {formatLabel && <QuantBadge>{formatLabel}</QuantBadge>}
          {quantLabel && <QuantBadge>{quantLabel}</QuantBadge>}
          {relationChip && (
            <QuantBadge title={model.base_model_repo ?? undefined}>{relationChip}</QuantBadge>
          )}
          {model.gated && (
            <InfoTooltip
              placement="bottom"
              delay={100}
              content={
                <div style={{ maxWidth: 240 }}>
                  {t(
                    'huggingFaceResult.gatedTooltip',
                    'Downloading requires accepting the license on Hugging Face and setting an HF token on the store host.',
                  )}
                </div>
              }
            >
              <GatedChip><FiLock size={9} /> {t('huggingFaceResult.gated', 'Gated')}</GatedChip>
            </InfoTooltip>
          )}
          <Author title={author}>{author}</Author>
          <MetaItem title={t('huggingFaceResult.downloads', 'Downloads')}>
            <FiDownload size={11} /> {formatCount(model.downloads)}
          </MetaItem>
          <MetaItem title={t('huggingFaceResult.likes', 'Likes')}>
            <FiHeart size={11} /> {formatCount(model.likes)}
          </MetaItem>
          {(model.param_count || model.total_file_size || model.context_length) && (
            <MetaItem>
              {[
                model.param_count ? formatParams(model.param_count) : null,
                model.total_file_size ? formatBytes(model.total_file_size) : null,
                model.context_length
                  ? t('modelPickerGroup.contextMeta', '{count}k context', {
                      count: Math.round(model.context_length / 1024),
                    })
                  : null,
              ].filter(Boolean).join(' · ')}
            </MetaItem>
          )}
          {model.matched_file && (
            <MatchedFileChip title={model.matched_file}>
              {model.matched_file.split('/').pop()}
            </MatchedFileChip>
          )}
        </MetaLine>
      </Info>

      {burst && <BurstChip info={burst} fleetMemoryBytes={fleetMemoryBytes} />}

      <HuggingFaceLink repoId={model.id} />

      {/* Info tooltip */}
      <InfoTooltip filled size={16} placement="left" delay={100} content={tooltipContent} />

      {/* Action */}
      {isInStore ? (
        <InStoreChip>
          <FiCheck size={12} strokeWidth={2.5} /> {t('huggingFaceResult.inStore', 'In Store')}
        </InStoreChip>
      ) : isAdded ? (
        <Button
          variant="primary"
          size="sm"
          onClick={onSelect}
          aria-label={t('huggingFaceResult.downloadModel', 'Download {modelId}', {
            modelId: model.id,
          })}
        >
          <FiDownload size={13} /> {t('huggingFaceResult.download', 'Download')}
        </Button>
      ) : (
        <Button
          variant="outline"
          size="sm"
          onClick={onAdd}
          disabled={isAdding}
          aria-label={t(
            'huggingFaceResult.addAndDownloadModel',
            'Add and download {modelId}',
            { modelId: model.id },
          )}
        >
          {isAdding ? '…' : <><FiDownload size={13} /> {t('huggingFaceResult.addAndDownload', 'Add & Download')}</>}
        </Button>
      )}

      {/* Quant chooser for GGUF repos: pick the exact artifact to download. */}
      {formatLabel === 'GGUF' && onSelectQuant && (
        <Chevron
          type="button"
          $open={quantsOpen}
          onClick={toggleQuants}
          aria-expanded={quantsOpen}
          aria-label={t('huggingFaceResult.showQuants', 'Show quantizations for {modelId}', { modelId: model.id })}
        >
          <FiChevronDown size={16} />
        </Chevron>
      )}
    </Row>

    {quantsOpen && (
      <QuantPanel>
        {quantOptions === null && (
          <QuantRow>{t('huggingFaceResult.loadingQuants', 'Listing quantizations…')}</QuantRow>
        )}
        {quantOptions !== null && quantOptions.length === 0 && (
          <QuantRow>{t('huggingFaceResult.noQuants', 'No downloadable quantizations found')}</QuantRow>
        )}
        {(quantOptions ?? []).map((option) => (
          <QuantRow key={option.gguf_file}>
            <QuantBadge>{option.label}</QuantBadge>
            <span style={{ fontWeight: 500, flex: 1 }}>
              {formatBytes(option.total_bytes)}
            </span>
            {fleetMemoryBytes !== undefined && option.total_bytes > fleetMemoryBytes && (
              <BurstChip
                info={{ reason: 'size', neededBytes: option.total_bytes }}
                fleetMemoryBytes={fleetMemoryBytes}
              />
            )}
            <Button
              variant="primary"
              size="sm"
              onClick={() => onSelectQuant?.(option.gguf_file)}
              aria-label={t('huggingFaceResult.downloadQuant', 'Download {label} of {modelId}', {
                label: option.label,
                modelId: model.id,
              })}
            >
              <FiDownload size={13} /> {t('huggingFaceResult.download', 'Download')}
            </Button>
          </QuantRow>
        ))}
      </QuantPanel>
    )}
    </div>
  );
}
