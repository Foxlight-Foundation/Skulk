import styled, { useTheme } from 'styled-components';
import { FiCheck, FiDownload, FiExternalLink, FiHeart, FiLock } from 'react-icons/fi';
import type { HuggingFaceModel } from '../../types/models';
import { Button } from '../common/Button';
import { InfoTooltip } from '../common/InfoTooltip';
import { FamilyAvatar } from './FamilyAvatar';
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

  const hfUrl = `https://huggingface.co/${model.id}`;
  const sizeTags = model.tags.filter((t) =>
    /^\d+[BMK]$|param|safetensor|gguf|mlx|fp16|bf16|\dbit|int[48]/i.test(t),
  );

  const tooltipContent = (
    <div style={{ minWidth: 220 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{ color: theme.colors.gold, fontWeight: 600 }}>{model.id}</span>
        <a
          href={hfUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: theme.colors.textMuted, display: 'flex', transition: 'color 0.15s' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = theme.colors.gold; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = theme.colors.textMuted; }}
          title={t('common.openOnHuggingFace', 'Open on HuggingFace')}
        >
          <FiExternalLink size={14} />
        </a>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        <span style={{ color: theme.colors.textMuted }}>{t('huggingFaceResult.author', 'Author')}</span>
        <span>{author}</span>
        <span style={{ color: theme.colors.textMuted }}>{t('huggingFaceResult.downloads', 'Downloads')}</span>
        <span>{formatCount(model.downloads)}</span>
        <span style={{ color: theme.colors.textMuted }}>{t('huggingFaceResult.likes', 'Likes')}</span>
        <span>{formatCount(model.likes)}</span>
        {model.license && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('huggingFaceResult.license', 'License')}</span>
            <span>{model.license}</span>
          </>
        )}
        {model.last_modified && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('huggingFaceResult.updated', 'Updated')}</span>
            <span>{new Date(model.last_modified).toLocaleDateString()}</span>
          </>
        )}
        {model.matched_file && (
          <>
            <span style={{ color: theme.colors.textMuted }}>{t('huggingFaceResult.matchedFile', 'Matched file')}</span>
            <span style={{ overflowWrap: 'anywhere' }}>{model.matched_file}</span>
          </>
        )}
      </div>
      {sizeTags.length > 0 && (
        <div style={{ marginTop: 8, borderTop: `1px solid ${theme.colors.borderLight}`, paddingTop: 6 }}>
          <div style={{ color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            {t('huggingFaceResult.tags', 'Tags')}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {sizeTags.map((t) => (
              <span key={t} style={{ padding: '1px 6px', borderRadius: 3, background: theme.colors.borderLight, color: theme.colors.textSecondary }}>{t}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <Row>
      <FamilyAvatar name={author || shortName} />

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
    </Row>
  );
}
