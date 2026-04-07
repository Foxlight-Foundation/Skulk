import styled, { useTheme } from 'styled-components';
import { FiCheck, FiDownload, FiExternalLink } from 'react-icons/fi';
import type { HuggingFaceModel } from '../../types/models';
import { Button } from '../common/Button';
import { InfoTooltip } from '../common/InfoTooltip';
import type { Theme } from '../../theme';

export interface HuggingFaceResultItemProps {
  model: HuggingFaceModel;
  isAdded: boolean;
  isAdding: boolean;
  isInStore?: boolean;
  onAdd: () => void;
  onSelect: () => void;
  downloadedOnNodes?: string[];
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

const Row = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  transition: background 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
  }
`;

const Info = styled.div`
  flex: 1;
  min-width: 0;
`;

const ModelName = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const Author = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const StatBadge = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textSecondary};
  display: flex;
  align-items: center;
  gap: 3px;
`;

const AddedBadge = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: ${({ theme }) => theme.fontSizes.label};
  padding: 2px 8px;
  border-radius: ${({ theme }) => theme.radii.sm};
  background: rgba(34, 197, 94, 0.15);
  color: ${({ theme }) => theme.colors.accent};
`;

const SelectBtn = styled(Button)`
  background: ${({ theme }) => theme.colors.goldBg};
  color: ${({ theme }) => theme.colors.gold};
  &:hover:not(:disabled) { background: rgba(255, 215, 0, 0.25); }
`;

const StoreDownloadIcon = () => <FiDownload size={14} />;

export function HuggingFaceResultItem({
  model,
  isAdded,
  isAdding,
  isInStore = false,
  onAdd,
  onSelect,
}: HuggingFaceResultItemProps) {
  const theme = useTheme() as Theme;
  const CheckIcon = () => <FiCheck size={16} color={theme.colors.accent} strokeWidth={2.5} />;
  const shortName = model.id.startsWith('mlx-community/')
    ? model.id.replace('mlx-community/', '')
    : model.id;

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
          style={{ color: 'rgba(255,255,255,0.5)', display: 'flex', transition: 'color 0.15s' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = theme.colors.gold; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
          title="Open on HuggingFace"
        >
          <FiExternalLink size={14} />
        </a>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Author</span>
        <span>{model.author}</span>
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Downloads</span>
        <span>{formatCount(model.downloads)}</span>
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Likes</span>
        <span>{formatCount(model.likes)}</span>
        <span style={{ color: 'rgba(255,255,255,0.45)' }}>Updated</span>
        <span>{new Date(model.last_modified).toLocaleDateString()}</span>
      </div>
      {sizeTags.length > 0 && (
        <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 6 }}>
          <div style={{ color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            Tags
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {sizeTags.map((t) => (
              <span key={t} style={{ padding: '1px 6px', borderRadius: 3, background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.7)' }}>{t}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <Row>
      <Info>
        <ModelName>{shortName}</ModelName>
        <Author>{model.author}</Author>
      </Info>

      {/* Info tooltip */}
      <InfoTooltip filled size={16} placement="left" delay={100} content={tooltipContent} />

      {/* Stats */}
      <StatBadge title="Downloads">↓ {formatCount(model.downloads)}</StatBadge>
      <StatBadge title="Likes">♥ {formatCount(model.likes)}</StatBadge>

      {/* Action */}
      {isInStore ? (
        <AddedBadge><CheckIcon /> In Store</AddedBadge>
      ) : isAdded ? (
        <SelectBtn variant="primary" size="sm" onClick={onSelect}>
          <StoreDownloadIcon /> Download
        </SelectBtn>
      ) : (
        <Button variant="outline" size="sm" onClick={onAdd} disabled={isAdding}>
          {isAdding ? '…' : <><StoreDownloadIcon /> Add & Download</>}
        </Button>
      )}
    </Row>
  );
}
