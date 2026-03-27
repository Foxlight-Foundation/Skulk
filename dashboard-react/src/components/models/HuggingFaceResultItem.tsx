import styled from 'styled-components';
import type { HuggingFaceModel } from '../../types/models';
import { Button } from '../common/Button';
import { InfoTooltip } from '../common/InfoTooltip';

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
  color: #22c55e;
`;

const SelectBtn = styled(Button)`
  background: rgba(255, 215, 0, 0.15);
  color: #ffd700;
  &:hover:not(:disabled) { background: rgba(255, 215, 0, 0.25); }
`;

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const StoreDownloadIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);

export function HuggingFaceResultItem({
  model,
  isAdded,
  isAdding,
  isInStore = false,
  onAdd,
  onSelect,
}: HuggingFaceResultItemProps) {
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
        <span style={{ color: '#FFD700', fontWeight: 600 }}>{model.id}</span>
        <a
          href={hfUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'rgba(255,255,255,0.5)', display: 'flex', transition: 'color 0.15s' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = '#FFD700'; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
          title="Open on HuggingFace"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <polyline points="15 3 21 3 21 9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </svg>
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
