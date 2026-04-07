import { useMemo, useState, useCallback, useRef } from 'react';
import styled, { css } from 'styled-components';
import type { ChatModelInfo } from '../../types/chat';
import { Button } from '../common/Button';

export interface ChatModelSelectorProps {
  models: ChatModelInfo[];
  clusterLabel: string;
  totalMemoryGB: number;
  onSelect: (modelId: string, category: string) => void;
  onAddModel: () => void;
  className?: string;
}

/* ================================================================
   Ranking data
   ================================================================ */

const CODING_RANKING = [
  'qwen3-coder-next', 'qwen3-coder-480b', 'qwen3-30b', 'qwen3-coder-240b',
  'gpt-oss', 'llama-4-maverick', 'llama-3.1-70b', 'qwen3-8b', 'llama-3.1-8b',
];

const WRITING_RANKING = [
  'kimi-k2.5', 'kimi-k2', 'qwen3-next-80b', 'qwen3-235b', 'qwen3-30b',
  'llama-4-maverick', 'llama-3.1-70b', 'qwen3-8b', 'step-3.5',
];

const AGENTIC_RANKING = [
  'deepseek-v3.1', 'glm-5', 'qwen3-235b', 'step-3.5', 'qwen3-30b',
  'kimi-k2', 'llama-4-maverick', 'llama-3.1-70b', 'qwen3-8b',
];

type Category = 'coding' | 'writing' | 'agentic' | 'biggest';

interface Recommendation {
  category: Category;
  label: string;
  model: ChatModelInfo | null;
  tooltip: string;
}

const CATEGORY_META: Record<Category, { label: string; tooltip: string }> = {
  coding: { label: 'Coding', tooltip: 'Best for code generation, debugging, and technical tasks' },
  writing: { label: 'Writing', tooltip: 'Best for creative writing, summarization, and general text' },
  agentic: { label: 'Agentic', tooltip: 'Best for reasoning, planning, and multi-step tasks' },
  biggest: { label: 'Biggest', tooltip: 'Largest model that fits in your cluster memory' },
};

/* ================================================================
   Selection logic
   ================================================================ */

function fitsInMemory(model: ChatModelInfo, memGB: number): boolean {
  const sizeGB = model.storage_size_megabytes / 1024;
  return sizeGB > 0 && sizeGB <= memGB;
}

function pickBestVariant(
  models: ChatModelInfo[],
  baseModel: string,
  memGB: number,
): ChatModelInfo | null {
  const variants = models
    .filter((m) => m.base_model === baseModel && fitsInMemory(m, memGB))
    .sort((a, b) => b.storage_size_megabytes - a.storage_size_megabytes);
  return variants[0] ?? null;
}

function pickFromRanking(
  models: ChatModelInfo[],
  ranking: string[],
  memGB: number,
): ChatModelInfo | null {
  for (const base of ranking) {
    const pick = pickBestVariant(models, base, memGB);
    if (pick) return pick;
  }
  return null;
}

function pickBiggest(models: ChatModelInfo[], memGB: number): ChatModelInfo | null {
  const fitting = models.filter((m) => fitsInMemory(m, memGB));
  if (fitting.length === 0) return null;
  fitting.sort((a, b) => b.storage_size_megabytes - a.storage_size_megabytes);
  return fitting[0];
}

/* ================================================================
   Styles
   ================================================================ */

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  max-width: 420px;
  margin: 0 auto;
  text-align: center;
`;

const Header = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ClusterName = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.gold};
  margin-top: 4px;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  width: 100%;
`;

const Card = styled.button<{ $disabled: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  transition: all 0.15s;
  text-align: left;

  &:hover {
    border-color: ${({ theme }) => theme.colors.goldDim};
    background: ${({ theme }) => theme.colors.surfaceHover};
  }

  ${({ $disabled }) =>
    $disabled &&
    css`
      opacity: 0.4;
      pointer-events: none;
    `}
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
`;

const CategoryLabel = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  flex: 1;
`;

const InfoBtn = styled.span`
  cursor: help;
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textMuted};
  &:hover { color: ${({ theme }) => theme.colors.text}; }
`;

const ModelName = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
`;

const ModelMeta = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;


const Hint = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textMuted};
  max-width: 300px;
`;

const Tooltip = styled.div<{ $x: number; $y: number }>`
  position: fixed;
  left: ${({ $x }) => $x}px;
  top: ${({ $y }) => $y}px;
  transform: translate(-50%, -100%);
  z-index: 9999;
  background: ${({ theme }) => theme.colors.bg};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 4px 8px;
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textSecondary};
  white-space: nowrap;
  pointer-events: none;
  margin-top: -4px;
`;

const ICONS: Record<Category, React.ReactNode> = {
  coding: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
    </svg>
  ),
  writing: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  ),
  agentic: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" />
      <line x1="10" y1="21" x2="14" y2="21" />
    </svg>
  ),
  biggest: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="2" width="20" height="8" rx="2" /><rect x="2" y="14" width="20" height="8" rx="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" /><line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  ),
};

/* ================================================================
   Component
   ================================================================ */

export function ChatModelSelector({
  models,
  clusterLabel,
  totalMemoryGB,
  onSelect,
  onAddModel,
  className,
}: ChatModelSelectorProps) {
  const [tooltip, setTooltip] = useState<{ cat: Category; x: number; y: number } | null>(null);

  const recommendations = useMemo<Recommendation[]>(() => {
    const cats: Category[] = ['coding', 'writing', 'agentic', 'biggest'];
    return cats.map((cat) => {
      let model: ChatModelInfo | null;
      if (cat === 'biggest') {
        model = pickBiggest(models, totalMemoryGB);
      } else {
        const ranking = cat === 'coding' ? CODING_RANKING : cat === 'writing' ? WRITING_RANKING : AGENTIC_RANKING;
        model = pickFromRanking(models, ranking, totalMemoryGB);
      }
      return { category: cat, label: CATEGORY_META[cat].label, model, tooltip: CATEGORY_META[cat].tooltip };
    });
  }, [models, totalMemoryGB]);

  const showTooltip = useCallback((cat: Category, e: React.MouseEvent) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setTooltip({ cat, x: rect.left + rect.width / 2, y: rect.top });
  }, []);

  const formatSize = (mb: number) => {
    const gb = mb / 1024;
    return gb >= 100 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
  };

  return (
    <Container className={className}>
      <div>
        <Header>Recommended for your</Header>
        <ClusterName>{clusterLabel}</ClusterName>
      </div>

      <Grid>
        {recommendations.map((rec) => (
          <Card
            key={rec.category}
            $disabled={!rec.model}
            onClick={() => rec.model && onSelect(rec.model.id, rec.category)}
          >
            <CardHeader>
              {ICONS[rec.category]}
              <CategoryLabel>{rec.label}</CategoryLabel>
              <InfoBtn
                onMouseEnter={(e) => showTooltip(rec.category, e)}
                onMouseLeave={() => setTooltip(null)}
              >
                ?
              </InfoBtn>
            </CardHeader>
            {rec.model ? (
              <>
                <ModelName>{rec.model.base_model}</ModelName>
                <ModelMeta>
                  {formatSize(rec.model.storage_size_megabytes)}
                  {rec.model.quantization && ` · ${rec.model.quantization}`}
                </ModelMeta>
              </>
            ) : (
              <ModelMeta>No model fits</ModelMeta>
            )}
          </Card>
        ))}
      </Grid>

      <Button variant="outline" size="sm" onClick={onAddModel}>+ Add Model</Button>
      <Hint>Or just start typing — we'll pick the best model automatically</Hint>

      {tooltip && (
        <Tooltip $x={tooltip.x} $y={tooltip.y}>
          {CATEGORY_META[tooltip.cat].tooltip}
        </Tooltip>
      )}
    </Container>
  );
}
