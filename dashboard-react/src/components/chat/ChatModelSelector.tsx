/**
 * ChatModelSelector
 *
 * Category recommendation cards shown on empty chat state.
 * Picks the best-fitting model per category based on available memory.
 * Ported from ChatModelSelector.svelte.
 */
import React, { useMemo, useState } from 'react';
import styled from 'styled-components';

// ─── Types and ranking data ────────────────────────────────────────────────────

export interface ChatModelInfo {
  id: string;
  name?: string;
  base_model: string;
  storage_size_megabytes: number;
  capabilities?: string[];
  family?: string;
  quantization?: string;
}

// Auto-mode tier list (best to worst)
export const AUTO_TIERS: string[][] = [
  ['DeepSeek V3.1', 'GLM-5', 'Kimi K2.5', 'Qwen3 Coder Next'],
  ['Kimi K2', 'Qwen3 235B', 'MiniMax M2.5', 'Step 3.5 Flash', 'Qwen3 Next 80B'],
  ['GLM 4.7', 'MiniMax M2.1', 'Qwen3 Coder 480B', 'GLM 4.5 Air', 'Llama 3.3 70B'],
  ['GPT-OSS 120B', 'Qwen3 30B', 'Llama 3.1 70B', 'GLM 4.7 Flash'],
  ['Llama 3.1 8B', 'GPT-OSS 20B', 'Llama 3.2 3B', 'Qwen3 0.6B', 'Llama 3.2 1B'],
];

export function getAutoTierIndex(baseModel: string): number {
  for (let i = 0; i < AUTO_TIERS.length; i++) {
    if (AUTO_TIERS[i]?.includes(baseModel)) return i;
  }
  return AUTO_TIERS.length;
}

export function pickAutoModel(
  modelList: ChatModelInfo[],
  memoryGB: number,
): ChatModelInfo | null {
  for (const tier of AUTO_TIERS) {
    const candidates: ChatModelInfo[] = [];
    for (const baseModel of tier) {
      const variants = modelList
        .filter(
          (m) =>
            m.base_model === baseModel &&
            (m.storage_size_megabytes || 0) / 1024 <= memoryGB &&
            (m.storage_size_megabytes || 0) > 0,
        )
        .sort((a, b) => (b.storage_size_megabytes || 0) - (a.storage_size_megabytes || 0));
      const best = variants[0];
      if (best) candidates.push(best);
    }
    if (candidates.length > 0) {
      candidates.sort((a, b) => (b.storage_size_megabytes || 0) - (a.storage_size_megabytes || 0));
      return candidates[0] ?? null;
    }
  }
  return null;
}

const CODING_RANKING = [
  'Qwen3 Coder Next', 'Qwen3 Coder 480B', 'Qwen3 30B',
  'GPT-OSS 20B', 'Llama 3.1 8B', 'Llama 3.2 3B', 'Qwen3 0.6B',
];
const WRITING_RANKING = [
  'Kimi K2.5', 'Kimi K2', 'Qwen3 Next 80B', 'Llama 3.3 70B',
  'MiniMax M2.5', 'GLM 4.5 Air', 'GLM 4.7 Flash', 'GPT-OSS 20B',
  'Llama 3.1 8B', 'Llama 3.2 3B', 'Qwen3 0.6B',
];
const AGENTIC_RANKING = [
  'DeepSeek V3.1', 'GLM-5', 'Qwen3 235B', 'Step 3.5 Flash', 'GLM 4.7',
  'MiniMax M2.1', 'GPT-OSS 120B', 'Llama 3.3 70B', 'Llama 3.1 70B',
  'GLM 4.7 Flash', 'GPT-OSS 20B', 'Qwen3 30B', 'Llama 3.1 8B', 'Llama 3.2 3B', 'Qwen3 0.6B',
];

const CATEGORY_ICONS: Record<string, string> = {
  coding:
    'M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z',
  writing:
    'M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z',
  agentic:
    'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
  biggest:
    'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
};

const CATEGORY_TOOLTIPS: Record<string, string> = {
  coding: 'Ranked by coding benchmark performance (LiveCodeBench, SWE-bench)',
  writing: 'Ranked by creative writing quality and instruction following',
  agentic: 'Ranked by reasoning, planning, and tool-use capability',
  biggest: 'Largest model that fits in your available memory',
};

// ─── Styled components ────────────────────────────────────────────────────────

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 24px;
`;

const Header = styled.div`
  text-align: center;
`;

const SubLabel = styled.p`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  text-transform: uppercase;
  letter-spacing: 0.2em;
  margin: 0 0 4px;
`;

const ClusterLabel = styled.p`
  font-size: 13px;
  color: ${({ theme }) => theme.colors.foreground};
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.05em;
  margin: 0;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  width: 100%;
  max-width: 400px;
`;

const Card = styled.button`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  padding: 14px;
  border-radius: ${({ theme }) => theme.radius.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: oklch(0.16 0 0 / 0.5);
  cursor: pointer;
  text-align: left;
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    border-color: oklch(0.85 0.18 85 / 0.4);
    background: oklch(0.16 0 0);
  }
`;

const EmptyCard = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  padding: 14px;
  border-radius: ${({ theme }) => theme.radius.md};
  border: 1px solid oklch(0.25 0 0 / 0.3);
  background: oklch(0.16 0 0 / 0.3);
  opacity: 0.5;
`;

const CardHeaderRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
`;

const CardIcon = styled.svg`
  width: 16px;
  height: 16px;
  color: oklch(0.85 0.18 85 / 0.7);
  flex-shrink: 0;
  transition: color ${({ theme }) => theme.transitions.fast};
  ${Card}:hover & { color: oklch(0.85 0.18 85); }
`;

const EmptyCardIcon = styled.svg`
  width: 16px;
  height: 16px;
  color: oklch(0.65 0 0 / 0.4);
  flex-shrink: 0;
`;

const CategoryName = styled.span`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: ${({ theme }) => theme.colors.lightGray};
  transition: color ${({ theme }) => theme.transitions.fast};
  ${Card}:hover & { color: ${({ theme }) => theme.colors.foreground}; }
`;

const InfoBtn = styled.span`
  margin-left: auto;
  color: oklch(0.65 0 0 / 0.4);
  cursor: help;
  line-height: 1;
  &:hover { color: ${({ theme }) => theme.colors.lightGray}; }
`;

const ModelName = styled.p`
  font-size: 13px;
  color: ${({ theme }) => theme.colors.foreground};
  font-family: ${({ theme }) => theme.fonts.mono};
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  width: 100%;
`;

const ModelMeta = styled.p`
  font-size: 11px;
  color: oklch(0.65 0 0 / 0.6);
  font-family: ${({ theme }) => theme.fonts.mono};
  margin: 2px 0 0;
`;

const NoModelText = styled.p`
  font-size: 11px;
  color: oklch(0.65 0 0 / 0.4);
  font-family: ${({ theme }) => theme.fonts.mono};
  margin: 0;
`;

const AddModelBtn = styled.button`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: ${({ theme }) => theme.colors.lightGray};
  border: 1px solid oklch(0.25 0 0 / 0.3);
  border-radius: ${({ theme }) => theme.radius.md};
  background: none;
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    color: ${({ theme }) => theme.colors.yellow};
    border-color: oklch(0.85 0.18 85 / 0.3);
  }
`;

const AutoHint = styled.p`
  font-size: 11px;
  color: oklch(0.65 0 0 / 0.4);
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.05em;
  text-align: center;
  margin: 0;
`;

const Tooltip = styled.div`
  position: fixed;
  z-index: 9999;
  padding: 6px 10px;
  background: oklch(0.08 0 0);
  border: 1px solid oklch(0.3 0 0 / 0.5);
  border-radius: ${({ theme }) => theme.radius.sm};
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  white-space: nowrap;
  box-shadow: 0 4px 16px oklch(0 0 0 / 0.5);
  pointer-events: none;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ChatModelSelectorProps {
  models: ChatModelInfo[];
  clusterLabel: string;
  totalMemoryGB: number;
  onSelect: (modelId: string, category: string) => void;
  onAddModel: () => void;
  className?: string;
}

function formatSize(mb: number): string {
  const gb = mb / 1024;
  if (gb >= 100) return `${Math.round(gb)} GB`;
  return `${gb.toFixed(1)} GB`;
}

export const ChatModelSelector: React.FC<ChatModelSelectorProps> = ({
  models,
  clusterLabel,
  totalMemoryGB,
  onSelect,
  onAddModel,
  className,
}) => {
  const [tooltip, setTooltip] = useState<{ category: string; x: number; y: number } | null>(null);

  function getModelSizeGB(m: ChatModelInfo): number {
    return (m.storage_size_megabytes || 0) / 1024;
  }

  function fitsInMemory(m: ChatModelInfo): boolean {
    return getModelSizeGB(m) <= totalMemoryGB && getModelSizeGB(m) > 0;
  }

  function pickBestVariant(baseModel: string): ChatModelInfo | null {
    const variants = models
      .filter((m) => m.base_model === baseModel && fitsInMemory(m))
      .sort((a, b) => getModelSizeGB(b) - getModelSizeGB(a));
    return variants[0] ?? null;
  }

  function pickFromRanking(ranking: string[]): ChatModelInfo | null {
    for (const baseModel of ranking) {
      const pick = pickBestVariant(baseModel);
      if (pick) return pick;
    }
    return null;
  }

  function pickBiggest(): ChatModelInfo | null {
    const fitting = models
      .filter((m) => fitsInMemory(m))
      .sort((a, b) => getModelSizeGB(b) - getModelSizeGB(a));
    return fitting[0] ?? null;
  }

  const recommendations = useMemo(
    () => [
      { category: 'coding', label: 'Best for Coding', model: pickFromRanking(CODING_RANKING) },
      { category: 'writing', label: 'Best for Writing', model: pickFromRanking(WRITING_RANKING) },
      { category: 'agentic', label: 'Best Agentic', model: pickFromRanking(AGENTIC_RANKING) },
      { category: 'biggest', label: 'Biggest', model: pickBiggest() },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [models, totalMemoryGB],
  );

  const showTooltip = (category: string, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltip({ category, x: rect.left + rect.width / 2, y: rect.top });
  };

  return (
    <Container className={className}>
      <Header>
        <SubLabel>Recommended for your</SubLabel>
        <ClusterLabel>{clusterLabel}</ClusterLabel>
      </Header>

      <Grid>
        {recommendations.map((rec) =>
          rec.model ? (
            <Card
              key={rec.category}
              type="button"
              onClick={() => onSelect(rec.model!.id, rec.category)}
            >
              <CardHeaderRow>
                <CardIcon fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={CATEGORY_ICONS[rec.category]} />
                </CardIcon>
                <CategoryName>{rec.label}</CategoryName>
                <InfoBtn
                  role="button"
                  tabIndex={-1}
                  onMouseEnter={(e) => showTooltip(rec.category, e)}
                  onMouseLeave={() => setTooltip(null)}
                  onClick={(e) => e.stopPropagation()}
                >
                  <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 16v-4m0-4h.01" />
                  </svg>
                </InfoBtn>
              </CardHeaderRow>
              <div style={{ width: '100%', overflow: 'hidden' }}>
                <ModelName>{rec.model.base_model}</ModelName>
                <ModelMeta>
                  {formatSize(rec.model.storage_size_megabytes)}
                  {rec.model.quantization && (
                    <span style={{ opacity: 0.5 }}> · {rec.model.quantization}</span>
                  )}
                </ModelMeta>
              </div>
            </Card>
          ) : (
            <EmptyCard key={rec.category}>
              <CardHeaderRow>
                <EmptyCardIcon fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={CATEGORY_ICONS[rec.category]} />
                </EmptyCardIcon>
                <CategoryName style={{ color: 'oklch(0.65 0 0 / 0.5)' }}>{rec.label}</CategoryName>
              </CardHeaderRow>
              <NoModelText>No model fits</NoModelText>
            </EmptyCard>
          ),
        )}
      </Grid>

      <AddModelBtn type="button" onClick={onAddModel}>
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
        </svg>
        Add Model
      </AddModelBtn>

      <AutoHint>Or just start typing — we&rsquo;ll pick the best model automatically</AutoHint>

      {/* Tooltip portal */}
      {tooltip && (
        <Tooltip
          style={{
            left: tooltip.x,
            top: tooltip.y - 8,
            transform: 'translate(-50%, -100%)',
          }}
        >
          {CATEGORY_TOOLTIPS[tooltip.category]}
        </Tooltip>
      )}
    </Container>
  );
};
