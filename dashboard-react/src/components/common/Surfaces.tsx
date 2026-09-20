import type { ReactNode } from 'react';
import styled from 'styled-components';
import { FiChevronRight } from 'react-icons/fi';

/** Repeated panel surface; layout remains the consuming component's responsibility. */
// eslint-disable-next-line react-refresh/only-export-components -- Exported styled primitives are React components.
export const Surface = styled.section`
  min-width: 0;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 12px;
  padding: 16px;
`;

/** Compact uppercase section label shared by the operator surfaces. */
// eslint-disable-next-line react-refresh/only-export-components -- Exported styled primitives are React components.
export const SectionLabel = styled.div`
  font: 600 10px ${({ theme }) => theme.fonts.mono};
  letter-spacing: .16em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.subtleText};
`;

/** Neutral identifying tile for tools, plugins and device placeholders. */
// eslint-disable-next-line react-refresh/only-export-components -- Exported styled primitives are React components.
export const Monogram = styled.span<{ $size?: number }>`
  width: ${({ $size = 44 }) => $size}px;
  height: ${({ $size = 44 }) => $size}px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: ${({ theme }) => theme.colors.selected};
  color: ${({ theme }) => theme.colors.text};
  border-radius: 8px;
  font: 700 13px ${({ theme }) => theme.fonts.mono};
`;

/** Semantic evidence state, deliberately separate from the displayed label. */
export type StatusTone = 'healthy' | 'live' | 'danger' | 'neutral';
const Pill = styled.span<{ $tone: StatusTone }>`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  width: fit-content;
  max-width: 100%;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid ${({ theme, $tone }) => ({ healthy: theme.colors.borderHealthy, live: theme.colors.borderLive, danger: theme.colors.borderDanger, neutral: theme.colors.border })[$tone]};
  background: ${({ theme, $tone }) => ({ healthy: theme.colors.accentBg, live: theme.colors.liveBg, danger: theme.colors.errorBg, neutral: theme.colors.surfaceHover })[$tone]};
  color: ${({ theme, $tone }) => ({ healthy: theme.colors.healthy, live: theme.colors.liveText, danger: theme.colors.error, neutral: theme.colors.textSecondary })[$tone]};
  font: 400 11px ${({ theme }) => theme.fonts.mono};
  &::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
`;

/** Labelled status pill: callers supply observed truth, never inferred presence. */
export function StatusPill({ tone = 'neutral', children }: { tone?: StatusTone; children: ReactNode }) {
  return <Pill $tone={tone}>{children}</Pill>;
}

const Disclosure = styled.details`
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
  &:last-child { border-bottom: none; }
  > summary {
    display: flex; align-items: center; gap: 10px; padding: 12px 14px;
    cursor: pointer; list-style: none; color: ${({ theme }) => theme.colors.text};
    font-size: 14px; font-weight: 600;
  }
  > summary::-webkit-details-marker { display: none; }
  > summary > svg { flex-shrink: 0; transition: transform 120ms; }
  &[open] > summary > svg { transform: rotate(90deg); }
`;
const SummaryValue = styled.span<{ $mono: boolean }>`
  margin-left: auto; text-align: right; color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 12.5px; font-weight: 400; overflow-wrap: anywhere;
  font-family: ${({ theme, $mono }) => $mono ? theme.fonts.mono : theme.fonts.body};
`;
const DisclosureBody = styled.div`
  padding: 0 14px 14px 36px; min-width: 0; display: flex; flex-direction: column; gap: 12px;
  @media (max-width: 480px) { padding-left: 16px; }
`;

/** Accessible disclosure with caller-owned expansion persistence and draft summary. */
export function CollapsibleSection({ title, summary, open, onOpenChange, children }: {
  title: string; summary?: ReactNode; open: boolean; onOpenChange: (open: boolean) => void; children: ReactNode;
}) {
  return <Disclosure open={open} onToggle={event => {
    if (event.currentTarget.open !== open) onOpenChange(event.currentTarget.open);
  }}>
    <summary><FiChevronRight aria-hidden="true" />{title}<SummaryValue $mono={typeof summary === 'string' && /[.:~]/.test(summary)}>{summary}</SummaryValue></summary>
    <DisclosureBody>{children}</DisclosureBody>
  </Disclosure>;
}
