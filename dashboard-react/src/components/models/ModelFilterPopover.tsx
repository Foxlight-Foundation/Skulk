/**
 * ModelFilterPopover
 *
 * Filter controls for the model picker: capability chips, size range,
 * and download status toggle.
 */
import React from 'react';
import styled, { css } from 'styled-components';
import { useTranslate } from '@tolgee/react';

// ─── Types ─────────────────────────────────────────────────────────────────────

export type CapabilityFilter = 'chat' | 'vision' | 'image_gen' | 'thinking';
export type DownloadFilter = 'all' | 'downloaded' | 'not_downloaded';

export interface ModelFilters {
  capabilities: Set<CapabilityFilter>;
  downloadStatus: DownloadFilter;
  minSizeGB: number | null;
  maxSizeGB: number | null;
}

export const DEFAULT_FILTERS: ModelFilters = {
  capabilities: new Set(),
  downloadStatus: 'all',
  minSizeGB: null,
  maxSizeGB: null,
};

// ─── Styled components ────────────────────────────────────────────────────────

const Popover = styled.div`
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.md};
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 220px;
  box-shadow: 0 8px 24px oklch(0 0 0 / 0.5);
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const SectionLabel = styled.div`
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.mutedForeground};
`;

const Chips = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
`;

const Chip = styled.button<{ $active: boolean }>`
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 9px;
  border-radius: ${({ theme }) => theme.radius.full};
  border: 1px solid ${({ theme, $active }) =>
    $active ? theme.colors.yellow : theme.colors.border};
  background: ${({ theme, $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.12)' : 'transparent'};
  color: ${({ theme, $active }) => ($active ? theme.colors.yellow : theme.colors.lightGray)};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    border-color: ${({ theme }) => theme.colors.yellowDarker};
    color: ${({ theme }) => theme.colors.yellow};
  }
`;

const ResetBtn = styled.button`
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: none;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.sm};
  color: ${({ theme }) => theme.colors.lightGray};
  padding: 4px 10px;
  cursor: pointer;
  align-self: flex-start;
  transition: color ${({ theme }) => theme.transitions.fast};
  &:hover { color: ${({ theme }) => theme.colors.foreground}; }
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ModelFilterPopoverProps {
  filters: ModelFilters;
  onChange: (filters: ModelFilters) => void;
}

export const ModelFilterPopover: React.FC<ModelFilterPopoverProps> = ({
  filters,
  onChange,
}) => {
  const { t } = useTranslate();

  const toggleCapability = (cap: CapabilityFilter) => {
    const next = new Set(filters.capabilities);
    if (next.has(cap)) next.delete(cap); else next.add(cap);
    onChange({ ...filters, capabilities: next });
  };

  const setDownloadStatus = (status: DownloadFilter) => {
    onChange({ ...filters, downloadStatus: status });
  };

  const isDefault =
    filters.capabilities.size === 0 &&
    filters.downloadStatus === 'all' &&
    filters.minSizeGB === null &&
    filters.maxSizeGB === null;

  const capabilities: Array<{ id: CapabilityFilter; label: string }> = [
    { id: 'chat',      label: t('models.capability_chat') },
    { id: 'vision',    label: t('models.capability_vision') },
    { id: 'image_gen', label: t('models.capability_image_gen') },
    { id: 'thinking',  label: t('models.capability_thinking') },
  ];

  const downloadOptions: Array<{ id: DownloadFilter; label: string }> = [
    { id: 'all',            label: t('models.all') },
    { id: 'downloaded',     label: t('models.downloaded') },
    { id: 'not_downloaded', label: t('models.not_downloaded') },
  ];

  return (
    <Popover onClick={(e) => e.stopPropagation()}>
      <Section>
        <SectionLabel>{t('models.capability')}</SectionLabel>
        <Chips>
          {capabilities.map((c) => (
            <Chip
              key={c.id}
              $active={filters.capabilities.has(c.id)}
              onClick={() => toggleCapability(c.id)}
            >
              {c.label}
            </Chip>
          ))}
        </Chips>
      </Section>

      <Section>
        <SectionLabel>{t('models.downloaded')}</SectionLabel>
        <Chips>
          {downloadOptions.map((o) => (
            <Chip
              key={o.id}
              $active={filters.downloadStatus === o.id}
              onClick={() => setDownloadStatus(o.id)}
            >
              {o.label}
            </Chip>
          ))}
        </Chips>
      </Section>

      {!isDefault && (
        <ResetBtn onClick={() => onChange(DEFAULT_FILTERS)}>
          {t('common.clear')} filters
        </ResetBtn>
      )}
    </Popover>
  );
};
