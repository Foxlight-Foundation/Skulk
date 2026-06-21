import { useEffect, useRef } from 'react';
import styled, { css } from 'styled-components';
import { CAPABILITIES, SIZE_RANGES, type FilterState } from '../../types/models';
import { Button } from '../common/Button';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

export interface ModelFilterPopoverProps {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
  onClear: () => void;
  onClose: () => void;
}

const Panel = styled.div`
  position: absolute;
  right: 0;
  top: 100%;
  margin-top: 4px;
  z-index: 20;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing.md};
  min-width: 260px;
  display: flex;
  flex-direction: column;
  gap: 14px;
`;

const SectionLabel = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textMuted};
  margin-bottom: 6px;
`;

const ChipRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
`;

const Chip = styled(Button)<{ $active: boolean }>`
  ${({ $active }) =>
    $active &&
    css`
      background: ${({ theme }) => theme.colors.goldBg};
      border-color: ${({ theme }) => theme.colors.gold};
      color: ${({ theme }) => theme.colors.gold};
    `}
`;

const ClearBtn = styled(Button)`
  align-self: flex-end;
`;

function capabilityLabel(capability: string, t: SkulkTranslate): string {
  const labels: Record<string, string> = {
    text: t('capability.text', 'Text'),
    thinking: t('capability.thinking', 'Thinking'),
    code: t('capability.code', 'Code'),
    vision: t('capability.vision', 'Vision'),
    image_gen: t('capability.imageGen', 'Image Gen'),
    image_edit: t('capability.imageEdit', 'Image Edit'),
  };
  return labels[capability] ?? capability;
}

function sizeRangeLabel(range: (typeof SIZE_RANGES)[number], t: SkulkTranslate): string {
  if (range.max === 10 * 1024) return t('modelFilter.sizeUnder10Gb', '< 10 GB');
  if (range.max === 50 * 1024) return t('modelFilter.size10To50Gb', '10-50 GB');
  if (range.max === 200 * 1024) return t('modelFilter.size50To200Gb', '50-200 GB');
  return t('modelFilter.sizeOver200Gb', '> 200 GB');
}

export function ModelFilterPopover({ filters, onChange, onClear, onClose }: ModelFilterPopoverProps) {
  const { t } = useSkulkTranslation();
  const ref = useRef<HTMLDivElement>(null);

  // Click-outside handler
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const toggleCapability = (cap: string) => {
    const caps = filters.capabilities.includes(cap)
      ? filters.capabilities.filter((c) => c !== cap)
      : [...filters.capabilities, cap];
    onChange({ ...filters, capabilities: caps });
  };

  const toggleSizeRange = (min: number, max: number) => {
    if (filters.sizeRange?.min === min && filters.sizeRange?.max === max) {
      onChange({ ...filters, sizeRange: null });
    } else {
      onChange({ ...filters, sizeRange: { min, max } });
    }
  };

  const hasActiveFilters =
    filters.capabilities.length > 0 ||
    filters.sizeRange !== null ||
    filters.downloadedOnly ||
    filters.readyOnly;

  return (
    <Panel ref={ref}>
      {/* Capabilities */}
      <div>
        <SectionLabel>{t('modelInfo.capabilities', 'Capabilities')}</SectionLabel>
        <ChipRow>
          {CAPABILITIES.map((cap) => (
            <Chip
              key={cap}
              variant="outline"
              size="sm"
              $active={filters.capabilities.includes(cap)}
              onClick={() => toggleCapability(cap)}
            >
              {capabilityLabel(cap, t)}
            </Chip>
          ))}
        </ChipRow>
      </div>

      {/* Size range */}
      <div>
        <SectionLabel>{t('common.size', 'Size')}</SectionLabel>
        <ChipRow>
          {SIZE_RANGES.map((r) => (
            <Chip
              key={`${r.min}-${r.max}`}
              variant="outline"
              size="sm"
              $active={filters.sizeRange?.min === r.min && filters.sizeRange?.max === r.max}
              onClick={() => toggleSizeRange(r.min, r.max)}
            >
              {sizeRangeLabel(r, t)}
            </Chip>
          ))}
        </ChipRow>
      </div>

      {/* Availability */}
      <div>
        <SectionLabel>{t('modelFilter.availability', 'Availability')}</SectionLabel>
        <ChipRow>
          <Chip
            variant="outline"
            size="sm"
            $active={filters.downloadedOnly}
            onClick={() => onChange({ ...filters, downloadedOnly: !filters.downloadedOnly })}
          >
            {t('modelFilter.downloaded', 'Downloaded')}
          </Chip>
          <Chip
            variant="outline"
            size="sm"
            $active={filters.readyOnly}
            onClick={() => onChange({ ...filters, readyOnly: !filters.readyOnly })}
          >
            {t('common.ready', 'Ready')}
          </Chip>
        </ChipRow>
      </div>

      {hasActiveFilters && <ClearBtn variant="ghost" size="sm" onClick={onClear}>{t('modelFilter.clearAll', 'Clear all')}</ClearBtn>}
    </Panel>
  );
}
