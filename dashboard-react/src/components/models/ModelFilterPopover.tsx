import { useEffect, useRef } from 'react';
import styled, { css } from 'styled-components';
import { CAPABILITIES, SIZE_RANGES, type FilterState } from '../../types/models';
import { Button } from '../common/Button';

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
  text-transform: uppercase;
  letter-spacing: 0.5px;
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
      background: rgba(255, 215, 0, 0.15);
      border-color: #ffd700;
      color: #ffd700;
    `}
`;

const ClearBtn = styled(Button)`
  align-self: flex-end;
`;

const CAPABILITY_LABELS: Record<string, string> = {
  text: 'Text',
  thinking: 'Thinking',
  code: 'Code',
  vision: 'Vision',
  image_gen: 'Image Gen',
  image_edit: 'Image Edit',
};

export function ModelFilterPopover({ filters, onChange, onClear, onClose }: ModelFilterPopoverProps) {
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
        <SectionLabel>Capabilities</SectionLabel>
        <ChipRow>
          {CAPABILITIES.map((cap) => (
            <Chip
              key={cap}
              variant="outline"
              size="sm"
              $active={filters.capabilities.includes(cap)}
              onClick={() => toggleCapability(cap)}
            >
              {CAPABILITY_LABELS[cap] ?? cap}
            </Chip>
          ))}
        </ChipRow>
      </div>

      {/* Size range */}
      <div>
        <SectionLabel>Size</SectionLabel>
        <ChipRow>
          {SIZE_RANGES.map((r) => (
            <Chip
              key={r.label}
              variant="outline"
              size="sm"
              $active={filters.sizeRange?.min === r.min && filters.sizeRange?.max === r.max}
              onClick={() => toggleSizeRange(r.min, r.max)}
            >
              {r.label}
            </Chip>
          ))}
        </ChipRow>
      </div>

      {/* Availability */}
      <div>
        <SectionLabel>Availability</SectionLabel>
        <ChipRow>
          <Chip
            variant="outline"
            size="sm"
            $active={filters.downloadedOnly}
            onClick={() => onChange({ ...filters, downloadedOnly: !filters.downloadedOnly })}
          >
            Downloaded
          </Chip>
          <Chip
            variant="outline"
            size="sm"
            $active={filters.readyOnly}
            onClick={() => onChange({ ...filters, readyOnly: !filters.readyOnly })}
          >
            Ready
          </Chip>
        </ChipRow>
      </div>

      {hasActiveFilters && <ClearBtn variant="ghost" size="sm" onClick={onClear}>Clear all</ClearBtn>}
    </Panel>
  );
}
