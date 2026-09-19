import { useEffect, useRef } from 'react';
import styled, { css } from 'styled-components';
import { CAPABILITIES, SIZE_RANGES, type FilterState } from '../../types/models';
import { Button } from '../common/Button';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

/** Controlled discovery facets, shown in a rail or standalone popover. */
export interface ModelFilterPopoverProps {
  /** Render inside a persistent facet rail instead of a floating popover. */
  inline?: boolean;
  filters: FilterState;
  onChange: (filters: FilterState) => void;
  onClear: () => void;
  onClose: () => void;
}

const Panel = styled.div<{ $inline: boolean }>`
  position: absolute;
  right: 0;
  top: 100%;
  margin-top: 4px;
  z-index: 20;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing.md};
  min-width: 0;
  width: ${({ $inline }) => $inline ? "auto" : "260px"};
  ${({ $inline }) => $inline && css`position: static; margin: 0; border: 0; border-radius: 0; padding: 16px; background: transparent;`}
  display: flex;
  flex-direction: column;
  gap: 14px;
`;

const SectionLabel = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
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
      border-color: ${({ theme }) => theme.colors.accentText};
      color: ${({ theme }) => theme.colors.accentText};
    `}
`;

const ClearBtn = styled(Button)`
  align-self: flex-end;
`;
const Availability = styled.div`
  display: flex; flex-direction: column; gap: 10px;
  label { display: flex; align-items: center; gap: 8px; font-size: 13px; color: ${({ theme }) => theme.colors.textSecondary}; cursor: pointer; }
`;

function capabilityLabel(capability: string, t: SkulkTranslate): string {
  const labels: Record<string, string> = {
    text: t('capability.text', 'Text'),
    thinking: t('capability.thinking', 'Thinking'),
    code: t('capability.code', 'Code'),
    vision: t('capability.vision', 'Vision'),
    image_gen: t('capability.imageGen', 'Image Gen'),
    image_edit: t('capability.imageEdit', 'Image Edit'),
    embedding: t('capability.embedding', 'Embedding'),
    tts: t('capability.tts', 'TTS'),
    stt: t('capability.stt', 'STT'),
  };
  return labels[capability] ?? capability;
}

function sizeRangeLabel(range: (typeof SIZE_RANGES)[number], t: SkulkTranslate): string {
  if (range.max === 10 * 1024) return t('modelFilter.sizeUnder10Gb', '< 10 GB');
  if (range.max === 50 * 1024) return t('modelFilter.size10To50Gb', '10-50 GB');
  if (range.max === 200 * 1024) return t('modelFilter.size50To200Gb', '50-200 GB');
  return t('modelFilter.sizeOver200Gb', '> 200 GB');
}

/** Edit capability, size and evidence-based availability filters. */
export function ModelFilterPopover({ inline = false, filters, onChange, onClear, onClose }: ModelFilterPopoverProps) {
  const { t } = useSkulkTranslation();
  const ref = useRef<HTMLDivElement>(null);

  // Click-outside handler
  useEffect(() => {
    if (inline) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose, inline]);

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
    <Panel ref={ref} $inline={inline}>
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
              aria-pressed={filters.capabilities.includes(cap)}
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
              aria-pressed={filters.sizeRange?.min === r.min && filters.sizeRange?.max === r.max}
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
        {inline ? <Availability>
          <label><input type="checkbox" checked={filters.downloadedOnly} onChange={event => onChange({ ...filters, downloadedOnly: event.target.checked })} />{t('modelPickerGroup.inStore', 'In store')}</label>
          <label><input type="checkbox" checked={filters.readyOnly} onChange={event => onChange({ ...filters, readyOnly: event.target.checked })} />{t('modelBrowser.readyNow', 'Ready now')}</label>
        </Availability> : <ChipRow>
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
        </ChipRow>}
      </div>

      {hasActiveFilters && <ClearBtn variant="ghost" size="sm" onClick={onClear}>{t('modelFilter.clearAll', 'Clear all')}</ClearBtn>}
    </Panel>
  );
}
