import styled from 'styled-components';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

/**
 * Normalized accelerator metrics as they arrive from the API (camelCase).
 * Mirrors the backend `AcceleratorMetrics`: any field a collector cannot measure
 * is `null`, which renders as "not reported" rather than a misleading zero.
 */
export interface AcceleratorMetrics {
  vendor?: string;
  name?: string;
  utilizationRatio?: number | null;
  vramTotalBytes?: number | null;
  vramUsedBytes?: number | null;
  powerWatts?: number | null;
  temperatureCelsius?: number | null;
  clockMhz?: number | null;
}

/** Props for {@link AcceleratorPanel}. */
export interface AcceleratorPanelProps {
  /** The node's normalized accelerator block, or undefined if none reported. */
  accelerator?: AcceleratorMetrics | null;
}

const Panel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const Header = styled.div`
  display: flex;
  align-items: baseline;
  gap: 8px;
`;

const VENDOR_COLORS: Record<string, string> = {
  amd: '#d32f2f',
  nvidia: '#5a8a00',
  apple: '#6e6e73',
  intel: '#0068b5',
};

const VendorPill = styled.span<{ $vendor?: string }>`
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 1px 6px;
  border-radius: 4px;
  /* Self-contained palette (not theme tokens) so the badge stays legible on
     any background and never collapses to one-color-on-itself. */
  color: #fff;
  background: ${({ $vendor }) => VENDOR_COLORS[$vendor ?? ''] ?? '#555'};
`;

const Name = styled.span`
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Row = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 12px;
`;

const Key = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Value = styled.div`
  font-family: 'SF Mono', Monaco, monospace;
  color: ${({ theme }) => theme.colors.text};
`;

const BarTrack = styled.div`
  height: 6px;
  border-radius: 3px;
  background: ${({ theme }) => theme.colors.gpuBarBg};
  overflow: hidden;
`;

const BarFill = styled.div<{ $ratio: number }>`
  height: 100%;
  width: ${({ $ratio }) => Math.min(100, Math.max(0, $ratio * 100))}%;
  background: ${({ theme }) => theme.colors.accent};
`;

function notReported(t: SkulkTranslate): string {
  return t('observability.accelerator.notReported', 'not reported');
}

function gib(bytes: number | null | undefined, t: SkulkTranslate): string {
  if (bytes == null) return notReported(t);
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

function pct(ratio: number | null | undefined, t: SkulkTranslate): string {
  if (ratio == null) return notReported(t);
  return `${Math.round(ratio * 100)}%`;
}

function watts(w: number | null | undefined, t: SkulkTranslate): string {
  if (w == null) return notReported(t);
  return `${w.toFixed(1)} W`;
}

function celsius(c: number | null | undefined, t: SkulkTranslate): string {
  if (c == null) return notReported(t);
  return `${Math.round(c)}°C`;
}

function mhz(m: number | null | undefined, t: SkulkTranslate): string {
  if (m == null) return notReported(t);
  return `${m} MHz`;
}

/**
 * Renders a node's normalized GPU/accelerator metrics uniformly across vendors
 * (Apple, AMD, NVIDIA). Driven entirely by the collector-agnostic
 * {@link AcceleratorMetrics} block, so a Strix Halo node and an Apple Silicon
 * node display the same shape; fields a collector cannot measure show
 * "not reported" instead of a fake zero. Returns null when no accelerator is
 * reported at all (e.g. a management node).
 */
export function AcceleratorPanel({ accelerator }: AcceleratorPanelProps) {
  const { t } = useSkulkTranslation();
  if (!accelerator) return null;
  const vendor = accelerator.vendor ?? 'unknown';
  const used = accelerator.vramUsedBytes;
  const total = accelerator.vramTotalBytes;
  const vramRatio = used != null && total != null && total > 0 ? used / total : null;

  return (
    <Panel>
      <Header>
        <VendorPill $vendor={vendor}>{vendor}</VendorPill>
        <Name>{accelerator.name ?? t('common.unknown', 'Unknown')}</Name>
      </Header>
      <Row>
        <Key>{t('observability.accelerator.utilization', 'Utilization')}</Key>
        <Value>{pct(accelerator.utilizationRatio, t)}</Value>
      </Row>
      <Row>
        <Key>{t('observability.accelerator.vram', 'VRAM')}</Key>
        <Value>
          {/* Collapse to a single "not reported" only when BOTH are absent;
              otherwise show each side via gib() so a partial reading isn't lost. */}
          {used == null && total == null
            ? notReported(t)
            : t('observability.accelerator.vramUsage', '{used} / {total}', {
                used: gib(used, t),
                total: gib(total, t),
              })}
        </Value>
      </Row>
      {vramRatio != null && (
        <BarTrack>
          <BarFill $ratio={vramRatio} />
        </BarTrack>
      )}
      <Row>
        <Key>{t('observability.accelerator.power', 'Power')}</Key>
        <Value>{watts(accelerator.powerWatts, t)}</Value>
      </Row>
      <Row>
        <Key>{t('observability.accelerator.temperature', 'Temperature')}</Key>
        <Value>{celsius(accelerator.temperatureCelsius, t)}</Value>
      </Row>
      <Row>
        <Key>{t('observability.accelerator.clock', 'Clock')}</Key>
        <Value>{mhz(accelerator.clockMhz, t)}</Value>
      </Row>
    </Panel>
  );
}
