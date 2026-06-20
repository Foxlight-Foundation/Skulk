import styled from 'styled-components';

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

const NOT_REPORTED = 'not reported';

function gib(bytes?: number | null): string {
  if (bytes == null) return NOT_REPORTED;
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

function pct(ratio?: number | null): string {
  if (ratio == null) return NOT_REPORTED;
  return `${Math.round(ratio * 100)}%`;
}

function watts(w?: number | null): string {
  if (w == null) return NOT_REPORTED;
  return `${w.toFixed(1)} W`;
}

function celsius(c?: number | null): string {
  if (c == null) return NOT_REPORTED;
  return `${Math.round(c)}°C`;
}

function mhz(m?: number | null): string {
  if (m == null) return NOT_REPORTED;
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
  if (!accelerator) return null;
  const vendor = accelerator.vendor ?? 'unknown';
  const used = accelerator.vramUsedBytes;
  const total = accelerator.vramTotalBytes;
  const vramRatio = used != null && total != null && total > 0 ? used / total : null;

  return (
    <Panel>
      <Header>
        <VendorPill $vendor={vendor}>{vendor}</VendorPill>
        <Name>{accelerator.name ?? 'Unknown'}</Name>
      </Header>
      <Row>
        <Key>Utilization</Key>
        <Value>{pct(accelerator.utilizationRatio)}</Value>
      </Row>
      <Row>
        <Key>VRAM</Key>
        <Value>
          {total == null ? NOT_REPORTED : `${gib(used)} / ${gib(total)}`}
        </Value>
      </Row>
      {vramRatio != null && (
        <BarTrack>
          <BarFill $ratio={vramRatio} />
        </BarTrack>
      )}
      <Row>
        <Key>Power</Key>
        <Value>{watts(accelerator.powerWatts)}</Value>
      </Row>
      <Row>
        <Key>Temperature</Key>
        <Value>{celsius(accelerator.temperatureCelsius)}</Value>
      </Row>
      <Row>
        <Key>Clock</Key>
        <Value>{mhz(accelerator.clockMhz)}</Value>
      </Row>
    </Panel>
  );
}
