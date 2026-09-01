import { useTheme } from 'styled-components';
import type { Theme } from '../../theme';
import type { DeviceModel } from '../../types/topology';

const APPLE_LOGO_PATH =
  'M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z';
const APPLE_LOGO_WIDTH = 814;
const APPLE_LOGO_HEIGHT = 1000;

/** Props for the compact vendor identity badge beside a topology node. */
export interface HardwareBadgeProps {
  /** Best-effort hardware family resolved from node telemetry. */
  model: DeviceModel;
}

function AppleMark({ color }: { color: string }) {
  const height = 18;
  const scale = height / APPLE_LOGO_HEIGHT;
  const width = APPLE_LOGO_WIDTH * scale;
  return (
    <path
      d={APPLE_LOGO_PATH}
      fill={color}
      transform={`translate(${(48 - width) / 2}, ${(34 - height) / 2}) scale(${scale})`}
    />
  );
}

/**
 * Flat hardware identity mark used by native-style topology nodes.
 *
 * The badge deliberately identifies only the vendor family; telemetry and
 * tooltips retain the exact model. This keeps the small SVG from pretending
 * to be a miniature computer illustration.
 */
export function HardwareBadge({ model }: HardwareBadgeProps) {
  const theme = useTheme() as Theme;
  const isApple = model === 'macbook-pro' || model === 'mac-studio' || model === 'mac-mini';

  return (
    <g data-hardware-badge data-hardware-model={model}>
      <rect
        fill={theme.colors.topologyNodeSurface}
        height={34}
        rx={8}
        stroke={theme.colors.topologyNodeComputeTrack}
        width={48}
      />
      {isApple ? <AppleMark color={theme.colors.topologyNodeText} /> : null}
      {model === 'amd-strix' ? (
        <text
          dominantBaseline="middle"
          fill={theme.colors.topologyNodeText}
          fontFamily="Arial, Helvetica, sans-serif"
          fontSize={11}
          fontWeight={700}
          textAnchor="middle"
          x={24}
          y={17}
        >
          AMD
        </text>
      ) : null}
      {model === 'nvidia-gpu' ? (
        <text
          dominantBaseline="middle"
          fill="#76B900"
          fontFamily="Arial, Helvetica, sans-serif"
          fontSize={8}
          fontWeight={700}
          letterSpacing={0.2}
          textAnchor="middle"
          x={24}
          y={17}
        >
          NVIDIA
        </text>
      ) : null}
      {model === 'unknown' ? (
        <text
          dominantBaseline="middle"
          fill={theme.colors.topologyNodeDetail}
          fontFamily={theme.fonts.mono}
          fontSize={12}
          fontWeight={600}
          textAnchor="middle"
          x={24}
          y={17}
        >
          ?
        </text>
      ) : null}
    </g>
  );
}
