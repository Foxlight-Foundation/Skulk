import { useTheme } from 'styled-components';
import { getTemperatureColor } from '../../utils/format';
import type { Theme } from '../../theme';

export interface GpuStatsBarProps {
  /** 0-100 */
  gpuPercent: number;
  /** degrees celsius, NaN if unavailable */
  gpuTemp: number;
  /** watts, null if unavailable */
  sysPower: number | null;
  width: number;
  height: number;
}

export function GpuStatsBar({ gpuPercent, gpuTemp, sysPower, width, height }: GpuStatsBarProps) {
  const theme = useTheme() as Theme;
  const fillHeight = (gpuPercent / 100) * height;
  // Always encode thermal state via the temperature gradient (cool-blue → amber → red).
  // The gradient reads correctly on both dark and light backgrounds, so we keep the
  // overheat signal in every theme rather than collapsing it to a flat fill.
  const fillColor = getTemperatureColor(gpuTemp);

  const fontSize = Math.min(16, Math.max(10, width * 0.55));
  const lineSpacing = fontSize * 1.25;
  const textX = width / 2;
  const textY = height / 2;

  const gpuText = `${gpuPercent.toFixed(0)}%`;
  const tempText = !isNaN(gpuTemp) ? `${gpuTemp.toFixed(0)}°C` : '-';
  const powerText = sysPower !== null ? `${sysPower.toFixed(0)}W` : '-';

  return (
    <g>
      {/* Background — dedicated token so light/dark can tune contrast independently. */}
      <rect x={0} y={0} width={width} height={height}
        fill={theme.colors.gpuBarBg} rx={2} />
      {/* Fill from bottom */}
      {gpuPercent > 0 && (
        <rect x={0} y={height - fillHeight} width={width} height={fillHeight}
          fill={fillColor} opacity={0.9} rx={2} />
      )}
      {/* GPU % */}
      <text x={textX} y={textY - lineSpacing} textAnchor="middle" dominantBaseline="middle"
        fill={theme.colors.text} fontSize={fontSize} fontWeight={700} fontFamily="SF Mono, Monaco, monospace">
        {gpuText}
      </text>
      {/* Temperature */}
      <text x={textX} y={textY} textAnchor="middle" dominantBaseline="middle"
        fill={theme.colors.text} fontSize={fontSize} fontWeight={700} fontFamily="SF Mono, Monaco, monospace">
        {tempText}
      </text>
      {/* Power */}
      <text x={textX} y={textY + lineSpacing} textAnchor="middle" dominantBaseline="middle"
        fill={theme.colors.text} fontSize={fontSize} fontWeight={700} fontFamily="SF Mono, Monaco, monospace">
        {powerText}
      </text>
    </g>
  );
}
