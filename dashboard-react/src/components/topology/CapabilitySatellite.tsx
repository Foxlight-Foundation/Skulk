import { useTheme } from 'styled-components';
import type { Theme } from '../../theme';
import {
  capabilityNodeHealth,
  capabilityNodeTitle,
  type CapabilityNodeSummary,
} from '../../types/capabilityNodes';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { SATELLITE_RADIUS } from './topologyLayout';
import { satelliteColor, satelliteStatusLabel } from './capabilityPresentation';

/** Props for one capability satellite drawn inside its host's node group. */
export interface CapabilitySatelliteProps {
  summary: CapabilityNodeSummary;
  /** Node-local center x (before the host group's scale). */
  x: number;
  /** Node-local center y (before the host group's scale). */
  y: number;
  /** Whether this satellite's flyout is open. */
  active?: boolean;
  /** Opens or closes the flyout for this satellite. */
  onSelect?: () => void;
}

/** Props for the overflow glyph that stands in for satellites beyond the cap. */
export interface CapabilityOverflowProps {
  /** How many satellites the glyph represents. */
  count: number;
  x: number;
  y: number;
  onSelect?: () => void;
}

const buttonStyle = {
  appearance: 'none',
  background: 'transparent',
  border: 0,
  borderRadius: '50%',
  cursor: 'pointer',
  height: '100%',
  margin: 0,
  outline: 'none',
  padding: 0,
  width: '100%',
} as const;

/**
 * A capability node drawn as a small disc orbiting its host: the ring color
 * is the node's own health (never the host's), the letter is its title's
 * initial, and the disc is a real button so keyboards reach the flyout.
 */
export function CapabilitySatellite({ summary, x, y, active = false, onSelect }: CapabilitySatelliteProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const title = capabilityNodeTitle(summary);
  const level = capabilityNodeHealth(summary);
  const color = satelliteColor(level, theme);
  const label = t('topology.capability.satelliteAria', '{title} capability node, {status}', {
    title,
    status: satelliteStatusLabel(summary, t),
  });
  const hit = SATELLITE_RADIUS * 2 + 8;
  return (
    <g
      className="topology-capability-satellite"
      data-capability-key={`${summary.pluginId}/${summary.nodeId}`}
      data-capability-health={level}
      transform={`translate(${x}, ${y})`}
    >
      <line
        stroke={theme.colors.topologyConnectionLine}
        strokeDasharray="2 3"
        strokeWidth={1}
        x1={0}
        x2={-x * 0.55}
        y1={0}
        y2={-y * 0.55}
      />
      <circle
        cx={0}
        cy={0}
        fill="none"
        opacity={active ? 0.9 : 0}
        r={SATELLITE_RADIUS + 4}
        stroke={theme.colors.topologyNodeSelection}
        strokeDasharray="2 3"
        strokeWidth={1.2}
      />
      <circle
        cx={0}
        cy={0}
        fill={theme.colors.topologyNodeSurface}
        opacity={level === 'muted' ? 0.6 : 1}
        r={SATELLITE_RADIUS}
        stroke={color}
        strokeWidth={2}
      />
      <text
        dominantBaseline="middle"
        fill={theme.colors.topologyNodeText}
        fontFamily={theme.fonts.body}
        fontSize={10}
        fontWeight={700}
        opacity={level === 'muted' ? 0.7 : 1}
        textAnchor="middle"
        x={0}
        y={0.5}
      >
        {title.slice(0, 1).toUpperCase()}
      </text>
      <foreignObject height={hit} width={hit} x={-hit / 2} y={-hit / 2}>
        <button
          aria-label={label}
          aria-pressed={active}
          onClick={(event) => {
            event.stopPropagation();
            onSelect?.();
          }}
          style={buttonStyle}
          title={title}
          type="button"
        />
      </foreignObject>
    </g>
  );
}

/** Stand-in glyph for satellites past the visible cap; opens the flyout on them. */
export function CapabilityOverflow({ count, x, y, onSelect }: CapabilityOverflowProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const label = t('topology.capability.overflowAria', '{count} more capability nodes', { count });
  const hit = SATELLITE_RADIUS * 2 + 8;
  return (
    <g className="topology-capability-overflow" transform={`translate(${x}, ${y})`}>
      <circle
        cx={0}
        cy={0}
        fill={theme.colors.topologyNodeSurface}
        r={SATELLITE_RADIUS}
        stroke={theme.colors.borderStrong}
        strokeDasharray="2 2"
        strokeWidth={1.5}
      />
      <text
        dominantBaseline="middle"
        fill={theme.colors.topologyNodeLabel}
        fontFamily={theme.fonts.body}
        fontSize={9}
        fontWeight={700}
        textAnchor="middle"
        x={0}
        y={0.5}
      >
        +{count}
      </text>
      <foreignObject height={hit} width={hit} x={-hit / 2} y={-hit / 2}>
        <button
          aria-label={label}
          onClick={(event) => {
            event.stopPropagation();
            onSelect?.();
          }}
          style={buttonStyle}
          title={label}
          type="button"
        />
      </foreignObject>
    </g>
  );
}
