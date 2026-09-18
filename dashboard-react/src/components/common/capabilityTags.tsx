import styled from 'styled-components';
import type { Theme } from '../../theme';

/**
 * Shared capability-tag dialect for model surfaces (store table, Find Models
 * modal). One source of truth for the tint palette so every surface renders
 * the same chip for the same capability.
 */
export interface CapabilityTagColors {
  color: string;
  bg: string;
  border: string;
}

/** Build the capability-tag tint palette for the active theme. */
export function buildTagColors(theme: Theme): Record<string, CapabilityTagColors> {
  const tinted = Object.fromEntries(Object.entries(theme.capabilityTints).map(([name, color]) => [name, {
    color, bg: `color-mix(in srgb, ${color} 10%, transparent)`, border: `color-mix(in srgb, ${color} 30%, transparent)`,
  }]));
  return {
    ...tinted,
    thinking: { color: theme.colors.gold, bg: theme.colors.goldBg, border: theme.colors.goldBg },
    vision: { color: theme.colors.tagVision, bg: `color-mix(in srgb, ${theme.colors.tagVision} 10%, transparent)`, border: `color-mix(in srgb, ${theme.colors.tagVision} 30%, transparent)` },
    tensor: { color: theme.colors.healthy, bg: theme.colors.accentBg, border: theme.colors.accentBg },
  };
}

/** Small tinted uppercase chip naming one model capability. */
export const CapabilityTagBadge = styled.span<{ $color: string; $bg: string; $border: string }>`
  flex-shrink: 0;
  font-size: 10px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  color: ${({ $color }) => $color};
  background: ${({ $bg }) => $bg};
  border: 1px solid ${({ $border }) => $border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 0 5px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
`;
