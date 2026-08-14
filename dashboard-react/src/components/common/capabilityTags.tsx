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
  return {
    optiq: { color: '#a78bfa', bg: 'rgba(167, 139, 250, 0.1)', border: 'rgba(167, 139, 250, 0.3)' },
    thinking: { color: theme.colors.info, bg: theme.colors.infoBg, border: theme.colors.infoBg },
    // Palette-aware (10px text over its own faint tint needs a deep value
    // on white); the tints derive from the token so the pair cannot drift.
    vision: {
      color: theme.colors.tagVision,
      bg: `color-mix(in srgb, ${theme.colors.tagVision} 10%, transparent)`,
      border: `color-mix(in srgb, ${theme.colors.tagVision} 30%, transparent)`,
    },
    tensor: { color: theme.colors.healthy, bg: theme.colors.accentBg, border: theme.colors.accentBg },
    embedding: { color: '#f472b6', bg: 'rgba(244, 114, 182, 0.1)', border: 'rgba(244, 114, 182, 0.3)' },
    tts: { color: '#38bdf8', bg: 'rgba(56, 189, 248, 0.1)', border: 'rgba(56, 189, 248, 0.3)' },
    stt: { color: '#34d399', bg: 'rgba(52, 211, 153, 0.1)', border: 'rgba(52, 211, 153, 0.3)' },
    code: { color: '#818cf8', bg: 'rgba(129, 140, 248, 0.1)', border: 'rgba(129, 140, 248, 0.3)' },
    image_gen: { color: '#fb923c', bg: 'rgba(251, 146, 60, 0.1)', border: 'rgba(251, 146, 60, 0.3)' },
    image_edit: { color: '#fb923c', bg: 'rgba(251, 146, 60, 0.1)', border: 'rgba(251, 146, 60, 0.3)' },
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
