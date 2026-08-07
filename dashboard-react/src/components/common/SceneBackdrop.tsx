import styled, { useTheme } from 'styled-components';
import type { Theme } from '../../theme';

/**
 * Brand scene behind the app: the valley at night shared with foxlight.ai
 * and the operator app. Two variants, from the design system:
 *
 * - `full`: edge-to-edge painting behind a sparse screen (the cluster view).
 * - `crown`: the top of the sky only, fading out beneath the header, so
 *   dense screens keep the night's warmth without a photograph competing
 *   with columns of identifiers.
 *
 * Renders only when the active palette declares a scene, so it is a token
 * decision, not a theme branch.
 */
export interface SceneBackdropProps {
  variant?: 'full' | 'crown';
}

const FullLayer = styled.div<{ $image: string }>`
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image: ${({ $image }) => $image};
  background-size: cover;
  background-position: center 65%;
`;

const CrownLayer = styled.div<{ $image: string }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 42vh;
  z-index: 0;
  pointer-events: none;
  background-image: ${({ $image }) => $image};
  background-size: cover;
  background-position: center top;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 0.75) 45%, transparent 100%);
`;

const Scrim = styled.div<{ $gradient: string }>`
  position: absolute;
  inset: 0;
  background: ${({ $gradient }) => $gradient};
`;

/** Render the scene layer, or nothing when the palette declares none. */
export function SceneBackdrop({ variant = 'full' }: SceneBackdropProps) {
  const theme = useTheme() as Theme;
  if (theme.colors.scene === 'none') return null;
  if (variant === 'crown') {
    return (
      <CrownLayer aria-hidden $image={theme.colors.scene}>
        <Scrim $gradient={theme.colors.sceneScrim} />
      </CrownLayer>
    );
  }
  return (
    <FullLayer aria-hidden $image={theme.colors.scene}>
      <Scrim $gradient={theme.colors.sceneScrim} />
    </FullLayer>
  );
}
