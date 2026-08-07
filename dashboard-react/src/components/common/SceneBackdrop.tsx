import styled, { useTheme } from 'styled-components';
import type { Theme } from '../../theme';

/**
 * Night sky crowning the app: the star field from the brand valley painting
 * (shared with foxlight.ai and the operator app), pinned to the top of the
 * viewport and dissolving to nothing on the way down. It sits above the
 * palette's background gradient and beneath everything interactive, so the
 * header floats over open sky while dense content keeps a plain ground.
 *
 * Renders only when the active palette declares a scene, so it is a token
 * decision, not a theme branch.
 */
const SkyLayer = styled.div<{ $image: string }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 50vh;
  z-index: 0;
  pointer-events: none;
  background-image: ${({ $image }) => $image};
  background-size: cover;
  background-position: center top;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.61) 0%, rgba(0, 0, 0, 0.29) 55%, transparent 100%);
`;

const Scrim = styled.div<{ $gradient: string }>`
  position: absolute;
  inset: 0;
  background: ${({ $gradient }) => $gradient};
`;

/** Render the night-sky layer, or nothing when the palette declares none. */
export function SceneBackdrop() {
  const theme = useTheme() as Theme;
  if (theme.colors.scene === 'none') return null;
  return (
    <SkyLayer aria-hidden $image={theme.colors.scene}>
      {theme.colors.sceneScrim !== 'none' && <Scrim $gradient={theme.colors.sceneScrim} />}
    </SkyLayer>
  );
}
