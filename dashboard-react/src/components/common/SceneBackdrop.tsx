import styled, { useTheme } from 'styled-components';
import type { Theme } from '../../theme';

/**
 * Full-bleed brand scene behind the whole app: the valley painting shared
 * with foxlight.ai and the operator app, under the design system's
 * structural scrim (sunk at the top so the header survives the starfield,
 * buried at the base so content keeps its ground). Renders only when the
 * active palette declares a scene, so it is a token decision, not a theme
 * branch; palettes without a scene keep the abstract mesh instead.
 */
const Layer = styled.div<{ $image: string }>`
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image: ${({ $image }) => $image};
  background-size: cover;
  background-position: center 65%;
`;

const Scrim = styled.div<{ $gradient: string }>`
  position: absolute;
  inset: 0;
  background: ${({ $gradient }) => $gradient};
`;

/** Render the scene layer, or nothing when the palette declares none. */
export function SceneBackdrop() {
  const theme = useTheme() as Theme;
  if (theme.colors.scene === 'none') return null;
  return (
    <Layer aria-hidden $image={theme.colors.scene}>
      <Scrim $gradient={theme.colors.sceneScrim} />
    </Layer>
  );
}
