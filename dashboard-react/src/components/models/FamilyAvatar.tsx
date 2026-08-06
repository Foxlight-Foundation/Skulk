import styled from 'styled-components';

/**
 * Rounded monogram tile identifying a model family (or author) in discovery
 * lists. Replaces the retired image-based family logos with a dependency-free
 * anchor: a deterministic hue derived from the name, rendered as a soft tint
 * that works over both palettes.
 */
export interface FamilyAvatarProps {
  /** Family or author name the tile represents. */
  name: string;
}

/** Well-known families keep stable, deliberately chosen hues. */
const PINNED_HUES: Readonly<Record<string, number>> = {
  qwen: 262,
  gemma: 214,
  llama: 24,
  'gpt-oss': 152,
  glm: 330,
  mistral: 12,
  deepseek: 200,
};

function familyHue(name: string): number {
  const pinned = PINNED_HUES[name.toLowerCase()];
  if (pinned !== undefined) return pinned;
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360;
  return h;
}

function monogram(name: string): string {
  const tokens = name.split(/[\s_-]+/).filter(Boolean);
  if (tokens.length === 0) return '?';
  if (tokens.length === 1) return tokens[0].slice(0, 1).toUpperCase();
  return (tokens[0].slice(0, 1) + tokens[1].slice(0, 1)).toUpperCase();
}

const Tile = styled.span<{ $hue: number }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  border-radius: ${({ theme }) => theme.radii.md};
  background: hsl(${({ $hue }) => $hue} 70% 50% / 0.13);
  border: 1px solid hsl(${({ $hue }) => $hue} 70% 50% / 0.22);
  color: hsl(${({ $hue }) => $hue} 60% 52%);
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: 700;
  letter-spacing: 0.5px;
`;

/** Render the monogram tile for one family/author name. */
export function FamilyAvatar({ name }: FamilyAvatarProps) {
  return <Tile aria-hidden $hue={familyHue(name)}>{monogram(name)}</Tile>;
}
