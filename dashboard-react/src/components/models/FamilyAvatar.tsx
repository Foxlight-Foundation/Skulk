import styled from 'styled-components';

/** Neutral family identity tile used by the specific Find Models row design. */
export interface FamilyAvatarProps {
  /** Family or author name used for the readable monogram. */
  name: string;
  /** Optional lineage names used to identify familiar model families. */
  markCandidates?: readonly string[];
  /** Family-colour ring for standalone identity specimens; result rows stay neutral. */
  ringColor?: string;
}

const familyMonograms: ReadonlyArray<readonly [RegExp, string]> = [
  [/minimax/i, 'MH'], [/qwen/i, 'Q3'], [/gemma/i, 'GM'], [/llama/i, 'LL'],
  [/gpt-oss|openai/i, 'OS'], [/deepseek/i, 'DS'],
];

function monogram(name: string, candidates: readonly string[]): string {
  for (const candidate of [name, ...candidates]) {
    const family = familyMonograms.find(([pattern]) => pattern.test(candidate));
    if (family) return family[1];
  }
  const words = name.split(/[\s_/-]+/).filter(Boolean);
  if (!words.length) return '?';
  return (words.length > 1 ? words[0][0] + words[1][0] : words[0].slice(0, 2)).toUpperCase();
}

const Tile = styled.span<{ $ringColor?: string }>`
  display: inline-flex; align-items: center; justify-content: center;
  width: 44px; height: 44px; flex-shrink: 0; border-radius: 10px;
  background: ${({ theme }) => theme.colors.selected};
  border: ${({ $ringColor, theme }) => $ringColor ? `2px solid ${$ringColor}` : `1px solid ${theme.colors.border}`};
  color: ${({ theme }) => theme.colors.body};
  font: 700 13px ${({ theme }) => theme.fonts.mono};
`;

/** Render the model monogram; the specific result-row design takes precedence over general identity specimens. */
export function FamilyAvatar({ name, markCandidates = [], ringColor }: FamilyAvatarProps) {
  return <Tile aria-hidden title={name} $ringColor={ringColor}>{monogram(name, markCandidates)}</Tile>;
}
