import styled from 'styled-components';

/**
 * Mono-spaced quantization tile shared by the catalog rows, the expanded
 * size panel, and Hugging Face results. The quantization is part of the
 * downloadable artifact's identity, so any row that represents exactly one
 * artifact must surface it.
 */
export const QuantBadge = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  background: ${({ theme }) => theme.colors.surfaceHover};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 7px;
  min-width: 42px;
  text-align: center;
  flex-shrink: 0;
`;

/* GGUF quant codes (Q4_K_M, IQ4_XS, Q8_0, F16...) as they appear in file
 * and repo names. */
const GGUF_QUANT = /(?:^|[-_./ (])(I?Q\d(?:_[K01])?(?:_(?:XXS|XS|S|M|L|XL|NL))?|F16|F32|BF16)(?=$|[-_./ )])/i;
/* Suffix-style quant markers used by MLX / safetensors repo names
 * (-4bit, -bf16, -nvfp4, -MXFP4, -int8...). */
const SUFFIX_QUANT = /(?:^|[-_./ (])((?:\d+)[-_]?bit|bf16|fp16|fp8|fp4|nvfp4|mxfp4|int[48])(?=$|[-_./ )])/i;

function normalize(raw: string): string {
  const lower = raw.toLowerCase().replace(/[-_]?bit$/, 'bit');
  // GGUF codes read as codes; keep their canonical uppercase form.
  if (/^i?q\d/i.test(raw) || /^(f16|f32)$/i.test(raw)) return raw.toUpperCase();
  return lower;
}

/**
 * Artifact-format label (GGUF, MLX) telling the user which runtime family
 * the artifact targets, derived from the matched file, repo id, or tags.
 * Returns null when the format cannot be determined.
 */
export function deriveFormatLabel(
  repoId: string,
  matchedFile?: string | null,
  tags?: readonly string[],
): string | null {
  if (matchedFile && matchedFile.toLowerCase().endsWith('.gguf')) return 'GGUF';
  if (/gguf/i.test(repoId)) return 'GGUF';
  if (/^mlx-community\//i.test(repoId) || /(^|[-_./])mlx([-_./]|$)/i.test(repoId)) return 'MLX';
  const lowered = (tags ?? []).map((tag) => tag.toLowerCase());
  if (lowered.includes('gguf')) return 'GGUF';
  if (lowered.includes('mlx')) return 'MLX';
  return null;
}

/**
 * Best-effort quantization label for a Hugging Face result, from the matched
 * GGUF filename first (exact artifact), then the repo name, then tags.
 * Returns null when nothing trustworthy is found.
 */
export function deriveQuantLabel(
  repoId: string,
  matchedFile?: string | null,
  tags?: readonly string[],
): string | null {
  if (matchedFile) {
    const m = GGUF_QUANT.exec(matchedFile);
    if (m) return normalize(m[1]);
  }
  const fromId = GGUF_QUANT.exec(repoId) ?? SUFFIX_QUANT.exec(repoId);
  if (fromId) return normalize(fromId[1]);
  for (const tag of tags ?? []) {
    const m = /^(\d+)-bit$/.exec(tag);
    if (m) return `${m[1]}bit`;
  }
  return null;
}
