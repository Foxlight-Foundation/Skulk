/**
 * Validate an external URL for safe use in anchor `href` attributes.
 *
 * Returns the normalized URL string only for explicit `http:` or `https:`
 * origins; rejects `javascript:`, `data:`, `vbscript:`, relative paths, and
 * malformed strings so untrusted capability metadata cannot inject executable
 * links.
 */
export function safeExternalHttpUrl(value: string | null | undefined): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (trimmed.length === 0) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      return null;
    }
    return parsed.href;
  } catch {
    return null;
  }
}
