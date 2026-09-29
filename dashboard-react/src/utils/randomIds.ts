/**
 * A 128-bit random identifier as 32 lowercase hex characters, the shape the
 * plugin and runtime APIs require for operation and plugin IDs.
 *
 * `crypto.randomUUID` exists only in secure contexts (HTTPS or localhost), so
 * a dashboard opened over plain HTTP from another machine on the LAN does not
 * have it. `crypto.getRandomValues` is available there, and is used instead.
 * The final fallback keeps the dashboard usable without Web Crypto at all:
 * these values identify idempotent operations and are never secrets.
 *
 * @param cryptoLike Web Crypto implementation; injectable for tests.
 * @returns 32 lowercase hex characters.
 */
export function randomHex32(cryptoLike: Partial<Crypto> | null = globalThis.crypto ?? null): string {
  if (cryptoLike && typeof cryptoLike.randomUUID === 'function') {
    // eslint-disable-next-line no-restricted-syntax -- guarded secure-context path
    return cryptoLike.randomUUID().replaceAll('-', '');
  }
  const bytes = new Uint8Array(16);
  if (cryptoLike && typeof cryptoLike.getRandomValues === 'function') {
    cryptoLike.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}
