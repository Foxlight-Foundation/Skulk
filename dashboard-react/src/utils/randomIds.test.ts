import { describe, expect, it } from 'vitest';
import { randomHex32 } from './randomIds';

const hex32 = /^[a-f0-9]{32}$/;

describe('randomHex32', () => {
  it('uses randomUUID where the page is a secure context', () => {
    const secure: Partial<Crypto> = {
      randomUUID: () => '0123abcd-4567-4890-abcd-ef0123456789',
    };
    expect(randomHex32(secure)).toBe('0123abcd45674890abcdef0123456789');
  });

  it('works without randomUUID, as on a plain-HTTP LAN dashboard', () => {
    const lanOnly: Partial<Crypto> = {
      getRandomValues: <T extends ArrayBufferView | null>(array: T): T => {
        if (array instanceof Uint8Array) array.fill(0x0f);
        return array;
      },
    };
    expect(randomHex32(lanOnly)).toBe('0f'.repeat(16));
  });

  it('still returns distinct API-valid ids without Web Crypto at all', () => {
    const first = randomHex32(null);
    const second = randomHex32(null);
    expect(first).toMatch(hex32);
    expect(second).toMatch(hex32);
    expect(first).not.toBe(second);
  });
});
