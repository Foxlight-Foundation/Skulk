import { describe, expect, it } from 'vitest';

import type { OperatorDevice } from '../../store/endpoints/devices';
import { isRefreshExpired } from './deviceState';

function device(overrides: Partial<OperatorDevice>): OperatorDevice {
  return {
    deviceId: '00000000-0000-4000-8000-000000000001',
    name: 'Phone',
    pairedAt: '2026-10-01T12:00:00.000Z',
    refreshExpiresAt: '2026-10-31T12:00:00.000Z',
    state: 'active',
    current: false,
    ...overrides,
  };
}

describe('isRefreshExpired', () => {
  const now = Date.parse('2026-10-05T12:00:00.000Z');

  it('marks active devices whose refresh credential lapsed', () => {
    expect(isRefreshExpired(device({ refreshExpiresAt: '2026-10-05T11:59:59.000Z' }), now)).toBe(true);
    expect(isRefreshExpired(device({ refreshExpiresAt: null }), now)).toBe(true);
  });

  it('leaves live and revoked devices alone', () => {
    expect(isRefreshExpired(device({}), now)).toBe(false);
    expect(isRefreshExpired(device({ state: 'revoked', refreshExpiresAt: null }), now)).toBe(false);
  });
});
