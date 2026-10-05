import { describe, expect, it } from 'vitest';

import { parseRemotePairingStatus } from './remotePairing';

describe('parseRemotePairingStatus', () => {
  it('accepts a complete status and fills absent optional fields', () => {
    expect(
      parseRemotePairingStatus({ state: 'not_set_up', registrationAvailable: true }),
    ).toEqual({
      state: 'not_set_up',
      registrationAvailable: true,
      registrationBlockedReason: null,
      lastFailure: null,
      retryAfterSeconds: null,
      managedOnNodeId: null,
      managedOnNodeName: null,
      relayHost: null,
    });
  });

  it('rejects malformed statuses so they never enable pairing controls', () => {
    for (const malformed of [
      null,
      [],
      { state: 'unknown', registrationAvailable: true },
      { state: 'connected' },
      { state: 'connected', registrationAvailable: true, registrationBlockedReason: 'nope' },
      { state: 'connected', registrationAvailable: true, retryAfterSeconds: -1 },
      { state: 'connected', registrationAvailable: true, relayHost: 42 },
    ]) {
      expect(() => parseRemotePairingStatus(malformed)).toThrow('invalid phone pairing status');
    }
  });
});
