import { describe, expect, it } from 'vitest';

import { pairingInvitationQueryErrorDetail, parsePairingCapacity } from './pairing';

describe('pairingInvitationQueryErrorDetail', () => {
  it('surfaces safe API guidance verbatim', () => {
    expect(
      pairingInvitationQueryErrorDetail({
        status: 403,
        data: {
          detail:
            'Pairing invitations require the configured gateway over Tailscale or localhost.',
        },
      }),
    ).toBe('Pairing invitations require the configured gateway over Tailscale or localhost.');
  });

  it('leaves transport and unknown failures to localized component guidance', () => {
    const transportFailure = pairingInvitationQueryErrorDetail({
      status: 'FETCH_ERROR',
      error: 'Load failed',
    });
    expect(transportFailure).toBeNull();
    expect(pairingInvitationQueryErrorDetail(undefined)).toBeNull();
  });
});

describe('parsePairingCapacity', () => {
  it('accepts a consistent capacity report', () => {
    expect(parsePairingCapacity({ activeDevices: 3, maximumDevices: 5, availableSlots: 2 })).toEqual({
      activeDevices: 3,
      maximumDevices: 5,
      availableSlots: 2,
    });
  });

  it('rejects malformed reports so they never enable pairing', () => {
    for (const malformed of [
      null,
      [],
      { activeDevices: 3, maximumDevices: 5 },
      { activeDevices: -1, maximumDevices: 5, availableSlots: 5 },
      { activeDevices: 1.5, maximumDevices: 5, availableSlots: 3 },
      { activeDevices: 0, maximumDevices: 0, availableSlots: 0 },
      { activeDevices: 0, maximumDevices: 5, availableSlots: 6 },
      { activeDevices: '1', maximumDevices: 5, availableSlots: 4 },
    ]) {
      expect(() => parsePairingCapacity(malformed)).toThrow('invalid pairing capacity');
    }
  });
});
