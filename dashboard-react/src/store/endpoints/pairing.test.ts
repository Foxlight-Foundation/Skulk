import { describe, expect, it } from 'vitest';

import { pairingInvitationQueryErrorDetail } from './pairing';

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
