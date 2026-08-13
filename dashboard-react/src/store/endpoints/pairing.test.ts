import { describe, expect, it } from 'vitest';

import { pairingInvitationQueryErrorMessage } from './pairing';

describe('pairingInvitationQueryErrorMessage', () => {
  it('surfaces safe API guidance verbatim', () => {
    expect(
      pairingInvitationQueryErrorMessage({
        status: 403,
        data: {
          detail:
            'Pairing invitations require the configured gateway over Tailscale or localhost.',
        },
      }),
    ).toBe('Pairing invitations require the configured gateway over Tailscale or localhost.');
  });

  it('provides a complete recovery path for transport and unknown failures', () => {
    const transportFailure = pairingInvitationQueryErrorMessage({
      status: 'FETCH_ERROR',
      error: 'Load failed',
    });
    expect(transportFailure).toContain('Skulk could not load invitation history: Load failed.');
    expect(transportFailure).toContain('MagicDNS name or Tailscale IP');
    expect(transportFailure).toContain('ordinary LAN access');

    const unknownFailure = pairingInvitationQueryErrorMessage(undefined);
    expect(unknownFailure).toContain('configured operator gateway');
    expect(unknownFailure).toContain('Public relay');
  });
});
