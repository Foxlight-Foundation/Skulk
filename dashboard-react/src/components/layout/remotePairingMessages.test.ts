import { describe, expect, it } from 'vitest';

import {
  remotePairingFailure,
  remotePairingFailureMessage,
  remotePairingStateLabel,
  type Translate,
} from './remotePairingMessages';

const t: Translate = (_key, fallback, parameters) =>
  fallback.replace(/\{(\w+)\}/g, (placeholder, name: string) =>
    parameters && name in parameters ? String(parameters[name]) : placeholder,
  );

const guidance = 'Open this node through localhost or Tailscale.';

describe('remotePairingFailure', () => {
  it('reads the stable code, detail, and delay from an RTK error', () => {
    expect(
      remotePairingFailure({
        status: 503,
        data: { code: 'registration_paused', detail: 'paused', retryAfterSeconds: 60 },
      }),
    ).toEqual({ status: 503, code: 'registration_paused', detail: 'paused', retryAfterSeconds: 60 });
  });

  it('tolerates transport failures and malformed bodies', () => {
    expect(remotePairingFailure({ status: 'FETCH_ERROR', error: 'Load failed' })).toEqual({
      status: 0,
      code: null,
      detail: null,
      retryAfterSeconds: null,
    });
    expect(remotePairingFailure(undefined).code).toBeNull();
    expect(remotePairingFailure({ status: 502, data: 'oops' }).code).toBeNull();
  });
});

describe('remotePairingFailureMessage', () => {
  it('explains known codes, including the relay-requested delay', () => {
    expect(
      remotePairingFailureMessage(
        { status: 503, code: 'registration_paused', detail: null, retryAfterSeconds: 60 },
        t,
        { authorityGuidance: guidance },
      ),
    ).toBe('The relay is not accepting new registrations right now. Try again in 60 seconds.');
    expect(
      remotePairingFailureMessage(
        { status: 409, code: 'managed_elsewhere', detail: null, retryAfterSeconds: null },
        t,
        { authorityGuidance: guidance, managedOnName: 'Kitchen Mac' },
      ),
    ).toContain('managed on Kitchen Mac');
  });

  it('uses access guidance for 403 and the node detail for unknown codes', () => {
    expect(
      remotePairingFailureMessage(
        { status: 403, code: null, detail: 'forbidden', retryAfterSeconds: null },
        t,
        { authorityGuidance: guidance },
      ),
    ).toBe(guidance);
    expect(
      remotePairingFailureMessage(
        { status: 500, code: 'something_new', detail: 'Node explanation.', retryAfterSeconds: null },
        t,
        { authorityGuidance: guidance },
      ),
    ).toBe('Node explanation.');
  });
});

describe('remotePairingStateLabel', () => {
  it('labels relay states and leaves neutral states unlabeled', () => {
    expect(remotePairingStateLabel('connected', t)).toBe('Connected to the relay');
    expect(remotePairingStateLabel('revoked', t)).toBe('Relay refused this cluster');
    expect(remotePairingStateLabel('not_set_up', t)).toBeNull();
    expect(remotePairingStateLabel('managed_elsewhere', t)).toBeNull();
  });
});
