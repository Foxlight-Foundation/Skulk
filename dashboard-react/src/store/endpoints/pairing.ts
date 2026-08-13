import { apiSlice } from '../api';

const dashboardPairingHeaders = {
  'X-Skulk-Dashboard': 'pairing-v1',
} as const;

const pairingAuthorityGuidance =
  "Open Settings on the configured operator gateway through Tailscale using its MagicDNS name or Tailscale IP, or through localhost. Public relay and ordinary LAN access cannot manage pairing invitations.";

/** Safe lifecycle state for a reusable operator pairing invitation. */
export type PairingInvitationState =
  | 'active'
  | 'expired'
  | 'exhausted'
  | 'attempt-limit'
  | 'revoked';

/** Secret-free invitation status returned by the direct dashboard API. */
export interface PairingInvitationSummary {
  invitationId: string;
  createdAt: string;
  expiresAt: string;
  successfulPairings: number;
  maxPairings: number;
  activeAttempts: number;
  totalAttempts: number;
  state: PairingInvitationState;
}

/** Bounded operator choices accepted when creating a pairing invitation. */
export interface CreatePairingInvitationRequest {
  validForSeconds: number;
  maxPairings: number;
}

/** One-time secret response retained only by the mounted pairing component. */
export interface CreatedPairingInvitation {
  invitation: PairingInvitationSummary;
  pairingCode: string;
}

/** Error returned by the direct dashboard pairing authority. */
export class PairingInvitationRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'PairingInvitationRequestError';
    this.status = status;
  }
}

/**
 * Create an invitation without placing its bearer pairing code in Redux.
 *
 * The safe invitation list uses RTK Query, but mutation responses intentionally
 * stay in mounted component memory because the pairing code contains secret
 * relay admission material and must disappear when the UI resets.
 */
export async function createPairingInvitation(
  request: CreatePairingInvitationRequest,
): Promise<CreatedPairingInvitation> {
  const response = await fetch('/v1/auth/pairing-invitations', {
    method: 'POST',
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {
      ...dashboardPairingHeaders,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new PairingInvitationRequestError(
      response.status,
      await pairingErrorMessage(response),
    );
  }
  return parseCreatedPairingInvitation(await response.json());
}

const pairingApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getPairingInvitations: build.query<PairingInvitationSummary[], void>({
      query: () => ({
        url: '/v1/auth/pairing-invitations',
        headers: dashboardPairingHeaders,
      }),
      transformResponse: (value: unknown) => {
        if (!Array.isArray(value)) {
          throw new PairingInvitationRequestError(502, 'Skulk returned an invalid invitation list.');
        }
        return value.map(parsePairingInvitationSummary);
      },
      providesTags: ['PairingInvitations'],
    }),
    revokePairingInvitation: build.mutation<void, string>({
      query: (invitationId) => ({
        url: `/v1/auth/pairing-invitations/${encodeURIComponent(invitationId)}`,
        method: 'DELETE',
        headers: dashboardPairingHeaders,
      }),
      invalidatesTags: ['PairingInvitations'],
    }),
  }),
});

export const {
  useGetPairingInvitationsQuery,
  useRevokePairingInvitationMutation,
} = pairingApi;

async function pairingErrorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (isObject(body) && typeof body.detail === 'string') return body.detail;
  } catch {
    // The status-based fallback below is intentionally payload-independent.
  }
  return response.status === 403 || response.status === 409
    ? pairingAuthorityGuidance
    : `Skulk could not manage pairing invitations. ${pairingAuthorityGuidance}`;
}

/** Return actionable dashboard copy for an RTK Query invitation-list failure. */
export function pairingInvitationQueryErrorMessage(value: unknown): string {
  if (isObject(value)) {
    const data = value.data;
    if (isObject(data) && typeof data.detail === 'string') return data.detail;
    if (typeof value.error === 'string' && value.error.trim()) {
      return `Skulk could not load invitation history: ${value.error}. ${pairingAuthorityGuidance}`;
    }
  }
  return `Skulk could not load invitation history. ${pairingAuthorityGuidance}`;
}

function parseCreatedPairingInvitation(value: unknown): CreatedPairingInvitation {
  if (!isObject(value) || typeof value.pairingCode !== 'string') {
    throw new PairingInvitationRequestError(502, 'Skulk returned an invalid pairing invitation.');
  }
  return {
    invitation: parsePairingInvitationSummary(value.invitation),
    pairingCode: value.pairingCode,
  };
}

function parsePairingInvitationSummary(value: unknown): PairingInvitationSummary {
  if (
    !isObject(value) ||
    typeof value.invitationId !== 'string' ||
    typeof value.createdAt !== 'string' ||
    typeof value.expiresAt !== 'string' ||
    typeof value.successfulPairings !== 'number' ||
    typeof value.maxPairings !== 'number' ||
    typeof value.activeAttempts !== 'number' ||
    typeof value.totalAttempts !== 'number' ||
    !isPairingInvitationState(value.state)
  ) {
    throw new PairingInvitationRequestError(502, 'Skulk returned invalid invitation status.');
  }
  return {
    invitationId: value.invitationId,
    createdAt: value.createdAt,
    expiresAt: value.expiresAt,
    successfulPairings: value.successfulPairings,
    maxPairings: value.maxPairings,
    activeAttempts: value.activeAttempts,
    totalAttempts: value.totalAttempts,
    state: value.state,
  };
}

function isPairingInvitationState(value: unknown): value is PairingInvitationState {
  return (
    value === 'active' ||
    value === 'expired' ||
    value === 'exhausted' ||
    value === 'attempt-limit' ||
    value === 'revoked'
  );
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
