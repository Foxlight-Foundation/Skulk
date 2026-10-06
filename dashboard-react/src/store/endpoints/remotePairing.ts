import { apiSlice } from '../api';

const dashboardPairingHeaders = {
  'X-Skulk-Dashboard': 'pairing-v1',
} as const;

/**
 * Phone-pairing state reported by the node.
 *
 * `not_set_up`: no relay route; Pair a phone registers one. `registering`: a
 * registration is in flight. `connecting`: a route is stored and its relay
 * session is starting. `connected`: the relay holds a live session from this
 * node. `relay_unreachable`: a route is stored but the node has not reached the
 * relay for a while. `revoked`: the relay permanently refused the route.
 * `managed_elsewhere`: another node holds the cluster's relay route.
 */
export type RemotePairingState =
  | 'not_set_up'
  | 'registering'
  | 'connecting'
  | 'connected'
  | 'relay_unreachable'
  | 'revoked'
  | 'managed_elsewhere';

/** Why this node cannot register a relay route. */
export type RegistrationBlockedReason = 'disabled' | 'offline' | 'not_configured';

/** Phone-pairing status shown in Devices & pairing. */
export interface RemotePairingStatus {
  state: RemotePairingState;
  /** Whether this node can register a relay route. */
  registrationAvailable: boolean;
  registrationBlockedReason: RegistrationBlockedReason | null;
  /** Stable code of the last failed registration, if any. */
  lastFailure: string | null;
  /** Seconds the relay asked to wait before registering again. */
  retryAfterSeconds: number | null;
  managedOnNodeId: string | null;
  managedOnNodeName: string | null;
  /** Host of the relay this node uses or would register with. */
  relayHost: string | null;
}

const remotePairingApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getRemotePairingStatus: build.query<RemotePairingStatus, void>({
      query: () => ({
        url: '/v1/auth/remote-pairing',
        headers: dashboardPairingHeaders,
      }),
      transformResponse: (value: unknown) => parseRemotePairingStatus(value),
      providesTags: ['RemotePairing'],
    }),
    enableRemotePairing: build.mutation<RemotePairingStatus, void>({
      query: () => ({
        url: '/v1/auth/remote-pairing',
        method: 'POST',
        headers: dashboardPairingHeaders,
      }),
      transformResponse: (value: unknown) => parseRemotePairingStatus(value),
      invalidatesTags: ['RemotePairing'],
    }),
    disableRemotePairing: build.mutation<RemotePairingStatus, void>({
      query: () => ({
        url: '/v1/auth/remote-pairing',
        method: 'DELETE',
        headers: dashboardPairingHeaders,
      }),
      transformResponse: (value: unknown) => parseRemotePairingStatus(value),
      // Turning pairing off revokes the route's invitations.
      invalidatesTags: ['RemotePairing', 'PairingInvitations'],
    }),
  }),
});

export const {
  useGetRemotePairingStatusQuery,
  useEnableRemotePairingMutation,
  useDisableRemotePairingMutation,
} = remotePairingApi;

/** Validate a status body so a malformed answer never enables pairing controls. */
export function parseRemotePairingStatus(value: unknown): RemotePairingStatus {
  if (
    !isObject(value) ||
    !isRemotePairingState(value.state) ||
    typeof value.registrationAvailable !== 'boolean' ||
    !isOptionalBlockedReason(value.registrationBlockedReason) ||
    !isOptionalString(value.lastFailure) ||
    !isOptionalCount(value.retryAfterSeconds) ||
    !isOptionalString(value.managedOnNodeId) ||
    !isOptionalString(value.managedOnNodeName) ||
    !isOptionalString(value.relayHost)
  ) {
    throw new Error('Skulk returned an invalid phone pairing status.');
  }
  return {
    state: value.state,
    registrationAvailable: value.registrationAvailable,
    registrationBlockedReason: value.registrationBlockedReason ?? null,
    lastFailure: value.lastFailure ?? null,
    retryAfterSeconds: value.retryAfterSeconds ?? null,
    managedOnNodeId: value.managedOnNodeId ?? null,
    managedOnNodeName: value.managedOnNodeName ?? null,
    relayHost: value.relayHost ?? null,
  };
}

function isRemotePairingState(value: unknown): value is RemotePairingState {
  return (
    value === 'not_set_up' ||
    value === 'registering' ||
    value === 'connecting' ||
    value === 'connected' ||
    value === 'relay_unreachable' ||
    value === 'revoked' ||
    value === 'managed_elsewhere'
  );
}

function isOptionalBlockedReason(
  value: unknown,
): value is RegistrationBlockedReason | null | undefined {
  return (
    value === null ||
    value === undefined ||
    value === 'disabled' ||
    value === 'offline' ||
    value === 'not_configured'
  );
}

function isOptionalString(value: unknown): value is string | null | undefined {
  return value === null || value === undefined || typeof value === 'string';
}

function isOptionalCount(value: unknown): value is number | null | undefined {
  return (
    value === null ||
    value === undefined ||
    (typeof value === 'number' && Number.isInteger(value) && value >= 0)
  );
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
