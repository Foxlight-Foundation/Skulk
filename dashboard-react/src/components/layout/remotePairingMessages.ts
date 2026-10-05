import type {
  RegistrationBlockedReason,
  RemotePairingState,
} from '../../store/endpoints/remotePairing';

/** Tolgee lookup with an English fallback and optional parameters. */
export type Translate = (
  key: string,
  fallback: string,
  parameters?: Record<string, string | number>,
) => string;

/** Stable failure returned when phone pairing cannot be turned on or off. */
export interface RemotePairingFailure {
  /** HTTP status, or 0 when the request did not complete. */
  status: number;
  /** Stable code: a registration failure code or a node-side refusal. */
  code: string | null;
  /** English explanation from the node, used when the code is unknown. */
  detail: string | null;
  retryAfterSeconds: number | null;
}

/** Read the stable failure from an RTK Query mutation error. */
export function remotePairingFailure(error: unknown): RemotePairingFailure {
  if (!isObject(error)) {
    return { status: 0, code: null, detail: null, retryAfterSeconds: null };
  }
  const status = typeof error.status === 'number' ? error.status : 0;
  const data = error.data;
  if (!isObject(data)) {
    return { status, code: null, detail: null, retryAfterSeconds: null };
  }
  return {
    status,
    code: typeof data.code === 'string' ? data.code : null,
    detail: typeof data.detail === 'string' ? data.detail : null,
    retryAfterSeconds:
      typeof data.retryAfterSeconds === 'number' ? data.retryAfterSeconds : null,
  };
}

/**
 * Explain why phone pairing could not be turned on or off.
 *
 * Known codes get a localized message; anything else falls back to the node's
 * English detail, then to a generic message.
 */
export function remotePairingFailureMessage(
  failure: RemotePairingFailure,
  t: Translate,
  options: { managedOnName?: string | null; authorityGuidance: string },
): string {
  if (failure.status === 403) return options.authorityGuidance;
  const seconds = failure.retryAfterSeconds;
  switch (failure.code) {
    case 'disabled':
    case 'offline':
    case 'not_configured':
      return blockedReasonMessage(failure.code, t);
    case 'rate_limited':
      return seconds !== null
        ? t(
            'settings.pairing.remote.failure.rateLimitedFor',
            'The relay is limiting new registrations from this network. Try again in {seconds} seconds.',
            { seconds },
          )
        : t(
            'settings.pairing.remote.failure.rateLimited',
            'The relay is limiting new registrations from this network. Try again later.',
          );
    case 'registration_paused':
      return seconds !== null
        ? t(
            'settings.pairing.remote.failure.pausedFor',
            'The relay is not accepting new registrations right now. Try again in {seconds} seconds.',
            { seconds },
          )
        : t(
            'settings.pairing.remote.failure.paused',
            'The relay is not accepting new registrations right now. Try again later.',
          );
    case 'capacity_exhausted':
      return t(
        'settings.pairing.remote.failure.capacity',
        'The relay has no room for new clusters right now. Try again later.',
      );
    case 'relay_busy':
      return t(
        'settings.pairing.remote.failure.busyRelay',
        'The relay is busy. Try again in a moment.',
      );
    case 'unreachable':
      return t(
        'settings.pairing.remote.failure.unreachable',
        "This node could not reach the relay. Check this machine's internet connection, then try again.",
      );
    case 'registration_unsupported':
      return t(
        'settings.pairing.remote.failure.unsupported',
        'This relay does not accept registrations. Check connectivity.relay.registration_url in skulk.yaml.',
      );
    case 'invalid_request':
    case 'already_registered':
    case 'invalid_response':
      return t(
        'settings.pairing.remote.failure.relayRefused',
        'The relay could not register this node. Try again; if it keeps failing, check the relay setting in skulk.yaml.',
      );
    case 'managed_elsewhere':
      return managedElsewhereMessage(options.managedOnName ?? null, t);
    case 'device_limit':
      return t(
        'settings.pairing.remote.failure.deviceLimit',
        'This cluster already has the most paired devices it allows. Revoke a device before pairing another.',
      );
    case 'busy':
      return t(
        'settings.pairing.remote.failure.inProgress',
        'Phone pairing is being turned on. Wait a moment, then try again.',
      );
    case 'pairing_state_unavailable':
      return t(
        'settings.pairing.remote.failure.stateUnavailable',
        "This node's pairing state is unavailable right now. Try again shortly.",
      );
    default:
      return (
        failure.detail ??
        t('settings.pairing.remote.failure.generic', 'Skulk could not set up phone pairing.')
      );
  }
}

/** Explain why this node cannot register a relay route. */
export function blockedReasonMessage(reason: RegistrationBlockedReason, t: Translate): string {
  switch (reason) {
    case 'disabled':
      return t(
        'settings.pairing.remote.blocked.disabled',
        'Phone pairing through the relay is turned off on this node (connectivity.relay.enabled in skulk.yaml).',
      );
    case 'offline':
      return t(
        'settings.pairing.remote.blocked.offline',
        'This node runs offline, so it cannot reach the relay to pair a phone.',
      );
    case 'not_configured':
      return t(
        'settings.pairing.remote.blocked.notConfigured',
        'No relay is configured for this node. Set connectivity.relay.registration_url in skulk.yaml to pair a phone.',
      );
  }
}

/** Name the node that manages phone pairing for this cluster. */
export function managedElsewhereMessage(nodeName: string | null, t: Translate): string {
  return nodeName
    ? t(
        'settings.pairing.remote.managedOn',
        "Phone pairing for this cluster is managed on {node}. Open that node's dashboard to pair a phone.",
        { node: nodeName },
      )
    : t(
        'settings.pairing.remote.managedElsewhere',
        "Phone pairing for this cluster is managed on another node. Open that node's dashboard to pair a phone.",
      );
}

/** Short label for a relay connection state, or null when none is shown. */
export function remotePairingStateLabel(state: RemotePairingState, t: Translate): string | null {
  switch (state) {
    case 'registering':
      return t('settings.pairing.remote.state.registering', 'Registering with the relay…');
    case 'connecting':
      return t('settings.pairing.remote.state.connecting', 'Connecting to the relay…');
    case 'connected':
      return t('settings.pairing.remote.state.connected', 'Connected to the relay');
    case 'relay_unreachable':
      return t('settings.pairing.remote.state.unreachable', 'Relay unreachable');
    case 'revoked':
      return t('settings.pairing.remote.state.revoked', 'Relay refused this cluster');
    case 'not_set_up':
    case 'managed_elsewhere':
      return null;
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
