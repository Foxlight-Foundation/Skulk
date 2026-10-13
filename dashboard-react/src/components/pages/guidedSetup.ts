import type { SetupAction, SetupActions } from '../../store/endpoints/plugins';

/** What a plugin asks of its guided setup, read from its setup action's form. */
export interface GuidedSetupHint {
  /** Credentials the card asks for before the form, in order. */
  credentials: string[];
  /** Whether the plugin should be turned on once the setup succeeds. */
  turnOn: boolean;
}

/** The schema keyword a plugin sets on the one setup action a newcomer should run. */
export const GUIDED_KEYWORD = 'x-skulk-guided';

/**
 * Read a setup action's guided-setup hint, or null when it is an ordinary action.
 *
 * The hint is advisory: a malformed one is ignored rather than trusted, so a
 * plugin's mistake falls back to the ordinary setup actions.
 */
export function guidedHint(action: SetupAction): GuidedSetupHint | null {
  const raw = action.parametersSchema[GUIDED_KEYWORD];
  if (raw === true) return { credentials: [], turnOn: false };
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null;
  const record = raw as Record<string, unknown>;
  const credentials = Array.isArray(record.credentials)
    ? record.credentials.filter((item): item is string => typeof item === 'string' && item.length > 0 && item.length <= 128).slice(0, 8)
    : [];
  return { credentials, turnOn: record.turn_on === true };
}

/** The first offered guided action and its hint, or null when there is none. */
export function guidedAction(actions: SetupActions | undefined): { action: SetupAction; hint: GuidedSetupHint } | null {
  for (const action of actions?.actions ?? []) {
    const hint = guidedHint(action);
    if (hint) return { action, hint };
  }
  return null;
}
