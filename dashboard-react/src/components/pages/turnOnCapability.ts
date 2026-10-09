import type { ConfigurationMutation, NodeAddress, NodeConfiguration } from '../../store/endpoints/plugins';
import type { Unwrappable } from './catalogJourney';

/** The host requests turning capability nodes on takes: a fresh settings read, then the fenced enable. */
export interface TurnOnRequests {
  readConfiguration: (address: NodeAddress) => Unwrappable<NodeConfiguration>;
  configure: (mutation: ConfigurationMutation) => Unwrappable<{ configuration: NodeConfiguration }>;
}

/** The node the host would not turn on, and the host's own sentence for why when it sent one. */
export interface TurnOnRefusal {
  nodeId: string;
  detail: string | null;
  /** The HTTP status of a decided refusal; null when the request may not have landed. */
  status: number | null;
}

/** What turning on did: the nodes the host turned on, and the first one it refused. */
export interface TurnOnOutcome {
  enabled: string[];
  refusal: TurnOnRefusal | null;
}

/**
 * Turn a capability's nodes on, one at a time, each against a fresh read.
 *
 * Enabling sends no settings: it is fenced only by the revision and schema
 * digest just read, so it works whatever the dashboard can render of the
 * node's settings form. The plugin runs its setup checks again before it
 * starts, so a refusal here means a setting, credential or check is not
 * ready; the first refusal stops the rest. A node already on is left alone.
 *
 * This is the whole "Turn on" step, kept apart from any screen so a later
 * caller can run it straight after an install that needs no settings.
 */
export async function turnOnNodes(
  requests: TurnOnRequests,
  pluginId: string,
  nodeIds: readonly string[],
  refusalDetail: (error: unknown) => string | null,
  refusedStatus: (error: unknown) => number | null,
): Promise<TurnOnOutcome> {
  const enabled: string[] = [];
  for (const nodeId of nodeIds) {
    try {
      const current = await requests.readConfiguration({ pluginId, nodeId }).unwrap();
      if (current.enabled) continue;
      await requests.configure({
        pluginId, nodeId, operation: 'enable', expectedRevision: current.revision, expectedSchemaDigest: current.schemaDigest,
      }).unwrap();
      enabled.push(nodeId);
    } catch (error) {
      return { enabled, refusal: { nodeId, detail: refusalDetail(error), status: refusedStatus(error) } };
    }
  }
  return { enabled, refusal: null };
}

/**
 * Start one node again that is on but stopped: send "enable" against a fresh read.
 *
 * A node stops at "needs settings" when a check fails while it starts, and it
 * stays stopped after the cause goes away (a check that failed while Skulk was
 * updating, for one) even though its checks pass when run again. The plugin
 * restarts a node on every settings change, including enabling one that is
 * already on, and runs its setup checks first, so this is the restart. It is
 * fenced like turning on, sends no settings, and returns the host's refusal
 * when a check still fails. A node the fresh read finds turned off is left
 * off: turning it on is the owner's choice, made with Turn on, and the
 * enable here adopts the latest revision, so the fence alone would not
 * protect a newer decision to turn it off.
 */
export async function startNodeAgain(
  requests: TurnOnRequests,
  address: NodeAddress,
  refusalDetail: (error: unknown) => string | null,
  refusedStatus: (error: unknown) => number | null,
): Promise<StartAgainOutcome> {
  try {
    const current = await requests.readConfiguration(address).unwrap();
    if (!current.enabled) return { kind: 'off' };
    await requests.configure({
      ...address, operation: 'enable', expectedRevision: current.revision, expectedSchemaDigest: current.schemaDigest,
    }).unwrap();
    return { kind: 'started' };
  } catch (error) {
    return { kind: 'refused', refusal: { nodeId: address.nodeId, detail: refusalDetail(error), status: refusedStatus(error) } };
  }
}

/** What starting a node again did: started it, found it turned off and left it, or the host refused. */
export type StartAgainOutcome =
  | { kind: 'started' }
  | { kind: 'off' }
  | { kind: 'refused'; refusal: TurnOnRefusal };
