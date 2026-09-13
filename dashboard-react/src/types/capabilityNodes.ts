/**
 * Capability nodes: managed extension children a host runs, drawn as
 * satellites of that host in the topology. The shapes mirror the
 * `capabilityNodes` projection of `GET /state` (camelCase wire form of the
 * Python `CapabilityNodeSummary`), plus the dashboard-side health mapping.
 */

/** Owner-reported lifecycle status of a capability node. */
export type CapabilityNodeStatus =
  | 'installed'
  | 'starting'
  | 'ready'
  | 'degraded'
  | 'disabled'
  | 'configuration_invalid'
  | 'failed';

/** What a flyout action does. */
export type CapabilityNodeActionKind = 'surface' | 'descriptor' | 'link';

/** One user-facing surface a capability node exposes. Only links today. */
export interface CapabilityNodeSurface {
  surfaceId: string;
  title: string;
  kind: 'link';
  url: string;
  ready: boolean;
}

/** One manifest-declared top-level action for the flyout. */
export interface CapabilityNodeAction {
  actionId: string;
  title: string;
  kind: CapabilityNodeActionKind;
  surfaceId?: string | null;
  capabilityId?: string | null;
  payload?: Record<string, unknown> | null;
  url?: string | null;
}

/** What a host says about one capability node it runs. */
export interface CapabilityNodeSummary {
  pluginId: string;
  nodeId: string;
  bundleId: string;
  version: string;
  title?: string | null;
  status: CapabilityNodeStatus;
  ownerAvailable: boolean;
  surfaces: CapabilityNodeSurface[];
  actions: CapabilityNodeAction[];
  operationsActive: number;
  /** Local receipt time (ISO 8601) of the host's last reading. */
  observedAt: string;
}

/**
 * Health of a satellite. `muted` covers nodes that are installed but not
 * running, disabled, or whose host has stopped publishing.
 */
export type CapabilityNodeHealthLevel = 'ok' | 'warn' | 'error' | 'muted';

/**
 * A host republishes its summaries every thirty seconds; a reading older
 * than this is treated as stale and the satellite is muted rather than
 * shown with a status nobody is vouching for.
 */
export const CAPABILITY_NODE_STALE_AFTER_MS = 90_000;

/** Host-local identity of a capability node: plugin plus node identifier. */
export function capabilityNodeKey(summary: Pick<CapabilityNodeSummary, 'pluginId' | 'nodeId'>): string {
  return `${summary.pluginId}/${summary.nodeId}`;
}

/** Display name for a capability node, falling back to its identifier. */
export function capabilityNodeTitle(summary: CapabilityNodeSummary): string {
  const title = summary.title?.trim();
  return title && title.length > 0 ? title : summary.nodeId;
}

/**
 * Maps owner-reported status and freshness onto a satellite health level.
 * Deliberately separate from the host's `nodeHealth`: a failed capability
 * node must never make its host look unhealthy, and vice versa.
 */
export function capabilityNodeHealth(
  summary: CapabilityNodeSummary,
  nowMs: number = Date.now(),
  staleAfterMs: number = CAPABILITY_NODE_STALE_AFTER_MS,
): CapabilityNodeHealthLevel {
  const observed = Date.parse(summary.observedAt);
  if (Number.isFinite(observed) && nowMs - observed > staleAfterMs) return 'muted';
  if (summary.status === 'installed' || summary.status === 'disabled') return 'muted';
  if (summary.status === 'failed' || !summary.ownerAvailable) return 'error';
  if (
    summary.status === 'starting' ||
    summary.status === 'degraded' ||
    summary.status === 'configuration_invalid'
  ) {
    return 'warn';
  }
  return 'ok';
}

/** Satellites are hidden for disabled nodes; everything else is drawn. */
export function isCapabilityNodeVisible(summary: CapabilityNodeSummary): boolean {
  return summary.status !== 'disabled';
}
