import type { CapabilityNodeSummary, CapabilityNodeSurface } from '../../types/capabilityNodes';
import { resolveSurfaceUrl } from '../topology/capabilityActions';

/** One capability node of a plugin, with the cluster node that hosts it. */
export interface HostedCapabilityNode {
  hostNodeId: string;
  summary: CapabilityNodeSummary;
}

/** A screen a plugin reported ready, resolved for this browser. */
export interface OpenableSurface {
  /** The surface, its URL resolved for this browser. */
  surface: CapabilityNodeSurface;
  /** Whether this browser can open it: a loopback screen opens only from a browser on its own host. */
  reachable: boolean;
}

/** What an installed capability's Browse card offers: open its screen, set it up, or manage it. */
export type InstalledCardAction =
  | { kind: 'open'; surface: CapabilityNodeSurface }
  | { kind: 'set-up' }
  | { kind: 'manage' };

/** Every capability node of one plugin the cluster reports, with its host. */
export function hostedCapabilityNodes(capabilityNodes: Record<string, CapabilityNodeSummary[]>, pluginId: string): HostedCapabilityNode[] {
  return Object.entries(capabilityNodes).flatMap(([hostNodeId, nodes]) => nodes
    .filter((node) => node.pluginId === pluginId).map((summary) => ({ hostNodeId, summary })));
}

/**
 * The screens a plugin's nodes report ready, once each, resolved for this
 * browser as the topology resolves them. The summary carries only surfaces
 * the plugin reported ready to open.
 */
export function openableSurfaces(hosted: HostedCapabilityNode[], localNodeId: string | null, dashboardHostname: string): OpenableSurface[] {
  const unique = new Map<string, OpenableSurface>();
  for (const { hostNodeId, summary } of hosted) {
    for (const surface of summary.surfaces) {
      if (surface.kind !== 'link' || !surface.url || unique.has(surface.url)) continue;
      const resolved = resolveSurfaceUrl(surface.url, { isLocalHost: hostNodeId === localNodeId, dashboardHostname });
      unique.set(surface.url, { surface: { ...surface, url: resolved.url }, reachable: resolved.reachable });
    }
  }
  return [...unique.values()];
}

/**
 * Whether a node needs the owner before it can run: turned off, waiting for
 * settings, or stopped. A node whose plugin service is not answering does not:
 * that is the service restarting (after every Skulk update, for one) or down,
 * nothing a setup step fixes, and its last status is stale until it answers.
 */
export function needsOwner(summary: CapabilityNodeSummary): boolean {
  return summary.ownerAvailable && (summary.status === 'disabled' || summary.status === 'configuration_invalid' || summary.status === 'failed');
}

/**
 * What an installed capability's Browse card offers, from what its nodes
 * report now: **Set up** when a node needs the owner, **Open** its first
 * screen this browser can open once every node is ready, and **Manage**
 * otherwise (still starting, not reported yet, or no screen to open here).
 */
export function installedCardAction(hosted: HostedCapabilityNode[], localNodeId: string | null, dashboardHostname: string): InstalledCardAction {
  if (hosted.some(({ summary }) => needsOwner(summary))) return { kind: 'set-up' };
  // A screen opens only while its plugin service answers; a silent one may be restarting.
  const ready = hosted.length > 0 && hosted.every(({ summary }) => summary.status === 'ready' && summary.ownerAvailable);
  const surface = ready ? openableSurfaces(hosted, localNodeId, dashboardHostname).find((item) => item.reachable)?.surface : undefined;
  return surface ? { kind: 'open', surface } : { kind: 'manage' };
}
