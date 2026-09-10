import type { CapabilityNodeAction, CapabilityNodeSummary } from '../../types/capabilityNodes';

/**
 * Resolved flyout entries for one capability node. The registry turns the
 * summary's surfaces and manifest actions into a flat, ordered list the
 * flyout and the capability panel both render, so the two never disagree
 * about what a node offers.
 */
export type CapabilityActionItem =
  | {
      kind: 'open-link';
      id: string;
      title: string;
      url: string;
      /** False when the surface reports it is not answering yet. */
      ready: boolean;
    }
  | {
      kind: 'call';
      id: string;
      title: string;
      capabilityId: string;
      payload: Record<string, unknown>;
      /** Descriptor calls run only against the host the dashboard is on. */
      enabled: boolean;
    }
  | { kind: 'details'; id: string; title: string }
  | { kind: 'manage-on-host'; id: string; title: string; hostName: string };

/** Where the dashboard sits relative to the node's host. */
export interface CapabilityActionContext {
  /** True when the dashboard is served by the node's host. */
  isLocalHost: boolean;
  /** Friendly name of the host, for the "manage on host" hint. */
  hostName: string;
  /** Translates a key with an English fallback. */
  t: (key: string, fallback: string, params?: Record<string, string | number>) => string;
}

function resolveAction(
  action: CapabilityNodeAction,
  summary: CapabilityNodeSummary,
  context: CapabilityActionContext,
): CapabilityActionItem | null {
  if (action.kind === 'surface') {
    const surface = summary.surfaces.find((candidate) => candidate.surfaceId === action.surfaceId);
    if (!surface) return null;
    return {
      kind: 'open-link',
      id: `action:${action.actionId}`,
      title: action.title,
      url: surface.url,
      ready: surface.ready,
    };
  }
  if (action.kind === 'link') {
    if (!action.url) return null;
    return { kind: 'open-link', id: `action:${action.actionId}`, title: action.title, url: action.url, ready: true };
  }
  if (!action.capabilityId) return null;
  return {
    kind: 'call',
    id: `action:${action.actionId}`,
    title: action.title,
    capabilityId: action.capabilityId,
    payload: action.payload ?? {},
    enabled: context.isLocalHost,
  };
}

/**
 * Builds the flyout's action list: every link surface, then the manifest
 * actions (surface actions that name a missing surface are dropped), then
 * the details entry, and finally "manage on host" when the dashboard is
 * connected to a different node than the one running the capability.
 */
export function buildCapabilityActions(
  summary: CapabilityNodeSummary,
  context: CapabilityActionContext,
): CapabilityActionItem[] {
  const items: CapabilityActionItem[] = [];
  const seenUrls = new Set<string>();
  for (const surface of summary.surfaces) {
    seenUrls.add(surface.url);
    items.push({
      kind: 'open-link',
      id: `surface:${surface.surfaceId}`,
      title: surface.title,
      url: surface.url,
      ready: surface.ready,
    });
  }
  for (const action of summary.actions) {
    const resolved = resolveAction(action, summary, context);
    if (!resolved) continue;
    // A manifest action that merely re-opens an already listed surface adds
    // nothing to the flyout; keep the surface entry and skip the duplicate.
    if (resolved.kind === 'open-link' && seenUrls.has(resolved.url)) continue;
    if (resolved.kind === 'open-link') seenUrls.add(resolved.url);
    items.push(resolved);
  }
  items.push({
    kind: 'details',
    id: 'details',
    title: context.t('topology.capability.details', 'Details'),
  });
  if (!context.isLocalHost) {
    items.push({
      kind: 'manage-on-host',
      id: 'manage-on-host',
      title: context.t('topology.capability.manageOnHost', 'Manage on {host}', {
        host: context.hostName,
      }),
      hostName: context.hostName,
    });
  }
  return items;
}

/**
 * Random call id that also works on plain-HTTP LAN dashboards, where the
 * page is not a secure context and `crypto.randomUUID` does not exist. Falls
 * back to `getRandomValues`, then to a clock-and-counter id so a call can
 * always be made. Exposed with an injectable crypto for tests.
 */
export function generateCallId(cryptoLike: Partial<Crypto> | null = globalThis.crypto ?? null): string {
  if (cryptoLike && typeof cryptoLike.randomUUID === 'function') return cryptoLike.randomUUID();
  if (cryptoLike && typeof cryptoLike.getRandomValues === 'function') {
    const bytes = cryptoLike.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  callIdFallbackCounter += 1;
  return `call-${Date.now().toString(16)}-${callIdFallbackCounter.toString(16)}`;
}

let callIdFallbackCounter = 0;

interface DescriptorListing {
  capabilities?: Array<{ id?: string; version?: string }>;
  revisions?: Record<string, string>;
}

interface CapabilityCallResult {
  ok?: boolean;
  result?: unknown;
  error?: { code?: string; message?: string } | null;
}

/**
 * Runs a descriptor action against the local host through
 * `POST /v1/capabilities/call`. The action names a capability by id (or
 * `id@version`); the dashboard resolves the exact version and descriptor
 * revision from `GET /v1/capabilities` first so the call cannot drift from
 * what the host currently serves. Returns the typed result envelope.
 */
export async function runDescriptorAction(
  localNodeId: string,
  capabilityId: string,
  payload: Record<string, unknown>,
  fetchImpl: typeof fetch = fetch,
): Promise<CapabilityCallResult> {
  const listingResponse = await fetchImpl('/v1/capabilities');
  if (!listingResponse.ok) {
    throw new Error(`capability listing failed with HTTP ${listingResponse.status}`);
  }
  const listing = (await listingResponse.json()) as DescriptorListing;
  const [wantedId, wantedVersion] = capabilityId.split('@', 2);
  const descriptor = (listing.capabilities ?? []).find(
    (candidate) =>
      candidate.id === wantedId && (wantedVersion === undefined || candidate.version === wantedVersion),
  );
  if (!descriptor?.id || !descriptor.version) {
    throw new Error(`capability ${capabilityId} is not served by this host`);
  }
  const qualifiedId = `${descriptor.id}@${descriptor.version}`;
  const revision = listing.revisions?.[qualifiedId];
  if (!revision) throw new Error(`no descriptor revision for ${qualifiedId}`);
  const response = await fetchImpl('/v1/capabilities/call', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      call_id: generateCallId(),
      capability_id: descriptor.id,
      version: descriptor.version,
      descriptor_revision: revision,
      caller_node: localNodeId,
      target_node: localNodeId,
      payload,
    }),
  });
  if (!response.ok) throw new Error(`capability call failed with HTTP ${response.status}`);
  return (await response.json()) as CapabilityCallResult;
}
