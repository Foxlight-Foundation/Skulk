import { operatorSession } from '../auth/operatorSession';

/** Server-authenticated caller; the issuer is supplied by Core, never a form. */
export interface InferencePrincipal {
  identifier: string;
  kind: 'operator' | 'client';
  issuer_node_id: string;
}

/** Exact signed implementation reviewed by the owner. Updates require reapproval. */
export interface InferenceGrant {
  plugin_id: string;
  runtime_digest: string;
  node_id: string;
  capability_id: string;
  version: string;
  descriptor_revision: string;
  behavior: 'read' | 'inference';
  processing_domain: 'local' | 'private_site' | 'external';
  io_mode: 'unary' | 'server_streaming' | 'client_streaming' | 'bidirectional';
}

/** One caller's permissions, including any already configured duration ceiling. */
export interface InferenceCallerAccess {
  principal: InferencePrincipal;
  grants: InferenceGrant[];
  maximum_seconds: number | null;
}

/** Revision-fenced host policy, independent of account connection and readiness. */
export interface InferenceAccessPolicy {
  protocol: 1;
  revision: string;
  enabled: boolean;
  callers: InferenceCallerAccess[];
}

/** Safe client metadata; credentials never appear in an inventory. */
export interface InferenceClient {
  identifier: string;
  label: string;
  expires_unix_ms: number | null;
}

/** Client-inventory revision for explicit creation and revocation. */
export interface InferenceClients {
  revision: string;
  clients: InferenceClient[];
}

/** Observed signed candidate. Observation alone supplies no execution permission. */
export interface InferenceListing {
  plugin_id: string;
  runtime_digest: string;
  target: Omit<InferenceGrant, 'plugin_id' | 'runtime_digest' | 'io_mode'>;
  descriptor: { title: string; description: string; io_mode: InferenceGrant['io_mode'] };
}

/** New bearer disclosure; keep only in component memory until dismissed. */
export interface InferenceClientCreated {
  client: InferenceClient;
  token: string;
  revision: string;
}

/** Send one management request. Mutations are never replayed or cached. */
export async function inferenceAccessRequest<T>(path: string, method = 'GET', body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await operatorSession.fetch(path, {
    method,
    headers: { 'X-Skulk-Dashboard': 'pairing-v1', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    credentials: 'omit', cache: 'no-store', redirect: 'error',
    body: body === undefined ? undefined : JSON.stringify(body), signal,
  });
  if (!response.ok) {
    if (response.status === 409) throw new Error('Access changed or a selected release is no longer current. Reload and review before saving again.');
    if (response.status === 403) throw new Error('Open this node directly through localhost or Tailscale. Paired devices cannot manage inference access.');
    throw new Error('Inference access is unavailable. Check the plugin service and reload.');
  }
  return response.json() as Promise<T>;
}

/** Complete equality key: two releases of the same capability are different grants. */
export function inferenceGrantKey(grant: InferenceGrant): string {
  return JSON.stringify([grant.plugin_id, grant.runtime_digest, grant.node_id, grant.capability_id, grant.version, grant.descriptor_revision, grant.behavior, grant.processing_domain, grant.io_mode]);
}

/** Project an observed candidate into the exact permission the backend revalidates. */
export function inferenceListingGrant(listing: InferenceListing): InferenceGrant {
  return { ...listing.target, plugin_id: listing.plugin_id, runtime_digest: listing.runtime_digest, io_mode: listing.descriptor.io_mode };
}

/** Select one implementation while replacing a prior release of that same capability. */
export function selectInferenceGrant(grants: InferenceGrant[], grant: InferenceGrant, enabled: boolean): InferenceGrant[] {
  const kept = grants.filter(existing => enabled
    ? !(existing.plugin_id === grant.plugin_id && existing.capability_id === grant.capability_id && existing.version === grant.version)
    : inferenceGrantKey(existing) !== inferenceGrantKey(grant));
  return enabled ? [...kept, grant] : kept;
}
