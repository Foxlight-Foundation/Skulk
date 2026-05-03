import { useEffect, useState } from 'react';

/** Snapshot of a node's Tailscale connectivity state as returned by GET /v1/connectivity/tailscale.
 *  FastAPI serializes FrozenModel fields via jsonable_encoder with by_alias=True,
 *  so snake_case Python field names arrive as camelCase JSON keys.
 */
export interface TailscaleStatus {
  running: boolean;
  selfIp: string | null;
  hostname: string | null;
  dnsName: string | null;
  tailnet: string | null;
  version: string | null;
}

/** Discriminated union result from useTailscaleStatus. */
export type TailscaleStatusResult =
  | { status: 'loading' }
  | { status: 'error' }
  | { status: 'ok'; data: TailscaleStatus };

/**
 * Fetches Tailscale status from the local node's API once on mount.
 *
 * @param nodeId - Optional cluster node ID. When provided, the request is
 *   proxied through the local API to the target node via ?node_id=<id>.
 *   Omit to query the local node directly.
 * @returns A discriminated union: loading → error → ok with data.
 */
export function useTailscaleStatus(nodeId?: string): TailscaleStatusResult {
  const [result, setResult] = useState<TailscaleStatusResult>({ status: 'loading' });

  useEffect(() => {
    setResult({ status: 'loading' });
    const url = nodeId
      ? `/v1/connectivity/tailscale?node_id=${encodeURIComponent(nodeId)}`
      : '/v1/connectivity/tailscale';
    fetch(url)
      .then(r => (r.ok ? (r.json() as Promise<TailscaleStatus>) : Promise.reject(r.status)))
      .then(data => setResult({ status: 'ok', data }))
      .catch(() => setResult({ status: 'error' }));
  }, [nodeId]);

  return result;
}
