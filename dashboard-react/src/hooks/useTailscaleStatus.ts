import { useEffect, useState } from 'react';

/** Snapshot of the local node's Tailscale connectivity state as returned by GET /v1/connectivity/tailscale.
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

/**
 * Fetches Tailscale status from the local node's API once on mount.
 * Returns null while loading or if the request fails.
 */
export function useTailscaleStatus(): TailscaleStatus | null {
  const [status, setStatus] = useState<TailscaleStatus | null>(null);

  useEffect(() => {
    fetch('/v1/connectivity/tailscale')
      .then(r => r.ok ? r.json() as Promise<TailscaleStatus> : null)
      .then(data => setStatus(data))
      .catch(() => setStatus(null));
  }, []);

  return status;
}
