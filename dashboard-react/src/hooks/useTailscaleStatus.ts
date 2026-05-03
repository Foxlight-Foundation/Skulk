import { useEffect, useState } from 'react';

/** Snapshot of the local node's Tailscale connectivity state as returned by GET /v1/connectivity/tailscale. */
export interface TailscaleStatus {
  running: boolean;
  self_ip: string | null;
  hostname: string | null;
  dns_name: string | null;
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
