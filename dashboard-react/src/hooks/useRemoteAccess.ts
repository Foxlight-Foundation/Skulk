import { useEffect, useState } from 'react';

/** Local LAN access details as returned by GET /v1/connectivity/remote-access. */
export interface LocalAccess {
  ip: string | null;
  port: number;
  url: string | null;
}

/** Tailscale overlay access details as returned by GET /v1/connectivity/remote-access. */
export interface TailscaleAccess {
  running: boolean;
  ip: string | null;
  dnsName: string | null;
  port: number;
  url: string | null;
}

/** Aggregated remote access info as returned by GET /v1/connectivity/remote-access. */
export interface RemoteAccessInfo {
  local: LocalAccess;
  tailscale: TailscaleAccess;
  preferredUrl: string | null;
  operatorUrl: string | null;
}

/** Discriminated union result from useRemoteAccess. */
export type RemoteAccessResult =
  | { status: 'loading' }
  | { status: 'error' }
  | { status: 'ok'; data: RemoteAccessInfo };

/**
 * Fetches remote access info from the local node's API once on mount.
 *
 * @returns A discriminated union: loading → error → ok with data.
 */
export function useRemoteAccess(): RemoteAccessResult {
  const [result, setResult] = useState<RemoteAccessResult>({ status: 'loading' });

  useEffect(() => {
    fetch('/v1/connectivity/remote-access')
      .then(r => (r.ok ? (r.json() as Promise<RemoteAccessInfo>) : Promise.reject(r.status)))
      .then(data => setResult({ status: 'ok', data }))
      .catch(() => setResult({ status: 'error' }));
  }, []);

  return result;
}
