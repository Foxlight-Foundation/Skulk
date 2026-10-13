import { skipToken } from '@reduxjs/toolkit/query';
import { useMemo } from 'react';
import { useGetBurstProvidersQuery } from '../store/endpoints/burst';

/** How often the provider list is read again; providers change only on install or removal. */
const PROVIDERS_POLL_MS = 60_000;

/**
 * Keys (`pluginId/nodeId`) of the burst providers installed on the host
 * serving this dashboard. The burst routes answer for the local API host only,
 * so a provider on another host never appears here.
 *
 * @param enabled Whether to read at all; false skips the request, for a
 *   dashboard whose host reports no capability nodes.
 */
export function useLocalBurstProviderKeys(enabled: boolean): ReadonlySet<string> {
  const providers = useGetBurstProvidersQuery(enabled ? undefined : skipToken, {
    pollingInterval: PROVIDERS_POLL_MS,
    skipPollingIfUnfocused: true,
  });
  const listed = providers.currentData?.providers;
  return useMemo(
    () => new Set((listed ?? []).map((provider) => `${provider.pluginId}/${provider.nodeId}`)),
    [listed],
  );
}
