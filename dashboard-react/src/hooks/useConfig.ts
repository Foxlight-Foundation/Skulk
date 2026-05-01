import { useCallback, useRef } from 'react';
import {
  useGetConfigQuery,
  useUpdateConfigMutation,
  type FullConfig,
  type EffectiveConfig,
  type StoreConfig,
} from '../store/endpoints/config';

export type {
  StoreConfig,
  InferenceConfig,
  LoggingConfig,
  FullConfig,
  EffectiveConfig,
  ConfigResponse,
} from '../store/endpoints/config';

/** State and actions exposed by {@link useConfig}. */
export interface UseConfigReturn {
  config: StoreConfig | null;
  fullConfig: FullConfig | null;
  effective: EffectiveConfig | null;
  configPath: string | null;
  loading: boolean;
  saving: boolean;
  error: string | null;
  fetchConfig: () => Promise<void>;
  saveFullConfig: (config: FullConfig) => Promise<boolean>;
}

/**
 * Thin compatibility wrapper around the RTK Query config endpoints.
 *
 * The previous hook owned its own loading/error state and a manual fetch
 * function; the wrapper preserves that surface so existing consumers
 * (SettingsPanel) port across without changes. Under the hood the GET is a
 * cached query and the PUT is a mutation that invalidates the cache, so a
 * successful save is followed by an automatic refetch.
 */
export function useConfig(): UseConfigReturn {
  const query = useGetConfigQuery();
  const [updateConfig, mutationState] = useUpdateConfigMutation();

  // The query result object is recreated on every render, so closing over
  // `query` directly would give `fetchConfig` a fresh identity each render.
  // Any consumer that puts `fetchConfig` in a useEffect dependency array
  // (SettingsPanel does, gated on `open`) would then loop. Stash the latest
  // query in a ref and read through it so `fetchConfig` keeps a single
  // stable identity for the lifetime of the hook subscription.
  const queryRef = useRef(query);
  queryRef.current = query;
  const fetchConfig = useCallback(async () => {
    await queryRef.current.refetch();
  }, []);

  const saveFullConfig = useCallback(
    async (config: FullConfig): Promise<boolean> => {
      try {
        await updateConfig(config).unwrap();
        return true;
      } catch {
        return false;
      }
    },
    [updateConfig],
  );

  const fullConfig = query.data?.config ?? null;
  const effective = query.data?.effective ?? null;
  const configPath = query.data?.configPath ?? null;
  const config = fullConfig?.model_store ?? null;

  // Surface query errors as readable strings; mutation errors flow through
  // the saveFullConfig boolean return rather than the shared `error` field
  // so the UI can keep showing the previously-loaded config while the user
  // retries the save.
  const error = query.isError
    ? (query.error as { error?: string; status?: number })?.error ?? 'Failed to fetch config'
    : null;

  return {
    config,
    fullConfig,
    effective,
    configPath,
    loading: query.isFetching,
    saving: mutationState.isLoading,
    error,
    fetchConfig,
    saveFullConfig,
  };
}
