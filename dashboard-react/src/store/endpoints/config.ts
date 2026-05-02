import { apiSlice } from '../api';

/** Shared model-store configuration returned by the Skulk config API. */
export interface StoreConfig {
  enabled: boolean;
  store_host: string;
  store_http_host: string;
  store_port: number;
  store_path: string;
  download: {
    allow_hf_fallback: boolean;
  };
  staging: {
    enabled: boolean;
    node_cache_path: string;
    cleanup_on_deactivate: boolean;
  };
}

export interface InferenceConfig {
  kv_cache_backend: string;
}

export interface LoggingConfig {
  enabled: boolean;
  ingest_url: string;
}

export interface FullConfig {
  model_store?: StoreConfig;
  inference?: InferenceConfig;
  logging?: LoggingConfig;
  hf_token?: string;
}

export interface EffectiveConfig {
  kv_cache_backend: string;
  has_hf_token?: boolean;
}

export interface ConfigResponse {
  config: FullConfig;
  configPath: string;
  fileExists: boolean;
  effective?: EffectiveConfig;
}

/**
 * Cluster-config endpoints.
 *
 * `updateConfig` invalidates the `Config` tag so the GET refetches the
 * authoritative server view after a successful save — operators see exactly
 * what was persisted, not an optimistic local copy.
 */
export const configApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getConfig: build.query<ConfigResponse, void>({
      query: () => '/config',
      providesTags: ['Config'],
    }),
    updateConfig: build.mutation<void, FullConfig>({
      query: (config) => ({
        url: '/config',
        method: 'PUT',
        body: { config },
      }),
      invalidatesTags: ['Config'],
    }),
  }),
});

export const { useGetConfigQuery, useUpdateConfigMutation } = configApi;
