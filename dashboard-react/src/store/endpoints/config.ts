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

/**
 * Persisted model-store configuration returned by `GET /config`.
 *
 * Skulk returns the YAML mapping exactly as written, so fields with Pydantic
 * defaults can be absent even though the running server has effective values
 * for them. Consumers must normalize this shape before rendering it.
 */
export interface PersistedStoreConfig
  extends Partial<Omit<StoreConfig, 'download' | 'staging'>> {
  download?: Partial<StoreConfig['download']>;
  staging?: Partial<StoreConfig['staging']>;
}

export interface InferenceConfig {
  kv_cache_backend: string;
}

export interface LoggingConfig {
  enabled: boolean;
  ingest_url: string;
}

export interface ExperimentsConfig {
  tts_streaming: boolean;
  /** Deprecated compatibility field; realtime STT is capability-gated. */
  stt_realtime: boolean;
  speech_translation: boolean;
}

/** Consent state for opt-in field telemetry (persisted in skulk.yaml). */
export type TelemetryConsent = 'unasked' | 'enabled' | 'disabled';

/**
 * Field-telemetry settings. `consent` gates anonymous performance +
 * reliability samples; `diagnostics_consent` is the SEPARATE gate for
 * future crash diagnostics. `install_id` is the anonymous rate-limit key
 * and the deletion capability (rotatable).
 */
export interface TelemetryConfig {
  consent: TelemetryConsent;
  diagnostics_consent: TelemetryConsent;
  install_id: string;
  consented_at: string;
  consented_version: string;
  ingest_url: string;
}

/** Intelligent-fabric (resident steward) settings. */
export interface IntelligentFabricConfig {
  /** Master switch: the fabric keeps a hidden steward placement while on. */
  enabled: boolean;
  /** Ordered model-card preference list for the steward brain. */
  steward_models?: string[];
}

export interface FullConfig {
  model_store?: PersistedStoreConfig;
  inference?: InferenceConfig;
  logging?: LoggingConfig;
  experiments?: ExperimentsConfig;
  telemetry?: TelemetryConfig;
  intelligent_fabric?: IntelligentFabricConfig;
  hf_token?: string;
}

export interface EffectiveConfig {
  kv_cache_backend: string;
  has_hf_token?: boolean;
  /**
   * True when the node runs with SKULK_ENABLE_EXPERIMENTAL_MODE. Gates the
   * dashboard's Experiments settings section; memory-agnostic.
   */
  experimental_mode_enabled?: boolean;
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
