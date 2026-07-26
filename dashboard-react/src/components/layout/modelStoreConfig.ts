import type {
  PersistedStoreConfig,
  StoreConfig,
} from '../../hooks/useConfig';

export const DEFAULT_MODEL_STORE_PORT = 12415;

function defaultStoreConfig(enabled: boolean): StoreConfig {
  return {
    enabled,
    store_host: '',
    store_http_host: '',
    store_port: DEFAULT_MODEL_STORE_PORT,
    store_path: '',
    download: {
      allow_hf_fallback: true,
    },
    staging: {
      enabled: true,
      node_cache_path: '~/.skulk/staging',
      cleanup_on_deactivate: true,
    },
  };
}

/**
 * Materialize the server defaults omitted from a persisted model-store mapping.
 *
 * The presence of the section enables the model store by default in Skulk's
 * Pydantic model. A completely absent section remains disabled in the UI.
 */
export function normalizeStoreConfig(
  config: PersistedStoreConfig | null | undefined,
): StoreConfig {
  const defaults = defaultStoreConfig(config != null);
  if (config == null) return defaults;
  return {
    ...defaults,
    ...config,
    download: {
      ...defaults.download,
      ...config.download,
    },
    staging: {
      ...defaults.staging,
      ...config.staging,
    },
  };
}
