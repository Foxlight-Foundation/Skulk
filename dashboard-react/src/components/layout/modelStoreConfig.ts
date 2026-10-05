import type {
  PersistedStoreConfig,
  StoreConfig,
} from '../../hooks/useConfig';
import type { ModelStoreDefaults } from '../../store/endpoints/config';

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
/**
 * Fill an enabled store's blank host and path with the node's defaults.
 *
 * Mirrors the server, which applies the same defaults to a blank save: the
 * node the dashboard talks to hosts the store at its default path. Values the
 * operator typed are kept, and a disabled store or a node that reports no
 * defaults is returned unchanged.
 */
export function withStoreDefaults(
  config: StoreConfig,
  defaults: ModelStoreDefaults | null | undefined,
): StoreConfig {
  if (!config.enabled || !defaults) return config;
  const blankHost = !config.store_host.trim();
  return {
    ...config,
    store_host: blankHost ? defaults.store_host : config.store_host,
    store_http_host:
      blankHost && !config.store_http_host.trim()
        ? defaults.store_http_host
        : config.store_http_host,
    store_path: config.store_path.trim() ? config.store_path : defaults.store_path,
  };
}

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
