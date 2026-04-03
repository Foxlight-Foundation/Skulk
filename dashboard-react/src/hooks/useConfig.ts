import { useCallback, useState } from 'react';

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

/** Inference-related config surfaced to the dashboard. */
export interface InferenceConfig {
  kv_cache_backend: string;
}

/** Logging / observability config synced across the cluster. */
export interface LoggingConfig {
  enabled: boolean;
  ingest_url: string;
  grafana_url: string;
}

/** Full editable config document as read from or written to `/config`. */
export interface FullConfig {
  model_store?: StoreConfig;
  inference?: InferenceConfig;
  logging?: LoggingConfig;
  hf_token?: string;
}

/** Effective runtime values derived from config and environment. */
export interface EffectiveConfig {
  kv_cache_backend: string;
  has_hf_token?: boolean;
}

/** Response shape returned by the dashboard config endpoint. */
export interface ConfigResponse {
  config: FullConfig;
  configPath: string;
  fileExists: boolean;
  effective?: EffectiveConfig;
}

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

/** Load and save cluster config through the Skulk dashboard config API. */
export function useConfig(): UseConfigReturn {
  const [fullConfig, setFullConfig] = useState<FullConfig | null>(null);
  const [effective, setEffective] = useState<EffectiveConfig | null>(null);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/config');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ConfigResponse = await res.json();
      setFullConfig(data.config);
      setEffective(data.effective ?? null);
      setConfigPath(data.configPath);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch config');
    } finally {
      setLoading(false);
    }
  }, []);

  const saveFullConfig = useCallback(async (updated: FullConfig): Promise<boolean> => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch('/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updated }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setFullConfig(updated);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save config');
      return false;
    } finally {
      setSaving(false);
    }
  }, []);

  const config = fullConfig?.model_store ?? null;

  return { config, fullConfig, effective, configPath, loading, saving, error, fetchConfig, saveFullConfig };
}
