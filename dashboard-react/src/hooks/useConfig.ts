import { useCallback, useState } from 'react';

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

export interface FullConfig {
  model_store?: StoreConfig;
  inference?: InferenceConfig;
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
