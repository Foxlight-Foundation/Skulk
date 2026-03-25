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

export interface ConfigResponse {
  config: {
    model_store: StoreConfig;
  };
  configPath: string;
  fileExists: boolean;
}

export interface UseConfigReturn {
  config: StoreConfig | null;
  configPath: string | null;
  loading: boolean;
  saving: boolean;
  error: string | null;
  fetchConfig: () => Promise<void>;
  saveConfig: (config: StoreConfig) => Promise<boolean>;
}

export function useConfig(): UseConfigReturn {
  const [config, setConfig] = useState<StoreConfig | null>(null);
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
      setConfig(data.config.model_store);
      setConfigPath(data.configPath);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch config');
    } finally {
      setLoading(false);
    }
  }, []);

  const saveConfig = useCallback(async (updated: StoreConfig): Promise<boolean> => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch('/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_store: updated }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setConfig(updated);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save config');
      return false;
    } finally {
      setSaving(false);
    }
  }, []);

  return { config, configPath, loading, saving, error, fetchConfig, saveConfig };
}
