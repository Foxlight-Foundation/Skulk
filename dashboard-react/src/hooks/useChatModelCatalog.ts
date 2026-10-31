import { useEffect, useState } from 'react';
import type { ModelInfo } from '../types/models';

/** Cached catalog shared by native chat and the Skulk conversation selector. */
export interface ChatModelCatalog {
  models: Record<string, ModelInfo>;
  loaded: boolean;
}

/** Keep one owner-scoped catalog alive while either chat presentation is mounted. */
export function useChatModelCatalog(provided?: ChatModelCatalog): ChatModelCatalog {
  const [catalog, setCatalog] = useState<ChatModelCatalog>({ models: {}, loaded: false });
  const shared = provided !== undefined;
  useEffect(() => {
    if (shared) return;
    let disposed = false;
    const refresh = async () => {
      try {
        const response = await fetch('/models');
        if (!response.ok) return;
        const value = await response.json() as { data?: ModelInfo[] };
        const models = Object.fromEntries((value.data ?? []).filter(model => model.id).map(model => [model.id, model]));
        if (!disposed) setCatalog({ models, loaded: true });
      } catch { /* Retain the last catalog; submission rechecks provider expiry. */ }
    };
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 15000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [shared]);
  return provided ?? catalog;
}

/** Return delegated model choices without inventing native mounted instances. */
export function capabilityChatChoices(models: Record<string, ModelInfo>): Array<{ modelId: string; name: string; capability: true }> {
  return Object.values(models)
    .filter(model => model.inference_capability && model.inference_capability.expires_at * 1000 > Date.now())
    .map(model => ({ modelId: model.id, name: model.name ?? model.id, capability: true }));
}
